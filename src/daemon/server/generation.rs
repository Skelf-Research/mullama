use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::super::models::{LoadedModel, RequestGuard};
use super::super::protocol::{generate_completion_id, ResponseFormat, StreamChunk};
use super::prompt::find_stop_in_recent_window;
use super::Daemon;
use crate::{MullamaError, SamplerParams};

impl Daemon {
    pub async fn generate_text(
        &self,
        loaded: &LoadedModel,
        prompt: &str,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
        response_format: Option<&ResponseFormat>,
    ) -> Result<(String, u32, u32), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        let grammar_gbnf = match response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                match crate::structured_output::JsonSchemaConverter::convert(&json_schema.schema) {
                    Ok(grammar) => Some(grammar.to_gbnf()),
                    Err(e) => {
                        tracing::warn!("Failed to convert JSON schema to grammar: {}", e);
                        None
                    }
                }
            }
            Some(ResponseFormat::JsonObject) => match crate::grammar::presets::json() {
                Ok(grammar) => Some(grammar.to_gbnf()),
                Err(e) => {
                    tracing::warn!("Failed to create JSON grammar: {}", e);
                    None
                }
            },
            Some(ResponseFormat::Text) | None => None,
        };

        let mut context = loaded.acquire_context().await;
        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let (generated, completion_tokens) = tokio::task::block_in_place(|| {
            context.kv_cache_clear();

            let mut sampler = sampler_params.build_chain(model.clone())?;

            if let Some(gbnf) = &grammar_gbnf {
                let grammar_sampler =
                    crate::sampling::Sampler::grammar(model.clone(), gbnf, "root")?;
                sampler.add(grammar_sampler);
            }

            context.decode(&tokens)?;

            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                let next_token = sampler.sample(&mut context, -1);

                if model.vocab_is_eog(next_token) {
                    break;
                }

                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    let previous_len = generated.len();
                    generated.push_str(&text);

                    if let Some(pos) = find_stop_in_recent_window(
                        &generated,
                        previous_len,
                        &stop_sequences,
                        max_stop_len,
                    ) {
                        generated.truncate(pos);
                        return Ok((generated, completion_tokens));
                    }
                }

                sampler.accept(next_token);
                context.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Generate text with streaming - yields tokens as they're generated
    pub async fn generate_text_streaming(
        &self,
        loaded: Arc<LoadedModel>,
        prompt: String,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(&prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let models_ref = self.models.clone();
        let cancellations = Arc::clone(&self.cancellations);
        let active_requests = Arc::clone(&self.active_requests);
        let request_id_cleanup = request_id.clone();
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;

            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();

                let mut sampler = sampler_params.build_chain(model.clone())?;

                context.decode(&tokens)?;

                let mut generated = String::new();
                let mut index = 0u32;
                let mut sent_len = 0usize;
                let hold_back = max_stop_len.saturating_sub(1);
                let mut tokens_generated = 0u32;
                let mut last_token_id = 0i32;

                for _ in 0..max_tokens {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let next_token = sampler.sample(&mut context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        tokens_generated += 1;
                        last_token_id = next_token;
                        let previous_len = generated.len();
                        generated.push_str(&text);

                        if let Some(pos) = find_stop_in_recent_window(
                            &generated,
                            previous_len,
                            &stop_sequences,
                            max_stop_len,
                        ) {
                            if pos > sent_len {
                                let partial = &generated[sent_len..pos];
                                let chunk = StreamChunk {
                                    request_id: request_id_arc.clone(),
                                    index,
                                    delta: partial.to_string(),
                                    token_id: next_token,
                                    thinking: None,
                                    tool_calls: None,
                                };
                                let _ = tx.blocking_send(chunk);
                            }
                            return Ok::<_, MullamaError>(tokens_generated);
                        }

                        let mut flush_end = generated.len().saturating_sub(hold_back);
                        while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                            flush_end -= 1;
                        }
                        if flush_end > sent_len {
                            let chunk = StreamChunk {
                                request_id: request_id_arc.clone(),
                                index,
                                delta: generated[sent_len..flush_end].to_string(),
                                token_id: next_token,
                                thinking: None,
                                tool_calls: None,
                            };

                            if tx.blocking_send(chunk).is_err() {
                                break;
                            }

                            sent_len = flush_end;
                            index += 1;
                        }
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                if sent_len < generated.len() {
                    let chunk = StreamChunk {
                        request_id: request_id_arc.clone(),
                        index,
                        delta: generated[sent_len..].to_string(),
                        token_id: last_token_id,
                        thinking: None,
                        tool_calls: None,
                    };
                    let _ = tx.blocking_send(chunk);
                }

                Ok::<_, MullamaError>(tokens_generated)
            });

            if let Ok(tokens_generated) = result {
                models_ref.add_tokens(tokens_generated as u64);
            }

            cancellations.remove(&request_id_cleanup);
            active_requests.fetch_sub(1, Ordering::Relaxed);
        });

        Ok((rx, prompt_tokens, request_id))
    }

    /// Generate text with vision input
    #[cfg(feature = "multimodal")]
    pub(super) async fn generate_vision_text(
        &self,
        loaded: &LoadedModel,
        prompt: &str,
        bitmaps: &[crate::Bitmap],
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
    ) -> Result<(String, u32, u32), MullamaError> {
        let mut ctx_guard = loaded.acquire_context().await;
        let mtmd_ref = loaded.mtmd_context.as_ref().ok_or_else(|| {
            MullamaError::MultimodalError("No multimodal context available".to_string())
        })?;
        let mut mtmd_guard = mtmd_ref.write().await;

        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let (generated, prompt_tokens, completion_tokens) = tokio::task::block_in_place(|| {
            ctx_guard.kv_cache_clear();

            let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();
            let chunks = mtmd_guard.tokenize(prompt, &bitmap_refs)?;

            let n_batch = 512;
            let n_past = mtmd_guard.eval_chunks(&mut ctx_guard, &chunks, 0, 0, n_batch, true)?;
            let prompt_tokens = n_past as u32;

            let mut sampler = sampler_params.build_chain(model.clone())?;
            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                let next_token = sampler.sample(&mut *ctx_guard, -1);

                if model.vocab_is_eog(next_token) {
                    break;
                }

                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    let previous_len = generated.len();
                    generated.push_str(&text);

                    if let Some(pos) = find_stop_in_recent_window(
                        &generated,
                        previous_len,
                        &stop_sequences,
                        max_stop_len,
                    ) {
                        generated.truncate(pos);
                        return Ok((generated, prompt_tokens, completion_tokens));
                    }
                }

                sampler.accept(next_token);
                ctx_guard.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, prompt_tokens, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Generate streaming text with vision input
    #[cfg(feature = "multimodal")]
    pub(super) async fn generate_vision_text_streaming(
        &self,
        loaded: Arc<LoadedModel>,
        prompt: String,
        bitmaps: Vec<crate::Bitmap>,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let models_ref = self.models.clone();
        let cancellations = Arc::clone(&self.cancellations);
        let active_requests = Arc::clone(&self.active_requests);
        let request_id_cleanup = request_id.clone();
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;
            let mtmd_ref = match loaded.mtmd_context.as_ref() {
                Some(r) => r,
                None => {
                    tracing::error!("No multimodal context available for streaming vision");
                    return;
                }
            };
            let mut mtmd_context = mtmd_ref.write().await;

            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();

                let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();
                let chunks = mtmd_context.tokenize(&prompt, &bitmap_refs)?;
                let n_batch = 512;
                let _n_past =
                    mtmd_context.eval_chunks(&mut context, &chunks, 0, 0, n_batch, true)?;

                let mut sampler = sampler_params.build_chain(model.clone())?;

                let mut generated = String::new();
                let mut index = 0u32;
                let mut sent_len = 0usize;
                let hold_back = max_stop_len.saturating_sub(1);
                let mut tokens_generated = 0u32;
                let mut last_token_id = 0i32;

                for _ in 0..max_tokens {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let next_token = sampler.sample(&mut *context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        tokens_generated += 1;
                        last_token_id = next_token;
                        let previous_len = generated.len();
                        generated.push_str(&text);

                        if let Some(pos) = find_stop_in_recent_window(
                            &generated,
                            previous_len,
                            &stop_sequences,
                            max_stop_len,
                        ) {
                            if pos > sent_len {
                                let partial = &generated[sent_len..pos];
                                let chunk = StreamChunk {
                                    request_id: request_id_arc.clone(),
                                    index,
                                    delta: partial.to_string(),
                                    token_id: next_token,
                                    thinking: None,
                                    tool_calls: None,
                                };
                                let _ = tx.blocking_send(chunk);
                            }
                            return Ok::<_, MullamaError>(tokens_generated);
                        }

                        let mut flush_end = generated.len().saturating_sub(hold_back);
                        while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                            flush_end -= 1;
                        }
                        if flush_end > sent_len {
                            let chunk = StreamChunk {
                                request_id: request_id_arc.clone(),
                                index,
                                delta: generated[sent_len..flush_end].to_string(),
                                token_id: next_token,
                                thinking: None,
                                tool_calls: None,
                            };
                            if tx.blocking_send(chunk).is_err() {
                                break;
                            }
                            sent_len = flush_end;
                            index += 1;
                        }
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                if sent_len < generated.len() {
                    let chunk = StreamChunk {
                        request_id: request_id_arc.clone(),
                        index,
                        delta: generated[sent_len..].to_string(),
                        token_id: last_token_id,
                        thinking: None,
                        tool_calls: None,
                    };
                    let _ = tx.blocking_send(chunk);
                }

                Ok::<_, MullamaError>(tokens_generated)
            });

            if let Ok(tokens) = result {
                models_ref.add_tokens(tokens as u64);
            }

            cancellations.remove(&request_id_cleanup);
            active_requests.fetch_sub(1, Ordering::Relaxed);
        });

        Ok((rx, 0, request_id))
    }
}
