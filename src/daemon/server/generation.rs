use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;

use super::super::models::{LoadedModel, ModelManager, RequestGuard};
use super::super::protocol::{generate_completion_id, ResponseFormat, StreamChunk};
use super::prompt::find_stop_in_recent_window;
use super::Daemon;
use crate::{Context, Model, MullamaError, SamplerChain, SamplerParams};

/// Controls whether tokens are buffered or streamed
enum TokenSink<'a> {
    /// Collect all tokens, return the full string at end
    Buffer,
    /// Stream tokens via channel with hold-back for stop detection
    Stream {
        tx: &'a mpsc::Sender<StreamChunk>,
        request_id: Arc<str>,
        cancel_flag: &'a AtomicBool,
    },
}

/// Result of the core token generation loop
struct GenerationResult {
    generated: String,
    completion_tokens: u32,
}

/// Core token generation loop — shared by all generation paths.
///
/// Callers are responsible for context setup (kv_cache_clear, prompt
/// encoding/decoding) and sampler construction before invoking this.
fn generate_tokens(
    context: &mut Context,
    model: &Model,
    sampler: &mut SamplerChain,
    max_tokens: u32,
    stop_sequences: &[String],
    max_stop_len: usize,
    sink: &TokenSink<'_>,
) -> Result<GenerationResult, MullamaError> {
    let mut generated = String::with_capacity(match sink {
        TokenSink::Buffer => (max_tokens as usize) * 6,
        TokenSink::Stream { .. } => 256,
    });
    let mut completion_tokens = 0u32;

    // Stream-specific state
    let mut index = 0u32;
    let mut sent_len = 0usize;
    let hold_back = max_stop_len.saturating_sub(1);
    let mut last_token_id = 0i32;

    for _ in 0..max_tokens {
        if let TokenSink::Stream { cancel_flag, .. } = sink {
            if cancel_flag.load(Ordering::Relaxed) {
                break;
            }
        }

        let next_token = sampler.sample(context, -1);

        if model.vocab_is_eog(next_token) {
            break;
        }

        if let Ok(text) = model.token_to_str(next_token, 0, false) {
            let previous_len = generated.len();
            generated.push_str(&text);

            if let Some(pos) = find_stop_in_recent_window(
                &generated,
                previous_len,
                stop_sequences,
                max_stop_len,
            ) {
                if let TokenSink::Stream { tx, request_id, .. } = sink {
                    if pos > sent_len {
                        let chunk = StreamChunk {
                            request_id: request_id.clone(),
                            index,
                            delta: generated[sent_len..pos].to_string(),
                            token_id: next_token,
                            thinking: None,
                            tool_calls: None,
                        };
                        let _ = tx.blocking_send(chunk);
                    }
                }
                generated.truncate(pos);
                return Ok(GenerationResult {
                    generated,
                    completion_tokens,
                });
            }

            if let TokenSink::Stream { tx, request_id, .. } = sink {
                let mut flush_end = generated.len().saturating_sub(hold_back);
                while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                    flush_end -= 1;
                }
                if flush_end > sent_len {
                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
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

            last_token_id = next_token;
        }

        sampler.accept(next_token);
        context.decode_single(next_token)?;
        completion_tokens += 1;
    }

    if let TokenSink::Stream { tx, request_id, .. } = sink {
        if sent_len < generated.len() {
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                index,
                delta: generated[sent_len..].to_string(),
                token_id: last_token_id,
                thinking: None,
                tool_calls: None,
            };
            let _ = tx.blocking_send(chunk);
        }
    }

    Ok(GenerationResult {
        generated,
        completion_tokens,
    })
}

/// Resolve a response format into an optional GBNF grammar string.
fn resolve_grammar(response_format: Option<&ResponseFormat>) -> Option<String> {
    match response_format {
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
    }
}

/// Pre-computed state for a streaming generation task.
///
/// Holds the channel, cancellation flag, and cleanup references shared by
/// both text and vision streaming paths. Call [`finish`](Self::finish) when
/// the generation loop completes to record stats and clean up tracking.
struct StreamingSetup {
    tx: mpsc::Sender<StreamChunk>,
    request_id_arc: Arc<str>,
    cancel_flag: Arc<AtomicBool>,
    models_ref: Arc<ModelManager>,
    cancellations: Arc<DashMap<String, Arc<AtomicBool>>>,
    active_requests: Arc<AtomicU32>,
    request_id_for_cleanup: String,
    stop_sequences: Vec<String>,
    max_stop_len: usize,
}

impl StreamingSetup {
    /// Record token stats and clean up cancellation/request tracking.
    fn finish(self, result: &Result<GenerationResult, MullamaError>) {
        if let Ok(r) = result {
            self.models_ref.add_tokens(r.completion_tokens as u64);
        }
        self.cancellations.remove(&self.request_id_for_cleanup);
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Daemon {
    /// Prepare common state for a streaming generation request.
    ///
    /// Returns `(setup, rx, request_id)` where `setup` is moved into the
    /// spawned task and `rx`/`request_id` are returned to the caller.
    fn prepare_streaming(
        &self,
        stop_sequences: Vec<String>,
    ) -> (StreamingSetup, mpsc::Receiver<StreamChunk>, String) {
        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let setup = StreamingSetup {
            tx,
            request_id_arc,
            cancel_flag,
            models_ref: self.models.clone(),
            cancellations: Arc::clone(&self.cancellations),
            active_requests: Arc::clone(&self.active_requests),
            request_id_for_cleanup: request_id.clone(),
            stop_sequences,
            max_stop_len,
        };

        (setup, rx, request_id)
    }

    /// Generate text (non-streaming)
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

        let grammar_gbnf = resolve_grammar(response_format);

        let mut context = loaded.acquire_context().await;
        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let result = tokio::task::block_in_place(|| {
            context.kv_cache_clear();

            let mut sampler = sampler_params.build_chain(model.clone())?;

            if let Some(gbnf) = &grammar_gbnf {
                let grammar_sampler =
                    crate::sampling::Sampler::grammar(model.clone(), gbnf, "root")?;
                sampler.add(grammar_sampler);
            }

            context.decode(&tokens)?;

            generate_tokens(
                &mut *context,
                &model,
                &mut sampler,
                max_tokens,
                &stop_sequences,
                max_stop_len,
                &TokenSink::Buffer,
            )
        })?;

        self.models.add_tokens(result.completion_tokens as u64);

        Ok((result.generated, prompt_tokens, result.completion_tokens))
    }

    /// Generate text with streaming — yields tokens as they're generated
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

        let (setup, rx, request_id) = self.prepare_streaming(stop_sequences);
        let model = loaded.model.clone();

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;

            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();
                let mut sampler = sampler_params.build_chain(model.clone())?;
                context.decode(&tokens)?;

                generate_tokens(
                    &mut *context,
                    &model,
                    &mut sampler,
                    max_tokens,
                    &setup.stop_sequences,
                    setup.max_stop_len,
                    &TokenSink::Stream {
                        tx: &setup.tx,
                        request_id: setup.request_id_arc.clone(),
                        cancel_flag: &setup.cancel_flag,
                    },
                )
            });

            setup.finish(&result);
        });

        Ok((rx, prompt_tokens, request_id))
    }

    /// Generate text with vision input (non-streaming)
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

        let (result, prompt_tokens) = tokio::task::block_in_place(|| {
            ctx_guard.kv_cache_clear();

            let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();
            let chunks = mtmd_guard.tokenize(prompt, &bitmap_refs)?;

            let n_batch = 512;
            let n_past =
                mtmd_guard.eval_chunks(&mut ctx_guard, &chunks, 0, 0, n_batch, true)?;
            let prompt_tokens = n_past as u32;

            let mut sampler = sampler_params.build_chain(model.clone())?;

            let result = generate_tokens(
                &mut *ctx_guard,
                &model,
                &mut sampler,
                max_tokens,
                &stop_sequences,
                max_stop_len,
                &TokenSink::Buffer,
            )?;

            Ok::<_, MullamaError>((result, prompt_tokens))
        })?;

        self.models.add_tokens(result.completion_tokens as u64);

        Ok((result.generated, prompt_tokens, result.completion_tokens))
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
        let (setup, rx, request_id) = self.prepare_streaming(stop_sequences);
        let model = loaded.model.clone();

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;
            let mtmd_ref = match loaded.mtmd_context.as_ref() {
                Some(r) => r,
                None => {
                    tracing::error!("No multimodal context available for streaming vision");
                    setup.finish(&Err(MullamaError::MultimodalError(
                        "No multimodal context".into(),
                    )));
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

                generate_tokens(
                    &mut *context,
                    &model,
                    &mut sampler,
                    max_tokens,
                    &setup.stop_sequences,
                    setup.max_stop_len,
                    &TokenSink::Stream {
                        tx: &setup.tx,
                        request_id: setup.request_id_arc.clone(),
                        cancel_flag: &setup.cancel_flag,
                    },
                )
            });

            setup.finish(&result);
        });

        Ok((rx, 0, request_id))
    }
}
