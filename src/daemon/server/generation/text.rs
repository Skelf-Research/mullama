use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use super::super::super::models::{LoadedModel, RequestGuard};
use super::super::super::protocol::{ResponseFormat, StreamChunk, Timings};
use super::super::Daemon;
use super::common::{generate_tokens, resolve_grammar, TokenSink};
use crate::{MullamaError, SamplerParams};

impl Daemon {
    /// Generate text without streaming.
    pub async fn generate_text(
        &self,
        loaded: &LoadedModel,
        prompt: &str,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
        response_format: Option<&ResponseFormat>,
    ) -> Result<(String, u32, u32, Timings), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let grammar_gbnf = resolve_grammar(response_format);
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut context = loaded.acquire_context().await;
        let model = loaded.model.clone();

        let result = tokio::task::block_in_place(|| {
            let tokens = model.tokenize(prompt, add_bos, true)?;
            let prompt_tokens = tokens.len() as u32;

            context.kv_cache_clear();

            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Repetition penalties include prompt history. Seed the base
            // sampler before adding grammar so prompt tokens affect penalties
            // without being consumed as generated grammar tokens.
            for &token in &tokens {
                sampler.accept(token);
            }

            if let Some(gbnf) = &grammar_gbnf {
                let grammar_sampler =
                    crate::sampling::Sampler::grammar(model.clone(), gbnf, "root")?;
                sampler.add(grammar_sampler);
            }

            let prompt_start = Instant::now();
            context.decode(&tokens)?;
            let prompt_eval_ns = prompt_start.elapsed().as_nanos() as u64;

            let gen_result = generate_tokens(
                &mut *context,
                &model,
                &mut sampler,
                max_tokens,
                &stop_sequences,
                max_stop_len,
                &TokenSink::Buffer,
            )?;

            Ok::<_, MullamaError>((gen_result, prompt_tokens, prompt_eval_ns))
        })?;

        self.models.add_tokens(result.0.completion_tokens as u64);

        let timings = Timings {
            prompt_eval_ns: result.2,
            eval_ns: result.0.eval_ns,
            prompt_tokens: result.1,
            completion_tokens: result.0.completion_tokens,
        };
        Ok((
            result.0.generated,
            result.1,
            result.0.completion_tokens,
            timings,
        ))
    }

    /// Generate text with streaming.
    pub async fn generate_text_streaming(
        &self,
        loaded: Arc<LoadedModel>,
        prompt: String,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let model_for_tokenize = loaded.model.clone();
        let tokens =
            tokio::task::block_in_place(|| model_for_tokenize.tokenize(&prompt, add_bos, false))?;
        let prompt_tokens = tokens.len() as u32;

        let (setup, rx, request_id) = self.prepare_streaming(stop_sequences);
        let model = loaded.model.clone();

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;

            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();
                let mut sampler = sampler_params.build_chain(model.clone())?;
                for &token in &tokens {
                    sampler.accept(token);
                }
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
}
