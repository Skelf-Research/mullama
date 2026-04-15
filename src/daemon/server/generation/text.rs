use std::sync::Arc;

use tokio::sync::mpsc;

use super::super::super::models::{LoadedModel, RequestGuard};
use super::super::super::protocol::{ResponseFormat, StreamChunk};
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
}
