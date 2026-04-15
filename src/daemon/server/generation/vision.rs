use std::sync::Arc;

use tokio::sync::mpsc;

use super::super::super::models::{LoadedModel, RequestGuard};
use super::super::super::protocol::StreamChunk;
use super::super::Daemon;
use super::common::{generate_tokens, TokenSink};
use crate::{Bitmap, MullamaError, SamplerParams};

impl Daemon {
    /// Generate text with vision input without streaming.
    pub(crate) async fn generate_vision_text(
        &self,
        loaded: &LoadedModel,
        prompt: &str,
        bitmaps: &[Bitmap],
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

            let bitmap_refs: Vec<&Bitmap> = bitmaps.iter().collect();
            let chunks = mtmd_guard.tokenize(prompt, &bitmap_refs)?;

            let n_batch = 512;
            let n_past = mtmd_guard.eval_chunks(&mut ctx_guard, &chunks, 0, 0, n_batch, true)?;
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

    /// Generate text with vision input and streaming.
    pub(crate) async fn generate_vision_text_streaming(
        &self,
        loaded: Arc<LoadedModel>,
        prompt: String,
        bitmaps: Vec<Bitmap>,
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

                let bitmap_refs: Vec<&Bitmap> = bitmaps.iter().collect();
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
