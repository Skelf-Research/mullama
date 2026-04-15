use mullama::{Context, ContextParams, SamplerParams};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

use crate::model::JsModel;
use crate::napi_error;
use crate::sampler::JsSamplerParams;

/// Result from streaming generation
#[napi(object)]
#[derive(Clone)]
pub struct StreamResult {
    pub pieces: Vec<String>,
    pub text: String,
}

/// Context creation parameters
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsContextParams {
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_threads: Option<i32>,
    pub embeddings: Option<bool>,
}

/// Context for model inference
#[napi]
pub struct JsContext {
    inner: Context,
    model: Arc<mullama::Model>,
}

#[napi]
impl JsContext {
    #[napi(constructor)]
    pub fn new(model: &JsModel, params: Option<JsContextParams>) -> Result<Self> {
        let p = params.unwrap_or_default();

        let n_threads = if p.n_threads.unwrap_or(0) <= 0 {
            num_cpus::get() as i32
        } else {
            p.n_threads.unwrap_or(0)
        };

        let ctx_params = ContextParams {
            n_ctx: p.n_ctx.unwrap_or(0),
            n_batch: p.n_batch.unwrap_or(2048),
            n_threads,
            n_threads_batch: n_threads,
            embeddings: p.embeddings.unwrap_or(false),
            ..Default::default()
        };

        let model_arc = model.get_inner();
        let context = Context::new(Arc::new((*model_arc).clone()), ctx_params)
            .map_err(|e| napi_error("Failed to create context", e))?;

        Ok(Self {
            inner: context,
            model: model_arc,
        })
    }

    #[napi]
    pub fn generate(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        params: Option<JsSamplerParams>,
    ) -> Result<String> {
        let tokens = self
            .model
            .tokenize(&prompt, true, false)
            .map_err(|e| napi_error("Tokenization failed", e))?;

        let sampler_params = params.as_ref().map(SamplerParams::from).unwrap_or_default();

        self.inner
            .generate_with_params(&tokens, max_tokens.unwrap_or(100) as usize, &sampler_params)
            .map_err(|e| napi_error("Generation failed", e))
    }

    #[napi]
    pub fn generate_from_tokens(
        &mut self,
        tokens: Vec<i32>,
        max_tokens: Option<u32>,
        params: Option<JsSamplerParams>,
    ) -> Result<String> {
        let sampler_params = params.as_ref().map(SamplerParams::from).unwrap_or_default();

        self.inner
            .generate_with_params(&tokens, max_tokens.unwrap_or(100) as usize, &sampler_params)
            .map_err(|e| napi_error("Generation failed", e))
    }

    #[napi]
    pub fn generate_stream(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        params: Option<JsSamplerParams>,
    ) -> Result<Vec<String>> {
        let result = self.generate_stream_full(prompt, max_tokens, params)?;
        Ok(result.pieces)
    }

    #[napi]
    pub fn generate_stream_full(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        params: Option<JsSamplerParams>,
    ) -> Result<StreamResult> {
        let tokens = self
            .model
            .tokenize(&prompt, true, false)
            .map_err(|e| napi_error("Tokenization failed", e))?;

        let sampler_params = params.as_ref().map(SamplerParams::from).unwrap_or_default();
        let mut pieces: Vec<String> = Vec::new();

        self.inner
            .generate_streaming(
                &tokens,
                max_tokens.unwrap_or(100) as usize,
                &sampler_params,
                |piece| {
                    pieces.push(piece.to_string());
                    true
                },
            )
            .map_err(|e| napi_error("Streaming failed", e))?;

        Ok(StreamResult {
            text: pieces.join(""),
            pieces,
        })
    }

    #[napi]
    pub fn clear_cache(&mut self) {
        self.inner.kv_cache_clear();
    }

    #[napi(getter)]
    pub fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    #[napi(getter)]
    pub fn n_batch(&self) -> u32 {
        self.inner.n_batch()
    }

    #[napi]
    pub fn get_embeddings(&self) -> Option<Vec<f64>> {
        self.inner
            .get_embeddings()
            .map(|embeddings| embeddings.iter().map(|&value| value as f64).collect())
    }
}
