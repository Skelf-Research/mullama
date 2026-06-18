use mullama::{Model, ModelParams};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

use crate::napi_error;

/// Model loading parameters
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsModelParams {
    pub n_gpu_layers: Option<i32>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub vocab_only: Option<bool>,
}

/// Model class for loading and managing LLM models
#[napi]
pub struct JsModel {
    inner: Arc<Model>,
}

#[napi]
impl JsModel {
    #[napi(factory)]
    pub fn load(path: String, params: Option<JsModelParams>) -> Result<Self> {
        let p = params.unwrap_or_default();

        let model_params = ModelParams {
            n_gpu_layers: p.n_gpu_layers.unwrap_or(0),
            use_mmap: p.use_mmap.unwrap_or(true),
            use_mlock: p.use_mlock.unwrap_or(false),
            vocab_only: p.vocab_only.unwrap_or(false),
            ..Default::default()
        };

        let model = Model::load_with_params(&path, model_params)
            .map_err(|e| napi_error("Failed to load model", e))?;

        Ok(Self {
            inner: Arc::new(model),
        })
    }

    #[napi]
    pub fn tokenize(
        &self,
        text: String,
        add_bos: Option<bool>,
        special: Option<bool>,
    ) -> Result<Vec<i32>> {
        self.inner
            .tokenize(&text, add_bos.unwrap_or(true), special.unwrap_or(false))
            .map_err(|e| napi_error("Tokenization failed", e))
    }

    #[napi]
    pub fn detokenize(
        &self,
        tokens: Vec<i32>,
        remove_special: Option<bool>,
        unparse_special: Option<bool>,
    ) -> Result<String> {
        self.inner
            .detokenize(
                &tokens,
                remove_special.unwrap_or(false),
                unparse_special.unwrap_or(false),
            )
            .map_err(|e| napi_error("Detokenization failed", e))
    }

    #[napi(getter)]
    pub fn n_ctx_train(&self) -> i32 {
        self.inner.n_ctx_train()
    }

    #[napi(getter)]
    pub fn n_embd(&self) -> i32 {
        self.inner.n_embd()
    }

    #[napi(getter)]
    pub fn n_vocab(&self) -> i32 {
        self.inner.vocab_size()
    }

    #[napi(getter)]
    pub fn n_layer(&self) -> i32 {
        self.inner.n_layer()
    }

    #[napi(getter)]
    pub fn n_head(&self) -> i32 {
        self.inner.n_head()
    }

    #[napi(getter)]
    pub fn token_bos(&self) -> i32 {
        self.inner.token_bos()
    }

    #[napi(getter)]
    pub fn token_eos(&self) -> i32 {
        self.inner.token_eos()
    }

    #[napi(getter)]
    pub fn size(&self) -> i64 {
        self.inner.size() as i64
    }

    #[napi(getter)]
    pub fn n_params(&self) -> i64 {
        self.inner.n_params() as i64
    }

    #[napi(getter)]
    pub fn description(&self) -> String {
        self.inner.desc()
    }

    #[napi(getter)]
    pub fn architecture(&self) -> Option<String> {
        self.inner.architecture()
    }

    #[napi(getter)]
    pub fn name(&self) -> Option<String> {
        self.inner.name()
    }

    #[napi]
    pub fn token_is_eog(&self, token: i32) -> bool {
        self.inner.token_is_eog(token)
    }

    #[napi]
    pub fn metadata(&self) -> std::collections::HashMap<String, String> {
        self.inner.metadata()
    }

    #[napi]
    pub fn apply_chat_template(
        &self,
        messages: Vec<(String, String)>,
        add_generation_prompt: Option<bool>,
    ) -> Result<String> {
        let msg_refs: Vec<(&str, &str)> = messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();

        self.inner
            .apply_chat_template(None, &msg_refs, add_generation_prompt.unwrap_or(true))
            .map_err(|e| napi_error("Chat template failed", e))
    }

    pub(crate) fn get_inner(&self) -> Arc<Model> {
        self.inner.clone()
    }
}
