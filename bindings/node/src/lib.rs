//! Node.js bindings for Mullama LLM library
//!
//! This module provides napi-rs based Node.js bindings for the Mullama library,
//! enabling high-performance LLM inference from JavaScript/TypeScript.

use mullama::{Context, ContextParams, Model, ModelParams, SamplerParams};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

/// Model loading parameters
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsModelParams {
    /// Number of layers to offload to GPU (0 = CPU only, -1 = all)
    pub n_gpu_layers: Option<i32>,
    /// Use memory mapping for model loading
    pub use_mmap: Option<bool>,
    /// Lock model in memory
    pub use_mlock: Option<bool>,
    /// Only load vocabulary (for tokenization only)
    pub vocab_only: Option<bool>,
}

/// Context creation parameters
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsContextParams {
    /// Context size (0 = use model default)
    pub n_ctx: Option<u32>,
    /// Batch size for prompt processing
    pub n_batch: Option<u32>,
    /// Number of threads (0 = auto)
    pub n_threads: Option<i32>,
    /// Enable embeddings mode
    pub embeddings: Option<bool>,
}

/// Sampler parameters for text generation
#[napi(object)]
#[derive(Clone)]
pub struct JsSamplerParams {
    /// Temperature (0.0 = deterministic, higher = more random)
    pub temperature: Option<f64>,
    /// Top-k sampling (0 = disabled)
    pub top_k: Option<i32>,
    /// Top-p/nucleus sampling (1.0 = disabled)
    pub top_p: Option<f64>,
    /// Min-p sampling (0.0 = disabled)
    pub min_p: Option<f64>,
    /// Typical sampling (1.0 = disabled)
    pub typical_p: Option<f64>,
    /// Repeat penalty (1.0 = disabled)
    pub penalty_repeat: Option<f64>,
    /// Frequency penalty (0.0 = disabled)
    pub penalty_freq: Option<f64>,
    /// Presence penalty (0.0 = disabled)
    pub penalty_present: Option<f64>,
    /// Tokens to consider for penalties
    pub penalty_last_n: Option<i32>,
    /// Random seed (0 = random)
    pub seed: Option<u32>,
}

impl Default for JsSamplerParams {
    fn default() -> Self {
        JsSamplerParams {
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: Some(0.05),
            typical_p: Some(1.0),
            penalty_repeat: Some(1.1),
            penalty_freq: Some(0.0),
            penalty_present: Some(0.0),
            penalty_last_n: Some(64),
            seed: Some(0),
        }
    }
}

impl From<&JsSamplerParams> for SamplerParams {
    fn from(p: &JsSamplerParams) -> Self {
        SamplerParams {
            temperature: p.temperature.unwrap_or(0.8) as f32,
            top_k: p.top_k.unwrap_or(40),
            top_p: p.top_p.unwrap_or(0.95) as f32,
            min_p: p.min_p.unwrap_or(0.05) as f32,
            typical_p: p.typical_p.unwrap_or(1.0) as f32,
            penalty_repeat: p.penalty_repeat.unwrap_or(1.1) as f32,
            penalty_freq: p.penalty_freq.unwrap_or(0.0) as f32,
            penalty_present: p.penalty_present.unwrap_or(0.0) as f32,
            penalty_last_n: p.penalty_last_n.unwrap_or(64),
            seed: p.seed.unwrap_or(0),
            ..Default::default()
        }
    }
}

/// Greedy sampler parameters (deterministic)
#[napi]
pub fn sampler_params_greedy() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        min_p: Some(0.0),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.0),
        penalty_freq: Some(0.0),
        penalty_present: Some(0.0),
        penalty_last_n: Some(0),
        seed: Some(0),
    }
}

/// Creative sampler parameters (high randomness)
#[napi]
pub fn sampler_params_creative() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(1.2),
        top_k: Some(100),
        top_p: Some(0.95),
        min_p: Some(0.02),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.15),
        penalty_freq: Some(0.1),
        penalty_present: Some(0.1),
        penalty_last_n: Some(128),
        seed: Some(0),
    }
}

/// Precise sampler parameters (low randomness)
#[napi]
pub fn sampler_params_precise() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(0.3),
        top_k: Some(20),
        top_p: Some(0.8),
        min_p: Some(0.1),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.05),
        penalty_freq: Some(0.0),
        penalty_present: Some(0.0),
        penalty_last_n: Some(32),
        seed: Some(0),
    }
}

/// Model class for loading and managing LLM models
#[napi]
pub struct JsModel {
    inner: Arc<Model>,
}

#[napi]
impl JsModel {
    /// Load a model from a GGUF file
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
            .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?;

        Ok(JsModel {
            inner: Arc::new(model),
        })
    }

    /// Tokenize text into token IDs
    #[napi]
    pub fn tokenize(
        &self,
        text: String,
        add_bos: Option<bool>,
        special: Option<bool>,
    ) -> Result<Vec<i32>> {
        self.inner
            .tokenize(&text, add_bos.unwrap_or(true), special.unwrap_or(false))
            .map_err(|e| Error::from_reason(format!("Tokenization failed: {}", e)))
    }

    /// Detokenize token IDs back to text
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
            .map_err(|e| Error::from_reason(format!("Detokenization failed: {}", e)))
    }

    /// Get the model's training context size
    #[napi(getter)]
    pub fn n_ctx_train(&self) -> i32 {
        self.inner.n_ctx_train()
    }

    /// Get the model's embedding dimension
    #[napi(getter)]
    pub fn n_embd(&self) -> i32 {
        self.inner.n_embd()
    }

    /// Get the vocabulary size
    #[napi(getter)]
    pub fn n_vocab(&self) -> i32 {
        self.inner.vocab_size()
    }

    /// Get the number of layers
    #[napi(getter)]
    pub fn n_layer(&self) -> i32 {
        self.inner.n_layer()
    }

    /// Get the number of attention heads
    #[napi(getter)]
    pub fn n_head(&self) -> i32 {
        self.inner.n_head()
    }

    /// Get the BOS (beginning of sequence) token ID
    #[napi(getter)]
    pub fn token_bos(&self) -> i32 {
        self.inner.token_bos()
    }

    /// Get the EOS (end of sequence) token ID
    #[napi(getter)]
    pub fn token_eos(&self) -> i32 {
        self.inner.token_eos()
    }

    /// Get the model size in bytes
    #[napi(getter)]
    pub fn size(&self) -> i64 {
        self.inner.size() as i64
    }

    /// Get the number of parameters
    #[napi(getter)]
    pub fn n_params(&self) -> i64 {
        self.inner.n_params() as i64
    }

    /// Get the model description
    #[napi(getter)]
    pub fn description(&self) -> String {
        self.inner.desc()
    }

    /// Get the model architecture
    #[napi(getter)]
    pub fn architecture(&self) -> Option<String> {
        self.inner.architecture()
    }

    /// Get the model name from metadata
    #[napi(getter)]
    pub fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// Check if a token is end-of-generation
    #[napi]
    pub fn token_is_eog(&self, token: i32) -> bool {
        self.inner.token_is_eog(token)
    }

    /// Get all metadata as an object
    #[napi]
    pub fn metadata(&self) -> std::collections::HashMap<String, String> {
        self.inner.metadata()
    }

    /// Apply chat template to format messages
    #[napi]
    pub fn apply_chat_template(
        &self,
        messages: Vec<(String, String)>,
        add_generation_prompt: Option<bool>,
    ) -> Result<String> {
        let msg_refs: Vec<(&str, &str)> = messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        self.inner
            .apply_chat_template(None, &msg_refs, add_generation_prompt.unwrap_or(true))
            .map_err(|e| Error::from_reason(format!("Chat template failed: {}", e)))
    }

    /// Get a reference to the inner model (for internal use)
    pub(crate) fn get_inner(&self) -> Arc<Model> {
        self.inner.clone()
    }
}

/// Context for model inference
#[napi]
pub struct JsContext {
    inner: Context,
    model: Arc<Model>,
}

#[napi]
impl JsContext {
    /// Create a new context from a model
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
            .map_err(|e| Error::from_reason(format!("Failed to create context: {}", e)))?;

        Ok(JsContext {
            inner: context,
            model: model_arc,
        })
    }

    /// Generate text from a prompt
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
            .map_err(|e| Error::from_reason(format!("Tokenization failed: {}", e)))?;

        let sampler_params = params.as_ref().map(SamplerParams::from).unwrap_or_default();

        self.inner
            .generate_with_params(&tokens, max_tokens.unwrap_or(100) as usize, &sampler_params)
            .map_err(|e| Error::from_reason(format!("Generation failed: {}", e)))
    }

    /// Generate text from token IDs
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
            .map_err(|e| Error::from_reason(format!("Generation failed: {}", e)))
    }

    /// Generate text with streaming (returns array of tokens)
    #[napi]
    pub fn generate_stream(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        params: Option<JsSamplerParams>,
    ) -> Result<Vec<String>> {
        let tokens = self
            .model
            .tokenize(&prompt, true, false)
            .map_err(|e| Error::from_reason(format!("Tokenization failed: {}", e)))?;

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
            .map_err(|e| Error::from_reason(format!("Streaming failed: {}", e)))?;

        Ok(pieces)
    }

    /// Clear the KV cache
    #[napi]
    pub fn clear_cache(&mut self) {
        self.inner.kv_cache_clear();
    }

    /// Get the context size
    #[napi(getter)]
    pub fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    /// Get the batch size
    #[napi(getter)]
    pub fn n_batch(&self) -> u32 {
        self.inner.n_batch()
    }

    /// Get embeddings (if embeddings mode is enabled)
    #[napi]
    pub fn get_embeddings(&self) -> Option<Vec<f64>> {
        self.inner
            .get_embeddings()
            .map(|e| e.iter().map(|&v| v as f64).collect())
    }
}

/// Embedding generator for creating text embeddings
#[napi]
pub struct JsEmbeddingGenerator {
    context: Context,
    model: Arc<Model>,
    normalize: bool,
}

#[napi]
impl JsEmbeddingGenerator {
    /// Create a new embedding generator
    #[napi(constructor)]
    pub fn new(model: &JsModel, n_ctx: Option<u32>, normalize: Option<bool>) -> Result<Self> {
        let params = ContextParams {
            n_ctx: n_ctx.unwrap_or(512),
            embeddings: true,
            pooling_type: mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            ..Default::default()
        };

        let model_arc = model.get_inner();
        let context = Context::new(Arc::new((*model_arc).clone()), params)
            .map_err(|e| Error::from_reason(format!("Failed to create context: {}", e)))?;

        Ok(JsEmbeddingGenerator {
            context,
            model: model_arc,
            normalize: normalize.unwrap_or(true),
        })
    }

    /// Generate embeddings for text
    #[napi]
    pub fn embed(&mut self, text: String) -> Result<Vec<f64>> {
        let tokens = self
            .model
            .tokenize(&text, true, false)
            .map_err(|e| Error::from_reason(format!("Tokenization failed: {}", e)))?;

        self.context.kv_cache_clear();
        self.context
            .decode(&tokens)
            .map_err(|e| Error::from_reason(format!("Decode failed: {}", e)))?;

        match self.context.get_embeddings() {
            Some(embeddings) => {
                let mut vec: Vec<f64> = embeddings.iter().map(|&v| v as f64).collect();

                if self.normalize {
                    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm > 0.0 {
                        for v in vec.iter_mut() {
                            *v /= norm;
                        }
                    }
                }

                Ok(vec)
            }
            None => Err(Error::from_reason("No embeddings available")),
        }
    }

    /// Generate embeddings for multiple texts
    #[napi]
    pub fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let emb = self.embed(text)?;
            embeddings.push(emb);
        }

        Ok(embeddings)
    }

    /// Get the embedding dimension
    #[napi(getter)]
    pub fn n_embd(&self) -> i32 {
        self.model.n_embd()
    }
}

/// Compute cosine similarity between two vectors
#[napi]
pub fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::from_reason("Vectors must have the same length"));
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm = norm_a.sqrt() * norm_b.sqrt();
    if norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / norm)
    }
}

/// Initialize the mullama backend
#[napi]
pub fn backend_init() {
    mullama::backend_init();
}

/// Free the mullama backend resources
#[napi]
pub fn backend_free() {
    mullama::backend_free();
}

/// Check if GPU offloading is supported
#[napi]
pub fn supports_gpu_offload() -> bool {
    mullama::supports_gpu_offload()
}

/// Get system information
#[napi]
pub fn system_info() -> String {
    mullama::print_system_info()
}

/// Get the maximum number of supported devices
#[napi]
pub fn max_devices() -> u32 {
    mullama::max_devices() as u32
}

/// Get the library version
#[napi]
pub fn version() -> String {
    "0.1.0".to_string()
}
