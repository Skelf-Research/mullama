use crate::{sys, model::Model, error::MullamaError, token::TokenId, batch::Batch};
use std::{sync::Arc};

/// Parameters for creating a context
#[derive(Debug, Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: sys::llama_rope_scaling_type,
    pub pooling_type: sys::llama_pooling_type,
    pub attention_type: sys::llama_attention_type,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub embeddings: bool,
    pub flash_attn: bool,
    pub offload_kqv: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 0, // Use model default
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: num_cpus::get() as u32,
            n_threads_batch: num_cpus::get() as u32,
            rope_scaling_type: sys::llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            pooling_type: sys::llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED,
            attention_type: sys::llama_attention_type::LLAMA_ATTENTION_TYPE_UNSPECIFIED,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_thold: -1.0,
            embeddings: false,
            flash_attn: false,
            offload_kqv: true,
            swa_full: true,
            kv_unified: false,
        }
    }
}

/// Represents a model context for inference
pub struct Context {
    pub model: Arc<Model>,
    pub ctx_ptr: *mut sys::llama_context,
}

impl Context {
    /// Create a new context from a model
    pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self, MullamaError> {
        // Get default context parameters
        let mut llama_params = unsafe { sys::llama_context_default_params() };

        // Apply our parameters
        llama_params.n_ctx = params.n_ctx;
        llama_params.n_batch = params.n_batch;
        llama_params.n_ubatch = params.n_ubatch;
        llama_params.n_seq_max = params.n_seq_max;
        llama_params.n_threads = params.n_threads;
        llama_params.n_threads_batch = params.n_threads_batch;
        llama_params.rope_scaling_type = params.rope_scaling_type;
        llama_params.pooling_type = params.pooling_type;
        llama_params.attention_type = params.attention_type;
        llama_params.rope_freq_base = params.rope_freq_base;
        llama_params.rope_freq_scale = params.rope_freq_scale;
        llama_params.yarn_ext_factor = params.yarn_ext_factor;
        llama_params.yarn_attn_factor = params.yarn_attn_factor;
        llama_params.yarn_beta_fast = params.yarn_beta_fast;
        llama_params.yarn_beta_slow = params.yarn_beta_slow;
        llama_params.yarn_orig_ctx = params.yarn_orig_ctx;
        llama_params.defrag_thold = params.defrag_thold;
        llama_params.embeddings = params.embeddings;
        llama_params.flash_attn = params.flash_attn;
        llama_params.offload_kqv = params.offload_kqv;
        llama_params.swa_full = params.swa_full;
        llama_params.kv_unified = params.kv_unified;
        
        // Create the context
        let ctx_ptr = unsafe {
            sys::llama_init_from_model(model.model_ptr, llama_params)
        };
        
        if ctx_ptr.is_null() {
            return Err(MullamaError::ContextError(
                "Failed to create context".to_string()
            ));
        }
        
        Ok(Context {
            model,
            ctx_ptr,
        })
    }
    
    /// Process a batch of tokens
    pub fn decode(&mut self, tokens: &[TokenId]) -> Result<(), MullamaError> {
        // Create a simple batch for these tokens
        let mut batch = Batch::from_tokens(tokens);
        
        // Get the llama_batch and call llama_decode
        if let Some(llama_batch) = batch.take_llama_batch() {
            let result = unsafe {
                sys::llama_decode(self.ctx_ptr, llama_batch)
            };
            
            if result != 0 {
                return Err(MullamaError::GenerationError(
                    format!("Decode failed with code: {}", result)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Simple text generation (placeholder - full implementation would use sampling)
    pub fn generate(&mut self, prompt_tokens: &[TokenId], max_tokens: usize) -> Result<String, MullamaError> {
        if prompt_tokens.is_empty() {
            return Err(MullamaError::GenerationError("Empty prompt tokens".to_string()));
        }

        // Create a batch for the prompt tokens
        let batch = Batch::from_tokens(prompt_tokens);

        // Process the prompt
        self.decode(prompt_tokens)?;

        // Note: A full implementation would:
        // 1. Get logits using self.logits()
        // 2. Apply sampling using a sampler
        // 3. Generate tokens one by one
        // 4. Convert tokens back to text
        // For now, return a meaningful placeholder
        Ok(format!(
            "[Placeholder] Generated {} tokens from prompt of {} tokens",
            max_tokens, prompt_tokens.len()
        ))
    }
    
    /// Get logits from the last evaluation
    pub fn logits(&self) -> Result<&[f32], MullamaError> {
        // In a real implementation, this would:
        // 1. Call llama_get_logits to get the raw pointer
        // 2. Determine the size (vocab size * batch size)
        // 3. Create a slice from the pointer
        // For now, return an empty slice as a placeholder
        Ok(&[])
    }
    
    /// Get embeddings (if enabled)
    pub fn embeddings(&self) -> Result<Option<&[f32]>, MullamaError> {
        // In a real implementation, this would:
        // 1. Call llama_get_embeddings to get the raw pointer
        // 2. Determine the size
        // 3. Create a slice from the pointer
        // For now, return None as a placeholder
        Ok(None)
    }
    
    /// Get the model associated with this context
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    
    /// Get the internal context pointer (for use by other modules)
    pub fn as_ptr(&self) -> *mut sys::llama_context {
        self.ctx_ptr
    }
}

// Contexts need to be freed when dropped
impl Drop for Context {
    fn drop(&mut self) {
        if !self.ctx_ptr.is_null() {
            unsafe {
                sys::llama_free(self.ctx_ptr);
            }
        }
    }
}