use crate::{sys, model::Model, error::MullamaError, token::TokenId, batch::Batch};
use std::{sync::Arc};

/// Parameters for creating a context
#[derive(Debug, Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub embeddings: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 512,
            n_threads: num_cpus::get() as i32,
            embeddings: false,
        }
    }
}

/// Represents a model context for inference
pub struct Context {
    model: Arc<Model>,
    ctx_ptr: *mut sys::llama_context,
}

impl Context {
    /// Create a new context from a model
    pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self, MullamaError> {
        // Get default context parameters
        let mut llama_params = unsafe { sys::llama_context_default_params() };
        
        // Apply our parameters
        llama_params.n_ctx = params.n_ctx;
        llama_params.n_batch = params.n_batch;
        llama_params.n_threads = params.n_threads;
        llama_params.embeddings = params.embeddings;
        
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
        self.decode(&batch)?;

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
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_context {
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