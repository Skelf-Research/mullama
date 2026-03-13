use crate::{sys, token::TokenId};

/// Represents a batch of tokens for processing
pub struct Batch {
    inner: Option<sys::llama_batch>,
}

impl Batch {
    /// Create a new batch
    pub fn new(max_tokens: usize, embd: i32, max_seq: usize) -> Self {
        let inner = unsafe { 
            sys::llama_batch_init(
                max_tokens as i32,
                embd,
                max_seq as i32,
            )
        };
        
        Self { inner: Some(inner) }
    }
    
    /// Create a batch from tokens
    pub fn from_tokens(tokens: &[TokenId]) -> Self {
        // For simple use cases, we can use the helper function
        // This creates a batch that's managed by llama.cpp
        // We'll create our own batch struct for more complex cases
        Self::new(tokens.len(), 0, 1)
    }
    
    /// Get the internal llama_batch struct
    pub(crate) fn as_llama_batch(&self) -> Option<&sys::llama_batch> {
        self.inner.as_ref()
    }
    
    /// Take the internal llama_batch struct (consuming the Batch)
    pub(crate) fn take_llama_batch(&mut self) -> Option<sys::llama_batch> {
        self.inner.take()
    }
    
    /// Get the number of tokens in the batch
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, |batch| batch.n_tokens as usize)
    }
    
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.inner.as_ref().map_or(true, |batch| batch.n_tokens == 0)
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new(512, 0, 1)
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if let Some(batch) = self.inner.take() {
            unsafe {
                sys::llama_batch_free(batch);
            }
        }
    }
}