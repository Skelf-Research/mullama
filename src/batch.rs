use crate::{
    sys,
    token::{TokenBuffer, TokenId},
};

/// Represents a batch of tokens for processing
///
/// ## Memory Safety Note
///
/// The `tokens_storage` uses `Vec<TokenId>` (heap-allocated) rather than `SmallVec`
/// because `llama_batch_get_one` stores a raw pointer to the token data internally.
/// If we used SmallVec's inline storage, the pointer would become invalid when the
/// Batch struct is moved (as SmallVec copies inline data during moves).
/// Vec's data is always heap-allocated, so the pointer remains stable.
#[allow(dead_code)]
pub struct Batch {
    inner: Option<sys::llama_batch>,
    /// Store tokens to ensure they outlive the batch (for llama_batch_get_one)
    /// Uses Vec for heap allocation to ensure stable pointer after struct moves
    tokens_storage: Option<Vec<TokenId>>,
    /// Whether this batch was created with llama_batch_init (needs to be freed)
    needs_free: bool,
}

impl Batch {
    /// Create a new batch with allocated memory
    pub fn new(max_tokens: usize, embd: i32, max_seq: usize) -> Self {
        let inner = unsafe { sys::llama_batch_init(max_tokens as i32, embd, max_seq as i32) };

        Self {
            inner: Some(inner),
            tokens_storage: None,
            needs_free: true,
        }
    }

    /// Create a batch from a TokenBuffer using llama_batch_get_one
    ///
    /// Converts to Vec for heap allocation to ensure pointer stability.
    pub fn from_token_buffer(tokens: TokenBuffer) -> Self {
        if tokens.is_empty() {
            return Self {
                inner: None,
                tokens_storage: None,
                needs_free: false,
            };
        }

        // Convert to Vec to ensure heap allocation - the pointer must remain
        // stable after the Batch struct is moved. SmallVec's inline storage
        // would invalidate the pointer on move.
        let mut vec_tokens: Vec<TokenId> = tokens.into_vec();

        let inner =
            unsafe { sys::llama_batch_get_one(vec_tokens.as_mut_ptr(), vec_tokens.len() as i32) };

        Self {
            inner: Some(inner),
            tokens_storage: Some(vec_tokens),
            needs_free: false,
        }
    }

    /// Create a batch from owned tokens using llama_batch_get_one
    /// This avoids copying when you already have a Vec
    pub fn from_tokens_owned(mut tokens: Vec<TokenId>) -> Self {
        if tokens.is_empty() {
            return Self {
                inner: None,
                tokens_storage: None,
                needs_free: false,
            };
        }

        let inner = unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };

        Self {
            inner: Some(inner),
            tokens_storage: Some(tokens),
            needs_free: false,
        }
    }

    /// Create a batch from a token slice using llama_batch_get_one
    ///
    /// Copies the tokens to ensure heap allocation for pointer stability.
    pub fn from_tokens(tokens: &[TokenId]) -> Self {
        Self::from_token_buffer(TokenBuffer::from_slice(tokens))
    }

    /// Get the internal llama_batch struct
    #[allow(dead_code)]
    pub(crate) fn as_llama_batch(&self) -> Option<&sys::llama_batch> {
        self.inner.as_ref()
    }

    /// Get the internal llama_batch struct (public for testing)
    pub fn get_llama_batch(&self) -> Option<&sys::llama_batch> {
        self.inner.as_ref()
    }

    /// Take the internal llama_batch struct (consuming the Batch)
    pub(crate) fn take_llama_batch(&mut self) -> Option<sys::llama_batch> {
        self.inner.take()
    }

    /// Get the number of tokens in the batch
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |batch| batch.n_tokens as usize)
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.inner
            .as_ref()
            .map_or(true, |batch| batch.n_tokens == 0)
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new(512, 0, 1)
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // Only free batches created with llama_batch_init
        // llama_batch_get_one doesn't allocate memory that needs freeing
        if self.needs_free {
            if let Some(batch) = self.inner.take() {
                unsafe {
                    sys::llama_batch_free(batch);
                }
            }
        }
    }
}
