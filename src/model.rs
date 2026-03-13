use crate::{sys, context::{Context, ContextParams}, error::MullamaError, token::TokenId};
use std::{path::Path, sync::Arc, ffi::CString};
use std::os::raw::c_char;

/// Represents a loaded LLM model
pub struct Model {
    pub(crate) model_ptr: *mut sys::llama_model,
}

// Models are safe to clone since the underlying C++ object is reference counted
impl Clone for Model {
    fn clone(&self) -> Self {
        // In a real implementation, we'd need to increment the reference count
        // For now, we'll just copy the pointer
        Self {
            model_ptr: self.model_ptr,
        }
    }
}

// Models need to be freed when dropped
impl Drop for Model {
    fn drop(&mut self) {
        if !self.model_ptr.is_null() {
            unsafe {
                sys::llama_model_free(self.model_ptr);
            }
        }
    }
}

/// Parameters for loading a model
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

impl Model {
    /// Load a model from a GGUF file
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError> {
        Self::load_with_params(path, ModelParams::default())
    }
    
    /// Load a model with specific parameters
    pub fn load_with_params(path: impl AsRef<Path>, params: ModelParams) -> Result<Self, MullamaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(MullamaError::ModelLoadError(format!(
                "Model file not found: {}", 
                path.display()
            )));
        }
        
        // Convert path to C string
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| MullamaError::ModelLoadError("Invalid path".to_string()))?;
        
        // Initialize the llama backend
        unsafe {
            sys::llama_backend_init();
        }
        
        // Get default model parameters
        let mut llama_params = unsafe { sys::llama_model_default_params() };
        
        // Apply our parameters
        llama_params.n_gpu_layers = params.n_gpu_layers;
        llama_params.use_mmap = params.use_mmap;
        llama_params.use_mlock = params.use_mlock;
        
        // Load the model
        let model_ptr = unsafe {
            sys::llama_model_load_from_file(c_path.as_ptr(), llama_params)
        };
        
        if model_ptr.is_null() {
            return Err(MullamaError::ModelLoadError(
                "Failed to load model".to_string()
            ));
        }
        
        Ok(Model { 
            model_ptr,
        })
    }
    
    /// Create a context for this model
    pub fn create_context(&self, params: ContextParams) -> Result<Context, MullamaError> {
        Context::new(Arc::new(self.clone()), params)
    }
    
    /// Tokenize text using this model's vocabulary
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<TokenId>, MullamaError> {
        // Convert text to C string
        let c_text = CString::new(text)
            .map_err(|_| MullamaError::TokenizationError("Invalid text".to_string()))?;
        
        // Get the vocabulary from the model
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.model_ptr) };
        
        // First, get the required buffer size
        let max_tokens = unsafe { 
            sys::llama_tokenize(
                vocab_ptr,
                c_text.as_ptr(),
                text.len() as i32,
                std::ptr::null_mut(),
                0,
                add_bos,
                special,
            )
        };
        
        if max_tokens < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Tokenization failed with code: {}", max_tokens)
            ));
        }
        
        // Allocate buffer and tokenize
        let mut tokens = vec![0i32; max_tokens as usize];
        let actual_tokens = unsafe {
            sys::llama_tokenize(
                vocab_ptr,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                max_tokens,
                add_bos,
                special,
            )
        };
        
        if actual_tokens < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Tokenization failed with code: {}", actual_tokens)
            ));
        }
        
        // Convert to TokenId and return
        Ok(tokens.into_iter().take(actual_tokens as usize).map(|t| t as TokenId).collect())
    }
    
    /// Convert a token to its text representation
    pub fn token_to_str(&self, token: TokenId, special: bool) -> Result<String, MullamaError> {
        // Get the vocabulary from the model
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.model_ptr) };
        
        let mut buf = vec![0u8; 32]; // Allocate buffer for token text
        
        let n_chars = unsafe {
            sys::llama_token_to_piece(
                vocab_ptr,
                token as sys::LlamaToken,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0, // lstrip
                special,
            )
        };
        
        if n_chars < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Failed to convert token to string: {}", n_chars)
            ));
        }
        
        // Resize buffer if needed
        if n_chars as usize > buf.len() {
            buf.resize(n_chars as usize, 0);
            let n_chars = unsafe {
                sys::llama_token_to_piece(
                    vocab_ptr,
                    token as sys::LlamaToken,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0, // lstrip
                    special,
                )
            };
            
            if n_chars < 0 {
                return Err(MullamaError::TokenizationError(
                    format!("Failed to convert token to string: {}", n_chars)
                ));
            }
        }
        
        // Convert to string, truncating at the actual length
        let result = String::from_utf8_lossy(&buf[..n_chars as usize]).to_string();
        Ok(result)
    }
    
    /// Get model context size
    pub fn n_ctx_train(&self) -> u32 {
        unsafe { sys::llama_model_n_ctx_train(self.model_ptr) as u32 }
    }
    
    /// Get the internal model pointer (for use by other modules)
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_model {
        self.model_ptr
    }
}

/// Represents a token with its metadata
#[derive(Debug, Clone)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
}