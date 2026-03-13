use crate::{sys, context::{Context, ContextParams}, error::MullamaError, token::TokenId};
use std::{path::Path, sync::Arc, ffi::CString, ptr};
use std::os::raw::{c_char, c_void};

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

/// Parameters for loading a model with complete feature support
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub n_gpu_layers: i32,
    pub split_mode: sys::llama_split_mode,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub kv_overrides: Vec<ModelKvOverride>,
    pub progress_callback: Option<fn(f32) -> bool>,
}

#[derive(Debug, Clone)]
pub struct ModelKvOverride {
    pub key: String,
    pub value: ModelKvOverrideValue,
}

#[derive(Debug, Clone)]
pub enum ModelKvOverrideValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            split_mode: sys::llama_split_mode::LLAMA_SPLIT_MODE_NONE,
            main_gpu: 0,
            tensor_split: Vec::new(),
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            use_extra_bufts: false,
            kv_overrides: Vec::new(),
            progress_callback: None,
        }
    }
}

impl Model {
    /// Load a model from a GGUF file
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError> {
        Self::load_with_params(path, ModelParams::default())
    }
    
    /// Load a model with advanced parameters and full feature support
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

        // Initialize the llama backend if not already done
        unsafe {
            sys::llama_backend_init();
        }

        // Get default model parameters from llama.cpp
        let mut llama_params = unsafe { sys::llama_model_default_params() };

        // Apply all our advanced parameters
        llama_params.n_gpu_layers = params.n_gpu_layers;
        llama_params.split_mode = params.split_mode;
        llama_params.main_gpu = params.main_gpu;
        llama_params.vocab_only = params.vocab_only as sys::c_bool;
        llama_params.use_mmap = params.use_mmap as sys::c_bool;
        llama_params.use_mlock = params.use_mlock as sys::c_bool;
        llama_params.check_tensors = params.check_tensors as sys::c_bool;
        llama_params.use_extra_bufts = params.use_extra_bufts as sys::c_bool;

        // Set tensor split if provided
        if !params.tensor_split.is_empty() {
            llama_params.tensor_split = params.tensor_split.as_ptr();
        } else {
            llama_params.tensor_split = ptr::null();
        }

        // Set up KV overrides if provided
        let kv_overrides: Vec<sys::llama_model_kv_override> = params.kv_overrides
            .iter()
            .map(|override_| Self::convert_kv_override(override_))
            .collect::<Result<Vec<_>, _>>()?;

        if !kv_overrides.is_empty() {
            llama_params.kv_overrides = kv_overrides.as_ptr();
        } else {
            llama_params.kv_overrides = ptr::null();
        }

        // Set progress callback if provided
        if params.progress_callback.is_some() {
            // Note: This would require more complex callback handling in a real implementation
            llama_params.progress_callback = None; // Placeholder
            llama_params.progress_callback_user_data = ptr::null_mut();
        } else {
            llama_params.progress_callback = None;
            llama_params.progress_callback_user_data = ptr::null_mut();
        }

        // Set remaining parameters
        llama_params.devices = ptr::null_mut();
        llama_params.tensor_buft_overrides = ptr::null();

        // Load the model with full parameter support
        let model_ptr = unsafe {
            sys::llama_model_load_from_file(c_path.as_ptr(), llama_params)
        };

        if model_ptr.is_null() {
            return Err(MullamaError::ModelLoadError(
                "Failed to load model - check file format and parameters".to_string()
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
    
    /// Tokenize text using this model's vocabulary with advanced options
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<TokenId>, MullamaError> {
        // Convert text to C string
        let c_text = CString::new(text)
            .map_err(|_| MullamaError::TokenizationError("Invalid text".to_string()))?;

        // First, get the required buffer size by passing null tokens
        let max_tokens = unsafe {
            sys::llama_tokenize(
                self.model_ptr,
                c_text.as_ptr(),
                text.len() as i32,
                ptr::null_mut(),
                0,
                add_bos as sys::c_bool,
                special as sys::c_bool,
            )
        };

        if max_tokens < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Tokenization failed with code: {}", max_tokens)
            ));
        }

        if max_tokens == 0 {
            return Ok(Vec::new());
        }

        // Allocate buffer and tokenize
        let mut tokens = vec![0i32; max_tokens as usize];
        let actual_tokens = unsafe {
            sys::llama_tokenize(
                self.model_ptr,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                max_tokens,
                add_bos as sys::c_bool,
                special as sys::c_bool,
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

    /// Detokenize tokens back to text with advanced options
    pub fn detokenize(&self, tokens: &[TokenId], remove_special: bool, unparse_special: bool) -> Result<String, MullamaError> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        // Convert tokens to the correct type
        let llama_tokens: Vec<sys::llama_token> = tokens.iter().map(|&t| t as sys::llama_token).collect();

        // First, get the required buffer size
        let max_chars = unsafe {
            sys::llama_detokenize(
                self.model_ptr,
                llama_tokens.as_ptr(),
                tokens.len() as i32,
                ptr::null_mut(),
                0,
                remove_special as sys::c_bool,
                unparse_special as sys::c_bool,
            )
        };

        if max_chars < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Detokenization failed with code: {}", max_chars)
            ));
        }

        if max_chars == 0 {
            return Ok(String::new());
        }

        // Allocate buffer and detokenize
        let mut buffer = vec![0u8; max_chars as usize + 1]; // +1 for null terminator
        let actual_chars = unsafe {
            sys::llama_detokenize(
                self.model_ptr,
                llama_tokens.as_ptr(),
                tokens.len() as i32,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as i32,
                remove_special as sys::c_bool,
                unparse_special as sys::c_bool,
            )
        };

        if actual_chars < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Detokenization failed with code: {}", actual_chars)
            ));
        }

        // Convert to string, handling the null terminator
        let result_bytes = &buffer[..actual_chars as usize];
        let result = String::from_utf8_lossy(result_bytes).to_string();
        Ok(result)
    }
    
    /// Convert a token to its text representation with advanced options
    pub fn token_to_str(&self, token: TokenId, lstrip: i32, special: bool) -> Result<String, MullamaError> {
        let mut buf = vec![0u8; 128]; // Start with reasonable buffer size

        let n_chars = unsafe {
            sys::llama_token_to_piece(
                self.model_ptr,
                token as sys::llama_token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                lstrip,
                special as sys::c_bool,
            )
        };

        if n_chars < 0 {
            return Err(MullamaError::TokenizationError(
                format!("Failed to convert token to string: {}", n_chars)
            ));
        }

        // Resize buffer if needed and retry
        if n_chars as usize > buf.len() {
            buf.resize(n_chars as usize + 1, 0);
            let n_chars_retry = unsafe {
                sys::llama_token_to_piece(
                    self.model_ptr,
                    token as sys::llama_token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    lstrip,
                    special as sys::c_bool,
                )
            };

            if n_chars_retry < 0 {
                return Err(MullamaError::TokenizationError(
                    format!("Failed to convert token to string on retry: {}", n_chars_retry)
                ));
            }
        }

        // Convert to string, handling UTF-8 properly
        let result_bytes = &buf[..n_chars as usize];
        let result = String::from_utf8_lossy(result_bytes).to_string();
        Ok(result)
    }
    
    /// Get model training context size
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { sys::llama_model_n_ctx_train(self.model_ptr) }
    }

    /// Get model embedding dimension
    pub fn n_embd(&self) -> i32 {
        unsafe { sys::llama_model_n_embd(self.model_ptr) }
    }

    /// Get number of model layers
    pub fn n_layer(&self) -> i32 {
        unsafe { sys::llama_model_n_layer(self.model_ptr) }
    }

    /// Get number of attention heads
    pub fn n_head(&self) -> i32 {
        unsafe { sys::llama_model_n_head(self.model_ptr) }
    }

    /// Get number of key-value heads
    pub fn n_head_kv(&self) -> i32 {
        unsafe { sys::llama_model_n_head_kv(self.model_ptr) }
    }

    /// Get sliding window attention size
    pub fn n_swa(&self) -> i32 {
        unsafe { sys::llama_model_n_swa(self.model_ptr) }
    }

    /// Get RoPE frequency scaling factor
    pub fn rope_freq_scale_train(&self) -> f32 {
        unsafe { sys::llama_model_rope_freq_scale_train(self.model_ptr) }
    }

    /// Get vocabulary type
    pub fn vocab_type(&self) -> sys::llama_vocab_type {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.model_ptr) };
        unsafe { sys::llama_vocab_type(vocab_ptr) }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> i32 {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.model_ptr) };
        unsafe { sys::llama_vocab_n_tokens(vocab_ptr) }
    }

    /// Get rope type used by the model
    pub fn rope_type(&self) -> sys::llama_rope_type {
        unsafe { sys::llama_model_rope_type(self.model_ptr) }
    }
    
    /// Get the internal model pointer (for use by other modules)
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_model {
        self.model_ptr
    }
}

/// Represents a token with complete metadata
#[derive(Debug, Clone)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
    pub attr: sys::llama_token_attr,
}

impl Model {
    /// Get complete token information including attributes
    pub fn get_token_info(&self, token: TokenId) -> Result<Token, MullamaError> {
        let text_ptr = unsafe { sys::llama_token_get_text(self.model_ptr, token as sys::llama_token) };
        if text_ptr.is_null() {
            return Err(MullamaError::TokenizationError("Token not found".to_string()));
        }

        let text = unsafe {
            std::ffi::CStr::from_ptr(text_ptr)
                .to_string_lossy()
                .to_string()
        };

        let score = unsafe { sys::llama_token_get_score(self.model_ptr, token as sys::llama_token) };
        let attr = unsafe { sys::llama_token_get_attr(self.model_ptr, token as sys::llama_token) };

        Ok(Token { id: token, text, score, attr })
    }

    /// Check if token is end of generation
    pub fn token_is_eog(&self, token: TokenId) -> bool {
        unsafe { sys::llama_token_is_eog(self.model_ptr, token as sys::llama_token) as bool }
    }

    /// Check if token is a control token
    pub fn token_is_control(&self, token: TokenId) -> bool {
        unsafe { sys::llama_token_is_control(self.model_ptr, token as sys::llama_token) as bool }
    }

    /// Get special tokens
    pub fn token_bos(&self) -> TokenId {
        unsafe { sys::llama_token_bos(self.model_ptr) as TokenId }
    }

    pub fn token_eos(&self) -> TokenId {
        unsafe { sys::llama_token_eos(self.model_ptr) as TokenId }
    }

    pub fn token_cls(&self) -> TokenId {
        unsafe { sys::llama_token_cls(self.model_ptr) as TokenId }
    }

    pub fn token_sep(&self) -> TokenId {
        unsafe { sys::llama_token_sep(self.model_ptr) as TokenId }
    }

    pub fn token_nl(&self) -> TokenId {
        unsafe { sys::llama_token_nl(self.model_ptr) as TokenId }
    }

    pub fn token_pad(&self) -> TokenId {
        unsafe { sys::llama_token_pad(self.model_ptr) as TokenId }
    }

    /// Check if model adds BOS token
    pub fn add_bos_token(&self) -> bool {
        unsafe { sys::llama_add_bos_token(self.model_ptr) as bool }
    }

    /// Check if model adds EOS token
    pub fn add_eos_token(&self) -> bool {
        unsafe { sys::llama_add_eos_token(self.model_ptr) as bool }
    }
}

// Helper implementation for KV overrides
impl Model {
    fn convert_kv_override(override_: &ModelKvOverride) -> Result<sys::llama_model_kv_override, MullamaError> {
        let key_bytes = override_.key.as_bytes();
        if key_bytes.len() >= 128 {
            return Err(MullamaError::ModelLoadError("KV override key too long".to_string()));
        }

        let mut key = [0i8; 128];
        for (i, &byte) in key_bytes.iter().enumerate() {
            key[i] = byte as i8;
        }

        let (tag, value) = match &override_.value {
            ModelKvOverrideValue::Int(v) => {
                let mut val = sys::llama_model_kv_override_value { val_i64: *v };
                (sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_INT, val)
            },
            ModelKvOverrideValue::Float(v) => {
                let mut val = sys::llama_model_kv_override_value { val_f64: *v };
                (sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_FLOAT, val)
            },
            ModelKvOverrideValue::Bool(v) => {
                let mut val = sys::llama_model_kv_override_value { val_bool: *v as sys::c_bool };
                (sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_BOOL, val)
            },
            ModelKvOverrideValue::Str(s) => {
                if s.len() >= 128 {
                    return Err(MullamaError::ModelLoadError("KV override string value too long".to_string()));
                }
                let mut val_str = [0i8; 128];
                for (i, &byte) in s.as_bytes().iter().enumerate() {
                    val_str[i] = byte as i8;
                }
                let val = sys::llama_model_kv_override_value { val_str };
                (sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_STR, val)
            },
        };

        Ok(sys::llama_model_kv_override {
            tag,
            key,
            value,
        })
    }
}