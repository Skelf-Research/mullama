//! Model FFI bindings
//!
//! This module provides C ABI functions for model loading, tokenization,
//! and model information retrieval.

use crate::error::{set_last_error, MullamaErrorCode};
use crate::handle::Handle;
use libc::{c_char, c_int, size_t};
use mullama::{Model, ModelParams};
use std::ffi::CStr;

/// Opaque model handle for FFI
pub type MullamaModel = Handle<Model>;

/// Model loading parameters
#[repr(C)]
#[derive(Debug)]
pub struct MullamaModelParams {
    /// Number of layers to offload to GPU (0 = CPU only, -1 = all)
    pub n_gpu_layers: c_int,
    /// Main GPU device index (for multi-GPU systems)
    pub main_gpu: c_int,
    /// Use memory mapping for model loading
    pub use_mmap: bool,
    /// Lock model in memory
    pub use_mlock: bool,
    /// Only load vocabulary (for tokenization only)
    pub vocab_only: bool,
    /// Check tensor data integrity
    pub check_tensors: bool,
}

impl Default for MullamaModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            main_gpu: 0,
            use_mmap: true,
            use_mlock: false,
            vocab_only: false,
            check_tensors: true,
        }
    }
}

/// Get default model parameters
#[no_mangle]
pub extern "C" fn mullama_model_default_params() -> MullamaModelParams {
    MullamaModelParams::default()
}

/// Load a model from a GGUF file
///
/// # Arguments
/// * `path` - Path to the GGUF model file
/// * `params` - Optional model parameters (NULL for defaults)
///
/// # Returns
/// A model handle on success, or NULL on failure.
/// Check `mullama_get_last_error()` for error details.
#[no_mangle]
pub extern "C" fn mullama_model_load(
    path: *const c_char,
    params: *const MullamaModelParams,
) -> *mut MullamaModel {
    if path.is_null() {
        set_last_error("Model path is null");
        return std::ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in model path");
            return std::ptr::null_mut();
        }
    };

    let model_params = if params.is_null() {
        ModelParams::default()
    } else {
        let p = unsafe { &*params };
        ModelParams {
            n_gpu_layers: p.n_gpu_layers,
            main_gpu: p.main_gpu,
            use_mmap: p.use_mmap,
            use_mlock: p.use_mlock,
            vocab_only: p.vocab_only,
            check_tensors: p.check_tensors,
            ..Default::default()
        }
    };

    match Model::load_with_params(path_str, model_params) {
        Ok(model) => Handle::new(model).into_raw(),
        Err(e) => {
            set_last_error(format!("Failed to load model: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free a model handle
///
/// # Safety
/// The model handle must have been created by `mullama_model_load`.
/// After calling this function, the handle is invalid.
#[no_mangle]
pub extern "C" fn mullama_model_free(model: *mut MullamaModel) {
    if !model.is_null() {
        unsafe {
            let _ = Handle::from_raw(model);
        }
    }
}

/// Tokenize text into token IDs
///
/// # Arguments
/// * `model` - Model handle
/// * `text` - Text to tokenize
/// * `tokens` - Output buffer for tokens
/// * `max_tokens` - Size of the output buffer
/// * `add_bos` - Whether to add beginning-of-sequence token
/// * `special` - Whether to parse special tokens
///
/// # Returns
/// Number of tokens written on success, or negative error code on failure.
/// If the buffer is too small, returns the required size as a negative number.
#[no_mangle]
pub extern "C" fn mullama_tokenize(
    model: *const MullamaModel,
    text: *const c_char,
    tokens: *mut c_int,
    max_tokens: c_int,
    add_bos: bool,
    special: bool,
) -> c_int {
    if model.is_null() {
        set_last_error("Model handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if text.is_null() {
        set_last_error("Text is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let model_ref = match unsafe { Handle::as_ref(model) } {
        Some(m) => m,
        None => {
            set_last_error("Invalid model handle");
            return MullamaErrorCode::NullPointer.to_i32();
        }
    };

    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in text");
            return MullamaErrorCode::Utf8Error.to_i32();
        }
    };

    match model_ref.tokenize(text_str, add_bos, special) {
        Ok(token_ids) => {
            let count = token_ids.len() as c_int;

            // If buffer is too small, return required size as negative
            if max_tokens < count || tokens.is_null() {
                return -count;
            }

            // Copy tokens to output buffer
            let output = unsafe { std::slice::from_raw_parts_mut(tokens, count as usize) };
            for (i, &token) in token_ids.iter().enumerate() {
                output[i] = token;
            }

            count
        }
        Err(e) => {
            set_last_error(format!("Tokenization failed: {}", e));
            MullamaErrorCode::Tokenization.to_i32()
        }
    }
}

/// Detokenize tokens back to text
///
/// # Arguments
/// * `model` - Model handle
/// * `tokens` - Token IDs to detokenize
/// * `n_tokens` - Number of tokens
/// * `output` - Output buffer for text
/// * `max_output` - Size of the output buffer
///
/// # Returns
/// Number of bytes written on success, or negative error code on failure.
/// If the buffer is too small, returns the required size as a negative number.
#[no_mangle]
pub extern "C" fn mullama_detokenize(
    model: *const MullamaModel,
    tokens: *const c_int,
    n_tokens: c_int,
    output: *mut c_char,
    max_output: c_int,
) -> c_int {
    if model.is_null() {
        set_last_error("Model handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if tokens.is_null() || n_tokens <= 0 {
        set_last_error("Invalid tokens array");
        return MullamaErrorCode::InvalidInput.to_i32();
    }

    let model_ref = match unsafe { Handle::as_ref(model) } {
        Some(m) => m,
        None => {
            set_last_error("Invalid model handle");
            return MullamaErrorCode::NullPointer.to_i32();
        }
    };

    let token_slice = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };

    match model_ref.detokenize(token_slice, false, false) {
        Ok(text) => {
            let bytes = text.as_bytes();
            let len = bytes.len() as c_int;

            // If buffer is too small or null, return required size as negative
            if max_output < len + 1 || output.is_null() {
                return -(len + 1);
            }

            // Copy text to output buffer
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, bytes.len());
                *output.add(bytes.len()) = 0; // Null terminator
            }

            len
        }
        Err(e) => {
            set_last_error(format!("Detokenization failed: {}", e));
            MullamaErrorCode::Tokenization.to_i32()
        }
    }
}

/// Get the model's training context size
#[no_mangle]
pub extern "C" fn mullama_model_n_ctx_train(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.n_ctx_train(),
        None => 0,
    }
}

/// Get the model's embedding dimension
#[no_mangle]
pub extern "C" fn mullama_model_n_embd(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.n_embd(),
        None => 0,
    }
}

/// Get the model's vocabulary size
#[no_mangle]
pub extern "C" fn mullama_model_n_vocab(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.vocab_size(),
        None => 0,
    }
}

/// Get the number of layers in the model
#[no_mangle]
pub extern "C" fn mullama_model_n_layer(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.n_layer(),
        None => 0,
    }
}

/// Get the number of attention heads
#[no_mangle]
pub extern "C" fn mullama_model_n_head(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.n_head(),
        None => 0,
    }
}

/// Get the BOS (beginning of sequence) token ID
#[no_mangle]
pub extern "C" fn mullama_model_token_bos(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return -1;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.token_bos(),
        None => -1,
    }
}

/// Get the EOS (end of sequence) token ID
#[no_mangle]
pub extern "C" fn mullama_model_token_eos(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return -1;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.token_eos(),
        None => -1,
    }
}

/// Get the newline token ID
#[no_mangle]
pub extern "C" fn mullama_model_token_nl(model: *const MullamaModel) -> c_int {
    if model.is_null() {
        return -1;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.token_nl(),
        None => -1,
    }
}

/// Check if a token is end-of-generation
#[no_mangle]
pub extern "C" fn mullama_model_token_is_eog(model: *const MullamaModel, token: c_int) -> bool {
    if model.is_null() {
        return false;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.token_is_eog(token),
        None => false,
    }
}

/// Get the model description
///
/// # Returns
/// Number of bytes written, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_model_desc(
    model: *const MullamaModel,
    output: *mut c_char,
    max_output: size_t,
) -> c_int {
    if model.is_null() {
        set_last_error("Model handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let model_ref = match unsafe { Handle::as_ref(model) } {
        Some(m) => m,
        None => {
            set_last_error("Invalid model handle");
            return MullamaErrorCode::NullPointer.to_i32();
        }
    };

    let desc = model_ref.desc();
    let bytes = desc.as_bytes();
    let len = bytes.len();

    if output.is_null() || max_output < len + 1 {
        return -(len as c_int + 1);
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, len);
        *output.add(len) = 0;
    }

    len as c_int
}

/// Get the model size in bytes
#[no_mangle]
pub extern "C" fn mullama_model_size(model: *const MullamaModel) -> u64 {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.size(),
        None => 0,
    }
}

/// Get the number of parameters in the model
#[no_mangle]
pub extern "C" fn mullama_model_n_params(model: *const MullamaModel) -> u64 {
    if model.is_null() {
        return 0;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.n_params(),
        None => 0,
    }
}

/// Check if model has an encoder (encoder-decoder models)
#[no_mangle]
pub extern "C" fn mullama_model_has_encoder(model: *const MullamaModel) -> bool {
    if model.is_null() {
        return false;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.has_encoder(),
        None => false,
    }
}

/// Check if model has a decoder
#[no_mangle]
pub extern "C" fn mullama_model_has_decoder(model: *const MullamaModel) -> bool {
    if model.is_null() {
        return false;
    }

    match unsafe { Handle::as_ref(model) } {
        Some(m) => m.has_decoder(),
        None => false,
    }
}

/// Clone the model Arc (increases reference count)
///
/// Returns a new handle that shares ownership with the original.
#[no_mangle]
pub extern "C" fn mullama_model_clone(model: *const MullamaModel) -> *mut MullamaModel {
    if model.is_null() {
        return std::ptr::null_mut();
    }

    match unsafe { Handle::clone_arc(model) } {
        Some(arc) => Handle::from_arc(arc).into_raw(),
        None => std::ptr::null_mut(),
    }
}

/// Get a metadata value from the model by key
///
/// # Returns
/// Number of bytes written, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_model_meta_val(
    model: *const MullamaModel,
    key: *const c_char,
    output: *mut c_char,
    max_output: size_t,
) -> c_int {
    if model.is_null() || key.is_null() {
        set_last_error("Null pointer");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let model_ref = match unsafe { Handle::as_ref(model) } {
        Some(m) => m,
        None => {
            set_last_error("Invalid model handle");
            return MullamaErrorCode::NullPointer.to_i32();
        }
    };

    let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in key");
            return MullamaErrorCode::Utf8Error.to_i32();
        }
    };

    match model_ref.meta_val(key_str) {
        Some(val) => {
            let bytes = val.as_bytes();
            let len = bytes.len();

            if output.is_null() || max_output < len + 1 {
                return -(len as c_int + 1);
            }

            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, len);
                *output.add(len) = 0;
            }

            len as c_int
        }
        None => {
            set_last_error(format!("Metadata key not found: {}", key_str));
            MullamaErrorCode::InvalidInput.to_i32()
        }
    }
}

/// Apply a chat template to format messages
///
/// # Arguments
/// * `model` - Model handle
/// * `messages` - Array of chat messages
/// * `n_messages` - Number of messages
/// * `add_generation_prompt` - Whether to add generation prompt
/// * `output` - Output buffer
/// * `max_output` - Output buffer size
///
/// # Returns
/// Number of bytes written, or negative error code.
#[repr(C)]
pub struct MullamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[no_mangle]
pub extern "C" fn mullama_model_apply_chat_template(
    model: *const MullamaModel,
    messages: *const MullamaChatMessage,
    n_messages: c_int,
    add_generation_prompt: bool,
    output: *mut c_char,
    max_output: size_t,
) -> c_int {
    if model.is_null() || messages.is_null() {
        set_last_error("Null pointer");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let model_ref = match unsafe { Handle::as_ref(model) } {
        Some(m) => m,
        None => {
            set_last_error("Invalid model handle");
            return MullamaErrorCode::NullPointer.to_i32();
        }
    };

    // Convert messages
    let mut rust_messages: Vec<(&str, &str)> = Vec::with_capacity(n_messages as usize);
    for i in 0..n_messages {
        let msg = unsafe { &*messages.add(i as usize) };

        if msg.role.is_null() || msg.content.is_null() {
            set_last_error("Message role or content is null");
            return MullamaErrorCode::NullPointer.to_i32();
        }

        let role = match unsafe { CStr::from_ptr(msg.role) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid UTF-8 in message role");
                return MullamaErrorCode::Utf8Error.to_i32();
            }
        };

        let content = match unsafe { CStr::from_ptr(msg.content) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid UTF-8 in message content");
                return MullamaErrorCode::Utf8Error.to_i32();
            }
        };

        rust_messages.push((role, content));
    }

    match model_ref.apply_chat_template(None, &rust_messages, add_generation_prompt) {
        Ok(formatted) => {
            let bytes = formatted.as_bytes();
            let len = bytes.len();

            if output.is_null() || max_output < len + 1 {
                return -(len as c_int + 1);
            }

            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, len);
                *output.add(len) = 0;
            }

            len as c_int
        }
        Err(e) => {
            set_last_error(format!("Chat template failed: {}", e));
            MullamaErrorCode::InvalidInput.to_i32()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = mullama_model_default_params();
        assert_eq!(params.n_gpu_layers, 0);
        assert!(params.use_mmap);
        assert!(!params.use_mlock);
    }

    #[test]
    fn test_null_model_handling() {
        assert!(mullama_model_load(std::ptr::null(), std::ptr::null()).is_null());
        assert_eq!(mullama_model_n_ctx_train(std::ptr::null()), 0);
        assert_eq!(mullama_model_n_embd(std::ptr::null()), 0);
        assert_eq!(mullama_model_n_vocab(std::ptr::null()), 0);
    }
}
