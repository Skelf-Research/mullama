//! Streaming FFI bindings
//!
//! This module provides C ABI functions for callback-based streaming
//! text generation.

use crate::context::MullamaContext;
use crate::error::{set_last_error, MullamaErrorCode};
use crate::handle::MutableHandle;
use crate::sampler::MullamaSamplerParams;
use libc::{c_char, c_int, c_void};
use mullama::SamplerParams;
use std::ffi::CString;

/// Callback function type for streaming token generation
///
/// # Arguments
/// * `token` - The generated token text (null-terminated UTF-8)
/// * `user_data` - User-provided data pointer
///
/// # Returns
/// Return `true` to continue generation, `false` to stop.
pub type MullamaStreamCallback =
    extern "C" fn(token: *const c_char, user_data: *mut c_void) -> bool;

/// Generate text with streaming callback
///
/// # Arguments
/// * `ctx` - Context handle
/// * `tokens` - Prompt token IDs
/// * `n_tokens` - Number of prompt tokens
/// * `max_tokens` - Maximum tokens to generate
/// * `params` - Optional sampler parameters (NULL for defaults)
/// * `callback` - Callback function for each generated token
/// * `user_data` - User data to pass to callback
///
/// # Returns
/// Number of tokens generated on success, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_generate_streaming(
    ctx: *mut MullamaContext,
    tokens: *const c_int,
    n_tokens: c_int,
    max_tokens: c_int,
    params: *const MullamaSamplerParams,
    callback: MullamaStreamCallback,
    user_data: *mut c_void,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if tokens.is_null() || n_tokens <= 0 {
        set_last_error("Invalid tokens array");
        return MullamaErrorCode::InvalidInput.to_i32();
    }

    let token_slice = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };

    let sampler_params = if params.is_null() {
        SamplerParams::default()
    } else {
        let p = unsafe { &*params };
        SamplerParams {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            penalty_repeat: p.penalty_repeat,
            penalty_freq: p.penalty_freq,
            penalty_present: p.penalty_present,
            penalty_last_n: p.penalty_last_n,
            seed: p.seed,
            ..Default::default()
        }
    };

    // Track generated tokens for return value
    let mut token_count = 0i32;

    let result = unsafe {
        MutableHandle::with_mut(ctx, |context| {
            context.generate_streaming(
                token_slice,
                max_tokens as usize,
                &sampler_params,
                |piece| {
                    // Convert piece to C string
                    match CString::new(piece) {
                        Ok(c_str) => {
                            let should_continue = callback(c_str.as_ptr(), user_data);
                            token_count += 1;
                            should_continue
                        }
                        Err(_) => {
                            // Skip tokens that can't be converted (shouldn't happen)
                            true
                        }
                    }
                },
            )
        })
    };

    match result {
        Some(Ok(_)) => token_count,
        Some(Err(e)) => {
            set_last_error(format!("Streaming generation failed: {}", e));
            MullamaErrorCode::Generation.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Extended streaming callback with additional metadata
pub type MullamaStreamCallbackEx = extern "C" fn(
    token: *const c_char,
    token_id: c_int,
    is_eos: bool,
    user_data: *mut c_void,
) -> bool;

/// Cancellation token for streaming operations
#[repr(C)]
pub struct MullamaCancelToken {
    cancelled: std::sync::atomic::AtomicBool,
}

impl MullamaCancelToken {
    fn new() -> Self {
        Self {
            cancelled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Create a new cancellation token
#[no_mangle]
pub extern "C" fn mullama_cancel_token_new() -> *mut MullamaCancelToken {
    Box::into_raw(Box::new(MullamaCancelToken::new()))
}

/// Cancel the operation associated with this token
#[no_mangle]
pub extern "C" fn mullama_cancel_token_cancel(token: *mut MullamaCancelToken) {
    if !token.is_null() {
        unsafe {
            (*token)
                .cancelled
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Check if the token has been cancelled
#[no_mangle]
pub extern "C" fn mullama_cancel_token_is_cancelled(token: *const MullamaCancelToken) -> bool {
    if token.is_null() {
        return false;
    }
    unsafe { (*token).is_cancelled() }
}

/// Free a cancellation token
#[no_mangle]
pub extern "C" fn mullama_cancel_token_free(token: *mut MullamaCancelToken) {
    if !token.is_null() {
        unsafe {
            let _ = Box::from_raw(token);
        }
    }
}

/// Generate text with streaming callback and cancellation support
///
/// # Arguments
/// * `ctx` - Context handle
/// * `tokens` - Prompt token IDs
/// * `n_tokens` - Number of prompt tokens
/// * `max_tokens` - Maximum tokens to generate
/// * `params` - Optional sampler parameters (NULL for defaults)
/// * `callback` - Callback function for each generated token
/// * `user_data` - User data to pass to callback
/// * `cancel_token` - Optional cancellation token (NULL to ignore)
///
/// # Returns
/// Number of tokens generated on success, or negative error code.
/// Returns MULLAMA_ERR_CANCELLED if generation was cancelled.
#[no_mangle]
pub extern "C" fn mullama_generate_streaming_cancellable(
    ctx: *mut MullamaContext,
    tokens: *const c_int,
    n_tokens: c_int,
    max_tokens: c_int,
    params: *const MullamaSamplerParams,
    callback: MullamaStreamCallback,
    user_data: *mut c_void,
    cancel_token: *const MullamaCancelToken,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if tokens.is_null() || n_tokens <= 0 {
        set_last_error("Invalid tokens array");
        return MullamaErrorCode::InvalidInput.to_i32();
    }

    let token_slice = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };

    let sampler_params = if params.is_null() {
        SamplerParams::default()
    } else {
        let p = unsafe { &*params };
        SamplerParams {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            penalty_repeat: p.penalty_repeat,
            penalty_freq: p.penalty_freq,
            penalty_present: p.penalty_present,
            penalty_last_n: p.penalty_last_n,
            seed: p.seed,
            ..Default::default()
        }
    };

    let mut token_count = 0i32;
    let mut was_cancelled = false;

    let result = unsafe {
        MutableHandle::with_mut(ctx, |context| {
            context.generate_streaming(
                token_slice,
                max_tokens as usize,
                &sampler_params,
                |piece| {
                    // Check cancellation
                    if !cancel_token.is_null() && (*cancel_token).is_cancelled() {
                        was_cancelled = true;
                        return false;
                    }

                    // Convert piece to C string and invoke callback
                    match CString::new(piece) {
                        Ok(c_str) => {
                            let should_continue = callback(c_str.as_ptr(), user_data);
                            token_count += 1;
                            should_continue
                        }
                        Err(_) => true,
                    }
                },
            )
        })
    };

    if was_cancelled {
        set_last_error("Generation was cancelled");
        return MullamaErrorCode::Cancelled.to_i32();
    }

    match result {
        Some(Ok(_)) => token_count,
        Some(Err(e)) => {
            set_last_error(format!("Streaming generation failed: {}", e));
            MullamaErrorCode::Generation.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_token() {
        let token = mullama_cancel_token_new();
        assert!(!mullama_cancel_token_is_cancelled(token));

        mullama_cancel_token_cancel(token);
        assert!(mullama_cancel_token_is_cancelled(token));

        mullama_cancel_token_free(token);
    }

    #[test]
    fn test_null_cancel_token() {
        assert!(!mullama_cancel_token_is_cancelled(std::ptr::null()));
    }
}
