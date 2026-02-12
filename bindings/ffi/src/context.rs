//! Context FFI bindings
//!
//! This module provides C ABI functions for creating contexts,
//! decoding tokens, and text generation.

use crate::error::{set_last_error, MullamaErrorCode};
use crate::handle::{Handle, MutableHandle};
use crate::model::MullamaModel;
use crate::sampler::MullamaSamplerParams;
use libc::{c_char, c_int, size_t};
use mullama::{Context, ContextParams, SamplerParams};
use std::sync::Arc;

/// Opaque context handle for FFI
pub type MullamaContext = MutableHandle<Context>;

/// Context creation parameters
#[repr(C)]
#[derive(Debug)]
pub struct MullamaContextParams {
    /// Context size (0 = use model default)
    pub n_ctx: u32,
    /// Logical batch size for prompt processing
    pub n_batch: u32,
    /// Physical batch size (memory allocation)
    pub n_ubatch: u32,
    /// Maximum number of sequences
    pub n_seq_max: u32,
    /// Number of threads for generation
    pub n_threads: c_int,
    /// Number of threads for batch processing
    pub n_threads_batch: c_int,
    /// Enable embeddings mode
    pub embeddings: bool,
    /// Offload KQV to GPU
    pub offload_kqv: bool,
    /// Flash attention type (0=auto, 1=disabled, 2=enabled)
    pub flash_attn: c_int,
}

impl Default for MullamaContextParams {
    fn default() -> Self {
        let n_threads = num_cpus::get() as c_int;
        Self {
            n_ctx: 0,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads,
            n_threads_batch: n_threads,
            embeddings: false,
            offload_kqv: true,
            flash_attn: 0,
        }
    }
}

/// Get default context parameters
#[no_mangle]
pub extern "C" fn mullama_context_default_params() -> MullamaContextParams {
    MullamaContextParams::default()
}

/// Create a new context from a model
///
/// # Arguments
/// * `model` - Model handle
/// * `params` - Optional context parameters (NULL for defaults)
///
/// # Returns
/// A context handle on success, or NULL on failure.
#[no_mangle]
pub extern "C" fn mullama_context_new(
    model: *const MullamaModel,
    params: *const MullamaContextParams,
) -> *mut MullamaContext {
    if model.is_null() {
        set_last_error("Model handle is null");
        return std::ptr::null_mut();
    }

    let model_arc = match unsafe { Handle::clone_arc(model) } {
        Some(arc) => arc,
        None => {
            set_last_error("Invalid model handle");
            return std::ptr::null_mut();
        }
    };

    let ctx_params = if params.is_null() {
        ContextParams::default()
    } else {
        let p = unsafe { &*params };
        ContextParams {
            n_ctx: p.n_ctx,
            n_batch: p.n_batch,
            n_ubatch: p.n_ubatch,
            n_seq_max: p.n_seq_max,
            n_threads: p.n_threads,
            n_threads_batch: p.n_threads_batch,
            embeddings: p.embeddings,
            offload_kqv: p.offload_kqv,
            ..Default::default()
        }
    };

    match Context::new(Arc::new((*model_arc).clone()), ctx_params) {
        Ok(context) => MutableHandle::new(context).into_raw(),
        Err(e) => {
            set_last_error(format!("Failed to create context: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free a context handle
#[no_mangle]
pub extern "C" fn mullama_context_free(ctx: *mut MullamaContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = MutableHandle::from_raw(ctx);
        }
    }
}

/// Decode tokens (process them through the model)
///
/// # Arguments
/// * `ctx` - Context handle
/// * `tokens` - Token IDs to decode
/// * `n_tokens` - Number of tokens
///
/// # Returns
/// 0 on success, negative error code on failure.
#[no_mangle]
pub extern "C" fn mullama_decode(
    ctx: *mut MullamaContext,
    tokens: *const c_int,
    n_tokens: c_int,
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

    let result = unsafe { MutableHandle::with_mut(ctx, |context| context.decode(token_slice)) };

    match result {
        Some(Ok(())) => MullamaErrorCode::Ok.to_i32(),
        Some(Err(e)) => {
            set_last_error(format!("Decode failed: {}", e));
            MullamaErrorCode::Generation.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Generate text from prompt tokens
///
/// # Arguments
/// * `ctx` - Context handle
/// * `tokens` - Prompt token IDs
/// * `n_tokens` - Number of prompt tokens
/// * `max_tokens` - Maximum tokens to generate
/// * `params` - Optional sampler parameters (NULL for defaults)
/// * `output` - Output buffer for generated text
/// * `max_output` - Size of output buffer
///
/// # Returns
/// Number of bytes written on success, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_generate(
    ctx: *mut MullamaContext,
    tokens: *const c_int,
    n_tokens: c_int,
    max_tokens: c_int,
    params: *const MullamaSamplerParams,
    output: *mut c_char,
    max_output: size_t,
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

    let result = unsafe {
        MutableHandle::with_mut(ctx, |context| {
            context.generate_with_params(token_slice, max_tokens as usize, &sampler_params)
        })
    };

    match result {
        Some(Ok(text)) => {
            let bytes = text.as_bytes();
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
        Some(Err(e)) => {
            set_last_error(format!("Generation failed: {}", e));
            MullamaErrorCode::Generation.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Get the context size
#[no_mangle]
pub extern "C" fn mullama_context_n_ctx(ctx: *const MullamaContext) -> u32 {
    if ctx.is_null() {
        return 0;
    }

    let result = unsafe { MutableHandle::with_ref(ctx, |context| context.n_ctx()) };
    result.unwrap_or(0)
}

/// Get the batch size
#[no_mangle]
pub extern "C" fn mullama_context_n_batch(ctx: *const MullamaContext) -> u32 {
    if ctx.is_null() {
        return 0;
    }

    let result = unsafe { MutableHandle::with_ref(ctx, |context| context.n_batch()) };
    result.unwrap_or(0)
}

/// Clear the KV cache
#[no_mangle]
pub extern "C" fn mullama_context_kv_cache_clear(ctx: *mut MullamaContext) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe { MutableHandle::with_mut(ctx, |context| context.kv_cache_clear()) };

    match result {
        Some(()) => MullamaErrorCode::Ok.to_i32(),
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Remove tokens from KV cache for a sequence
///
/// # Arguments
/// * `ctx` - Context handle
/// * `seq_id` - Sequence ID (-1 for all sequences)
/// * `p0` - Start position (inclusive)
/// * `p1` - End position (exclusive)
///
/// # Returns
/// 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn mullama_context_kv_cache_seq_rm(
    ctx: *mut MullamaContext,
    seq_id: c_int,
    p0: c_int,
    p1: c_int,
) -> c_int {
    if ctx.is_null() {
        return 0;
    }

    let result =
        unsafe { MutableHandle::with_mut(ctx, |context| context.kv_cache_seq_rm(seq_id, p0, p1)) };

    match result {
        Some(success) => {
            if success {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Get the number of threads used for generation
#[no_mangle]
pub extern "C" fn mullama_context_n_threads(ctx: *const MullamaContext) -> c_int {
    if ctx.is_null() {
        return 0;
    }

    let result = unsafe { MutableHandle::with_ref(ctx, |context| context.n_threads()) };
    result.unwrap_or(0)
}

/// Set the number of threads
#[no_mangle]
pub extern "C" fn mullama_context_set_n_threads(
    ctx: *mut MullamaContext,
    n_threads: c_int,
    n_threads_batch: c_int,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe {
        MutableHandle::with_mut(ctx, |context| {
            context.set_n_threads(n_threads, n_threads_batch)
        })
    };

    match result {
        Some(()) => MullamaErrorCode::Ok.to_i32(),
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Get logits for the last token
///
/// # Arguments
/// * `ctx` - Context handle
/// * `output` - Output buffer for logits
/// * `max_output` - Size of output buffer (should be >= n_vocab)
///
/// # Returns
/// Number of floats written, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_context_get_logits(
    ctx: *const MullamaContext,
    output: *mut f32,
    max_output: size_t,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if output.is_null() {
        set_last_error("Output buffer is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe {
        MutableHandle::with_ref(ctx, |context| {
            let logits = context.get_logits();
            if logits.is_empty() {
                return Err("No logits available");
            }

            let len = logits.len();
            if max_output < len {
                return Err("Buffer too small");
            }

            std::ptr::copy_nonoverlapping(logits.as_ptr(), output, len);
            Ok(len)
        })
    };

    match result {
        Some(Ok(len)) => len as c_int,
        Some(Err(msg)) => {
            set_last_error(msg);
            MullamaErrorCode::BufferTooSmall.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Get embeddings for the last token (requires embeddings mode)
///
/// # Arguments
/// * `ctx` - Context handle
/// * `output` - Output buffer for embeddings
/// * `max_output` - Size of output buffer (should be >= n_embd)
///
/// # Returns
/// Number of floats written, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_context_get_embeddings(
    ctx: *const MullamaContext,
    output: *mut f32,
    max_output: size_t,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if output.is_null() {
        set_last_error("Output buffer is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe {
        MutableHandle::with_ref(ctx, |context| match context.get_embeddings() {
            Some(embeddings) => {
                let len = embeddings.len();
                if max_output < len {
                    return Err("Buffer too small");
                }

                std::ptr::copy_nonoverlapping(embeddings.as_ptr(), output, len);
                Ok(len)
            }
            None => Err("No embeddings available - enable embeddings mode"),
        })
    };

    match result {
        Some(Ok(len)) => len as c_int,
        Some(Err(msg)) => {
            set_last_error(msg);
            MullamaErrorCode::NotAvailable.to_i32()
        }
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Save context state to a buffer
///
/// # Arguments
/// * `ctx` - Context handle
/// * `output` - Output buffer
/// * `max_output` - Size of output buffer
///
/// # Returns
/// Number of bytes written, or negative required size.
#[no_mangle]
pub extern "C" fn mullama_context_save_state(
    ctx: *const MullamaContext,
    output: *mut u8,
    max_output: size_t,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe {
        MutableHandle::with_ref(ctx, |context| {
            let state = context.save_state();
            let len = state.len();

            if output.is_null() || max_output < len {
                return Err(-(len as c_int));
            }

            std::ptr::copy_nonoverlapping(state.as_ptr(), output, len);
            Ok(len as c_int)
        })
    };

    match result {
        Some(Ok(len)) => len,
        Some(Err(neg_len)) => neg_len,
        None => {
            set_last_error("Failed to acquire context lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Load context state from a buffer
///
/// # Arguments
/// * `ctx` - Context handle
/// * `data` - State data buffer
/// * `data_size` - Size of state data
///
/// # Returns
/// Number of bytes read, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_context_load_state(
    ctx: *mut MullamaContext,
    data: *const u8,
    data_size: size_t,
) -> c_int {
    if ctx.is_null() {
        set_last_error("Context handle is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if data.is_null() || data_size == 0 {
        set_last_error("Invalid state data");
        return MullamaErrorCode::InvalidInput.to_i32();
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, data_size) };

    let result = unsafe { MutableHandle::with_mut(ctx, |context| context.load_state(data_slice)) };

    match result {
        Some(Ok(read)) => read as c_int,
        Some(Err(e)) => {
            set_last_error(format!("Failed to load state: {}", e));
            MullamaErrorCode::Context.to_i32()
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
    fn test_default_params() {
        let params = mullama_context_default_params();
        assert!(params.n_batch > 0);
        assert!(params.n_threads > 0);
    }

    #[test]
    fn test_null_handling() {
        assert!(mullama_context_new(std::ptr::null(), std::ptr::null()).is_null());
        assert_eq!(mullama_context_n_ctx(std::ptr::null()), 0);
    }
}
