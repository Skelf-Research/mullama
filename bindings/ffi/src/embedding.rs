//! Embedding FFI bindings
//!
//! This module provides C ABI functions for generating text embeddings.

use crate::error::{set_last_error, MullamaErrorCode};
use crate::handle::{Handle, MutableHandle};
use crate::model::MullamaModel;
use libc::{c_char, c_int, size_t};
use mullama::{Context, ContextParams, Model};
use std::ffi::CStr;
use std::sync::Arc;

/// Embedding generator handle
pub struct EmbeddingGeneratorInner {
    context: Context,
    _model: Arc<Model>,
}

pub type MullamaEmbeddingGenerator = MutableHandle<EmbeddingGeneratorInner>;

/// Configuration for embedding generation
#[repr(C)]
#[derive(Debug)]
pub struct MullamaEmbeddingConfig {
    /// Context size (0 = use model default)
    pub n_ctx: u32,
    /// Batch size for processing
    pub n_batch: u32,
    /// Number of threads
    pub n_threads: c_int,
    /// Pooling type: 0=none, 1=mean, 2=cls
    pub pooling_type: c_int,
    /// Normalize embeddings
    pub normalize: bool,
}

impl Default for MullamaEmbeddingConfig {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 512,
            n_threads: num_cpus::get() as c_int,
            pooling_type: 1, // Mean pooling
            normalize: true,
        }
    }
}

/// Get default embedding configuration
#[no_mangle]
pub extern "C" fn mullama_embedding_default_config() -> MullamaEmbeddingConfig {
    MullamaEmbeddingConfig::default()
}

/// Create a new embedding generator
///
/// # Arguments
/// * `model` - Model handle
/// * `config` - Optional configuration (NULL for defaults)
///
/// # Returns
/// An embedding generator handle on success, or NULL on failure.
#[no_mangle]
pub extern "C" fn mullama_embedding_generator_new(
    model: *const MullamaModel,
    config: *const MullamaEmbeddingConfig,
) -> *mut MullamaEmbeddingGenerator {
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

    let cfg = if config.is_null() {
        MullamaEmbeddingConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Create context with embeddings enabled
    let mut ctx_params = ContextParams::default();
    ctx_params.n_ctx = cfg.n_ctx;
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;
    ctx_params.embeddings = true; // Enable embeddings mode

    // Set pooling type
    ctx_params.pooling_type = match cfg.pooling_type {
        0 => mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_NONE,
        1 => mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
        2 => mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_CLS,
        3 => mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_LAST,
        _ => mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
    };

    match Context::new(Arc::new((*model_arc).clone()), ctx_params) {
        Ok(context) => {
            let inner = EmbeddingGeneratorInner {
                context,
                _model: model_arc,
            };
            MutableHandle::new(inner).into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to create embedding context: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free an embedding generator handle
#[no_mangle]
pub extern "C" fn mullama_embedding_generator_free(gen: *mut MullamaEmbeddingGenerator) {
    if !gen.is_null() {
        unsafe {
            let _ = MutableHandle::from_raw(gen);
        }
    }
}

/// Generate embeddings for text
///
/// # Arguments
/// * `gen` - Embedding generator handle
/// * `text` - Text to embed
/// * `output` - Output buffer for embeddings
/// * `max_output` - Size of output buffer (should be >= n_embd)
///
/// # Returns
/// Number of floats written on success, or negative error code.
#[no_mangle]
pub extern "C" fn mullama_embed_text(
    gen: *mut MullamaEmbeddingGenerator,
    text: *const c_char,
    output: *mut f32,
    max_output: size_t,
) -> c_int {
    if gen.is_null() {
        set_last_error("Embedding generator is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if text.is_null() {
        set_last_error("Text is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if output.is_null() {
        set_last_error("Output buffer is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in text");
            return MullamaErrorCode::Utf8Error.to_i32();
        }
    };

    let result = unsafe {
        MutableHandle::with_mut(gen, |inner| {
            // Tokenize the text
            let tokens = inner.context.model.tokenize(text_str, true, false)?;

            // Clear cache and decode
            inner.context.kv_cache_clear();
            inner.context.decode(&tokens)?;

            // Get embeddings
            match inner.context.get_embeddings() {
                Some(embeddings) => {
                    let len = embeddings.len();
                    if max_output < len {
                        return Err(mullama::MullamaError::InvalidInput(
                            "Buffer too small".to_string(),
                        ));
                    }

                    std::ptr::copy_nonoverlapping(embeddings.as_ptr(), output, len);
                    Ok(len as c_int)
                }
                None => Err(mullama::MullamaError::InvalidInput(
                    "No embeddings available".to_string(),
                )),
            }
        })
    };

    match result {
        Some(Ok(len)) => len,
        Some(Err(e)) => {
            set_last_error(format!("Embedding failed: {}", e));
            MullamaErrorCode::Embedding.to_i32()
        }
        None => {
            set_last_error("Failed to acquire lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Generate embeddings for multiple texts in a batch
///
/// # Arguments
/// * `gen` - Embedding generator handle
/// * `texts` - Array of text pointers
/// * `n_texts` - Number of texts
/// * `output` - Output buffer for embeddings (flattened: n_texts * n_embd)
/// * `max_output` - Size of output buffer
///
/// # Returns
/// Number of floats written on success (n_texts * n_embd), or negative error code.
#[no_mangle]
pub extern "C" fn mullama_embed_batch(
    gen: *mut MullamaEmbeddingGenerator,
    texts: *const *const c_char,
    n_texts: c_int,
    output: *mut f32,
    max_output: size_t,
) -> c_int {
    if gen.is_null() {
        set_last_error("Embedding generator is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    if texts.is_null() || n_texts <= 0 {
        set_last_error("Invalid texts array");
        return MullamaErrorCode::InvalidInput.to_i32();
    }

    if output.is_null() {
        set_last_error("Output buffer is null");
        return MullamaErrorCode::NullPointer.to_i32();
    }

    let result = unsafe {
        MutableHandle::with_mut(gen, |inner| {
            let n_embd = inner.context.model.n_embd() as usize;
            let total_floats = n_embd * (n_texts as usize);

            if max_output < total_floats {
                return Err(mullama::MullamaError::InvalidInput(
                    "Buffer too small for batch embeddings".to_string(),
                ));
            }

            let mut offset = 0usize;

            for i in 0..n_texts {
                let text_ptr = *texts.add(i as usize);
                if text_ptr.is_null() {
                    return Err(mullama::MullamaError::InvalidInput(format!(
                        "Text {} is null",
                        i
                    )));
                }

                let text_str = match CStr::from_ptr(text_ptr).to_str() {
                    Ok(s) => s,
                    Err(_) => {
                        return Err(mullama::MullamaError::InvalidInput(format!(
                            "Invalid UTF-8 in text {}",
                            i
                        )))
                    }
                };

                // Tokenize and decode
                let tokens = inner.context.model.tokenize(text_str, true, false)?;
                inner.context.kv_cache_clear();
                inner.context.decode(&tokens)?;

                // Get embeddings
                match inner.context.get_embeddings() {
                    Some(embeddings) => {
                        let out_ptr = output.add(offset);
                        std::ptr::copy_nonoverlapping(
                            embeddings.as_ptr(),
                            out_ptr,
                            embeddings.len(),
                        );
                        offset += embeddings.len();
                    }
                    None => {
                        return Err(mullama::MullamaError::InvalidInput(format!(
                            "No embeddings for text {}",
                            i
                        )))
                    }
                }
            }

            Ok(offset as c_int)
        })
    };

    match result {
        Some(Ok(len)) => len,
        Some(Err(e)) => {
            set_last_error(format!("Batch embedding failed: {}", e));
            MullamaErrorCode::Embedding.to_i32()
        }
        None => {
            set_last_error("Failed to acquire lock");
            MullamaErrorCode::LockError.to_i32()
        }
    }
}

/// Get the embedding dimension for the generator
#[no_mangle]
pub extern "C" fn mullama_embedding_generator_n_embd(
    gen: *const MullamaEmbeddingGenerator,
) -> c_int {
    if gen.is_null() {
        return 0;
    }

    let result = unsafe { MutableHandle::with_ref(gen, |inner| inner.context.model.n_embd()) };

    result.unwrap_or(0)
}

/// Compute cosine similarity between two embedding vectors
///
/// # Arguments
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
/// * `n` - Dimension of the vectors
///
/// # Returns
/// Cosine similarity value between -1 and 1
#[no_mangle]
pub extern "C" fn mullama_embedding_cosine_similarity(
    a: *const f32,
    b: *const f32,
    n: size_t,
) -> f32 {
    if a.is_null() || b.is_null() || n == 0 {
        return 0.0;
    }

    let a_slice = unsafe { std::slice::from_raw_parts(a, n) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, n) };

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..n {
        dot += a_slice[i] * b_slice[i];
        norm_a += a_slice[i] * a_slice[i];
        norm_b += b_slice[i] * b_slice[i];
    }

    let norm = norm_a.sqrt() * norm_b.sqrt();
    if norm == 0.0 {
        0.0
    } else {
        dot / norm
    }
}

/// Normalize an embedding vector in-place
///
/// # Arguments
/// * `embedding` - Embedding vector to normalize
/// * `n` - Dimension of the vector
#[no_mangle]
pub extern "C" fn mullama_embedding_normalize(embedding: *mut f32, n: size_t) {
    if embedding.is_null() || n == 0 {
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(embedding, n) };

    let mut norm = 0.0f32;
    for &val in slice.iter() {
        norm += val * val;
    }
    norm = norm.sqrt();

    if norm > 0.0 {
        for val in slice.iter_mut() {
            *val /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let sim = mullama_embedding_cosine_similarity(a.as_ptr(), b.as_ptr(), 3);
        assert!((sim - 1.0).abs() < 0.001);

        let c = [0.0f32, 1.0, 0.0];
        let sim2 = mullama_embedding_cosine_similarity(a.as_ptr(), c.as_ptr(), 3);
        assert!(sim2.abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let mut vec = [3.0f32, 4.0, 0.0];
        mullama_embedding_normalize(vec.as_mut_ptr(), 3);

        // Should be [0.6, 0.8, 0.0]
        assert!((vec[0] - 0.6).abs() < 0.001);
        assert!((vec[1] - 0.8).abs() < 0.001);
        assert!(vec[2].abs() < 0.001);
    }

    #[test]
    fn test_default_config() {
        let config = mullama_embedding_default_config();
        assert!(config.n_ctx > 0);
        assert!(config.n_batch > 0);
        assert!(config.normalize);
    }
}
