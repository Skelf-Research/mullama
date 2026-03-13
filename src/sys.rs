//! Raw FFI bindings to llama.cpp
//!
//! This module contains simplified manual FFI bindings to the most important
//! functions from llama.cpp. These bindings are not meant to be used directly
//! by most users. Instead, use the safe Rust API provided by the other modules.

use std::os::raw::{c_char, c_int, c_uint, c_float};

// Forward declarations for opaque types
#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone)]
pub struct llama_batch {
    pub n_tokens: c_int,
    pub token: *mut c_int,
    pub embd: *mut c_float,
    pub pos: *mut c_int,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut c_int,
    pub logits: *mut c_char,
}

// Type aliases
pub type LlamaToken = c_int;
pub type LlamaPos = c_int;
pub type LlamaSeqId = c_int;

// Function declarations
extern "C" {
    // Backend initialization
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    
    // Model functions
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> c_uint;
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    
    // Context functions
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_init_from_model(
        model: *const llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;
    
    // Tokenization functions
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut LlamaToken,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;
    
    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;
    
    // Batch functions
    pub fn llama_batch_get_one(
        tokens: *mut LlamaToken,
        n_tokens: c_int,
    ) -> llama_batch;
    
    pub fn llama_batch_init(
        n_tokens: c_int,
        embd: c_int,
        n_seq_max: c_int,
    ) -> llama_batch;
    
    pub fn llama_batch_free(batch: llama_batch);
    
    // State functions
    pub fn llama_state_get_size(ctx: *const llama_context) -> usize;
    pub fn llama_state_get_data(
        ctx: *const llama_context,
        dst: *mut u8,
        size: usize,
    ) -> usize;
    
    pub fn llama_state_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
    ) -> usize;
}

// Parameter structures
#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: c_int,
    pub use_mmap: bool,
    pub use_mlock: bool,
    // In a real implementation, there would be more fields here
    pub _padding: [u8; 100], // Placeholder for other fields
}

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: c_uint,
    pub n_batch: c_uint,
    pub n_threads: c_int,
    pub embeddings: bool,
    // In a real implementation, there would be more fields here
    pub _padding: [u8; 200], // Placeholder for other fields
}

// Forward declaration for vocabulary
#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}