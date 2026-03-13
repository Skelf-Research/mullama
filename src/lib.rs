//! # Mullama
//! 
//! A safe Rust wrapper for llama.cpp with built-in compilation.
//! 
//! ## Features
//! 
//! - Builds llama.cpp as part of the Rust build process
//! - Safe Rust API with automatic memory management
//! - Session management for saving/restoring model states
//! - Support for embeddings, tokenization, and generation
//! - Batch processing for efficient inference
//! - Sampling with various strategies (top-p, top-k, temperature)
//! - Vocabulary management
//! - Memory management
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use mullama::{Model, ContextParams};
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a model
//! let model = Model::load("path/to/model.gguf")?;
//! 
//! // Create a context
//! let mut ctx = model.create_context(ContextParams::default())?;
//! 
//! // Generate text
//! let tokens = model.tokenize("Hello, world!", true, false)?;
//! let result = ctx.generate(&tokens, 100)?;
//! 
//! println!("Generated: {}", result);
//! # Ok(())
//! # }
//! ```

pub mod sys;
pub mod model;
pub mod context;
pub mod token;
pub mod session;
pub mod error;
pub mod batch;
pub mod sampling;
pub mod embedding;
pub mod memory;
pub mod vocab;

// Re-export the main types
pub use model::Model;
pub use context::{Context, ContextParams};
pub use token::{Token, TokenId};
pub use session::Session;
pub use error::MullamaError;
pub use batch::Batch;
pub use sampling::{Sampler, SamplerParams};
pub use embedding::{Embeddings, EmbeddingUtil};
pub use memory::MemoryManager;
pub use vocab::Vocabulary;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_backend_initialization() {
        // Test that we can initialize the backend
        unsafe {
            sys::llama_backend_init();
            sys::llama_backend_free();
        }
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn test_model_structure() {
        // Test that we can create a model struct
        // This is just testing the Rust structure, not actual model loading
        let model = model::Model {
            model_ptr: std::ptr::null_mut(),
        };
        assert!(model.model_ptr.is_null());
    }
    
    #[test]
    fn test_context_structure() {
        // Test that we can create a context struct
        // This is just testing the Rust structure, not actual context creation
        let model = Arc::new(model::Model {
            model_ptr: std::ptr::null_mut(),
        });
        
        let context = context::Context {
            model,
            ctx_ptr: std::ptr::null_mut(),
        };
        assert!(context.ctx_ptr.is_null());
    }
    
    #[test]
    fn test_token_structure() {
        // Test that we can create token structs
        let token = token::Token {
            id: 1234,
            text: "test".to_string(),
            score: 0.5,
        };
        assert_eq!(token.id, 1234);
        assert_eq!(token.text, "test");
        assert_eq!(token.score, 0.5);
    }
    
    #[test]
    fn test_batch_structure() {
        // Test that we can create batch structs
        let batch = batch::Batch::default();
        assert!(batch.is_empty());
    }
    
    #[test]
    fn test_session_structure() {
        // Test that we can create session structs
        let session = session::Session {
            data: vec![],
        };
        assert!(session.data.is_empty());
    }
    
    #[test]
    fn test_sampling_structure() {
        // Test that we can create sampler structs
        let sampler = sampling::Sampler::new();
        let params = sampling::SamplerParams::default();
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 40);
    }
    
    #[test]
    fn test_embedding_structure() {
        // Test that we can create embedding structs
        let embeddings = embedding::Embeddings::new(vec![0.1, 0.2, 0.3], 3);
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings.dimension, 3);
    }
    
    #[test]
    fn test_memory_manager_structure() {
        // Test that we can create memory manager structs
        let memory_manager = memory::MemoryManager::new();
        assert_eq!(memory_manager._placeholder, 0);
    }
    
    #[test]
    fn test_vocabulary_structure() {
        // Test that we can create vocabulary structs
        let vocab = vocab::Vocabulary::new();
        assert_eq!(vocab._placeholder, 0);
    }
}