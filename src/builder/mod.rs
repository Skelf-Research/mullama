//! Builder patterns for complex configurations
//!
//! This module provides fluent, ergonomic builder patterns for configuring
//! Mullama components. Builders enforce correct construction order and
//! provide sensible defaults while allowing fine-grained control.
//!
//! ## Features
//!
//! - **Fluent API**: Chainable method calls for readable configuration
//! - **Type safety**: Compile-time guarantees about required parameters
//! - **Validation**: Built-in validation with helpful error messages
//! - **Defaults**: Sensible defaults for all optional parameters
//! - **Presets**: Quick configuration for common use cases
//! - **Progressive disclosure**: Start simple, add complexity as needed
//!
//! ## Example
//!
//! ```rust,ignore
//! use mullama::builder::{ModelBuilder, ContextBuilder, SamplerBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     // Build a complete model with fluent API
//!     let model = ModelBuilder::new()
//!         .path("path/to/model.gguf")
//!         .gpu_layers(32)
//!         .context_size(4096)
//!         .memory_mapping(true)
//!         .build()
//!         .await?;
//!
//!     // Build context with advanced configuration
//!     let context = ContextBuilder::new(model.clone())
//!         .context_size(4096)
//!         .batch_size(512)
//!         .threads(8)
//!         .embeddings(true)
//!         .flash_attention(true)
//!         .build()
//!         .await?;
//!
//!     // Build sophisticated sampling strategy
//!     let sampler = SamplerBuilder::new()
//!         .temperature(0.8)
//!         .top_k(50)
//!         .nucleus(0.95)
//!         .penalties(|p| p
//!             .repetition(1.1)
//!             .frequency(0.1)
//!             .presence(0.1)
//!         )
//!         .build(model.clone())?;
//!
//!     Ok(())
//! }
//! ```

mod model;
mod context;
mod sampler;

pub use model::ModelBuilder;
pub use context::ContextBuilder;
pub use sampler::{SamplerBuilder, PenaltyBuilder};

/// Preset configurations for builders
pub mod presets {
    use super::*;

    /// Creative writing model configuration
    pub fn creative_model(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(24)
            .context_size(4096)
            .memory_mapping(true)
    }

    /// Performance optimized model configuration
    pub fn performance_optimized(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(99) // Offload as much as possible
            .memory_mapping(true)
            .memory_locking(false)
            .tensor_validation(false) // Skip validation for speed
    }

    /// Memory optimized model configuration
    pub fn memory_optimized(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(0) // Use CPU only
            .memory_mapping(true)
            .memory_locking(false)
    }

    /// Creative sampling configuration
    pub fn creative_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.9)
            .top_k(60)
            .nucleus(0.95)
            .penalties(|p| p.repetition(1.15))
    }

    /// Precise sampling configuration
    pub fn precise_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.2)
            .top_k(10)
            .nucleus(0.85)
            .penalties(|p| p.repetition(1.05))
    }

    /// Balanced sampling configuration
    pub fn balanced_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.7)
            .top_k(40)
            .nucleus(0.9)
            .penalties(|p| p.repetition(1.1).frequency(0.1).presence(0.1))
    }
}
