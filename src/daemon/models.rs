//! Multi-model manager for the daemon.
//!
//! The public daemon surface stays the same, but the internals are now split
//! into focused modules so runtime state, load configuration, and manager
//! orchestration can evolve independently.

mod config;
mod loaded;
mod manager;
mod pool;
mod stats;

pub use config::{ModelConfig, ModelLoadConfig, DEFAULT_CONTEXT_POOL_SIZE};
pub use loaded::{LoadedModel, RequestGuard};
pub use manager::ModelManager;
pub use stats::{estimate_model_memory, MemoryEstimate, ModelStats};
