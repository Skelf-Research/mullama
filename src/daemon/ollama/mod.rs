//! Ollama Registry Client
//!
//! Downloads and caches models from Ollama's registry (registry.ollama.ai).
//! Provides full compatibility with Ollama's model naming and configuration.
//!
//! ## Model Specification Formats
//!
//! ```text
//! # Official library models
//! llama3              -> registry.ollama.ai/library/llama3:latest
//! llama3:1b           -> registry.ollama.ai/library/llama3:1b
//! llama3:70b-instruct -> registry.ollama.ai/library/llama3:70b-instruct
//!
//! # User models
//! user/mymodel:v1     -> registry.ollama.ai/user/mymodel:v1
//!
//! # Explicit ollama: prefix
//! ollama:llama3:1b    -> registry.ollama.ai/library/llama3:1b
//! ```

mod client;
mod types;

pub use client::*;
pub use types::*;

/// Ollama registry URL
const OLLAMA_REGISTRY_URL: &str = "https://registry.ollama.ai";

/// Cache directory for Ollama models
const OLLAMA_CACHE_DIR: &str = "ollama";

/// Layer media types
pub const LAYER_MODEL: &str = "application/vnd.ollama.image.model";
pub const LAYER_TEMPLATE: &str = "application/vnd.ollama.image.template";
pub const LAYER_PARAMS: &str = "application/vnd.ollama.image.params";
pub const LAYER_SYSTEM: &str = "application/vnd.ollama.image.system";
pub const LAYER_PROJECTOR: &str = "application/vnd.ollama.image.projector";
pub const LAYER_LICENSE: &str = "application/vnd.ollama.image.license";
pub const LAYER_MESSAGES: &str = "application/vnd.ollama.image.messages";
