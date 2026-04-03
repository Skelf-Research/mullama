//! OpenAI-compatible HTTP API
//!
//! Provides REST endpoints compatible with the OpenAI API specification.

mod chat;
mod completions;
mod defaults;
mod embeddings;
mod error;
mod helpers;
mod middleware;
mod models;
mod router;
mod system;
mod types;
mod ui;

use std::sync::Arc;

use super::server::Daemon;

/// Shared state for the HTTP server
pub type AppState = Arc<Daemon>;

pub use error::ApiError;
pub use router::create_openai_router;
pub use types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, CompletionChoice,
    CompletionRequest, CompletionResponse, EmbeddingObject, EmbeddingsRequest,
    EmbeddingsResponse, ErrorDetail, ErrorResponse, ModelObject, ModelsResponse,
};
