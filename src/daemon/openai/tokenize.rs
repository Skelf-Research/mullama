//! POST /v1/tokenize
//!
//! Non-standard extension (mullama-only). Returns the token ids the loaded
//! model's tokenizer assigns to the input text, so tooling such as the parity
//! bench can compare two engines' outputs at the token level using a single
//! shared tokenizer (the same GGUF) rather than guessing from detokenized text.

use axum::extract::{Json, State};
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::AppState;

#[derive(Debug, Deserialize)]
pub(super) struct TokenizeRequest {
    /// Model alias (optional: resolves to the default model).
    pub model: Option<String>,
    /// Text to tokenize.
    pub text: String,
}

#[derive(Debug, Serialize)]
pub(super) struct TokenizeResponse {
    pub model: String,
    pub tokens: Vec<i32>,
    pub count: u32,
}

/// POST /v1/tokenize
pub(super) async fn tokenize(
    State(daemon): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    match daemon.handle_tokenize(req.model.clone(), &req.text).await {
        crate::daemon::protocol::Response::Tokens { tokens, count } => Ok(Json(TokenizeResponse {
            model: req.model.unwrap_or_default(),
            tokens,
            count: count as u32,
        })),
        crate::daemon::protocol::Response::Error { message, .. } => Err(ApiError::new(&message)),
        _ => Err(ApiError::new("Unexpected response from tokenize handler")),
    }
}