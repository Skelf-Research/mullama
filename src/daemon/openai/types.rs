use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::error::ApiError;
use crate::daemon::protocol::{ChatMessage, EmbeddingInput, Usage};
use crate::daemon::protocol::ResponseFormat;

/// Chat completion request (OpenAI compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
    /// Response format for structured outputs (JSON Schema validation)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

/// Chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Streaming chat completion chunk (OpenAI compatible)
#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatChoiceDelta {
    pub index: u32,
    pub delta: DeltaContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Text completion request
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(super) struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Embeddings request
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// Embeddings response
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

pub(super) fn default_max_tokens() -> u32 {
    512
}

pub(super) fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) fn validate_n_parameter(n: Option<u32>, endpoint: &str) -> Result<(), ApiError> {
    if n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request(format!(
            "Only n=1 is currently supported for {}",
            endpoint
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn validate_n_parameter_accepts_default_and_one() {
        assert!(validate_n_parameter(None, "chat completions").is_ok());
        assert!(validate_n_parameter(Some(1), "chat completions").is_ok());
    }

    #[test]
    fn validate_n_parameter_rejects_multiple_choices() {
        let err = validate_n_parameter(Some(2), "chat completions").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("n=1"));
    }
}
