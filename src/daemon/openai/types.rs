use serde::{Deserialize, Serialize};

use super::error::ApiError;
use crate::daemon::protocol::ResponseFormat;
use crate::daemon::protocol::{ChatMessage, EmbeddingInput, Timings, Usage};

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
    /// Cross-turn KV-reuse session id (mullama extension). Same id across
    /// requests reuses the pinned context's KV cache (delta-only prefill).
    #[serde(default)]
    pub session: Option<String>,
    /// Sliding-window turn bound for the session (mullama extension). Keeps
    /// the last N user turns, bounding prompt + KV for long sessions.
    #[serde(default)]
    pub session_keep_turns: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
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

pub fn unix_timestamp_secs() -> u64 {
    crate::daemon::protocol::unix_timestamp_secs()
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

// ── From impls for protocol → OpenAI type conversions ──

impl From<crate::daemon::protocol::ChatCompletionResponse> for ChatCompletionResponse {
    fn from(resp: crate::daemon::protocol::ChatCompletionResponse) -> Self {
        Self {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp.choices.into_iter().map(ChatChoice::from).collect(),
            usage: resp.usage,
            timings: resp.timings,
        }
    }
}

impl From<crate::daemon::protocol::ChatChoice> for ChatChoice {
    fn from(c: crate::daemon::protocol::ChatChoice) -> Self {
        Self {
            index: c.index,
            message: c.message,
            finish_reason: c.finish_reason,
        }
    }
}

impl From<crate::daemon::protocol::CompletionResponse> for CompletionResponse {
    fn from(resp: crate::daemon::protocol::CompletionResponse) -> Self {
        Self {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp
                .choices
                .into_iter()
                .map(CompletionChoice::from)
                .collect(),
            usage: resp.usage,
            timings: resp.timings,
        }
    }
}

impl From<crate::daemon::protocol::CompletionChoice> for CompletionChoice {
    fn from(c: crate::daemon::protocol::CompletionChoice) -> Self {
        Self {
            index: c.index,
            text: c.text,
            finish_reason: c.finish_reason,
        }
    }
}

// ── From impls for OpenAI request → protocol param conversions ──

impl From<ChatCompletionRequest> for crate::daemon::protocol::ChatCompletionParams {
    fn from(req: ChatCompletionRequest) -> Self {
        Self {
            model: req.model,
            messages: req.messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: None,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            stream: req.stream,
            stop: req.stop.unwrap_or_default(),
            response_format: req.response_format,
            tools: None,
            tool_choice: None,
            thinking: None,
            session: req.session,
            session_keep_turns: req.session_keep_turns,
        }
    }
}

impl From<CompletionRequest> for crate::daemon::protocol::CompletionParams {
    fn from(req: CompletionRequest) -> Self {
        Self {
            model: req.model,
            prompt: req.prompt,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: None,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            stream: req.stream,
            stop: req.stop.unwrap_or_default(),
        }
    }
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
