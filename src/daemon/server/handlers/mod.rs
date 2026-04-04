mod embeddings;
mod management;
mod text;

#[cfg(feature = "multimodal")]
mod vision;

use crate::daemon::protocol::{
    generate_completion_id, unix_timestamp_secs, ChatChoice, ChatCompletionResponse, ChatMessage,
    CompletionChoice, CompletionResponse, Response, Usage,
};

/// Build a chat completion response from generation output.
///
/// Shared by text and vision chat completion handlers.
fn build_chat_completion_response(
    model_alias: &str,
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Response {
    Response::ChatCompletion(ChatCompletionResponse {
        id: generate_completion_id(),
        object: "chat.completion".to_string(),
        created: unix_timestamp_secs(),
        model: model_alias.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text.into(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        thinking: None,
    })
}

/// Build a text completion response from generation output.
///
/// Shared by text completion handlers.
fn build_completion_response(
    model_alias: &str,
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Response {
    Response::Completion(CompletionResponse {
        id: generate_completion_id(),
        object: "text_completion".to_string(),
        created: unix_timestamp_secs(),
        model: model_alias.to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}
