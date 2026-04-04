use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response},
};
use futures::stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;

use super::error::ApiError;
use super::helpers::{protocol_err_to_api, sse_response};
use super::types::{
    validate_n_parameter, ChatChoiceDelta, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, DeltaContent,
};
use super::AppState;

/// POST /v1/chat/completions
pub(super) async fn chat_completions(
    State(daemon): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_n_parameter(req.n, "chat completions")?;

    let has_images = req.messages.iter().any(|m| m.content.has_images());

    if req.stream {
        if has_images {
            #[cfg(feature = "multimodal")]
            return chat_completions_vision_stream(daemon, req).await;
            #[cfg(not(feature = "multimodal"))]
            return Err(ApiError::new("Vision support requires multimodal feature"));
        }
        return chat_completions_stream(daemon, req).await;
    }

    if has_images {
        #[cfg(feature = "multimodal")]
        {
            let params = crate::daemon::protocol::ChatCompletionParams::from(req);
            match daemon.handle_vision_chat_completion(params).await {
                crate::daemon::protocol::Response::ChatCompletion(resp) => {
                    return Ok(Json(ChatCompletionResponse::from(resp)).into_response());
                }
                crate::daemon::protocol::Response::Error { code, message, .. } => {
                    return Err(ApiError::from_protocol_error(code, message));
                }
                _ => return Err(ApiError::new("Unexpected response")),
            }
        }
        #[cfg(not(feature = "multimodal"))]
        return Err(ApiError::new("Vision support requires multimodal feature"));
    }

    let params = crate::daemon::protocol::ChatCompletionParams::from(req);
    let request = crate::daemon::protocol::Request::ChatCompletion(params);

    match daemon.handle_request(request).await {
        crate::daemon::protocol::Response::ChatCompletion(resp) => {
            Ok(Json(ChatCompletionResponse::from(resp)).into_response())
        }
        crate::daemon::protocol::Response::Error { code, message, .. } => {
            Err(ApiError::from_protocol_error(code, message))
        }
        _ => Err(ApiError::new("Unexpected response")),
    }
}

/// Build an SSE response from a streaming chat completion receiver.
///
/// Shared by both text and vision chat completion streaming paths.
fn chat_stream_to_sse(
    rx: tokio::sync::mpsc::Receiver<crate::daemon::protocol::StreamChunk>,
    request_id: String,
    model_alias: String,
) -> Response {
    use futures::stream;

    let created = crate::daemon::protocol::unix_timestamp_secs();

    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let event_stream = stream
        .map(move |chunk| {
            let sse_chunk = ChatCompletionChunk {
                id: request_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_clone.clone(),
                choices: vec![ChatChoiceDelta {
                    index: chunk.index,
                    delta: DeltaContent {
                        role: if chunk.index == 0 {
                            Some("assistant".to_string())
                        } else {
                            None
                        },
                        content: Some(chunk.delta),
                    },
                    finish_reason: None,
                }],
            };
            Event::default().data(serde_json::to_string(&sse_chunk).unwrap_or_default())
        })
        .chain(stream::once(async move {
            let final_chunk = ChatCompletionChunk {
                id: request_id,
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_alias,
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: DeltaContent {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            Event::default().data(serde_json::to_string(&final_chunk).unwrap_or_default())
        }));

    sse_response(event_stream)
}

async fn chat_completions_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let params = crate::daemon::protocol::ChatCompletionParams::from(req);
    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_chat_completion_streaming(params)
        .await
        .map_err(protocol_err_to_api)?;

    Ok(chat_stream_to_sse(rx, request_id, model_alias))
}

#[cfg(feature = "multimodal")]
async fn chat_completions_vision_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let params = crate::daemon::protocol::ChatCompletionParams::from(req);
    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_vision_chat_completion_streaming(params)
        .await
        .map_err(protocol_err_to_api)?;

    Ok(chat_stream_to_sse(rx, request_id, model_alias))
}
