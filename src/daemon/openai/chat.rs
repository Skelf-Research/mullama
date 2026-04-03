use std::convert::Infallible;
use std::time::Duration;

use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response, Sse},
};
use futures::stream::{self, StreamExt as _};
use tokio_stream::wrappers::ReceiverStream;

use super::error::ApiError;
use super::types::{
    validate_n_parameter, ChatChoice, ChatChoiceDelta, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, DeltaContent,
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
            match daemon
                .handle_vision_chat_completion(
                    req.model,
                    req.messages,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    None,
                    req.frequency_penalty,
                    req.presence_penalty,
                    req.stop.unwrap_or_default(),
                )
                .await
            {
                crate::daemon::protocol::Response::ChatCompletion(resp) => {
                    return Ok(Json(ChatCompletionResponse {
                        id: resp.id,
                        object: resp.object,
                        created: resp.created,
                        model: resp.model,
                        choices: resp
                            .choices
                            .into_iter()
                            .map(|c| ChatChoice {
                                index: c.index,
                                message: c.message,
                                finish_reason: c.finish_reason,
                            })
                            .collect(),
                        usage: resp.usage,
                    })
                    .into_response());
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

    let request = crate::daemon::protocol::Request::ChatCompletion {
        model: req.model,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        stream: false,
        stop: req.stop.unwrap_or_default(),
        response_format: req.response_format,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    match daemon.handle_request(request).await {
        crate::daemon::protocol::Response::ChatCompletion(resp) => Ok(Json(ChatCompletionResponse {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp
                .choices
                .into_iter()
                .map(|c| ChatChoice {
                    index: c.index,
                    message: c.message,
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: resp.usage,
        })
        .into_response()),
        crate::daemon::protocol::Response::Error { code, message, .. } => {
            Err(ApiError::from_protocol_error(code, message))
        }
        _ => Err(ApiError::new("Unexpected response")),
    }
}

async fn chat_completions_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_chat_completion_streaming(
            req.model,
            req.messages,
            req.max_tokens,
            req.temperature,
            req.top_p,
            None,
            req.frequency_penalty,
            req.presence_penalty,
            req.stop.unwrap_or_default(),
        )
        .await
        .map_err(|resp| {
            if let crate::daemon::protocol::Response::Error { message, .. } = resp {
                ApiError::new(message)
            } else {
                ApiError::new("Failed to start streaming")
            }
        })?;

    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let sse_stream = stream
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
        }))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);

    Ok(Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response())
}

#[cfg(feature = "multimodal")]
async fn chat_completions_vision_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_vision_chat_completion_streaming(
            req.model,
            req.messages,
            req.max_tokens,
            req.temperature,
            req.top_p,
            None,
            req.frequency_penalty,
            req.presence_penalty,
            req.stop.unwrap_or_default(),
        )
        .await
        .map_err(|resp| {
            if let crate::daemon::protocol::Response::Error { message, .. } = resp {
                ApiError::new(message)
            } else {
                ApiError::new("Failed to start vision streaming")
            }
        })?;

    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let sse_stream = stream
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
        }))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);

    Ok(Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response())
}
