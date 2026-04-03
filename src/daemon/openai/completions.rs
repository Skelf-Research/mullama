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
    validate_n_parameter, CompletionChoice, CompletionChunk, CompletionChunkChoice,
    CompletionRequest, CompletionResponse,
};
use super::AppState;

/// POST /v1/completions
pub(super) async fn completions(
    State(daemon): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    validate_n_parameter(req.n, "text completions")?;

    if req.stream {
        return completions_stream(daemon, req).await;
    }

    let request = crate::daemon::protocol::Request::Completion {
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
    };

    match daemon.handle_request(request).await {
        crate::daemon::protocol::Response::Completion(resp) => Ok(Json(CompletionResponse {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp
                .choices
                .into_iter()
                .map(|c| CompletionChoice {
                    index: c.index,
                    text: c.text,
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

async fn completions_stream(
    daemon: AppState,
    req: CompletionRequest,
) -> Result<Response, ApiError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_completion_streaming(
            req.model,
            req.prompt,
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
                ApiError::new("Failed to start completion streaming")
            }
        })?;

    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let sse_stream = stream
        .map(move |chunk| {
            let sse_chunk = CompletionChunk {
                id: request_id_clone.clone(),
                object: "text_completion".to_string(),
                created,
                model: model_clone.clone(),
                choices: vec![CompletionChunkChoice {
                    index: chunk.index,
                    text: chunk.delta,
                    finish_reason: None,
                }],
            };

            Event::default().data(serde_json::to_string(&sse_chunk).unwrap_or_default())
        })
        .chain(stream::once(async move {
            let final_chunk = CompletionChunk {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: model_alias,
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: String::new(),
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
