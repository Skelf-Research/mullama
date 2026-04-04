use axum::{
    extract::{Json, State},
    response::{sse::Event, IntoResponse, Response},
};
use futures::stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;

use super::error::ApiError;
use super::helpers::{protocol_err_to_api, sse_response};
use super::types::{
    validate_n_parameter, CompletionChunk, CompletionChunkChoice, CompletionRequest,
    CompletionResponse,
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

    let params = crate::daemon::protocol::CompletionParams::from(req);
    let request = crate::daemon::protocol::Request::Completion(params);

    match daemon.handle_request(request).await {
        crate::daemon::protocol::Response::Completion(resp) => {
            Ok(Json(CompletionResponse::from(resp)).into_response())
        }
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
    let created = crate::daemon::protocol::unix_timestamp_secs();

    let params = crate::daemon::protocol::CompletionParams::from(req);
    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_completion_streaming(params)
        .await
        .map_err(protocol_err_to_api)?;

    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let event_stream = stream
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
        .chain(futures::stream::once(async move {
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
        }));

    Ok(sse_response(event_stream))
}
