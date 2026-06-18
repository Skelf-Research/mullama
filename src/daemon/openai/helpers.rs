use std::convert::Infallible;
use std::time::Duration;

use axum::response::{sse::Event, IntoResponse, Response, Sse};
use futures::stream::{self, StreamExt as _};

/// Build an SSE response from a stream of events.
///
/// Appends the `[DONE]` sentinel and configures keep-alive. Shared by
/// chat completion, vision chat completion, and text completion streaming.
pub(super) fn sse_response(
    event_stream: impl futures::Stream<Item = Event> + Send + 'static,
) -> Response {
    let sse_stream = event_stream
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);

    Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

/// Convert a protocol `Response::Error` into an `ApiError`.
///
/// Shared by all streaming endpoints that call `handle_*_streaming` methods
/// which return `Result<_, Response>`.
pub(super) fn protocol_err_to_api(
    resp: crate::daemon::protocol::Response,
) -> super::error::ApiError {
    if let crate::daemon::protocol::Response::Error { message, .. } = resp {
        super::error::ApiError::new(message)
    } else {
        super::error::ApiError::new("Failed to start streaming")
    }
}

pub(super) use crate::daemon::protocol::format_size;

pub(super) fn model_config_from_modelfile(
    modelfile: &crate::modelfile::Modelfile,
) -> crate::daemon::models::ModelConfig {
    let mut stop_sequences = modelfile.stop_sequences.clone();
    if stop_sequences.is_empty() {
        if let Some(stop) = modelfile.stop() {
            stop_sequences = stop;
        }
    }

    crate::daemon::models::ModelConfig {
        stop_sequences,
        system_prompt: modelfile.system.clone(),
        temperature: modelfile.temperature().map(|v| v as f32),
        top_p: modelfile.top_p().map(|v| v as f32),
        top_k: modelfile.top_k().and_then(|v| i32::try_from(v).ok()),
        context_size: modelfile.num_ctx().and_then(|v| u32::try_from(v).ok()),
    }
}
