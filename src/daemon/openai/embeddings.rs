use axum::extract::{Json, State};

use super::error::ApiError;
use super::types::{EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse};
use super::AppState;

/// POST /v1/embeddings
pub(super) async fn embeddings(
    State(daemon): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    match daemon.handle_embeddings(req.model, req.input).await {
        crate::daemon::protocol::Response::Embeddings(resp) => {
            let data = resp
                .data
                .into_iter()
                .map(|d| EmbeddingObject {
                    object: d.object,
                    embedding: d.embedding,
                    index: d.index,
                })
                .collect();
            Ok(Json(EmbeddingsResponse {
                object: resp.object,
                data,
                model: resp.model,
                usage: resp.usage,
            }))
        }
        crate::daemon::protocol::Response::Error { message, .. } => Err(ApiError::new(&message)),
        _ => Err(ApiError::new("Unexpected response from embeddings handler")),
    }
}
