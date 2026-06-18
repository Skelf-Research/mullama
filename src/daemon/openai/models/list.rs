use axum::extract::{Json, Path, State};

use super::super::helpers::format_size;
use super::super::types::{ModelObject, ModelsResponse};
use super::super::AppState;
use super::types::ModelOperationResponse;

/// GET /v1/models
pub(in crate::daemon::openai) async fn list_models(
    State(daemon): State<AppState>,
) -> Json<ModelsResponse> {
    let models = daemon.models.list();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|(alias, _info, _, _)| ModelObject {
                id: alias,
                object: "model".to_string(),
                created: super::super::types::unix_timestamp_secs(),
                owned_by: "local".to_string(),
            })
            .collect(),
    })
}

/// GET /v1/models/:model
pub(in crate::daemon::openai) async fn get_model(
    State(daemon): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelObject>, super::super::ApiError> {
    match daemon.models.get(Some(&model_id)).await {
        Ok(model) => Ok(Json(ModelObject {
            id: model.alias.clone(),
            object: "model".to_string(),
            created: super::super::types::unix_timestamp_secs(),
            owned_by: "local".to_string(),
        })),
        Err(_) => Err(super::super::ApiError::not_found(&model_id)),
    }
}

/// List all models (cached + running)
pub(in crate::daemon::openai) async fn api_list_models(
    State(daemon): State<AppState>,
) -> Json<serde_json::Value> {
    use crate::daemon::hf::HfDownloader;
    use crate::daemon::registry::registry;

    let mut models = Vec::new();

    if let Ok(downloader) = HfDownloader::new() {
        for cached in downloader.list_cached() {
            let short_name = format!(
                "{}:{}",
                cached
                    .repo_id
                    .split('/')
                    .next_back()
                    .unwrap_or(&cached.repo_id),
                cached.filename.trim_end_matches(".gguf")
            );

            models.push(serde_json::json!({
                "name": short_name,
                "source": "huggingface",
                "repo_id": cached.repo_id,
                "filename": cached.filename,
                "size": cached.size_bytes,
                "size_formatted": format_size(cached.size_bytes),
                "path": cached.local_path.display().to_string(),
                "downloaded": cached.downloaded_at,
                "loaded": false,
            }));
        }
    }

    let loaded = daemon.models.list();
    for (alias, info, is_default, active_requests) in loaded {
        let already_listed = models.iter().any(|m| {
            m.get("path")
                .and_then(|p| p.as_str())
                .map(|p| p == info.path)
                .unwrap_or(false)
        });

        if already_listed {
            for model in &mut models {
                if model.get("path").and_then(|p| p.as_str()) == Some(info.path.as_str()) {
                    model["loaded"] = serde_json::json!(true);
                    model["is_default"] = serde_json::json!(is_default);
                    model["active_requests"] = serde_json::json!(active_requests);
                    model["context_size"] = serde_json::json!(info.context_size);
                    model["gpu_layers"] = serde_json::json!(info.gpu_layers);
                }
            }
        } else {
            models.push(serde_json::json!({
                "name": alias,
                "source": "local",
                "size": 0,
                "size_formatted": "unknown",
                "path": info.path,
                "loaded": true,
                "is_default": is_default,
                "active_requests": active_requests,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
            }));
        }
    }

    let reg = registry();
    let aliases: Vec<_> = reg.list_aliases().iter().map(|a| a.to_string()).collect();

    Json(serde_json::json!({
        "models": models,
        "available_aliases": aliases,
        "total_cached": models.len(),
    }))
}

/// Get model details
pub(in crate::daemon::openai) async fn api_get_model(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<ModelOperationResponse>)> {
    use crate::daemon::hf::HfDownloader;

    let loaded = daemon.models.list();
    for (alias, info, is_default, active_requests) in &loaded {
        if alias == &name {
            return Ok(Json(serde_json::json!({
                "name": alias,
                "source": "loaded",
                "path": info.path,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
                "is_default": is_default,
                "active_requests": active_requests,
                "loaded": true,
            })));
        }
    }

    if let Ok(downloader) = HfDownloader::new() {
        for model in downloader.list_cached() {
            let short_name = format!(
                "{}:{}",
                model
                    .repo_id
                    .split('/')
                    .next_back()
                    .unwrap_or(&model.repo_id),
                model.filename.trim_end_matches(".gguf")
            );

            if model.filename == name
                || model.repo_id == name
                || short_name == name
                || model.filename.trim_end_matches(".gguf") == name
            {
                return Ok(Json(serde_json::json!({
                    "name": short_name,
                    "source": "huggingface",
                    "repo_id": model.repo_id,
                    "filename": model.filename,
                    "size": model.size_bytes,
                    "size_formatted": format_size(model.size_bytes),
                    "path": model.local_path.display().to_string(),
                    "downloaded": model.downloaded_at,
                    "loaded": false,
                })));
            }
        }
    }

    Err((
        axum::http::StatusCode::NOT_FOUND,
        Json(ModelOperationResponse {
            success: false,
            message: format!("Model '{}' not found", name),
            model: None,
        }),
    ))
}
