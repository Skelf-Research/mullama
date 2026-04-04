use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};

use super::super::helpers::format_size;
use super::super::AppState;
use super::types::{ModelOperationResponse, PullModelRequest};

/// Pull a model from HuggingFace
pub(in crate::daemon::openai) async fn api_pull_model(
    State(_daemon): State<AppState>,
    Json(request): Json<PullModelRequest>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use crate::daemon::hf::{HfDownloader, HfModelSpec};
    use crate::daemon::registry::{resolve_model_name, ResolvedModel};

    let resolved = resolve_model_name(&request.name);

    let hf_spec = match resolved {
        ResolvedModel::HuggingFace { spec, .. } => spec,
        ResolvedModel::LocalPath(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ModelOperationResponse {
                    success: false,
                    message: "Cannot pull a local path".to_string(),
                    model: None,
                }),
            ));
        }
        ResolvedModel::Ollama { name, tag } => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ModelOperationResponse {
                    success: false,
                    message: format!(
                        "Ollama model '{}:{}' detected. Use CLI: mullama pull {}:{}",
                        name, tag, name, tag
                    ),
                    model: None,
                }),
            ));
        }
        ResolvedModel::Unknown(name) => {
            if name.starts_with("hf:") || name.contains('/') {
                if name.starts_with("hf:") {
                    name
                } else {
                    format!("hf:{}", name)
                }
            } else {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!(
                            "Unknown model '{}'. Use hf:owner/repo format or a known alias.",
                            name
                        ),
                        model: None,
                    }),
                ));
            }
        }
    };

    let spec = HfModelSpec::parse(&hf_spec).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Invalid HuggingFace spec: {}", hf_spec),
                model: None,
            }),
        )
    })?;

    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to initialize downloader: {}", e),
                model: None,
            }),
        )
    })?;

    let path = downloader.download_spec(&spec, false).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Download failed: {}", e),
                model: None,
            }),
        )
    })?;

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    Ok(Json(ModelOperationResponse {
        success: true,
        message: format!("Model '{}' downloaded successfully", request.name),
        model: Some(serde_json::json!({
            "name": spec.get_alias(),
            "source": "huggingface",
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "size": size,
            "size_formatted": format_size(size),
            "path": path.display().to_string(),
            "downloaded": chrono::Utc::now().to_rfc3339(),
        })),
    }))
}

/// Delete a model
pub(in crate::daemon::openai) async fn api_delete_model(
    State(_daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use crate::daemon::hf::HfDownloader;

    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to initialize: {}", e),
                model: None,
            }),
        )
    })?;

    let cached = downloader.list_cached();
    let mut found = None;
    for model in &cached {
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
            found = Some(model);
            break;
        }
    }

    let model = found.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Model '{}' not found", name),
                model: None,
            }),
        )
    })?;

    let size = model.size_bytes;
    let path = model.local_path.display().to_string();

    std::fs::remove_file(&model.local_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to delete: {}", e),
                model: None,
            }),
        )
    })?;

    Ok(Json(ModelOperationResponse {
        success: true,
        message: format!("Model '{}' deleted, freed {}", name, format_size(size)),
        model: Some(serde_json::json!({
            "name": name,
            "source": "huggingface",
            "repo_id": model.repo_id,
            "filename": model.filename,
            "size": size,
            "size_formatted": format_size(size),
            "path": path,
        })),
    }))
}
