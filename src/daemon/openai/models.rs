use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use super::helpers::format_size;
use super::types::{ModelObject, ModelsResponse};
use super::AppState;
use crate::daemon::models::ModelConfig;

/// Request to pull a model
#[derive(Debug, Deserialize)]
pub(super) struct PullModelRequest {
    /// Model name or HuggingFace spec
    pub name: String,
}

/// Request to load a model into the daemon
#[derive(Debug, Deserialize)]
pub(super) struct LoadModelRequest {
    pub name: String,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub use_mmap: Option<bool>,
    #[serde(default)]
    pub use_mlock: bool,
    #[serde(default)]
    pub rope_freq_base: Option<f32>,
    #[serde(default)]
    pub rope_freq_scale: Option<f32>,
    #[serde(default)]
    pub n_batch: Option<u32>,
    #[serde(default)]
    pub defrag_thold: Option<f32>,
    #[serde(default)]
    pub split_mode: Option<String>,
}

/// Response for model operations
#[derive(Debug, Serialize)]
pub(super) struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<serde_json::Value>,
}

/// Detailed model information
#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub(super) struct ModelDetails {
    pub name: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub size: u64,
    pub size_formatted: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded: Option<String>,
}

/// GET /v1/models
pub(super) async fn list_models(State(daemon): State<AppState>) -> Json<ModelsResponse> {
    let models = daemon.models.list();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|(alias, _info, _, _)| ModelObject {
                id: alias,
                object: "model".to_string(),
                created: super::types::unix_timestamp_secs(),
                owned_by: "local".to_string(),
            })
            .collect(),
    })
}

/// GET /v1/models/:model
pub(super) async fn get_model(
    State(daemon): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelObject>, super::ApiError> {
    match daemon.models.get(Some(&model_id)).await {
        Ok(model) => Ok(Json(ModelObject {
            id: model.alias.clone(),
            object: "model".to_string(),
            created: super::types::unix_timestamp_secs(),
            owned_by: "local".to_string(),
        })),
        Err(_) => Err(super::ApiError::not_found(&model_id)),
    }
}

/// List all models (cached + running)
pub(super) async fn api_list_models(State(daemon): State<AppState>) -> Json<serde_json::Value> {
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

/// Pull a model from HuggingFace
pub(super) async fn api_pull_model(
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
pub(super) async fn api_delete_model(
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

/// Get model details
pub(super) async fn api_get_model(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ModelOperationResponse>)> {
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
        StatusCode::NOT_FOUND,
        Json(ModelOperationResponse {
            success: false,
            message: format!("Model '{}' not found", name),
            model: None,
        }),
    ))
}

/// Load a model into the daemon
pub(super) async fn api_load_model(
    State(daemon): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use crate::daemon::hf::HfDownloader;
    use crate::daemon::registry::{resolve_model_name, ResolvedModel};

    let resolved = resolve_model_name(&request.name);

    let (path, alias, model_config): (String, String, Option<ModelConfig>) = match resolved {
        ResolvedModel::HuggingFace { spec, .. } => {
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

            if let Some(hf_spec) = crate::daemon::hf::HfModelSpec::parse(&spec) {
                let cached = downloader.list_cached();
                let found = cached.iter().find(|m| {
                    m.repo_id == hf_spec.repo_id
                        && (hf_spec.filename.is_none()
                            || Some(&m.filename) == hf_spec.filename.as_ref())
                });

                if let Some(model) = found {
                    let model_alias = hf_spec.alias.unwrap_or_else(|| request.name.clone());
                    (model.local_path.display().to_string(), model_alias, None)
                } else {
                    return Err((
                        StatusCode::NOT_FOUND,
                        Json(ModelOperationResponse {
                            success: false,
                            message: format!(
                                "Model '{}' not downloaded. Pull it first.",
                                request.name
                            ),
                            model: None,
                        }),
                    ));
                }
            } else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Invalid model spec: {}", spec),
                        model: None,
                    }),
                ));
            }
        }
        ResolvedModel::LocalPath(path) => (path.display().to_string(), request.name.clone(), None),
        ResolvedModel::Ollama { name, tag } => {
            let client = crate::daemon::ollama::OllamaClient::new().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Failed to initialize Ollama client: {}", e),
                        model: None,
                    }),
                )
            })?;

            let model_name = format!("{}:{}", name, tag);
            if let Some(ollama_model) = client.get_cached(&model_name) {
                let config = ModelConfig {
                    stop_sequences: ollama_model.get_stop_sequences(),
                    system_prompt: ollama_model.system_prompt.clone(),
                    temperature: ollama_model.parameters.temperature,
                    top_p: ollama_model.parameters.top_p,
                    top_k: ollama_model.parameters.top_k,
                    context_size: ollama_model.parameters.num_ctx,
                };
                (
                    ollama_model.gguf_path.display().to_string(),
                    model_name,
                    Some(config),
                )
            } else {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!(
                            "Ollama model '{}' not downloaded. Pull it first: mullama pull {}",
                            model_name, model_name
                        ),
                        model: None,
                    }),
                ));
            }
        }
        ResolvedModel::Unknown(name) => {
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
            let found = cached.iter().find(|m| {
                let short_name = format!(
                    "{}:{}",
                    m.repo_id.split('/').next_back().unwrap_or(&m.repo_id),
                    m.filename.trim_end_matches(".gguf")
                );
                m.filename == name
                    || m.repo_id == name
                    || short_name == name
                    || m.filename.trim_end_matches(".gguf") == name
            });

            if let Some(model) = found {
                let short_name = format!(
                    "{}:{}",
                    model
                        .repo_id
                        .split('/')
                        .next_back()
                        .unwrap_or(&model.repo_id),
                    model.filename.trim_end_matches(".gguf")
                );
                (model.local_path.display().to_string(), short_name, None)
            } else if std::path::Path::new(&name).exists() {
                (name.clone(), name, None)
            } else {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!(
                            "Model '{}' not found. Pull it first or provide a valid path.",
                            name
                        ),
                        model: None,
                    }),
                ));
            }
        }
    };

    let gpu_layers = request
        .gpu_layers
        .unwrap_or(daemon.config.default_gpu_layers);
    let context_size = request
        .context_size
        .or_else(|| model_config.as_ref().and_then(|c| c.context_size))
        .unwrap_or(daemon.config.default_context_size);

    let config = crate::daemon::models::ModelLoadConfig {
        alias: alias.clone(),
        path: path.clone(),
        gpu_layers,
        context_size,
        threads: daemon.config.threads_per_model,
        context_pool_size: daemon.config.default_context_pool_size,
        mmproj_path: None,
        model_config,
        use_mmap: request.use_mmap.or(daemon.config.default_use_mmap),
        use_mlock: request.use_mlock || daemon.config.default_use_mlock,
        flash_attn: request.flash_attn || daemon.config.default_flash_attn,
        cache_type_k: request.cache_type_k.or_else(|| daemon.config.default_cache_type_k.clone()),
        cache_type_v: request.cache_type_v.or_else(|| daemon.config.default_cache_type_v.clone()),
        rope_freq_base: request.rope_freq_base.or(daemon.config.default_rope_freq_base),
        rope_freq_scale: request.rope_freq_scale.or(daemon.config.default_rope_freq_scale),
        n_batch: request.n_batch.or(daemon.config.default_n_batch),
        defrag_thold: request.defrag_thold.or(daemon.config.default_defrag_thold),
        split_mode: request.split_mode.or_else(|| daemon.config.default_split_mode.clone()),
    };

    match daemon.models.load(config).await {
        Ok(info) => Ok(Json(ModelOperationResponse {
            success: true,
            message: format!("Model '{}' loaded successfully", alias),
            model: Some(serde_json::json!({
                "alias": alias,
                "path": path,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
            })),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to load model: {}", e),
                model: None,
            }),
        )),
    }
}

/// Unload a model from the daemon
pub(super) async fn api_unload_model(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    match daemon.models.unload(&name).await {
        Ok(_) => Ok(Json(ModelOperationResponse {
            success: true,
            message: format!("Model '{}' unloaded successfully", name),
            model: None,
        })),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            Err((
                status,
                Json(ModelOperationResponse {
                    success: false,
                    message: format!("Failed to unload model: {}", e),
                    model: None,
                }),
            ))
        }
    }
}
