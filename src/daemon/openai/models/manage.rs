use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};

use super::super::AppState;
use super::types::{LoadModelRequest, ModelOperationResponse};
use crate::daemon::models::ModelConfig;

fn validate_local_path(path: &std::path::Path) -> Result<std::path::PathBuf, (StatusCode, String)> {
    let canonical = std::fs::canonicalize(path).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid path '{}': {}", path.display(), e),
        )
    })?;

    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/"));
    let allowed_dirs = [
        home.join(".mullama"),
        dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp/mullama")),
        std::path::PathBuf::from("/tmp"),
    ];

    let allowed = allowed_dirs.iter().any(|d| canonical.starts_with(d));

    if !allowed && !canonical.extension().map_or(false, |ext| ext == "gguf") {
        return Err((
            StatusCode::FORBIDDEN,
            format!(
                "Path '{}' is outside allowed model directories. Use ~/.mullama/models/ or the model cache directory.",
                path.display()
            ),
        ));
    }

    Ok(canonical)
}

/// Load a model into the daemon
pub(in crate::daemon::openai) async fn api_load_model(
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
        ResolvedModel::LocalPath(path) => {
            let validated = validate_local_path(&path).map_err(|(status, msg)| {
                (status, Json(ModelOperationResponse {
                    success: false,
                    message: msg,
                    model: None,
                }))
            })?;
            (validated.display().to_string(), request.name.clone(), None)
        }
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
            } else {
                let local_path = std::path::Path::new(&name);
                if local_path.exists() {
                    let validated = validate_local_path(local_path).map_err(|(status, msg)| {
                        (status, Json(ModelOperationResponse {
                            success: false,
                            message: msg,
                            model: None,
                        }))
                    })?;
                    (validated.display().to_string(), name, None)
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
        }
    };

    let md = &daemon.config.model_defaults;
    let gpu_layers = request.gpu_layers.unwrap_or(md.gpu_layers);
    let context_size = request
        .context_size
        .or_else(|| model_config.as_ref().and_then(|c| c.context_size))
        .unwrap_or(md.context_size);

    let config = crate::daemon::models::ModelLoadConfig {
        alias: alias.clone(),
        path: path.clone(),
        gpu_layers,
        context_size,
        threads: md.threads_per_model,
        context_pool_size: md.context_pool_size,
        mmproj_path: None,
        model_config,
        use_mmap: request.use_mmap.or(md.use_mmap),
        use_mlock: request.use_mlock || md.use_mlock,
        flash_attn: request.flash_attn || md.flash_attn,
        cache_type_k: request.cache_type_k.or_else(|| md.cache_type_k.clone()),
        cache_type_v: request.cache_type_v.or_else(|| md.cache_type_v.clone()),
        rope_freq_base: request.rope_freq_base.or(md.rope_freq_base),
        rope_freq_scale: request.rope_freq_scale.or(md.rope_freq_scale),
        n_batch: request.n_batch.or(md.n_batch),
        defrag_thold: request.defrag_thold.or(md.defrag_thold),
        split_mode: request.split_mode.or_else(|| md.split_mode.clone()),
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
pub(in crate::daemon::openai) async fn api_unload_model(
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
