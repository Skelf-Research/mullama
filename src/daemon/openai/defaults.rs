use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};
use serde::Serialize;

use super::helpers::model_config_from_modelfile;
use super::AppState;

/// Response for listing default models
#[derive(Debug, Serialize)]
pub(super) struct DefaultsResponse {
    models: Vec<crate::daemon::defaults::DefaultModelInfo>,
}

/// Response for using a default model
#[derive(Debug, Serialize)]
pub(super) struct UseDefaultResponse {
    success: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<serde_json::Value>,
}

/// List all available default model templates
pub(super) async fn api_list_defaults() -> Json<DefaultsResponse> {
    let infos = crate::daemon::defaults::list_default_infos();
    Json(DefaultsResponse { models: infos })
}

/// Use a default model (download if needed and load)
pub(super) async fn api_use_default(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<UseDefaultResponse>, (StatusCode, Json<UseDefaultResponse>)> {
    use crate::daemon::defaults::get_default;
    use crate::daemon::hf::{HfDownloader, HfModelSpec};

    let default = get_default(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Default model '{}' not found", name),
                model: None,
            }),
        )
    })?;

    let from = &default.modelfile.from;
    if !from.starts_with("hf:") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Default model '{}' is not a HuggingFace model", name),
                model: None,
            }),
        ));
    }

    let spec = HfModelSpec::parse(from).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Invalid HuggingFace spec in modelfile: {}", from),
                model: None,
            }),
        )
    })?;

    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to initialize downloader: {}", e),
                model: None,
            }),
        )
    })?;

    let model_path = downloader.download_spec(&spec, false).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to download model: {}", e),
                model: None,
            }),
        )
    })?;

    let context_size = default
        .modelfile
        .num_ctx()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(4096);

    let gpu_layers = default.modelfile.gpu_layers.unwrap_or(0);

    let mmproj_path = default.modelfile.vision_projector.as_ref().map(|p| {
        if p.is_relative() {
            model_path
                .parent()
                .map(|parent| parent.join(p).display().to_string())
                .unwrap_or_else(|| p.display().to_string())
        } else {
            p.display().to_string()
        }
    });

    let load_config = crate::daemon::models::ModelLoadConfig {
        path: model_path.display().to_string(),
        alias: name.clone(),
        context_size,
        gpu_layers,
        threads: num_cpus::get() as i32,
        context_pool_size: daemon.config.model_defaults.context_pool_size,
        mmproj_path,
        model_config: Some(model_config_from_modelfile(&default.modelfile)),
        use_mmap: daemon.config.model_defaults.use_mmap,
        use_mlock: daemon.config.model_defaults.use_mlock,
        flash_attn: daemon.config.model_defaults.flash_attn,
        cache_type_k: daemon.config.model_defaults.cache_type_k.clone(),
        cache_type_v: daemon.config.model_defaults.cache_type_v.clone(),
        rope_freq_base: daemon.config.model_defaults.rope_freq_base,
        rope_freq_scale: daemon.config.model_defaults.rope_freq_scale,
        n_batch: daemon.config.model_defaults.n_batch,
        defrag_thold: daemon.config.model_defaults.defrag_thold,
        split_mode: daemon.config.model_defaults.split_mode.clone(),
    };

    daemon.models.load(load_config).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to load model: {}", e),
                model: None,
            }),
        )
    })?;

    daemon.models.set_default(&name).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to set default model: {}", e),
                model: None,
            }),
        )
    })?;

    let model_info = daemon.models.get(Some(name.as_str())).await.ok().map(|_m| {
        serde_json::json!({
            "alias": name,
            "path": model_path.display().to_string(),
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "description": default.info.description,
            "has_thinking": default.info.has_thinking,
            "has_vision": default.info.has_vision,
        })
    });

    Ok(Json(UseDefaultResponse {
        success: true,
        message: format!("Model '{}' is now ready to use", name),
        model: model_info,
    }))
}
