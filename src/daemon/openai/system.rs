use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;

use super::AppState;

/// GET /health
pub(super) async fn health() -> &'static str {
    "ok"
}

/// GET /status
pub(super) async fn status(State(daemon): State<AppState>) -> Json<serde_json::Value> {
    let models = daemon.models.list();
    let default = daemon.models.default_alias();

    Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": daemon.start_time.elapsed().as_secs(),
        "models_loaded": models.len(),
        "default_model": default,
        "models": models.iter().map(|(alias, info, is_default, active)| {
            serde_json::json!({
                "alias": alias,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "is_default": is_default,
                "active_requests": active,
            })
        }).collect::<Vec<_>>(),
        "stats": {
            "total_requests": daemon.total_requests.load(std::sync::atomic::Ordering::Relaxed),
            "tokens_generated": daemon.models.total_tokens(),
            "gpu_available": crate::supports_gpu_offload(),
        }
    }))
}

/// Prometheus-compatible metrics endpoint
pub(super) async fn metrics(State(daemon): State<AppState>) -> impl IntoResponse {
    let models = daemon.models.list();
    let uptime = daemon.start_time.elapsed().as_secs();
    let total_requests = daemon
        .total_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let active_requests = daemon
        .active_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let tokens_generated = daemon.models.total_tokens();

    let mut output = String::new();
    output.push_str("# HELP mullama_info Mullama daemon information\n");
    output.push_str("# TYPE mullama_info gauge\n");
    output.push_str(&format!(
        "mullama_info{{version=\"{}\"}} 1\n",
        env!("CARGO_PKG_VERSION")
    ));

    output.push_str("\n# HELP mullama_uptime_seconds Daemon uptime in seconds\n");
    output.push_str("# TYPE mullama_uptime_seconds counter\n");
    output.push_str(&format!("mullama_uptime_seconds {}\n", uptime));

    output.push_str("\n# HELP mullama_models_loaded Number of loaded models\n");
    output.push_str("# TYPE mullama_models_loaded gauge\n");
    output.push_str(&format!("mullama_models_loaded {}\n", models.len()));

    output.push_str("\n# HELP mullama_requests_total Total number of requests processed\n");
    output.push_str("# TYPE mullama_requests_total counter\n");
    output.push_str(&format!("mullama_requests_total {}\n", total_requests));

    output.push_str("\n# HELP mullama_requests_active Number of active requests\n");
    output.push_str("# TYPE mullama_requests_active gauge\n");
    output.push_str(&format!("mullama_requests_active {}\n", active_requests));

    output.push_str("\n# HELP mullama_tokens_generated_total Total tokens generated\n");
    output.push_str("# TYPE mullama_tokens_generated_total counter\n");
    output.push_str(&format!(
        "mullama_tokens_generated_total {}\n",
        tokens_generated
    ));

    output.push_str("\n# HELP mullama_gpu_available Whether GPU offload is available\n");
    output.push_str("# TYPE mullama_gpu_available gauge\n");
    output.push_str(&format!(
        "mullama_gpu_available {}\n",
        if crate::supports_gpu_offload() { 1 } else { 0 }
    ));

    if !models.is_empty() {
        output.push_str("\n# HELP mullama_model_parameters Model parameter count\n");
        output.push_str("# TYPE mullama_model_parameters gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_parameters{{model=\"{}\"}} {}\n",
                alias, info.parameters
            ));
        }

        output.push_str("\n# HELP mullama_model_context_size Model context size\n");
        output.push_str("# TYPE mullama_model_context_size gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_context_size{{model=\"{}\"}} {}\n",
                alias, info.context_size
            ));
        }

        output.push_str("\n# HELP mullama_model_gpu_layers Model GPU layers\n");
        output.push_str("# TYPE mullama_model_gpu_layers gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_gpu_layers{{model=\"{}\"}} {}\n",
                alias, info.gpu_layers
            ));
        }

        output.push_str("\n# HELP mullama_model_active_requests Active requests per model\n");
        output.push_str("# TYPE mullama_model_active_requests gauge\n");
        for (alias, _, _, active) in &models {
            output.push_str(&format!(
                "mullama_model_active_requests{{model=\"{}\"}} {}\n",
                alias, active
            ));
        }
    }

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
}

/// System status response
#[derive(Debug, Serialize)]
pub(super) struct SystemStatus {
    pub version: String,
    pub uptime_secs: u64,
    pub models_loaded: usize,
    pub http_endpoint: Option<String>,
}

/// Get system status
pub(super) async fn api_system_status(State(daemon): State<AppState>) -> Json<SystemStatus> {
    let models = daemon.models.list();
    let uptime = daemon.start_time.elapsed().as_secs();

    let http_endpoint = daemon
        .config
        .http
        .port
        .map(|port| format!("http://{}:{}", daemon.config.http.addr, port));

    Json(SystemStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        models_loaded: models.len(),
        http_endpoint,
    })
}
