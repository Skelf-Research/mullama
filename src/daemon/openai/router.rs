use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::{
    extract::DefaultBodyLimit,
    middleware::from_fn_with_state,
    routing::{delete, get, post},
    Router,
};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::trace::TraceLayer;

use super::AppState;

/// Create the OpenAI-compatible router
pub fn create_openai_router(daemon: AppState) -> Router {
    let mut protected = Router::new()
        .route("/v1/chat/completions", post(super::chat::chat_completions))
        .route("/v1/completions", post(super::completions::completions))
        .route("/v1/models", get(super::models::list_models))
        .route("/v1/models/:model", get(super::models::get_model))
        .route("/v1/embeddings", post(super::embeddings::embeddings))
        .route(
            "/v1/messages",
            post(crate::daemon::anthropic::messages_handler),
        )
        .route("/api/models", get(super::models::api_list_models))
        .route("/api/models/pull", post(super::models::api_pull_model))
        .route("/api/models/load", post(super::models::api_load_model))
        .route("/api/models/:name/unload", post(super::models::api_unload_model))
        .route("/api/models/:name", delete(super::models::api_delete_model))
        .route("/api/models/:name", get(super::models::api_get_model))
        .route("/api/system/status", get(super::system::api_system_status))
        .route("/api/defaults", get(super::defaults::api_list_defaults))
        .route("/api/defaults/:name/use", post(super::defaults::api_use_default))
        .route("/status", get(super::system::status))
        .route("/metrics", get(super::system::metrics))
        .with_state(daemon.clone())
        .layer(DefaultBodyLimit::max(daemon.config.max_request_body_bytes))
        .layer(ConcurrencyLimitLayer::new(
            daemon.config.max_concurrent_http_requests,
        ));

    if daemon.config.max_requests_per_second > 0 {
        let rate_limit_state = super::middleware::HttpRateLimitState {
            limit: daemon.config.max_requests_per_second,
            second: Arc::new(AtomicU64::new(super::types::unix_timestamp_secs())),
            count: Arc::new(AtomicU64::new(0)),
        };
        protected = protected.layer(from_fn_with_state(
            rate_limit_state,
            super::middleware::enforce_rate_limit,
        ));
    }

    if daemon.config.enforce_http_api_key {
        if let Some(api_key) = daemon.config.http_api_key.as_deref() {
            let auth = super::middleware::HttpAuthState {
                api_key: Arc::<str>::from(api_key),
            };
            protected = protected.layer(from_fn_with_state(
                auth,
                super::middleware::require_api_key,
            ));
        }
    }

    let ollama_router = crate::daemon::ollama_api::create_ollama_router(daemon.clone());

    Router::new()
        .route("/health", get(super::system::health))
        .route("/ui", get(super::ui::ui_redirect))
        .route("/ui/", get(super::ui::serve_ui_handler))
        .route("/ui/*path", get(super::ui::serve_ui_handler))
        .with_state(daemon)
        .merge(protected)
        .merge(ollama_router)
        .layer(TraceLayer::new_for_http())
}
