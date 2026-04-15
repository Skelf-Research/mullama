//! Minimal web service example using Mullama's web feature stack.
//!
//! The production router requires a loaded model. This example stays model-free so it
//! can compile and run as a health/configuration service in development.
//!
//! Run with: cargo run --example web_service --features web

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mullama::prelude::*;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;

#[derive(Clone)]
struct DemoState {
    config: MullamaConfig,
    metrics: Arc<RwLock<DemoMetrics>>,
}

#[derive(Default, Serialize)]
struct DemoMetrics {
    requests: u64,
}

#[derive(Deserialize)]
struct DemoGenerateRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

#[derive(Serialize)]
struct DemoGenerateResponse {
    text: String,
    tokens_requested: usize,
}

fn default_max_tokens() -> usize {
    64
}

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("Mullama Web Service Example");
    println!("===========================");

    #[cfg(feature = "web")]
    {
        let state = DemoState {
            config: MullamaConfig::default(),
            metrics: Arc::new(RwLock::new(DemoMetrics::default())),
        };

        let app: Router = Router::new()
            .route("/health", get(health))
            .route("/config", get(config))
            .route("/metrics", get(metrics))
            .route("/generate", post(generate))
            .with_state(state);

        println!("Configured endpoints:");
        println!("  GET  /health");
        println!("  GET  /config");
        println!("  GET  /metrics");
        println!("  POST /generate");

        let addr: SocketAddr = "127.0.0.1:3000"
            .parse()
            .map_err(|e| MullamaError::ConfigError(format!("Invalid listen address: {}", e)))?;
        println!("To run the service, bind a listener and serve this router at http://{}", addr);

        let _app = app;
    }

    #[cfg(not(feature = "web"))]
    {
        println!("This example requires the web feature.");
        println!("Run with: cargo run --example web_service --features web");
    }

    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "mullama-demo"
    }))
}

async fn config(State(state): State<DemoState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "context_size": state.config.context.n_ctx,
        "temperature": state.config.sampling.temperature,
        "top_k": state.config.sampling.top_k
    }))
}

async fn metrics(State(state): State<DemoState>) -> impl IntoResponse {
    let metrics = state.metrics.read().await;
    Json(serde_json::json!({
        "requests": metrics.requests
    }))
}

async fn generate(
    State(state): State<DemoState>,
    Json(request): Json<DemoGenerateRequest>,
) -> impl IntoResponse {
    let mut metrics = state.metrics.write().await;
    metrics.requests += 1;

    Json(DemoGenerateResponse {
        text: format!("Model-free demo response for: {}", request.prompt),
        tokens_requested: request.max_tokens,
    })
}
