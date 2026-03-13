//! # Web Service Example
//!
//! This example demonstrates how to create a production-ready web service
//! using Mullama's web framework integration with Axum.
//!
//! Features demonstrated:
//! - REST API endpoints for text generation
//! - Server-sent events for streaming
//! - Health checks and metrics
//! - CORS and middleware configuration
//! - Error handling and logging
//!
//! Run with: cargo run --example web_service --features web,async,streaming

use mullama::prelude::*;
use std::sync::Arc;

#[cfg(all(feature = "web", feature = "async"))]
use mullama::{
    create_router, ApiMetrics, AppState, AsyncModel, GenerateRequest, GenerateResponse,
    TokenizeRequest, TokenizeResponse,
};

#[cfg(all(feature = "web", feature = "async"))]
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    Router,
};

#[cfg(all(feature = "web", feature = "async"))]
use tokio::net::TcpListener;

#[cfg(all(feature = "web", feature = "async"))]
use tower_http::cors::CorsLayer;

#[cfg(all(feature = "web", feature = "async"))]
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Mullama Web Service Example");
    println!("==============================");

    #[cfg(all(feature = "web", feature = "async"))]
    {
        // Initialize logging
        tracing_subscriber::fmt::init();

        // Load configuration
        let config = load_configuration().await?;
        println!("📋 Configuration loaded");

        // Load model
        println!("📂 Loading model...");
        // In real scenario:
        // let model = AsyncModel::load(&config.model.path).await
        //     .map_err(|e| format!("Failed to load model: {}", e))?;

        // Create mock model for example
        println!("✅ Model loaded (using mock for example)");

        // Initialize application state
        let app_state = create_app_state(config).await;
        println!("🏗️  Application state initialized");

        // Create router with all endpoints
        let app = create_complete_router(app_state).await;
        println!("🛤️  Router created with all endpoints");

        // Print available endpoints
        print_endpoints();

        // Start server
        let port = std::env::var("PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .unwrap_or(3000);

        let addr = format!("0.0.0.0:{}", port);
        println!("🚀 Starting server on http://{}", addr);

        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;

        println!("✅ Server ready! Try these commands:");
        print_example_commands(port);

        // Start the server
        axum::serve(listener, app)
            .await
            .map_err(|e| format!("Server error: {}", e))?;
    }

    #[cfg(not(all(feature = "web", feature = "async")))]
    {
        println!("❌ This example requires 'web' and 'async' features");
        println!("Run with: cargo run --example web_service --features web,async,streaming");
    }

    Ok(())
}

#[cfg(all(feature = "web", feature = "async"))]
async fn load_configuration() -> Result<MullamaConfig, Box<dyn std::error::Error>> {
    // Try to load from environment first
    let mut config = MullamaConfig::from_env().unwrap_or_default();

    // Override with some web service specific settings
    config.model.path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "path/to/model.gguf".to_string());

    config.context.n_ctx = 2048;
    config.context.n_batch = 512;
    config.context.n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);

    config.sampling.temperature = 0.7;
    config.sampling.top_k = 40;
    config.sampling.top_p = 0.9;

    // Validate configuration
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    Ok(config)
}

#[cfg(all(feature = "web", feature = "async"))]
async fn create_app_state(config: MullamaConfig) -> AppState {
    // In real scenario, load actual model:
    // let model = AsyncModel::load(&config.model.path).await.unwrap();

    // For this example, we'll create a mock state structure
    // Note: This would normally use the actual loaded model
    AppState {
        // model,
        default_config: config,
        metrics: Arc::new(tokio::sync::RwLock::new(ApiMetrics::default())),
    }
}

#[cfg(all(feature = "web", feature = "async"))]
async fn create_complete_router(app_state: AppState) -> Router {
    // Create the main router with all endpoints
    let main_router = create_router(app_state.clone());

    // Add additional custom endpoints
    let custom_router = Router::new()
        .route("/custom/status", axum::routing::get(custom_status))
        .route("/custom/config", axum::routing::get(get_config))
        .with_state(app_state);

    // Combine routers
    Router::new()
        .nest("/api/v1", main_router)
        .merge(custom_router)
        .layer(CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

#[cfg(all(feature = "web", feature = "async"))]
async fn custom_status(State(_state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "mullama-web",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "features": {
            "async": cfg!(feature = "async"),
            "streaming": cfg!(feature = "streaming"),
            "web": cfg!(feature = "web"),
        }
    }))
}

#[cfg(all(feature = "web", feature = "async"))]
async fn get_config(State(state): State<AppState>) -> Json<MullamaConfig> {
    Json(state.default_config.clone())
}

#[cfg(all(feature = "web", feature = "async"))]
fn print_endpoints() {
    println!("\n📍 Available Endpoints:");
    println!("====================");

    // Main API endpoints
    println!("🔗 Main API (prefix: /api/v1):");
    println!("   POST   /generate     - Generate text from prompt");
    println!("   POST   /tokenize     - Tokenize input text");
    println!("   GET    /stream/:prompt - Server-sent events streaming");
    println!("   GET    /health       - Health check endpoint");
    println!("   GET    /metrics      - API usage metrics");

    // Custom endpoints
    println!("\n🔗 Custom endpoints:");
    println!("   GET    /custom/status - Service status with features");
    println!("   GET    /custom/config - Current configuration");
}

#[cfg(all(feature = "web", feature = "async"))]
fn print_example_commands(port: u16) {
    println!("\n💡 Example API calls:");
    println!("====================");

    println!("📡 Health check:");
    println!("   curl http://localhost:{}/api/v1/health", port);

    println!("\n🤖 Text generation:");
    println!(
        "   curl -X POST http://localhost:{}/api/v1/generate \\",
        port
    );
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{");
    println!("       \"prompt\": \"The future of AI is\",");
    println!("       \"max_tokens\": 100,");
    println!("       \"temperature\": 0.7");
    println!("     }}'");

    println!("\n🔤 Tokenization:");
    println!(
        "   curl -X POST http://localhost:{}/api/v1/tokenize \\",
        port
    );
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{\"text\": \"Hello, world!\"}}'");

    println!("\n🌊 Streaming (in browser or with curl):");
    println!(
        "   curl http://localhost:{}/api/v1/stream/Hello%20world",
        port
    );

    println!("\n📊 Metrics:");
    println!("   curl http://localhost:{}/api/v1/metrics", port);

    println!("\n⚙️ Custom status:");
    println!("   curl http://localhost:{}/custom/status", port);

    println!("\n📋 Configuration:");
    println!("   curl http://localhost:{}/custom/config", port);
}

#[cfg(all(feature = "web", feature = "async"))]
async fn demonstrate_client_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔌 Client Usage Examples");
    println!("========================");

    // Example of how clients would use the API
    use reqwest;

    let client = reqwest::Client::new();
    let base_url = "http://localhost:3000";

    // 1. Health check
    println!("1️⃣ Health check...");
    let health_response = client
        .get(&format!("{}/api/v1/health", base_url))
        .send()
        .await?;

    if health_response.status().is_success() {
        println!("   ✅ Service is healthy");
    } else {
        println!("   ❌ Service is not healthy");
    }

    // 2. Text generation
    println!("2️⃣ Text generation...");
    let generate_request = GenerateRequest {
        prompt: "The benefits of renewable energy include".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        stream: Some(false),
    };

    let generate_response = client
        .post(&format!("{}/api/v1/generate", base_url))
        .json(&generate_request)
        .send()
        .await?;

    if generate_response.status().is_success() {
        let response: GenerateResponse = generate_response.json().await?;
        println!("   ✅ Generated: {}", response.text);
    } else {
        println!("   ❌ Generation failed");
    }

    // 3. Tokenization
    println!("3️⃣ Tokenization...");
    let tokenize_request = TokenizeRequest {
        text: "Hello, artificial intelligence!".to_string(),
        add_bos: Some(true),
        special: Some(false),
    };

    let tokenize_response = client
        .post(&format!("{}/api/v1/tokenize", base_url))
        .json(&tokenize_request)
        .send()
        .await?;

    if tokenize_response.status().is_success() {
        let response: TokenizeResponse = tokenize_response.json().await?;
        println!("   ✅ Tokens: {:?}", response.tokens);
        println!("   📊 Token count: {}", response.tokens.len());
    } else {
        println!("   ❌ Tokenization failed");
    }

    Ok(())
}

// Integration with monitoring and observability
#[cfg(all(feature = "web", feature = "async"))]
async fn setup_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Monitoring Setup");
    println!("===================");

    // Example of how to set up monitoring
    println!("🔍 Metrics collection:");
    println!("   - Request counts and latencies");
    println!("   - Token generation rates");
    println!("   - Error rates and types");
    println!("   - Model performance metrics");

    println!("\n📈 Observability:");
    println!("   - Structured logging with tracing");
    println!("   - Health check endpoints");
    println!("   - Prometheus metrics (can be added)");
    println!("   - Custom monitoring endpoints");

    Ok(())
}
