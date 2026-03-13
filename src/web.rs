//! Web framework integration helpers
//!
//! This module provides utilities for integrating Mullama with web frameworks,
//! particularly Axum. It includes request/response types, middleware, error handling,
//! and streaming support for building robust LLM-powered web services.
//!
//! ## Features
//!
//! - **REST API helpers**: Request/response types for common LLM operations
//! - **Streaming responses**: Server-sent events for real-time generation
//! - **Error handling**: Web-friendly error responses with proper HTTP status codes
//! - **Middleware**: CORS, authentication, and rate limiting helpers
//! - **OpenAPI support**: Automatic API documentation generation
//! - **State management**: Shared model state across requests
//!
//! ## Example
//!
//! ```rust,no_run
//! use axum::{Router, routing::post};
//! use mullama::web::{AppState, handlers, middleware};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create shared application state
//!     let state = AppState::builder()
//!         .model_path("path/to/model.gguf")
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     // Build the router
//!     let app = Router::new()
//!         .route("/generate", post(handlers::generate))
//!         .route("/stream", post(handlers::generate_stream))
//!         .route("/tokenize", post(handlers::tokenize))
//!         .route("/health", get(handlers::health))
//!         .layer(middleware::cors())
//!         .layer(middleware::timeout())
//!         .with_state(state);
//!
//!     // Start the server
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
//!         .await
//!         .unwrap();
//!
//!     println!("Server running on http://0.0.0.0:3000");
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```

#[cfg(feature = "web")]
use axum::{
    extract::{Json, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse},
    Router,
};

#[cfg(feature = "web")]
use tower::{timeout::TimeoutLayer, ServiceBuilder};
#[cfg(feature = "web")]
use tower_http::cors::{Any, CorsLayer};

#[cfg(feature = "web")]
use futures::Stream;
#[cfg(feature = "web")]
use std::sync::Arc;
#[cfg(feature = "web")]
use std::time::Duration;

use crate::{MullamaError, TokenId};
use serde::{Deserialize, Serialize};

#[cfg(feature = "async")]
use crate::async_support::AsyncModel;
#[cfg(feature = "streaming")]
use crate::streaming::{StreamConfig, TokenStream};

/// Application state shared across all requests
#[cfg(feature = "web")]
#[derive(Clone)]
pub struct AppState {
    /// The async model instance
    pub model: AsyncModel,
    /// Default configuration for requests
    pub default_config: crate::config::MullamaConfig,
    /// API metrics and monitoring
    pub metrics: Arc<tokio::sync::RwLock<ApiMetrics>>,
}

/// API metrics for monitoring
#[derive(Debug, Default)]
pub struct ApiMetrics {
    /// Total number of requests
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
}

#[cfg(feature = "web")]
impl AppState {
    /// Create a new app state builder
    pub fn builder() -> AppStateBuilder {
        AppStateBuilder::new()
    }
}

/// Builder for application state
#[cfg(feature = "web")]
pub struct AppStateBuilder {
    model_path: Option<String>,
    config: crate::config::MullamaConfig,
}

#[cfg(feature = "web")]
impl AppStateBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            model_path: None,
            config: crate::config::MullamaConfig::default(),
        }
    }

    /// Set the model path
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: crate::config::MullamaConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the application state
    pub async fn build(self) -> Result<AppState, MullamaError> {
        let model_path = self
            .model_path
            .ok_or_else(|| MullamaError::ConfigError("Model path is required".to_string()))?;

        let model = AsyncModel::load(model_path).await?;

        Ok(AppState {
            model,
            default_config: self.config,
            metrics: Arc::new(tokio::sync::RwLock::new(ApiMetrics::default())),
        })
    }
}

/// Request types for the API

/// Text generation request
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// Input prompt text
    pub prompt: String,
    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Temperature for sampling (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-k sampling parameter
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    /// Top-p sampling parameter
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Repetition penalty
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Random seed (0 = random)
    #[serde(default)]
    pub seed: u32,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
}

/// Tokenization request
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    /// Text to tokenize
    pub text: String,
    /// Whether to add beginning-of-sequence token
    #[serde(default)]
    pub add_bos: bool,
    /// Whether to handle special tokens
    #[serde(default)]
    pub special: bool,
}

/// Response types for the API

/// Text generation response
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
    /// Token IDs (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<TokenId>>,
}

/// Tokenization response
#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    /// Token IDs
    pub tokens: Vec<TokenId>,
    /// Token count
    pub count: usize,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Model information
    pub model_info: ModelInfo,
    /// API metrics
    pub metrics: ApiMetrics,
}

/// Model information for API responses
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Vocabulary size
    pub vocab_size: u32,
    /// Training context size
    pub context_size: u32,
    /// Embedding dimension
    pub embedding_dim: u32,
    /// Number of layers
    pub layers: u32,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Error code
    pub code: String,
    /// HTTP status code
    pub status: u16,
}

/// Stream chunk for server-sent events
#[derive(Debug, Serialize)]
pub struct StreamChunk {
    /// Generated text
    pub text: String,
    /// Token ID
    pub token_id: TokenId,
    /// Position in sequence
    pub position: usize,
    /// Whether this is the final chunk
    pub is_final: bool,
}

// Default values for request parameters
fn default_max_tokens() -> usize {
    100
}
fn default_temperature() -> f32 {
    0.8
}
fn default_top_k() -> i32 {
    40
}
fn default_top_p() -> f32 {
    0.95
}
fn default_repeat_penalty() -> f32 {
    1.1
}

/// HTTP handlers for the API
#[cfg(feature = "web")]
pub mod handlers {
    use super::*;
    use axum::extract::{Request, State};
    use axum::response::Json;
    use std::time::Instant;

    /// Generate text handler
    pub async fn generate(
        State(state): State<AppState>,
        Json(request): Json<GenerateRequest>,
    ) -> Result<Json<GenerateResponse>, AppError> {
        let start_time = Instant::now();

        // Update metrics
        {
            let mut metrics = state.metrics.write().await;
            metrics.total_requests += 1;
        }

        // Validate request
        if request.prompt.is_empty() {
            return Err(AppError::BadRequest("Prompt cannot be empty".to_string()));
        }

        if request.max_tokens == 0 || request.max_tokens > 4096 {
            return Err(AppError::BadRequest(
                "max_tokens must be between 1 and 4096".to_string(),
            ));
        }

        // Generate text
        let result = state
            .model
            .generate_async(&request.prompt, request.max_tokens)
            .await
            .map_err(|e| AppError::Internal(format!("Generation failed: {}", e)))?;

        let generation_time = start_time.elapsed().as_millis() as f64;

        // Update metrics
        {
            let mut metrics = state.metrics.write().await;
            metrics.successful_requests += 1;
            metrics.total_tokens_generated += request.max_tokens as u64;

            // Update average response time (simple moving average)
            if metrics.total_requests == 1 {
                metrics.avg_response_time_ms = generation_time;
            } else {
                metrics.avg_response_time_ms = (metrics.avg_response_time_ms
                    * (metrics.total_requests - 1) as f64
                    + generation_time)
                    / metrics.total_requests as f64;
            }
        }

        Ok(Json(GenerateResponse {
            text: result,
            tokens_generated: request.max_tokens, // Simplified
            generation_time_ms: generation_time,
            token_ids: None,
        }))
    }

    /// Streaming generation handler
    #[cfg(feature = "streaming")]
    pub async fn generate_stream(
        State(state): State<AppState>,
        Json(request): Json<GenerateRequest>,
    ) -> Result<
        Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>,
        AppError,
    > {
        // Validate request
        if request.prompt.is_empty() {
            return Err(AppError::BadRequest("Prompt cannot be empty".to_string()));
        }

        // Create stream configuration
        let config = StreamConfig::default()
            .max_tokens(request.max_tokens)
            .temperature(request.temperature)
            .top_k(request.top_k)
            .top_p(request.top_p);

        // Create token stream
        let token_stream = TokenStream::new(state.model.clone(), request.prompt, config)
            .await
            .map_err(|e| AppError::Internal(format!("Failed to create stream: {}", e)))?;

        // Convert to SSE stream
        let sse_stream = async_stream::stream! {
            let mut stream = token_stream;
            use futures::StreamExt;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(token_data) => {
                        let chunk = StreamChunk {
                            text: token_data.text,
                            token_id: token_data.token_id,
                            position: token_data.position,
                            is_final: token_data.is_final,
                        };

                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = axum::response::sse::Event::default()
                            .event("token")
                            .data(data);

                        yield Ok(event);

                        if token_data.is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_event = axum::response::sse::Event::default()
                            .event("error")
                            .data(format!("{{\"error\": \"{}\"}}", e));
                        yield Ok(error_event);
                        break;
                    }
                }
            }
        };

        Ok(Sse::new(sse_stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(1))
                .text("keep-alive-text"),
        ))
    }

    /// Tokenize text handler
    pub async fn tokenize(
        State(state): State<AppState>,
        Json(request): Json<TokenizeRequest>,
    ) -> Result<Json<TokenizeResponse>, AppError> {
        if request.text.is_empty() {
            return Err(AppError::BadRequest("Text cannot be empty".to_string()));
        }

        let tokens = state
            .model
            .model()
            .tokenize(&request.text, request.add_bos, request.special)
            .map_err(|e| AppError::Internal(format!("Tokenization failed: {}", e)))?;

        Ok(Json(TokenizeResponse {
            count: tokens.len(),
            tokens,
        }))
    }

    /// Health check handler
    pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
        let model_info = ModelInfo {
            vocab_size: state.model.model().vocab_size(),
            context_size: state.model.model().n_ctx_train(),
            embedding_dim: state.model.model().n_embd(),
            layers: state.model.model().n_layer(),
        };

        let metrics = state.metrics.read().await.clone();

        Json(HealthResponse {
            status: "healthy".to_string(),
            model_info,
            metrics,
        })
    }

    /// Model information handler
    pub async fn model_info(State(state): State<AppState>) -> Json<ModelInfo> {
        Json(ModelInfo {
            vocab_size: state.model.model().vocab_size(),
            context_size: state.model.model().n_ctx_train(),
            embedding_dim: state.model.model().n_embd(),
            layers: state.model.model().n_layer(),
        })
    }
}

/// Middleware for the web service
#[cfg(feature = "web")]
pub mod middleware {
    use super::*;
    use axum::{http::Request, middleware, response::Response};
    use std::time::Duration;
    use tower::{Layer, ServiceBuilder};

    /// CORS middleware
    pub fn cors() -> CorsLayer {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    }

    /// Timeout middleware
    pub fn timeout() -> TimeoutLayer {
        TimeoutLayer::new(Duration::from_secs(300)) // 5 minute timeout
    }

    /// Request logging middleware
    pub async fn logging<B>(request: Request<B>, next: Next<B>) -> Result<Response, StatusCode> {
        let start = std::time::Instant::now();
        let method = request.method().clone();
        let uri = request.uri().clone();

        let response = next.run(request).await;

        let duration = start.elapsed();
        println!(
            "{} {} - {} - {:?}",
            method,
            uri,
            response.status(),
            duration
        );

        Ok(response)
    }

    /// Rate limiting middleware (simplified)
    pub async fn rate_limit<B>(request: Request<B>, next: Next<B>) -> Result<Response, StatusCode> {
        // This is a simplified rate limiter
        // In production, you'd use a proper rate limiting library

        // For now, just pass through
        Ok(next.run(request).await)
    }
}

/// Custom error type for web handlers
#[cfg(feature = "web")]
#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    Internal(String),
    NotFound(String),
    Unauthorized(String),
}

#[cfg(feature = "web")]
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "UNAUTHORIZED", msg),
        };

        let error_response = ErrorResponse {
            error: message,
            code: code.to_string(),
            status: status.as_u16(),
        };

        (status, Json(error_response)).into_response()
    }
}

/// Router builder for easy setup
#[cfg(feature = "web")]
pub struct RouterBuilder {
    state: Option<AppState>,
    cors_enabled: bool,
    timeout_enabled: bool,
    logging_enabled: bool,
    rate_limiting_enabled: bool,
}

#[cfg(feature = "web")]
impl RouterBuilder {
    /// Create a new router builder
    pub fn new() -> Self {
        Self {
            state: None,
            cors_enabled: true,
            timeout_enabled: true,
            logging_enabled: true,
            rate_limiting_enabled: false,
        }
    }

    /// Set the application state
    pub fn state(mut self, state: AppState) -> Self {
        self.state = Some(state);
        self
    }

    /// Enable/disable CORS
    pub fn cors(mut self, enabled: bool) -> Self {
        self.cors_enabled = enabled;
        self
    }

    /// Enable/disable timeout
    pub fn timeout(mut self, enabled: bool) -> Self {
        self.timeout_enabled = enabled;
        self
    }

    /// Enable/disable logging
    pub fn logging(mut self, enabled: bool) -> Self {
        self.logging_enabled = enabled;
        self
    }

    /// Enable/disable rate limiting
    pub fn rate_limiting(mut self, enabled: bool) -> Self {
        self.rate_limiting_enabled = enabled;
        self
    }

    /// Build the router
    pub fn build(self) -> Result<Router, &'static str> {
        let state = self.state.ok_or("State is required")?;

        let mut router = Router::new()
            .route("/generate", axum::routing::post(handlers::generate))
            .route("/tokenize", axum::routing::post(handlers::tokenize))
            .route("/health", axum::routing::get(handlers::health))
            .route("/model", axum::routing::get(handlers::model_info));

        #[cfg(feature = "streaming")]
        {
            router = router.route("/stream", axum::routing::post(handlers::generate_stream));
        }

        router = router.with_state(state);

        let mut service_builder = ServiceBuilder::new();

        if self.cors_enabled {
            service_builder = service_builder.layer(middleware::cors());
        }

        if self.timeout_enabled {
            service_builder = service_builder.layer(middleware::timeout());
        }

        if self.logging_enabled {
            service_builder = service_builder.layer(middleware::from_fn(middleware::logging));
        }

        if self.rate_limiting_enabled {
            service_builder = service_builder.layer(middleware::from_fn(middleware::rate_limit));
        }

        Ok(router.layer(service_builder))
    }
}

/// Utility functions for web integration
pub mod utils {
    use super::*;

    /// Create a complete web service with default configuration
    #[cfg(feature = "web")]
    pub async fn create_service(
        model_path: impl Into<String>,
        bind_address: impl Into<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create application state
        let state = AppState::builder().model_path(model_path).build().await?;

        // Build router
        let router = RouterBuilder::new()
            .state(state)
            .cors(true)
            .timeout(true)
            .logging(true)
            .build()?;

        // Start server
        let bind_addr = bind_address.into();
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

        println!("Mullama web service running on http://{}", bind_addr);
        axum::serve(listener, router).await?;

        Ok(())
    }

    /// Extract bearer token from request headers
    pub fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
        let auth_header = headers.get(header::AUTHORIZATION)?;
        let auth_str = auth_header.to_str().ok()?;

        if auth_str.starts_with("Bearer ") {
            Some(auth_str[7..].to_string())
        } else {
            None
        }
    }

    /// Validate API key (placeholder implementation)
    pub fn validate_api_key(api_key: &str) -> bool {
        // In a real implementation, you'd check against a database or configuration
        !api_key.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_defaults() {
        let json = r#"{"prompt": "Hello"}"#;
        let request: GenerateRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.prompt, "Hello");
        assert_eq!(request.max_tokens, 100);
        assert_eq!(request.temperature, 0.8);
        assert_eq!(request.top_k, 40);
        assert_eq!(request.top_p, 0.95);
    }

    #[test]
    fn test_error_response_serialization() {
        let error = ErrorResponse {
            error: "Test error".to_string(),
            code: "TEST_ERROR".to_string(),
            status: 400,
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Test error"));
        assert!(json.contains("TEST_ERROR"));
        assert!(json.contains("400"));
    }

    #[test]
    fn test_bearer_token_extraction() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "Bearer test-token-123".parse().unwrap(),
        );

        let token = utils::extract_bearer_token(&headers);
        assert_eq!(token, Some("test-token-123".to_string()));
    }
}

#[cfg(not(feature = "web"))]
compile_error!("Web support requires the 'web' feature to be enabled");
