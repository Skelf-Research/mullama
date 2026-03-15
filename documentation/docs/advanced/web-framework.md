# Web Framework (Axum)

Build production-ready REST APIs for LLM inference using Mullama's integrated Axum web framework support.

!!! info "Feature Gate"
    This feature requires the `web` feature flag, which transitively enables `async`.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["web"] }
    ```

## Overview

Mullama provides first-class integration with the [Axum](https://github.com/tokio-rs/axum) web framework, offering:

- **AppState** for shared model and context management
- **AppStateBuilder** pattern for configuration
- **RouterBuilder** for auto-generated REST endpoints
- **Built-in endpoints** for generate, tokenize, health, and metrics
- **Middleware** support (CORS, timeout, rate limiting, logging)
- **SSE streaming** for real-time token delivery
- **AppError** for structured error responses

---

## AppState

`AppState` manages shared resources across all request handlers, including the loaded model, metrics, and configuration.

=== "Node.js"

    ```javascript
    const { createServer } = require('mullama');

    const server = await createServer({
      model: 'model.gguf',
      port: 3000,
      streaming: true,
      metrics: true,
      maxConcurrentRequests: 64,
      rateLimit: { requests: 100, windowSecs: 60 }
    });

    await server.start();
    ```

=== "Python"

    ```python
    from mullama import create_server

    server = await create_server(
        model="model.gguf",
        port=3000,
        streaming=True,
        metrics=True,
        max_concurrent_requests=64,
        rate_limit={"requests": 100, "window_secs": 60}
    )

    await server.start()
    ```

=== "Rust"

    ```rust
    use mullama::{AppState, AsyncModel, ApiMetrics};
    use std::sync::Arc;

    let model = Arc::new(AsyncModel::load("model.gguf").await?);

    let app_state = AppState::new(model)
        .enable_streaming()
        .enable_metrics()
        .max_concurrent_requests(64)
        .rate_limit(100, Duration::from_secs(60))
        .build();
    ```

=== "CLI"

    ```bash
    # Start server with default configuration
    mullama serve --model model.gguf --port 3000

    # With rate limiting and metrics
    mullama serve --model model.gguf \
      --port 3000 \
      --max-concurrent 64 \
      --rate-limit 100 \
      --metrics
    ```

### AppStateBuilder Methods

| Method | Description | Default |
|--------|-------------|---------|
| `enable_streaming()` | Enable SSE streaming endpoints | Disabled |
| `enable_metrics()` | Enable `/metrics` endpoint | Disabled |
| `max_concurrent_requests(n)` | Limit concurrent requests | 32 |
| `rate_limit(requests, window)` | Rate limiting per client | None |

### Accessing State in Handlers

```rust
use axum::extract::State;
use std::sync::Arc;

async fn my_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let model = state.model();
    let metrics = state.metrics();
    let is_streaming = state.streaming_enabled();
    // ...
}
```

---

## RouterBuilder

The `create_router` function generates a fully-configured Axum `Router` with default endpoints.

```rust
use mullama::{create_router, AppState};

let app = create_router(app_state);
```

### Default Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Text generation |
| `POST` | `/tokenize` | Text tokenization |
| `POST` | `/embeddings` | Generate embeddings |
| `GET` | `/metrics` | Performance metrics |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket streaming |

### Adding Custom Endpoints

```rust
use axum::{routing::{get, post}, Router};

let app = create_router(app_state)
    .route("/custom/summarize", post(summarize_handler))
    .route("/custom/status", get(status_handler));
```

---

## Request/Response Types

### GenerateRequest

```rust
#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}
```

### GenerateResponse

```rust
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub processing_time_ms: u64,
    pub model_info: Option<ModelInfo>,
}
```

### TokenizeRequest / TokenizeResponse

```rust
#[derive(Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    pub add_bos: Option<bool>,
    pub special_tokens: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<TokenId>,
    pub token_count: usize,
    pub text_length: usize,
}
```

---

## Middleware Integration

Mullama's web layer integrates seamlessly with tower-http middleware for production deployments.

### CORS

=== "Node.js"

    ```javascript
    const server = await createServer({
      model: 'model.gguf',
      cors: {
        origins: ['https://myapp.com'],
        methods: ['GET', 'POST'],
        headers: ['Content-Type', 'Authorization']
      }
    });
    ```

=== "Python"

    ```python
    server = await create_server(
        model="model.gguf",
        cors={
            "origins": ["https://myapp.com"],
            "methods": ["GET", "POST"],
            "headers": ["Content-Type", "Authorization"]
        }
    )
    ```

=== "Rust"

    ```rust
    use tower_http::cors::{CorsLayer, Any};

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = create_router(app_state).layer(cors);
    ```

### Timeout and Rate Limiting

```rust
use tower_http::timeout::TimeoutLayer;
use std::time::Duration;

let app = create_router(app_state)
    .layer(TimeoutLayer::new(Duration::from_secs(30)));
```

### Combined Middleware Stack

```rust
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
};
use std::time::Duration;

let app = create_router(app_state)
    .layer(CorsLayer::permissive())
    .layer(TraceLayer::new_for_http())
    .layer(CompressionLayer::new())
    .layer(TimeoutLayer::new(Duration::from_secs(30)));
```

---

## ApiMetrics

Built-in metrics collection for monitoring API performance.

```rust
let metrics: &ApiMetrics = state.metrics();

// Metrics include:
// - Total requests served
// - Average response time
// - Tokens generated per second
// - Active connections
// - Error rate
```

!!! tip "Prometheus Integration"
    When using the `daemon` feature, metrics are automatically exposed in Prometheus format at `/metrics`.

---

## SSE Streaming

Stream generated tokens to clients in real-time using Server-Sent Events.

=== "Node.js"

    ```javascript
    // Client-side consumption
    const response = await fetch('http://localhost:3000/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: 'Tell me a story:',
        max_tokens: 500,
        stream: true
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      process.stdout.write(decoder.decode(value));
    }
    ```

=== "Python"

    ```python
    import httpx

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:3000/generate",
            json={
                "prompt": "Tell me a story:",
                "max_tokens": 500,
                "stream": True
            }
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
    ```

=== "Rust"

    ```rust
    use axum::response::sse::{Event, Sse};
    use futures::stream::{self, Stream};
    use tokio::sync::mpsc;
    use std::convert::Infallible;

    async fn stream_generate(
        State(state): State<Arc<AppState>>,
        Json(req): Json<GenerateRequest>,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let model = state.model().clone();
        let max_tokens = req.max_tokens.unwrap_or(200);
        let (tx, mut rx) = mpsc::channel::<String>(32);

        tokio::task::spawn_blocking(move || {
            let mut ctx = Context::new(model, ContextParams::default()).unwrap();
            ctx.generate_streaming(&req.prompt, max_tokens, |token| {
                tx.blocking_send(token.to_string()).is_ok()
            }).ok();
        });

        let stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|token| {
                let event = Event::default().data(token);
                (Ok(event), rx)
            })
        });

        Sse::new(stream)
    }
    ```

---

## Error Handling with AppError

Mullama errors map cleanly to HTTP status codes for API responses.

```rust
use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde::Serialize;

#[derive(Serialize)]
struct ApiError {
    error: String,
    code: u32,
    details: Option<String>,
}

impl IntoResponse for MullamaError {
    fn into_response(self) -> Response {
        let (status, error) = match &self {
            MullamaError::ModelLoadError(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                ApiError { error: "Model unavailable".into(), code: 503, details: Some(msg.clone()) }
            ),
            MullamaError::TokenizationError(msg) => (
                StatusCode::BAD_REQUEST,
                ApiError { error: "Invalid input".into(), code: 400, details: Some(msg.clone()) }
            ),
            MullamaError::GenerationError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiError { error: "Generation failed".into(), code: 500, details: Some(msg.clone()) }
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiError { error: "Internal error".into(), code: 500, details: None }
            ),
        };

        (status, Json(error)).into_response()
    }
}
```

---

## Complete Example: REST API Server

=== "Node.js"

    ```javascript
    const { createServer } = require('mullama');

    async function main() {
      const server = await createServer({
        model: 'model.gguf',
        port: 3000,
        streaming: true,
        metrics: true,
        maxConcurrentRequests: 32,
        cors: { origins: '*' }
      });

      // Add custom endpoint
      server.addRoute('POST', '/summarize', async (req) => {
        const { text, maxLength } = req.body;
        const prompt = `Summarize in ${maxLength || 100} words:\n\n${text}\n\nSummary:`;
        return await server.generate(prompt, { maxTokens: maxLength || 100 });
      });

      await server.start();
      console.log('Server running on http://localhost:3000');
    }

    main();
    ```

=== "Python"

    ```python
    from mullama import create_server

    async def main():
        server = await create_server(
            model="model.gguf",
            port=3000,
            streaming=True,
            metrics=True,
            max_concurrent_requests=32,
            cors={"origins": "*"}
        )

        @server.route("POST", "/summarize")
        async def summarize(request):
            text = request.body["text"]
            max_length = request.body.get("max_length", 100)
            prompt = f"Summarize in {max_length} words:\n\n{text}\n\nSummary:"
            return await server.generate(prompt, max_tokens=max_length)

        await server.start()
        print("Server running on http://localhost:3000")

    import asyncio
    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{create_router, AppState, AsyncModel, GenerateRequest, GenerateResponse};
    use axum::{extract::{Json, State}, routing::post, Router};
    use tower_http::cors::CorsLayer;
    use std::sync::Arc;

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        tracing_subscriber::init();

        let model = Arc::new(AsyncModel::load("model.gguf").await?);

        let app_state = AppState::new(model)
            .enable_streaming()
            .enable_metrics()
            .max_concurrent_requests(32)
            .build();

        let app = create_router(app_state)
            .layer(CorsLayer::permissive());

        let addr = "0.0.0.0:3000".parse()?;
        println!("Server listening on http://{}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;

        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # Start server
    mullama serve --model model.gguf --port 3000 --metrics

    # Test endpoints
    curl http://localhost:3000/health

    curl -X POST http://localhost:3000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Explain quantum computing:", "max_tokens": 200}'

    curl -X POST http://localhost:3000/tokenize \
      -H "Content-Type: application/json" \
      -d '{"text": "Hello, world!", "add_bos": true}'
    ```

---

## Performance Tuning

!!! warning "Production Checklist"
    Ensure your production configuration addresses these concerns:

- Set `max_concurrent_requests` to match your hardware capacity
- Enable CORS only for trusted origins in production
- Add authentication middleware for public-facing APIs
- Configure request timeouts to prevent resource exhaustion
- Enable compression for large responses
- Set up health checks for load balancer integration
- Monitor `ApiMetrics` for performance degradation
- Use TLS termination (reverse proxy recommended)

### Connection Limits

| Hardware | Recommended Concurrent | Timeout |
|----------|----------------------|---------|
| 4-core CPU | 4-8 | 60s |
| 8-core CPU | 8-16 | 45s |
| GPU (single) | 16-32 | 30s |
| GPU (multi) | 32-64 | 30s |

---

## See Also

- [WebSockets](websockets.md) - Real-time bidirectional communication
- [Streaming Guide](../guide/streaming.md) - Token streaming patterns
- [Async Integration](../guide/async.md) - Async/await fundamentals
- [REST API (Daemon)](../daemon/rest-api.md) - Pre-built daemon REST API
