# Web Service Examples

Build web APIs with Mullama.

## Simple REST API

```rust
use axum::{
    extract::{Json, State},
    routing::post,
    Router,
};
use mullama::{Model, Context, ContextParams};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    tokens: usize,
}

struct AppState {
    model: Arc<Model>,
    context: Arc<Mutex<Context>>,
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let max_tokens = req.max_tokens.unwrap_or(100);
    let prompt = req.prompt.clone();

    let mut ctx = state.context.lock().await;

    let text = tokio::task::spawn_blocking(move || {
        ctx.generate(&prompt, max_tokens)
    })
    .await
    .unwrap()
    .unwrap_or_default();

    let tokens = text.split_whitespace().count();

    Json(GenerateResponse { text, tokens })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf")?);
    let context = Arc::new(Mutex::new(
        Context::new(model.clone(), ContextParams::default())?
    ));

    let state = Arc::new(AppState { model, context });

    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(state);

    println!("Server running on http://localhost:3000");
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

## Streaming with SSE

```rust
use axum::{
    extract::{Json, State},
    response::sse::{Event, Sse},
    routing::post,
    Router,
};
use futures::stream::{self, Stream};
use mullama::{Model, Context, ContextParams};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Deserialize)]
struct StreamRequest {
    prompt: String,
    max_tokens: Option<usize>,
}

struct AppState {
    model: Arc<Model>,
}

async fn stream_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StreamRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let model = state.model.clone();
    let max_tokens = req.max_tokens.unwrap_or(200);
    let prompt = req.prompt;

    tokio::task::spawn_blocking(move || {
        let mut ctx = Context::new(model, ContextParams::default()).unwrap();
        ctx.generate_streaming(&prompt, max_tokens, |token| {
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf")?);
    let state = Arc::new(AppState { model });

    let app = Router::new()
        .route("/stream", post(stream_generate))
        .with_state(state);

    println!("Server running on http://localhost:3000");
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

**Client usage:**
```javascript
const eventSource = new EventSource('/stream', {
    method: 'POST',
    body: JSON.stringify({ prompt: 'Hello!' }),
});

eventSource.onmessage = (e) => {
    document.getElementById('output').innerText += e.data;
};
```

## WebSocket Chat

```rust
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;
use tokio::sync::mpsc;

struct AppState {
    model: Arc<Model>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let model = state.model.clone();

    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(prompt) = msg {
            let (tx, mut rx) = mpsc::channel::<String>(32);
            let model = model.clone();

            // Generate in background
            tokio::task::spawn_blocking(move || {
                let mut ctx = Context::new(model, ContextParams::default()).unwrap();
                ctx.generate_streaming(&prompt, 200, |token| {
                    tx.blocking_send(token.to_string()).is_ok()
                }).ok();
            });

            // Stream tokens to client
            while let Some(token) = rx.recv().await {
                if sender.send(Message::Text(token)).await.is_err() {
                    break;
                }
            }

            // Signal completion
            sender.send(Message::Text("[DONE]".to_string())).await.ok();
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf")?);
    let state = Arc::new(AppState { model });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state);

    println!("WebSocket server on ws://localhost:3000/ws");
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

## Connection Pool

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

pub struct LlmPool {
    model: Arc<Model>,
    contexts: Vec<Arc<Mutex<Context>>>,
    semaphore: Arc<Semaphore>,
}

impl LlmPool {
    pub fn new(model_path: &str, pool_size: usize) -> Result<Self, mullama::MullamaError> {
        let model = Arc::new(Model::load(model_path)?);

        let contexts: Vec<_> = (0..pool_size)
            .map(|_| {
                let ctx = Context::new(model.clone(), ContextParams::default())?;
                Ok(Arc::new(Mutex::new(ctx)))
            })
            .collect::<Result<_, mullama::MullamaError>>()?;

        Ok(Self {
            model,
            contexts,
            semaphore: Arc::new(Semaphore::new(pool_size)),
        })
    }

    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, mullama::MullamaError> {
        let _permit = self.semaphore.acquire().await.unwrap();

        for ctx in &self.contexts {
            if let Ok(mut guard) = ctx.try_lock() {
                let prompt = prompt.to_string();
                return tokio::task::spawn_blocking(move || {
                    guard.clear()?;
                    guard.generate(&prompt, max_tokens)
                }).await.unwrap();
            }
        }

        Err(mullama::MullamaError::ContextError("No context available".into()))
    }
}

// Usage in handler
async fn generate_handler(
    State(pool): State<Arc<LlmPool>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, String> {
    let text = pool.generate(&req.prompt, 100)
        .await
        .map_err(|e| e.to_string())?;

    Ok(Json(GenerateResponse { text }))
}
```

## Multimodal API

```rust
use axum::{
    extract::{Multipart, State},
    routing::post,
    Json, Router,
};
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

struct AppState {
    model: Arc<Model>,
    context: Arc<Mutex<Context>>,
    mtmd: Arc<Mutex<MtmdContext>>,
}

#[derive(Serialize)]
struct ImageResponse {
    description: String,
}

async fn describe_image(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ImageResponse>, String> {
    // Get image from multipart
    let mut image_data = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(|e| e.to_string())? {
        if field.name() == Some("image") {
            image_data = field.bytes().await.map_err(|e| e.to_string())?.to_vec();
        }
    }

    if image_data.is_empty() {
        return Err("No image provided".into());
    }

    let mut ctx = state.context.lock().await;
    let mut mtmd = state.mtmd.lock().await;

    let description = tokio::task::spawn_blocking(move || {
        ctx.clear()?;

        let image = mtmd.bitmap_from_buffer(&image_data)?;
        let chunks = mtmd.tokenize("Describe this image: <__media__>", &[&image])?;
        let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

        ctx.generate_continue(n_past, 200)
    })
    .await
    .unwrap()
    .map_err(|e: mullama::MullamaError| e.to_string())?;

    Ok(Json(ImageResponse { description }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let context = Arc::new(Mutex::new(
        Context::new(model.clone(), ContextParams::default())?
    ));
    let mtmd = Arc::new(Mutex::new(
        MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?
    ));

    let state = Arc::new(AppState { model, context, mtmd });

    let app = Router::new()
        .route("/describe", post(describe_image))
        .with_state(state);

    println!("Server running on http://localhost:3000");
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

**Client usage:**
```bash
curl -X POST http://localhost:3000/describe \
  -F "image=@photo.jpg"
```

## Health Check Endpoint

```rust
use axum::{routing::get, Json, Router};
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
    context_count: usize,
}

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: true,
        context_count: state.contexts.len(),
    })
}

// Add to router
let app = Router::new()
    .route("/health", get(health_check))
    .route("/generate", post(generate))
    .with_state(state);
```
