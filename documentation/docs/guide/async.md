# Async Support

Run LLM inference asynchronously with Tokio integration.

## Enabling Async

```toml
[dependencies]
mullama = { version = "0.1", features = ["async"] }
tokio = { version = "1", features = ["full"] }
```

## Basic Async Usage

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    // Load model (blocking operation, use spawn_blocking)
    let model = tokio::task::spawn_blocking(|| {
        Model::load("model.gguf")
    }).await.unwrap()?;

    let model = Arc::new(model);
    let mut context = Context::new(model, ContextParams::default())?;

    // Generate asynchronously
    let response = tokio::task::spawn_blocking(move || {
        context.generate("Hello!", 100)
    }).await.unwrap()?;

    println!("{}", response);
    Ok(())
}
```

## Concurrent Requests

Handle multiple requests concurrently:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

struct LlmService {
    model: Arc<Model>,
    context: Arc<Mutex<Context>>,
}

impl LlmService {
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, mullama::MullamaError> {
        let prompt = prompt.to_string();
        let mut context = self.context.lock().await;

        // Run generation in blocking task
        let response = tokio::task::spawn_blocking(move || {
            context.generate(&prompt, max_tokens)
        }).await.unwrap()?;

        Ok(response)
    }
}
```

## Connection Pool

For high-throughput applications:

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

struct LlmPool {
    model: Arc<Model>,
    contexts: Vec<Arc<Mutex<Context>>>,
    semaphore: Arc<Semaphore>,
}

impl LlmPool {
    fn new(model: Arc<Model>, pool_size: usize) -> Result<Self, mullama::MullamaError> {
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

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, mullama::MullamaError> {
        // Acquire permit
        let _permit = self.semaphore.acquire().await.unwrap();

        // Find available context
        for ctx in &self.contexts {
            if let Ok(mut guard) = ctx.try_lock() {
                let prompt = prompt.to_string();
                return tokio::task::spawn_blocking(move || {
                    guard.generate(&prompt, max_tokens)
                }).await.unwrap();
            }
        }

        Err(mullama::MullamaError::ContextError("No context available".to_string()))
    }
}
```

## Async Streaming

Stream tokens asynchronously:

```rust
use tokio::sync::mpsc;

async fn generate_streaming(
    context: Arc<Mutex<Context>>,
    prompt: String,
    max_tokens: usize,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let mut ctx = context.blocking_lock();
        ctx.generate_streaming(&prompt, max_tokens, |token| {
            tx.blocking_send(token.to_string()).is_ok()
        }).ok();
    });

    rx
}

// Usage
let mut rx = generate_streaming(context, "Hello:".to_string(), 200).await;
while let Some(token) = rx.recv().await {
    print!("{}", token);
}
```

## Timeout Handling

Add timeouts to prevent hanging:

```rust
use tokio::time::{timeout, Duration};

async fn generate_with_timeout(
    context: &mut Context,
    prompt: &str,
    max_tokens: usize,
    time_limit: Duration,
) -> Result<String, mullama::MullamaError> {
    let prompt = prompt.to_string();

    match timeout(time_limit, tokio::task::spawn_blocking(move || {
        context.generate(&prompt, max_tokens)
    })).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => Err(mullama::MullamaError::GenerationError(e.to_string())),
        Err(_) => Err(mullama::MullamaError::GenerationError("Timeout".to_string())),
    }
}
```

## Cancellation

Cancel generation on signal:

```rust
use tokio::sync::oneshot;
use std::sync::atomic::{AtomicBool, Ordering};

async fn cancellable_generate(
    context: Arc<Mutex<Context>>,
    prompt: String,
    max_tokens: usize,
) -> (mpsc::Receiver<String>, oneshot::Sender<()>) {
    let (tx, rx) = mpsc::channel(32);
    let (cancel_tx, cancel_rx) = oneshot::channel();

    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = cancelled.clone();

    // Watch for cancellation
    tokio::spawn(async move {
        let _ = cancel_rx.await;
        cancelled_clone.store(true, Ordering::SeqCst);
    });

    tokio::task::spawn_blocking(move || {
        let mut ctx = context.blocking_lock();
        ctx.generate_streaming(&prompt, max_tokens, |token| {
            if cancelled.load(Ordering::SeqCst) {
                return false;
            }
            tx.blocking_send(token.to_string()).is_ok()
        }).ok();
    });

    (rx, cancel_tx)
}

// Usage
let (mut rx, cancel) = cancellable_generate(context, prompt, 500).await;

tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(5)).await;
    let _ = cancel.send(());  // Cancel after 5 seconds
});

while let Some(token) = rx.recv().await {
    print!("{}", token);
}
```

## Web Server Example

Full async web server:

```rust
use axum::{
    extract::Json,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: usize,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
}

struct AppState {
    model: Arc<Model>,
    context: Arc<Mutex<Context>>,
}

async fn generate_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let mut ctx = state.context.lock().await;
    let prompt = req.prompt.clone();

    let text = tokio::task::spawn_blocking(move || {
        ctx.generate(&prompt, req.max_tokens)
    }).await.unwrap().unwrap_or_default();

    Json(GenerateResponse { text })
}

#[tokio::main]
async fn main() {
    let model = Arc::new(Model::load("model.gguf").unwrap());
    let context = Arc::new(Mutex::new(
        Context::new(model.clone(), ContextParams::default()).unwrap()
    ));

    let state = Arc::new(AppState { model, context });

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(state);

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

## Best Practices

1. **Use `spawn_blocking`** - LLM inference is CPU-bound
2. **Pool contexts** - Reuse contexts for better performance
3. **Add timeouts** - Prevent runaway generation
4. **Handle cancellation** - Allow stopping long operations
5. **Limit concurrency** - Use semaphores to control load
