---
title: Async API
description: Asynchronous model loading, inference, and Tokio runtime integration
---

# Async API

The async module provides non-blocking wrappers around Mullama's core operations, enabling integration with Tokio-based async applications. It includes async model loading, non-blocking inference, model pooling, and advanced runtime management.

!!! info "Feature Gates"
    - `async` -- Enables `AsyncModel`, `AsyncContext`, `AsyncConfig`, `ModelInfo`
    - `tokio-runtime` -- Enables `MullamaRuntime`, `ModelPool`, `TaskManager`, `RuntimeMetrics`

    ```toml
    mullama = { version = "0.1", features = ["async"] }
    # Or for full runtime management:
    mullama = { version = "0.1", features = ["async", "tokio-runtime"] }
    ```

## AsyncModel

Async-first model interface with full Tokio integration. Wraps the synchronous `Model` and offloads CPU-intensive work to a blocking thread pool.

```rust
pub struct AsyncModel {
    inner: Arc<Model>,
    // Runtime management fields
}

impl AsyncModel {
    /// Load model asynchronously (non-blocking file I/O + parsing)
    pub async fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError>;

    /// Load with full configuration
    pub async fn load_with_config(config: AsyncConfig) -> Result<Self, MullamaError>;

    /// Generate text asynchronously
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, MullamaError>;

    /// Generate with streaming output
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: StreamConfig,
    ) -> Result<TokenStream, MullamaError>;

    /// Tokenize asynchronously
    pub async fn tokenize_async(
        &self,
        text: &str,
        add_bos: bool,
    ) -> Result<Vec<TokenId>, MullamaError>;

    /// Get model metadata
    pub async fn model_info(&self) -> Result<ModelInfo, MullamaError>;

    /// Check if model is ready for inference
    pub async fn is_ready(&self) -> bool;

    /// Clone for concurrent use (Arc-based, cheap)
    pub fn clone(&self) -> Self;
}
```

**Thread Safety:** `AsyncModel` implements `Send + Sync` and is designed for concurrent use across multiple async tasks.

### Example

```rust
use mullama::AsyncModel;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    // Load model (non-blocking)
    let model = AsyncModel::load("model.gguf").await?;

    // Generate text
    let response = model.generate("Tell me about Rust", 200).await?;
    println!("{}", response);

    // Model info
    let info = model.model_info().await?;
    println!("Model: {} ({} params)", info.name, info.n_params);

    Ok(())
}
```

!!! note "Node.js / Python Equivalents"
    - **Node.js:** `const model = await Model.load("model.gguf"); const result = await model.generate("prompt", 200);`
    - **Python:** `model = await Model.load("model.gguf"); result = await model.generate("prompt", 200)`

## AsyncContext

Async context management for non-blocking inference operations. Each `AsyncContext` manages its own synchronous `Context` internally via `spawn_blocking`.

```rust
pub struct AsyncContext {
    // Internal synchronous context managed via spawn_blocking
}

impl AsyncContext {
    /// Create async context from model
    pub async fn new(model: Arc<AsyncModel>) -> Result<Self, MullamaError>;

    /// Generate with async streaming
    pub async fn generate_stream(
        &mut self,
        tokens: &[TokenId],
    ) -> Result<TokenStream, MullamaError>;

    /// Process batch asynchronously
    pub async fn process_batch(
        &mut self,
        batches: Vec<Batch>,
    ) -> Result<Vec<Vec<TokenId>>, MullamaError>;

    /// Save context state to file
    pub async fn save_state(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), MullamaError>;

    /// Load context state from file
    pub async fn load_state(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), MullamaError>;
}
```

## AsyncConfig

Configuration for async model loading and behavior.

```rust
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    pub model_path: String,
    pub gpu_layers: i32,
    pub context_size: u32,
    pub threads: i32,
    pub flash_attention: bool,
    pub max_concurrent: usize,
    pub timeout_ms: u64,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_path` | `String` | `""` | Path to the GGUF model file |
| `gpu_layers` | `i32` | `0` | Number of layers to offload to GPU |
| `context_size` | `u32` | `2048` | Context window size in tokens |
| `threads` | `i32` | CPU count | Number of threads for inference |
| `flash_attention` | `bool` | `false` | Enable flash attention |
| `max_concurrent` | `usize` | `4` | Maximum concurrent inference requests |
| `timeout_ms` | `u64` | `30000` | Request timeout in milliseconds |

**Example:**

```rust
use mullama::{AsyncModel, AsyncConfig};

let config = AsyncConfig {
    model_path: "model.gguf".to_string(),
    gpu_layers: 32,
    context_size: 4096,
    threads: 8,
    flash_attention: true,
    max_concurrent: 4,
    timeout_ms: 30000,
};

let model = AsyncModel::load_with_config(config).await?;
```

## Cancellation

Async operations support cancellation via Tokio's `CancellationToken`:

```rust
use tokio_util::sync::CancellationToken;
use tokio::select;

let model = AsyncModel::load("model.gguf").await?;
let cancel = CancellationToken::new();

let cancel_clone = cancel.clone();
let generate_handle = tokio::spawn(async move {
    select! {
        result = model.generate("Write a long essay", 5000) => result,
        _ = cancel_clone.cancelled() => {
            Err(MullamaError::OperationFailed("Cancelled".to_string()))
        }
    }
});

// Cancel after 5 seconds
tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
cancel.cancel();
```

## ModelInfo

Metadata about a loaded model, available through async access.

```rust
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub n_params: u64,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ctx_train: usize,
    pub size_bytes: u64,
}
```

## MullamaRuntime

Advanced Tokio runtime management with builder pattern. Provides a managed runtime specifically configured for LLM inference workloads.

!!! info "Feature Gate"
    Requires the `tokio-runtime` feature flag.

```rust
pub struct MullamaRuntime {
    // Tokio runtime and management state
}

impl MullamaRuntime {
    /// Create new runtime builder
    pub fn new() -> MullamaRuntimeBuilder;

    /// Spawn async task on runtime
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawn blocking task (for CPU-intensive inference)
    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Get runtime performance metrics
    pub async fn metrics(&self) -> RuntimeMetrics;

    /// Graceful shutdown with timeout
    pub async fn shutdown(self);

    /// Check if runtime is still running
    pub fn is_running(&self) -> bool;
}
```

### MullamaRuntimeBuilder

```rust
let runtime = MullamaRuntime::new()
    .worker_threads(4)
    .max_blocking_threads(8)
    .thread_stack_size(4 * 1024 * 1024)
    .thread_name("mullama-worker")
    .enable_io()
    .enable_time()
    .build()?;
```

| Method | Parameter | Description |
|--------|-----------|-------------|
| `worker_threads(n)` | `usize` | Number of async worker threads |
| `max_blocking_threads(n)` | `usize` | Maximum blocking thread pool size |
| `thread_stack_size(n)` | `usize` | Stack size per thread in bytes |
| `thread_name(s)` | `impl Into<String>` | Thread name prefix for debugging |
| `enable_io()` | -- | Enable I/O driver (network, files) |
| `enable_time()` | -- | Enable time driver (timers, delays) |
| `enable_all()` | -- | Enable all drivers |
| `build()` | -- | Build the runtime |

## ModelPool

Manages multiple model/context instances for concurrent request handling. Implements connection pooling semantics.

```rust
pub struct ModelPool {
    // Pool of model instances
}

impl ModelPool {
    /// Create a pool with N model instances
    pub async fn new(
        config: AsyncConfig,
        pool_size: usize,
    ) -> Result<Self, MullamaError>;

    /// Acquire a model instance (blocks if all in use)
    pub async fn acquire(&self) -> Result<PooledModel, MullamaError>;

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats;

    /// Resize the pool dynamically
    pub async fn resize(&self, new_size: usize) -> Result<(), MullamaError>;
}
```

**Example:**

```rust
use mullama::{AsyncConfig, ModelPool};

let config = AsyncConfig {
    model_path: "model.gguf".to_string(),
    gpu_layers: 32,
    ..Default::default()
};

// Create pool with 4 model instances
let pool = ModelPool::new(config, 4).await?;

// Acquire and use (model returned to pool when PooledModel is dropped)
let model = pool.acquire().await?;
let result = model.generate("Hello", 100).await?;
drop(model); // Explicitly return to pool (or let it drop naturally)

// Check pool stats
let stats = pool.stats();
println!("Available: {}, In use: {}", stats.available, stats.in_use);
```

## TaskManager

Coordinates async tasks with lifecycle management and health monitoring.

```rust
pub struct TaskManager {
    // Task tracking state
}

impl TaskManager {
    /// Create new task manager
    pub fn new(runtime: &Arc<MullamaRuntime>) -> Self;

    /// Spawn a generation worker
    pub async fn spawn_generation_worker(&mut self) -> Result<(), MullamaError>;

    /// Spawn a metrics collection worker
    pub async fn spawn_metrics_collector(&mut self) -> Result<(), MullamaError>;

    /// Spawn a custom named worker
    pub async fn spawn_worker<F>(
        &mut self,
        name: &str,
        worker: F,
    ) -> Result<(), MullamaError>
    where
        F: Future<Output = Result<(), MullamaError>> + Send + 'static;

    /// Stop all managed workers
    pub async fn stop_all(&mut self) -> Result<(), MullamaError>;

    /// Get status of all workers
    pub fn worker_status(&self) -> HashMap<String, WorkerStatus>;
}
```

## RuntimeMetrics

Performance monitoring data for the async runtime.

```rust
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    pub active_tasks: usize,
    pub completed_tasks: u64,
    pub worker_threads: usize,
    pub blocking_threads: usize,
    pub total_generations: u64,
    pub avg_generation_ms: f64,
    pub uptime_seconds: u64,
}
```

## Async/Sync Duality

The core API is synchronous for simplicity. Async wrappers use `spawn_blocking` internally to avoid blocking the Tokio runtime:

```rust
// Synchronous (core API -- no feature required)
let model = Model::load("model.gguf")?;
let tokens = model.tokenize("Hello", true, false)?;

// Asynchronous (requires "async" feature)
let model = AsyncModel::load("model.gguf").await?;
let tokens = model.tokenize_async("Hello", true).await?;
```

Both approaches use the same underlying llama.cpp operations. The async wrappers offload CPU-intensive work (model loading, token generation) to a blocking thread pool so they do not block async worker threads.

## Complete Example

```rust
use mullama::{AsyncModel, AsyncConfig, MullamaRuntime, StreamConfig};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    // Create runtime
    let runtime = Arc::new(
        MullamaRuntime::new()
            .worker_threads(4)
            .enable_all()
            .build()?
    );

    // Load model with configuration
    let model = AsyncModel::load_with_config(AsyncConfig {
        model_path: "model.gguf".to_string(),
        gpu_layers: 32,
        context_size: 4096,
        threads: 8,
        flash_attention: true,
        max_concurrent: 4,
        timeout_ms: 30000,
    }).await?;

    // Simple generation
    let response = model.generate("What is Rust?", 200).await?;
    println!("Response: {}", response);

    // Streaming generation
    let config = StreamConfig::new()
        .max_tokens(200)
        .temperature(0.7);

    let mut stream = model.generate_stream("Tell me a story", config).await?;

    print!("Story: ");
    while let Some(token_result) = stream.next().await {
        let token_data = token_result?;
        print!("{}", token_data.text);
    }
    println!();

    // Runtime metrics
    let metrics = runtime.metrics().await;
    println!("Completed tasks: {}", metrics.completed_tasks);

    Ok(())
}
```
