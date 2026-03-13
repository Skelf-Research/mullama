//! Tokio runtime integration with advanced utilities
//!
//! This module provides comprehensive Tokio integration for Mullama, including
//! runtime management, task scheduling, and coordination helpers.
//!
//! ## Features
//!
//! - **Runtime Management**: Create and manage Tokio runtimes
//! - **Task Coordination**: Advanced task spawning and coordination
//! - **Resource Pooling**: Async resource pools for models and contexts
//! - **Background Processing**: Long-running background tasks
//! - **Graceful Shutdown**: Clean shutdown coordination
//! - **Performance Monitoring**: Task and runtime metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::tokio_integration::{MullamaRuntime, TaskManager, ModelPool};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     // Create managed runtime
//!     let runtime = MullamaRuntime::new()
//!         .worker_threads(8)
//!         .max_blocking_threads(16)
//!         .enable_all()
//!         .build()?;
//!
//!     // Create model pool
//!     let pool = ModelPool::new()
//!         .max_size(4)
//!         .min_idle(1)
//!         .build(&runtime).await?;
//!
//!     // Spawn coordinated tasks
//!     let task_manager = TaskManager::new(&runtime);
//!     task_manager.spawn_generation_worker().await?;
//!
//!     Ok(())
//! }
//! ```

#[cfg(feature = "tokio-runtime")]
use tokio::{
    runtime::{Builder, Handle, Runtime},
    sync::{broadcast, mpsc, oneshot, RwLock, Semaphore},
    task::{JoinHandle, JoinSet},
    time::{interval, Duration, Instant},
};

#[cfg(feature = "tokio-runtime")]
use tokio_util::{sync::CancellationToken, task::TaskTracker};

#[cfg(feature = "tokio-runtime")]
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

#[cfg(feature = "async")]
use crate::{AsyncContext, AsyncModel, MullamaError};

/// Managed Tokio runtime for Mullama operations
#[cfg(feature = "tokio-runtime")]
pub struct MullamaRuntime {
    runtime: Runtime,
    task_tracker: TaskTracker,
    shutdown_token: CancellationToken,
    metrics: Arc<RuntimeMetrics>,
}

#[cfg(feature = "tokio-runtime")]
impl MullamaRuntime {
    /// Create a new runtime builder
    pub fn new() -> MullamaRuntimeBuilder {
        MullamaRuntimeBuilder::new()
    }

    /// Get the runtime handle
    pub fn handle(&self) -> &Handle {
        self.runtime.handle()
    }

    /// Get the task tracker
    pub fn task_tracker(&self) -> &TaskTracker {
        &self.task_tracker
    }

    /// Get the shutdown token
    pub fn shutdown_token(&self) -> &CancellationToken {
        &self.shutdown_token
    }

    /// Get runtime metrics
    pub fn metrics(&self) -> Arc<RuntimeMetrics> {
        self.metrics.clone()
    }

    /// Spawn a tracked task
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let handle = self.runtime.spawn(future);
        self.task_tracker.track(&handle);
        self.metrics.tasks_spawned.fetch_add(1, Ordering::Relaxed);
        handle
    }

    /// Spawn a blocking task
    pub fn spawn_blocking<F, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let handle = self.runtime.spawn_blocking(f);
        self.task_tracker.track(&handle);
        self.metrics
            .blocking_tasks_spawned
            .fetch_add(1, Ordering::Relaxed);
        handle
    }

    /// Block on a future
    pub fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// Gracefully shutdown the runtime
    pub async fn shutdown(self) {
        println!("🔄 Initiating graceful shutdown...");

        // Signal shutdown
        self.shutdown_token.cancel();

        // Wait for all tasks to complete
        self.task_tracker.close();
        self.task_tracker.wait().await;

        // Shutdown the runtime
        self.runtime.shutdown_timeout(Duration::from_secs(30)).await;

        println!("✅ Runtime shutdown complete");
    }
}

/// Builder for MullamaRuntime
#[cfg(feature = "tokio-runtime")]
pub struct MullamaRuntimeBuilder {
    builder: Builder,
}

#[cfg(feature = "tokio-runtime")]
impl MullamaRuntimeBuilder {
    pub fn new() -> Self {
        Self {
            builder: Builder::new_multi_thread(),
        }
    }

    /// Set the number of worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.builder.worker_threads(threads);
        self
    }

    /// Set the maximum number of blocking threads
    pub fn max_blocking_threads(mut self, threads: usize) -> Self {
        self.builder.max_blocking_threads(threads);
        self
    }

    /// Enable all Tokio features
    pub fn enable_all(mut self) -> Self {
        self.builder.enable_all();
        self
    }

    /// Set thread name prefix
    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.builder.thread_name(name);
        self
    }

    /// Set thread stack size
    pub fn thread_stack_size(mut self, size: usize) -> Self {
        self.builder.thread_stack_size(size);
        self
    }

    /// Build the runtime
    pub fn build(self) -> Result<MullamaRuntime, MullamaError> {
        let runtime = self
            .builder
            .build()
            .map_err(|e| MullamaError::ConfigError(format!("Failed to build runtime: {}", e)))?;

        Ok(MullamaRuntime {
            runtime,
            task_tracker: TaskTracker::new(),
            shutdown_token: CancellationToken::new(),
            metrics: Arc::new(RuntimeMetrics::new()),
        })
    }
}

/// Runtime metrics
#[cfg(feature = "tokio-runtime")]
#[derive(Debug)]
pub struct RuntimeMetrics {
    pub tasks_spawned: AtomicU64,
    pub blocking_tasks_spawned: AtomicU64,
    pub tasks_completed: AtomicU64,
    pub tasks_failed: AtomicU64,
    pub generation_requests: AtomicU64,
    pub average_generation_time: RwLock<Duration>,
}

#[cfg(feature = "tokio-runtime")]
impl RuntimeMetrics {
    pub fn new() -> Self {
        Self {
            tasks_spawned: AtomicU64::new(0),
            blocking_tasks_spawned: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            generation_requests: AtomicU64::new(0),
            average_generation_time: RwLock::new(Duration::from_millis(0)),
        }
    }

    pub async fn record_generation(&self, duration: Duration) {
        let count = self.generation_requests.fetch_add(1, Ordering::Relaxed) + 1;
        let mut avg = self.average_generation_time.write().await;

        // Calculate rolling average
        *avg = Duration::from_nanos(
            ((avg.as_nanos() as u64 * (count - 1)) + duration.as_nanos() as u64) / count,
        );
    }

    pub async fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            tasks_spawned: self.tasks_spawned.load(Ordering::Relaxed),
            blocking_tasks_spawned: self.blocking_tasks_spawned.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
            tasks_failed: self.tasks_failed.load(Ordering::Relaxed),
            generation_requests: self.generation_requests.load(Ordering::Relaxed),
            average_generation_time: *self.average_generation_time.read().await,
        }
    }
}

#[cfg(feature = "tokio-runtime")]
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub tasks_spawned: u64,
    pub blocking_tasks_spawned: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub generation_requests: u64,
    pub average_generation_time: Duration,
}

/// Task manager for coordinated operations
#[cfg(feature = "tokio-runtime")]
pub struct TaskManager {
    runtime: Arc<MullamaRuntime>,
    join_set: JoinSet<Result<(), MullamaError>>,
}

#[cfg(feature = "tokio-runtime")]
impl TaskManager {
    pub fn new(runtime: &Arc<MullamaRuntime>) -> Self {
        Self {
            runtime: runtime.clone(),
            join_set: JoinSet::new(),
        }
    }

    /// Spawn a generation worker task
    pub async fn spawn_generation_worker(&mut self) -> Result<(), MullamaError> {
        let runtime = self.runtime.clone();
        let shutdown_token = runtime.shutdown_token().clone();

        self.join_set.spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    _ = shutdown_token.cancelled() => {
                        println!("🔄 Generation worker shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        // Process generation queue (placeholder)
                        // In real implementation, this would process queued requests
                    }
                }
            }

            Ok(())
        });

        Ok(())
    }

    /// Spawn a metrics collection task
    pub async fn spawn_metrics_collector(&mut self) -> Result<(), MullamaError> {
        let runtime = self.runtime.clone();
        let shutdown_token = runtime.shutdown_token().clone();

        self.join_set.spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                tokio::select! {
                    _ = shutdown_token.cancelled() => {
                        println!("📊 Metrics collector shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        let summary = runtime.metrics().summary().await;
                        println!("📊 Runtime metrics: {:?}", summary);
                    }
                }
            }

            Ok(())
        });

        Ok(())
    }

    /// Wait for all managed tasks to complete
    pub async fn wait_all(
        &mut self,
    ) -> Vec<Result<Result<(), MullamaError>, tokio::task::JoinError>> {
        let mut results = Vec::new();

        while let Some(result) = self.join_set.join_next().await {
            results.push(result);
        }

        results
    }
}

/// Model pool for efficient resource management
#[cfg(all(feature = "tokio-runtime", feature = "async"))]
pub struct ModelPool {
    models: RwLock<Vec<Arc<AsyncModel>>>,
    semaphore: Semaphore,
    max_size: usize,
    min_idle: usize,
}

#[cfg(all(feature = "tokio-runtime", feature = "async"))]
impl ModelPool {
    pub fn new() -> ModelPoolBuilder {
        ModelPoolBuilder::new()
    }

    /// Get a model from the pool
    pub async fn get(&self) -> Result<PooledModel, MullamaError> {
        let _permit =
            self.semaphore.acquire().await.map_err(|_| {
                MullamaError::ConfigError("Failed to acquire semaphore".to_string())
            })?;

        let models = self.models.read().await;
        if let Some(model) = models.first() {
            Ok(PooledModel {
                model: model.clone(),
                _permit,
            })
        } else {
            Err(MullamaError::ConfigError(
                "No models available in pool".to_string(),
            ))
        }
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let models = self.models.read().await;
        PoolStats {
            total_models: models.len(),
            available_permits: self.semaphore.available_permits(),
            max_size: self.max_size,
            min_idle: self.min_idle,
        }
    }
}

/// Builder for ModelPool
#[cfg(all(feature = "tokio-runtime", feature = "async"))]
pub struct ModelPoolBuilder {
    max_size: usize,
    min_idle: usize,
}

#[cfg(all(feature = "tokio-runtime", feature = "async"))]
impl ModelPoolBuilder {
    pub fn new() -> Self {
        Self {
            max_size: 4,
            min_idle: 1,
        }
    }

    pub fn max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    pub fn min_idle(mut self, idle: usize) -> Self {
        self.min_idle = idle;
        self
    }

    pub async fn build(self, _runtime: &MullamaRuntime) -> Result<ModelPool, MullamaError> {
        // In real implementation, this would load models
        let models = Vec::new(); // Placeholder

        Ok(ModelPool {
            models: RwLock::new(models),
            semaphore: Semaphore::new(self.max_size),
            max_size: self.max_size,
            min_idle: self.min_idle,
        })
    }
}

/// A model checked out from the pool
#[cfg(all(feature = "tokio-runtime", feature = "async"))]
pub struct PooledModel {
    model: Arc<AsyncModel>,
    _permit: tokio::sync::SemaphorePermit<'static>,
}

#[cfg(all(feature = "tokio-runtime", feature = "async"))]
impl PooledModel {
    pub fn model(&self) -> &Arc<AsyncModel> {
        &self.model
    }
}

/// Pool statistics
#[cfg(feature = "tokio-runtime")]
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_models: usize,
    pub available_permits: usize,
    pub max_size: usize,
    pub min_idle: usize,
}

/// Background task coordinator
#[cfg(feature = "tokio-runtime")]
pub struct BackgroundCoordinator {
    tasks: HashMap<String, JoinHandle<()>>,
    shutdown_token: CancellationToken,
}

#[cfg(feature = "tokio-runtime")]
impl BackgroundCoordinator {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            shutdown_token: CancellationToken::new(),
        }
    }

    /// Start a background task
    pub fn start_task<F>(&mut self, name: String, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let shutdown_token = self.shutdown_token.clone();
        let handle = tokio::spawn(async move {
            tokio::select! {
                _ = future => {},
                _ = shutdown_token.cancelled() => {
                    println!("🔄 Background task '{}' cancelled", name);
                }
            }
        });

        self.tasks.insert(name, handle);
    }

    /// Stop a specific task
    pub fn stop_task(&mut self, name: &str) -> Option<JoinHandle<()>> {
        self.tasks.remove(name)
    }

    /// Stop all tasks
    pub async fn stop_all(&mut self) {
        self.shutdown_token.cancel();

        for (name, handle) in self.tasks.drain() {
            if let Err(e) = handle.await {
                eprintln!("❌ Error stopping task '{}': {}", name, e);
            }
        }
    }

    /// Get running task names
    pub fn running_tasks(&self) -> Vec<String> {
        self.tasks.keys().cloned().collect()
    }
}

/// High-level async coordination utilities
#[cfg(feature = "tokio-runtime")]
pub mod coordination {
    use super::*;

    /// Run multiple generation tasks concurrently with rate limiting
    pub async fn concurrent_generation<F, Fut>(
        tasks: Vec<F>,
        max_concurrent: usize,
    ) -> Vec<Result<String, MullamaError>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<String, MullamaError>>,
    {
        let semaphore = Semaphore::new(max_concurrent);
        let mut handles = Vec::new();

        for task in tasks {
            let permit = semaphore.clone();
            let handle = tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                task().await
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(MullamaError::GenerationError(format!(
                    "Task failed: {}",
                    e
                )))),
            }
        }

        results
    }

    /// Create a generation pipeline with backpressure
    pub async fn generation_pipeline(
        input: mpsc::Receiver<String>,
        output: mpsc::Sender<Result<String, MullamaError>>,
        concurrency: usize,
    ) {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut input = input;

        while let Some(prompt) = input.recv().await {
            let semaphore = semaphore.clone();
            let output = output.clone();

            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Simulate generation (placeholder)
                let result = Ok(format!("Generated response for: {}", prompt));

                let _ = output.send(result).await;
            });
        }
    }
}

#[cfg(not(feature = "tokio-runtime"))]
compile_error!("Tokio integration requires the 'tokio-runtime' feature to be enabled");
