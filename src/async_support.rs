//! Async/await support for non-blocking operations
//!
//! This module provides async versions of potentially long-running operations
//! like model loading and text generation, allowing for better integration
//! with async Rust applications.
//!
//! ## Features
//!
//! - **Non-blocking model loading**: Load models without blocking the async runtime
//! - **Async generation**: Generate text without blocking other tasks
//! - **Cancellation support**: Operations can be cancelled via tokio cancellation tokens
//! - **Progress reporting**: Optional progress callbacks for long operations
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::async_support::AsyncModel;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     // Load model asynchronously
//!     let model = AsyncModel::load("path/to/model.gguf").await?;
//!
//!     // Generate text without blocking
//!     let result = model.generate_async("Hello, world!", 100).await?;
//!     println!("Generated: {}", result);
//!
//!     Ok(())
//! }
//! ```

#[cfg(feature = "async")]
use futures::future::BoxFuture;
#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use tokio::task;

use crate::{Context, ContextParams, Model, MullamaError, SamplerChain, SamplerParams, TokenId};

/// Async wrapper for Model with non-blocking operations
#[cfg(feature = "async")]
#[derive(Clone)]
pub struct AsyncModel {
    inner: Arc<Model>,
}

#[cfg(feature = "async")]
impl AsyncModel {
    /// Load a model asynchronously without blocking the current thread
    ///
    /// This spawns the model loading operation on a blocking thread pool,
    /// allowing the async runtime to continue processing other tasks.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::async_support::AsyncModel;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), mullama::MullamaError> {
    ///     let model = AsyncModel::load("path/to/model.gguf").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn load(path: impl AsRef<str> + Send + 'static) -> Result<Self, MullamaError> {
        let path = path.as_ref().to_string();
        let model = task::spawn_blocking(move || Model::load(&path))
            .await
            .map_err(|e| MullamaError::ModelLoadError(format!("Async task failed: {}", e)))?;

        match model {
            Ok(model) => Ok(AsyncModel {
                inner: Arc::new(model),
            }),
            Err(e) => Err(e),
        }
    }

    /// Load a model with custom parameters asynchronously
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    /// * `params` - Model loading parameters
    pub async fn load_with_params(
        path: impl AsRef<str> + Send + 'static,
        params: crate::ModelParams,
    ) -> Result<Self, MullamaError> {
        let path = path.as_ref().to_string();
        let model = task::spawn_blocking(move || Model::load_with_params(&path, params))
            .await
            .map_err(|e| MullamaError::ModelLoadError(format!("Async task failed: {}", e)))?;

        match model {
            Ok(model) => Ok(AsyncModel {
                inner: Arc::new(model),
            }),
            Err(e) => Err(e),
        }
    }

    /// Create an async context for this model
    ///
    /// # Arguments
    ///
    /// * `params` - Context parameters
    pub async fn create_context_async(
        &self,
        params: ContextParams,
    ) -> Result<AsyncContext, MullamaError> {
        let model = self.inner.clone();
        let context = task::spawn_blocking(move || Context::new(model, params))
            .await
            .map_err(|e| MullamaError::ContextError(format!("Async task failed: {}", e)))?;

        match context {
            Ok(context) => Ok(AsyncContext {
                inner: context,
                model: self.inner.clone(),
            }),
            Err(e) => Err(e),
        }
    }

    /// Generate text asynchronously
    ///
    /// This is a convenience method that creates a context, configures sampling,
    /// and generates text all in one async operation.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::async_support::AsyncModel;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), mullama::MullamaError> {
    ///     let model = AsyncModel::load("path/to/model.gguf").await?;
    ///     let result = model.generate_async("The future of AI is", 100).await?;
    ///     println!("Generated: {}", result);
    ///     Ok(())
    /// }
    /// ```
    pub async fn generate_async(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, MullamaError> {
        let model = self.inner.clone();
        let prompt = prompt.to_string();

        task::spawn_blocking(move || {
            // Create context
            let mut ctx_params = ContextParams::default();
            ctx_params.n_ctx = 2048;
            let mut context = Context::new(model.clone(), ctx_params)?;

            // Configure sampling
            let mut sampler_params = SamplerParams::default();
            sampler_params.temperature = 0.7;
            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Tokenize prompt
            let tokens = model.tokenize(&prompt, true, false)?;
            if tokens.is_empty() {
                return Err(MullamaError::InvalidInput(
                    "Prompt produced no tokens".to_string(),
                ));
            }
            context.decode(&tokens)?;

            // Generate
            let mut result = String::new();
            let eos = model.token_eos();
            for _ in 0..max_tokens {
                // Use -1 to sample from the last token's logits
                let next_token = sampler.sample(&mut context, -1);
                if next_token == eos {
                    break;
                }

                let text = model.token_to_str(next_token, 0, false)?;
                result.push_str(&text);
                sampler.accept(next_token);
                context.decode(std::slice::from_ref(&next_token))?;
            }

            Ok(result)
        })
        .await
        .map_err(|e| MullamaError::GenerationError(format!("Async task failed: {}", e)))?
    }

    /// Get the underlying model
    pub fn model(&self) -> &Arc<Model> {
        &self.inner
    }

    /// Get model information asynchronously
    pub async fn info_async(&self) -> ModelInfo {
        let model = self.inner.clone();
        task::spawn_blocking(move || ModelInfo {
            vocab_size: model.vocab_size(),
            n_ctx_train: model.n_ctx_train(),
            n_embd: model.n_embd(),
            n_layer: model.n_layer(),
        })
        .await
        .unwrap_or_default()
    }
}

/// Async wrapper for Context
#[cfg(feature = "async")]
pub struct AsyncContext {
    inner: Context,
    model: Arc<Model>,
}

#[cfg(feature = "async")]
impl AsyncContext {
    /// Create a new AsyncContext from a Context and model
    ///
    /// # Arguments
    ///
    /// * `inner` - The wrapped Context
    /// * `model` - The model Arc
    pub fn new(inner: Context, model: Arc<Model>) -> Self {
        Self { inner, model }
    }

    /// Get a mutable reference to the inner context
    pub fn inner_mut(&mut self) -> &mut Context {
        &mut self.inner
    }

    /// Get a reference to the inner context
    pub fn inner(&self) -> &Context {
        &self.inner
    }

    /// Generate text asynchronously with fine-grained control
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input tokens
    /// * `max_tokens` - Maximum tokens to generate
    /// * `sampler` - Sampling strategy
    pub async fn generate_with_sampler_async(
        mut self,
        tokens: &[TokenId],
        max_tokens: usize,
        mut sampler: SamplerChain,
    ) -> Result<String, MullamaError> {
        let model = self.model.clone();
        let tokens = tokens.to_vec();

        task::spawn_blocking(move || {
            let mut result = String::new();
            if tokens.is_empty() {
                return Err(MullamaError::InvalidInput(
                    "Cannot generate from empty prompt".to_string(),
                ));
            }
            self.inner.decode(&tokens)?;
            let eos = model.token_eos();

            for _ in 0..max_tokens {
                // Use -1 to sample from the last token's logits
                let next_token = sampler.sample(&mut self.inner, -1);
                if next_token == eos {
                    break;
                }

                let text = model.token_to_str(next_token, 0, false)?;
                result.push_str(&text);
                sampler.accept(next_token);
                self.inner.decode(std::slice::from_ref(&next_token))?;
            }

            Ok(result)
        })
        .await
        .map_err(|e| MullamaError::GenerationError(format!("Async task failed: {}", e)))?
    }

    /// Get the underlying context (consumes self)
    pub fn into_inner(self) -> Context {
        self.inner
    }
}

/// Model information structure for async operations
#[cfg(feature = "async")]
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub vocab_size: i32,
    pub n_ctx_train: i32,
    pub n_embd: i32,
    pub n_layer: i32,
}

/// Progress callback type for long-running async operations
#[cfg(feature = "async")]
pub type ProgressCallback = Box<dyn Fn(f32) -> BoxFuture<'static, ()> + Send + Sync>;

/// Configuration for async operations
#[cfg(feature = "async")]
#[derive(Clone)]
pub struct AsyncConfig {
    /// Optional progress callback
    pub progress_callback: Option<Arc<ProgressCallback>>,
    /// Cancellation token
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

#[cfg(feature = "async")]
impl std::fmt::Debug for AsyncConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncConfig")
            .field("progress_callback", &self.progress_callback.as_ref().map(|_| "<callback>"))
            .field("cancellation_token", &self.cancellation_token)
            .finish()
    }
}

#[cfg(feature = "async")]
impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            progress_callback: None,
            cancellation_token: None,
        }
    }
}

#[cfg(not(feature = "async"))]
compile_error!("Async support requires the 'async' feature to be enabled");
