//! Model provider trait for unified model resolution and downloading.
//!
//! Abstracts over different model sources (HuggingFace, Ollama registry,
//! local files) with a unified async interface.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use crate::error::MullamaError;

/// Result of resolving a model spec to a local path.
#[derive(Debug, Clone)]
pub struct ResolvedModelPath {
    /// Local filesystem path to the GGUF model file
    pub path: PathBuf,
    /// Human-readable alias for the model
    pub alias: String,
    /// Whether the model was already cached locally
    pub was_cached: bool,
}

/// Trait for model providers that can resolve model specs to local paths.
///
/// Implementors handle the full lifecycle: check cache, download if needed,
/// and return a local path. This enables a unified model resolution pipeline
/// regardless of the source registry.
///
/// Uses `Pin<Box<dyn Future>>` for `resolve` so the trait is dyn-compatible,
/// allowing `Vec<Box<dyn ModelProvider>>` for multi-source resolution.
///
/// # Example
///
/// ```rust,no_run
/// use mullama::daemon::provider::ModelProvider;
///
/// async fn load_model(providers: &[Box<dyn ModelProvider>], spec: &str) {
///     for provider in providers {
///         if provider.supports(spec) {
///             match provider.resolve(spec).await {
///                 Ok(resolved) => {
///                     println!("Model at: {}", resolved.path.display());
///                     return;
///                 }
///                 Err(e) => eprintln!("Failed: {}", e),
///             }
///         }
///     }
///     eprintln!("No provider supports: {}", spec);
/// }
/// ```
pub trait ModelProvider: Send + Sync {
    /// Check if this provider can handle the given model spec.
    ///
    /// Returns `true` if the spec matches this provider's format.
    /// For example, `OllamaProvider` returns true for "llama3:1b",
    /// while `HuggingFaceProvider` returns true for "hf:owner/repo".
    fn supports(&self, spec: &str) -> bool;

    /// Resolve a model spec to a local path, downloading if necessary.
    ///
    /// This is an async operation since it may involve network downloads.
    fn resolve(
        &self,
        spec: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedModelPath, MullamaError>> + Send + '_>>;

    /// Check if the model is already cached locally.
    fn is_cached(&self, spec: &str) -> bool;

    /// Provider name for display/logging purposes.
    fn name(&self) -> &str;
}
