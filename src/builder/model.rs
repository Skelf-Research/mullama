use crate::{Model, ModelParams, MullamaError};
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::async_support::AsyncModel;

/// Builder for creating models with fluent API
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    path: Option<String>,
    gpu_layers: i32,
    context_size: Option<u32>,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
    vocab_only: bool,
}

impl ModelBuilder {
    /// Create a new model builder
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            path: None,
            gpu_layers: 0,
            context_size: None,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            vocab_only: false,
        }
    }

    /// Set the path to the model file (required)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .path("path/to/model.gguf");
    /// ```
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the number of GPU layers to offload
    ///
    /// # Arguments
    ///
    /// * `layers` - Number of layers to offload to GPU (0 = CPU only)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .gpu_layers(32); // Offload 32 layers to GPU
    /// ```
    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Set the context size for the model
    ///
    /// # Arguments
    ///
    /// * `size` - Context size in tokens
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .context_size(4096); // 4K context
    /// ```
    pub fn context_size(mut self, size: u32) -> Self {
        self.context_size = Some(size);
        self
    }

    /// Enable or disable memory mapping
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to use memory mapping
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .memory_mapping(true); // Enable mmap
    /// ```
    pub fn memory_mapping(mut self, enable: bool) -> Self {
        self.use_mmap = enable;
        self
    }

    /// Enable or disable memory locking
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to use memory locking
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .memory_locking(true); // Enable mlock
    /// ```
    pub fn memory_locking(mut self, enable: bool) -> Self {
        self.use_mlock = enable;
        self
    }

    /// Enable or disable tensor validation
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to validate tensors
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .tensor_validation(false); // Disable validation for faster loading
    /// ```
    pub fn tensor_validation(mut self, enable: bool) -> Self {
        self.check_tensors = enable;
        self
    }

    /// Set vocabulary-only mode
    ///
    /// In vocabulary-only mode, only the tokenizer is loaded,
    /// which is useful for tokenization-only tasks.
    ///
    /// # Arguments
    ///
    /// * `vocab_only` - Whether to load only vocabulary
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .vocabulary_only(true); // Load only tokenizer
    /// ```
    pub fn vocabulary_only(mut self, vocab_only: bool) -> Self {
        self.vocab_only = vocab_only;
        self
    }

    /// Apply a preset configuration
    ///
    /// # Arguments
    ///
    /// * `preset` - Preset configuration function
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, presets};
    ///
    /// let builder = ModelBuilder::new()
    ///     .preset(presets::performance_optimized);
    /// ```
    pub fn preset<F>(self, preset: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        preset(self)
    }

    /// Build the model synchronously
    ///
    /// # Returns
    ///
    /// An `Arc<Model>` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if model loading fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let model = ModelBuilder::new()
    ///     .path("model.gguf")
    ///     .gpu_layers(16)
    ///     .build()?;
    /// # Ok::<(), mullama::MullamaError>(())
    /// ```
    pub fn build(self) -> Result<Arc<Model>, MullamaError> {
        let path = self
            .path
            .ok_or_else(|| MullamaError::ConfigError("Model path is required".to_string()))?;

        let params = ModelParams {
            n_gpu_layers: self.gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            check_tensors: self.check_tensors,
            vocab_only: self.vocab_only,
            ..Default::default()
        };

        let model = Model::load_with_params(&path, params)?;
        Ok(Arc::new(model))
    }

    /// Build the model asynchronously
    ///
    /// # Returns
    ///
    /// An `AsyncModel` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if model loading fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), mullama::MullamaError> {
    ///     let model = ModelBuilder::new()
    ///         .path("model.gguf")
    ///         .gpu_layers(16)
    ///         .build_async()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "async")]
    pub async fn build_async(self) -> Result<AsyncModel, MullamaError> {
        let path = self
            .path
            .ok_or_else(|| MullamaError::ConfigError("Model path is required".to_string()))?;

        let params = ModelParams {
            n_gpu_layers: self.gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            check_tensors: self.check_tensors,
            vocab_only: self.vocab_only,
            ..Default::default()
        };

        AsyncModel::load_with_params(path, params).await
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_builder() {
        let builder = ModelBuilder::new()
            .path("test.gguf")
            .gpu_layers(16)
            .context_size(2048);

        assert_eq!(builder.path, Some("test.gguf".to_string()));
        assert_eq!(builder.gpu_layers, 16);
        assert_eq!(builder.context_size, Some(2048));
    }
}
