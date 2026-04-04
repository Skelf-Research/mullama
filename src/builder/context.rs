use crate::{context::KvCacheType, Context, ContextParams, Model, MullamaError};
use std::sync::Arc;

/// Builder for creating contexts with fluent API
#[derive(Debug, Clone)]
pub struct ContextBuilder {
    model: Arc<Model>,
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    embeddings: bool,
    flash_attn_type: crate::sys::llama_flash_attn_type,
    offload_kqv: bool,
    type_k: KvCacheType,
    type_v: KvCacheType,
}

impl ContextBuilder {
    /// Create a new context builder
    ///
    /// # Arguments
    ///
    /// * `model` - The model to create a context for
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, ContextBuilder};
    ///
    /// # async fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let builder = ContextBuilder::new(model);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            n_ctx: 2048,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            embeddings: false,
            flash_attn_type: crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
            offload_kqv: false,
            type_k: KvCacheType::default(),
            type_v: KvCacheType::default(),
        }
    }

    /// Set the context size
    ///
    /// # Arguments
    ///
    /// * `size` - Context size in tokens
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .context_size(4096);
    /// ```
    pub fn context_size(mut self, size: u32) -> Self {
        self.n_ctx = size;
        self
    }

    /// Set the batch size
    ///
    /// # Arguments
    ///
    /// * `size` - Batch size for prompt processing
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .batch_size(1024);
    /// ```
    pub fn batch_size(mut self, size: u32) -> Self {
        self.n_batch = size;
        self
    }

    /// Set the physical batch size
    ///
    /// # Arguments
    ///
    /// * `size` - Physical batch size
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .physical_batch_size(256);
    /// ```
    pub fn physical_batch_size(mut self, size: u32) -> Self {
        self.n_ubatch = size;
        self
    }

    /// Set maximum number of sequences
    ///
    /// # Arguments
    ///
    /// * `max_seq` - Maximum number of parallel sequences
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .max_sequences(4);
    /// ```
    pub fn max_sequences(mut self, max_seq: u32) -> Self {
        self.n_seq_max = max_seq;
        self
    }

    /// Set number of threads for generation
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .threads(8);
    /// ```
    pub fn threads(mut self, threads: i32) -> Self {
        self.n_threads = threads;
        self
    }

    /// Set number of threads for batch processing
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads for batch processing
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .batch_threads(4);
    /// ```
    pub fn batch_threads(mut self, threads: i32) -> Self {
        self.n_threads_batch = threads;
        self
    }

    /// Enable or disable embeddings
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable embeddings
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .embeddings(true);
    /// ```
    pub fn embeddings(mut self, enable: bool) -> Self {
        self.embeddings = enable;
        self
    }

    /// Enable or disable flash attention
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable flash attention
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .flash_attention(true);
    /// ```
    pub fn flash_attention(mut self, enable: bool) -> Self {
        self.flash_attn_type = if enable {
            crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
        } else {
            crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED
        };
        self
    }

    /// Enable or disable KQV offloading
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to offload KQV operations to GPU
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .kqv_offload(true);
    /// ```
    pub fn kqv_offload(mut self, enable: bool) -> Self {
        self.offload_kqv = enable;
        self
    }

    /// Set KV-cache quantization type for both K and V caches
    ///
    /// Lower precision types reduce memory usage but may slightly affect quality:
    /// - F16 (default): Best quality, baseline memory
    /// - Q8_0: ~50% memory savings
    /// - Q4_0: ~75% memory savings
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for both K and V caches
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use mullama::context::KvCacheType;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .kv_cache_type(KvCacheType::Q8_0); // 50% memory savings
    /// ```
    pub fn kv_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_k = cache_type;
        self.type_v = cache_type;
        self
    }

    /// Set Key cache quantization type separately
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for the K cache
    pub fn key_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_k = cache_type;
        self
    }

    /// Set Value cache quantization type separately
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for the V cache
    pub fn value_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_v = cache_type;
        self
    }

    /// Apply performance optimizations
    ///
    /// This enables flash attention, optimizes thread counts,
    /// and sets efficient batch sizes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_performance();
    /// ```
    pub fn optimize_for_performance(mut self) -> Self {
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.n_batch = 1024;
        self.n_ubatch = 512;
        self.offload_kqv = true;
        self
    }

    /// Optimize for memory usage
    ///
    /// This reduces batch sizes, uses Q4 quantized KV cache (~75% memory savings),
    /// and disables memory-intensive features.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_memory();
    /// ```
    pub fn optimize_for_memory(mut self) -> Self {
        self.n_ctx = 2048;
        self.n_batch = 256;
        self.n_ubatch = 256;
        self.type_k = KvCacheType::Q4_0;
        self.type_v = KvCacheType::Q4_0;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self
    }

    /// Balanced optimization (Q8 KV cache)
    ///
    /// Uses Q8 quantized KV cache for ~50% memory savings with minimal quality loss.
    /// Good balance between memory usage and output quality.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_balanced();
    /// ```
    pub fn optimize_balanced(mut self) -> Self {
        self.type_k = KvCacheType::Q8_0;
        self.type_v = KvCacheType::Q8_0;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.offload_kqv = true;
        self
    }

    /// Optimize for quality (F16 KV cache)
    ///
    /// Uses full F16 precision KV cache for best output quality.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_quality();
    /// ```
    pub fn optimize_for_quality(mut self) -> Self {
        self.type_k = KvCacheType::F16;
        self.type_v = KvCacheType::F16;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO;
        self
    }

    /// Build the context
    ///
    /// # Returns
    ///
    /// A `Context` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if context creation fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::{ModelBuilder, ContextBuilder};
    /// # fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let context = ContextBuilder::new(model)
    ///     .context_size(2048)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Context, MullamaError> {
        let params = ContextParams {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embeddings: self.embeddings,
            flash_attn_type: self.flash_attn_type,
            offload_kqv: self.offload_kqv,
            type_k: self.type_k,
            type_v: self.type_v,
            ..Default::default()
        };

        Context::new(self.model, params)
    }

    /// Build the context asynchronously
    ///
    /// # Returns
    ///
    /// An `AsyncContext` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if context creation fails
    #[cfg(feature = "async")]
    pub async fn build_async(self) -> Result<crate::async_support::AsyncContext, MullamaError> {
        use tokio::task;

        let params = ContextParams {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embeddings: self.embeddings,
            flash_attn_type: self.flash_attn_type,
            offload_kqv: self.offload_kqv,
            type_k: self.type_k,
            type_v: self.type_v,
            ..Default::default()
        };

        let model = self.model.clone();
        let context = task::spawn_blocking(move || Context::new(model.clone(), params))
            .await
            .map_err(|e| MullamaError::ContextError(format!("Async task failed: {}", e)))?;

        match context {
            Ok(ctx) => Ok(crate::async_support::AsyncContext::new(ctx, self.model)),
            Err(e) => Err(e),
        }
    }
}
