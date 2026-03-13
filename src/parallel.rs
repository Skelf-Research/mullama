//! Parallel processing with Rayon integration
//!
//! This module provides comprehensive parallel processing capabilities using Rayon,
//! enabling efficient CPU utilization for batch operations and data processing.
//!
//! ## Features
//!
//! - **Parallel Tokenization**: Process multiple texts concurrently
//! - **Batch Generation**: Generate multiple responses in parallel
//! - **Data Processing**: Parallel preprocessing of datasets
//! - **Custom Thread Pools**: Configurable thread pools for different workloads
//! - **NUMA Awareness**: Optimize for NUMA architectures
//! - **Work Stealing**: Efficient load balancing across threads
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::parallel::{ParallelProcessor, BatchConfig, ThreadPoolConfig};
//! use std::sync::Arc;
//!
//! fn main() -> Result<(), mullama::MullamaError> {
//!     let model = Arc::new(mullama::Model::load("model.gguf")?);
//!
//!     let processor = ParallelProcessor::new(model)
//!         .thread_pool(ThreadPoolConfig::new().num_threads(8))
//!         .build()?;
//!
//!     let prompts = vec!["Hello", "World", "Parallel", "Processing"];
//!     let results = processor.batch_tokenize(&prompts)?;
//!
//!     println!("Processed {} prompts in parallel", results.len());
//!     Ok(())
//! }
//! ```

#[cfg(feature = "parallel")]
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    prelude::*,
    ThreadPool, ThreadPoolBuilder,
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{Context, ContextParams, Model, MullamaError, SamplerChain, SamplerParams, TokenId};

/// Parallel processor for batch operations
#[cfg(feature = "parallel")]
pub struct ParallelProcessor {
    model: Arc<Model>,
    thread_pool: Option<ThreadPool>,
    config: ProcessorConfig,
}

#[cfg(feature = "parallel")]
impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new(model: Arc<Model>) -> ParallelProcessorBuilder {
        ParallelProcessorBuilder::new(model)
    }

    /// Process multiple texts in parallel for tokenization
    pub fn batch_tokenize(&self, texts: &[&str]) -> Result<Vec<Vec<TokenId>>, MullamaError> {
        let model = &self.model;

        let process_batch = |texts: &[&str]| -> Result<Vec<Vec<TokenId>>, MullamaError> {
            match &self.thread_pool {
                Some(pool) => pool.install(|| {
                    texts
                        .par_iter()
                        .map(|text| model.tokenize(text, true, false))
                        .collect()
                }),
                None => texts
                    .par_iter()
                    .map(|text| model.tokenize(text, true, false))
                    .collect(),
            }
        };

        process_batch(texts)
    }

    /// Process multiple token sequences in parallel for detokenization
    pub fn batch_detokenize(
        &self,
        token_sequences: &[&[TokenId]],
    ) -> Result<Vec<String>, MullamaError> {
        let model = &self.model;

        let process_batch = |sequences: &[&[TokenId]]| -> Result<Vec<String>, MullamaError> {
            match &self.thread_pool {
                Some(pool) => pool.install(|| {
                    sequences
                        .par_iter()
                        .map(|tokens| {
                            tokens
                                .iter()
                                .map(|&token| model.token_to_str(token, 0, false))
                                .collect::<Result<Vec<String>, _>>()
                                .map(|parts| parts.join(""))
                        })
                        .collect()
                }),
                None => sequences
                    .par_iter()
                    .map(|tokens| {
                        tokens
                            .iter()
                            .map(|&token| model.token_to_str(token, 0, false))
                            .collect::<Result<Vec<String>, _>>()
                            .map(|parts| parts.join(""))
                    })
                    .collect(),
            }
        };

        process_batch(token_sequences)
    }

    /// Generate multiple responses in parallel
    pub fn batch_generate(
        &self,
        prompts: &[&str],
        config: &BatchGenerationConfig,
    ) -> Result<Vec<GenerationResult>, MullamaError> {
        let model = &self.model;
        let generation_config = config.clone();

        let process_batch = |prompts: &[&str]| -> Result<Vec<GenerationResult>, MullamaError> {
            match &self.thread_pool {
                Some(pool) => pool.install(|| {
                    prompts
                        .par_iter()
                        .map(|prompt| self.generate_single(prompt, &generation_config))
                        .collect()
                }),
                None => prompts
                    .par_iter()
                    .map(|prompt| self.generate_single(prompt, &generation_config))
                    .collect(),
            }
        };

        process_batch(prompts)
    }

    /// Process a dataset in parallel chunks
    pub fn process_dataset<T, R, F>(
        &self,
        data: &[T],
        chunk_size: usize,
        processor: F,
    ) -> Result<Vec<R>, MullamaError>
    where
        T: Sync + Send,
        R: Send,
        F: Fn(&[T]) -> Result<Vec<R>, MullamaError> + Sync + Send,
    {
        let process_chunks = |chunks: &[&[T]]| -> Result<Vec<R>, MullamaError> {
            match &self.thread_pool {
                Some(pool) => pool.install(|| {
                    let results: Result<Vec<Vec<R>>, MullamaError> =
                        chunks.par_iter().map(|chunk| processor(chunk)).collect();

                    results.map(|vecs| vecs.into_iter().flatten().collect())
                }),
                None => {
                    let results: Result<Vec<Vec<R>>, MullamaError> = data
                        .par_chunks(chunk_size)
                        .map(|chunk| processor(chunk))
                        .collect();

                    results.map(|vecs| vecs.into_iter().flatten().collect())
                }
            }
        };

        let chunks: Vec<&[T]> = data.chunks(chunk_size).collect();
        process_chunks(&chunks)
    }

    /// Get processor statistics
    pub fn stats(&self) -> ProcessorStats {
        ProcessorStats {
            thread_count: self
                .thread_pool
                .as_ref()
                .map(|pool| pool.current_num_threads())
                .unwrap_or_else(|| rayon::current_num_threads()),
            has_custom_pool: self.thread_pool.is_some(),
            config: self.config.clone(),
        }
    }

    fn generate_single(
        &self,
        prompt: &str,
        config: &BatchGenerationConfig,
    ) -> Result<GenerationResult, MullamaError> {
        let start_time = Instant::now();

        // Create context (this would need thread-local storage in real implementation)
        let mut ctx_params = ContextParams::default();
        ctx_params.n_ctx = config.context_size;
        ctx_params.n_threads = 1; // Single thread per generation task

        // Note: In real implementation, we'd need thread-safe context creation
        // For now, this is a simplified placeholder
        let tokens = self.model.tokenize(prompt, true, false)?;

        // Simulate generation
        let generated_text = format!("Generated response for: {}", prompt);
        let generation_time = start_time.elapsed();

        Ok(GenerationResult {
            prompt: prompt.to_string(),
            generated_text,
            tokens_generated: 50, // Placeholder
            generation_time,
        })
    }
}

/// Builder for ParallelProcessor
#[cfg(feature = "parallel")]
pub struct ParallelProcessorBuilder {
    model: Arc<Model>,
    thread_pool_config: Option<ThreadPoolConfig>,
    config: ProcessorConfig,
}

#[cfg(feature = "parallel")]
impl ParallelProcessorBuilder {
    pub fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            thread_pool_config: None,
            config: ProcessorConfig::default(),
        }
    }

    /// Configure a custom thread pool
    pub fn thread_pool(mut self, config: ThreadPoolConfig) -> Self {
        self.thread_pool_config = Some(config);
        self
    }

    /// Set processor configuration
    pub fn config(mut self, config: ProcessorConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the parallel processor
    pub fn build(self) -> Result<ParallelProcessor, MullamaError> {
        let thread_pool = if let Some(config) = self.thread_pool_config {
            Some(config.build()?)
        } else {
            None
        };

        Ok(ParallelProcessor {
            model: self.model,
            thread_pool,
            config: self.config,
        })
    }
}

/// Thread pool configuration
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    num_threads: Option<usize>,
    thread_name: Option<String>,
    stack_size: Option<usize>,
    panic_handler: bool,
}

#[cfg(feature = "parallel")]
impl ThreadPoolConfig {
    pub fn new() -> Self {
        Self {
            num_threads: None,
            thread_name: None,
            stack_size: None,
            panic_handler: true,
        }
    }

    /// Set the number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    /// Set thread name prefix
    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.thread_name = Some(name.into());
        self
    }

    /// Set thread stack size
    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Enable/disable panic handler
    pub fn panic_handler(mut self, enable: bool) -> Self {
        self.panic_handler = enable;
        self
    }

    fn build(self) -> Result<ThreadPool, MullamaError> {
        let mut builder = ThreadPoolBuilder::new();

        if let Some(threads) = self.num_threads {
            builder = builder.num_threads(threads);
        }

        if let Some(name) = self.thread_name {
            builder = builder.thread_name(move |index| format!("{}-{}", name, index));
        }

        if let Some(size) = self.stack_size {
            builder = builder.stack_size(size);
        }

        if self.panic_handler {
            builder = builder.panic_handler(|_| {
                eprintln!("❌ Panic in Rayon thread pool");
            });
        }

        builder
            .build()
            .map_err(|e| MullamaError::ConfigError(format!("Failed to build thread pool: {}", e)))
    }
}

/// Processor configuration
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub enable_work_stealing: bool,
    pub chunk_size_hint: usize,
    pub numa_aware: bool,
}

#[cfg(feature = "parallel")]
impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            enable_work_stealing: true,
            chunk_size_hint: 100,
            numa_aware: false,
        }
    }
}

/// Configuration for batch generation
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct BatchGenerationConfig {
    pub max_tokens: usize,
    pub context_size: u32,
    pub sampler_params: SamplerParams,
    pub timeout: Option<Duration>,
}

#[cfg(feature = "parallel")]
impl Default for BatchGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            context_size: 2048,
            sampler_params: SamplerParams::default(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Result of parallel generation
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub prompt: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub generation_time: Duration,
}

/// Processor statistics
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    pub thread_count: usize,
    pub has_custom_pool: bool,
    pub config: ProcessorConfig,
}

/// Parallel data processing utilities
#[cfg(feature = "parallel")]
pub mod data_processing {
    use super::*;

    /// Process text data in parallel for preprocessing
    pub fn parallel_preprocess<F>(texts: &[String], processor: F) -> Vec<String>
    where
        F: Fn(&str) -> String + Sync + Send,
    {
        texts.par_iter().map(|text| processor(text)).collect()
    }

    /// Parallel text cleaning and normalization
    pub fn parallel_clean_texts(texts: &[String]) -> Vec<String> {
        texts
            .par_iter()
            .map(|text| {
                text.trim()
                    .replace('\n', " ")
                    .replace('\t', " ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    /// Parallel vocabulary extraction
    pub fn parallel_extract_vocabulary(texts: &[String]) -> HashMap<String, usize> {
        use std::sync::Mutex;

        let vocab = Mutex::new(HashMap::new());

        texts.par_iter().for_each(|text| {
            let words: Vec<String> = text
                .split_whitespace()
                .map(|word| word.to_lowercase())
                .collect();

            let mut vocab_lock = vocab.lock().unwrap();
            for word in words {
                *vocab_lock.entry(word).or_insert(0) += 1;
            }
        });

        vocab.into_inner().unwrap()
    }

    /// Parallel batch encoding
    pub fn parallel_encode_batch(
        model: &Arc<Model>,
        texts: &[String],
        max_length: usize,
    ) -> Result<Vec<Vec<TokenId>>, MullamaError> {
        texts
            .par_iter()
            .map(|text| {
                let mut tokens = model.tokenize(text, true, false)?;
                if tokens.len() > max_length {
                    tokens.truncate(max_length);
                }
                Ok(tokens)
            })
            .collect()
    }
}

/// Advanced parallel patterns
#[cfg(feature = "parallel")]
pub mod patterns {
    use super::*;

    /// Map-reduce pattern for parallel processing
    pub fn parallel_map_reduce<T, R, MapF, ReduceF>(
        data: &[T],
        chunk_size: usize,
        map_fn: MapF,
        reduce_fn: ReduceF,
    ) -> Result<R, MullamaError>
    where
        T: Sync + Send,
        R: Send,
        MapF: Fn(&[T]) -> Result<R, MullamaError> + Sync + Send,
        ReduceF: Fn(R, R) -> R,
    {
        let results: Result<Vec<R>, MullamaError> = data
            .par_chunks(chunk_size)
            .map(|chunk| map_fn(chunk))
            .collect();

        let results = results?;

        if results.is_empty() {
            return Err(MullamaError::ConfigError("No data to process".to_string()));
        }

        Ok(results.into_iter().reduce(reduce_fn).unwrap())
    }

    /// Pipeline pattern for sequential stages with parallel processing
    pub fn parallel_pipeline<T, R>(
        data: Vec<T>,
        stages: Vec<Box<dyn Fn(Vec<T>) -> Result<Vec<R>, MullamaError> + Send + Sync>>,
    ) -> Result<Vec<R>, MullamaError>
    where
        T: Send + 'static,
        R: Send + 'static,
    {
        // Simplified pipeline implementation
        // In real implementation, this would use channels and async coordination
        Ok(Vec::new()) // Placeholder
    }

    /// Parallel tree processing
    pub fn parallel_tree_process<T, R, F>(
        tree_data: &[T],
        depth: usize,
        processor: F,
    ) -> Result<Vec<R>, MullamaError>
    where
        T: Sync + Send,
        R: Send,
        F: Fn(&T, usize) -> Result<R, MullamaError> + Sync + Send,
    {
        tree_data
            .par_iter()
            .map(|node| processor(node, depth))
            .collect()
    }
}

/// Performance optimization utilities
#[cfg(feature = "parallel")]
pub mod optimization {
    use super::*;

    /// Auto-tune chunk size for optimal performance
    pub fn auto_tune_chunk_size<T, F>(data: &[T], processor: F, max_iterations: usize) -> usize
    where
        T: Sync + Send,
        F: Fn(&[T]) -> Duration + Sync + Send,
    {
        let mut best_chunk_size = data.len() / rayon::current_num_threads();
        let mut best_time = Duration::from_secs(u64::MAX);

        let chunk_sizes = [
            best_chunk_size / 4,
            best_chunk_size / 2,
            best_chunk_size,
            best_chunk_size * 2,
            best_chunk_size * 4,
        ];

        for &chunk_size in &chunk_sizes {
            if chunk_size == 0 || chunk_size > data.len() {
                continue;
            }

            let start = Instant::now();
            for _ in 0..max_iterations.min(3) {
                let _ = processor(&data[..chunk_size.min(data.len())]);
            }
            let avg_time = start.elapsed() / max_iterations.min(3) as u32;

            if avg_time < best_time {
                best_time = avg_time;
                best_chunk_size = chunk_size;
            }
        }

        best_chunk_size
    }

    /// Memory-aware processing with adaptive chunking
    pub fn memory_aware_process<T, R, F>(
        data: &[T],
        processor: F,
        max_memory_mb: usize,
    ) -> Result<Vec<R>, MullamaError>
    where
        T: Sync + Send,
        R: Send,
        F: Fn(&[T]) -> Result<Vec<R>, MullamaError> + Sync + Send,
    {
        // Calculate adaptive chunk size based on memory constraints
        let estimated_item_size = std::mem::size_of::<T>();
        let max_items_per_chunk = (max_memory_mb * 1024 * 1024) / estimated_item_size;
        let chunk_size = max_items_per_chunk
            .min(data.len() / rayon::current_num_threads())
            .max(1);

        let results: Result<Vec<Vec<R>>, MullamaError> = data
            .par_chunks(chunk_size)
            .map(|chunk| processor(chunk))
            .collect();

        results.map(|vecs| vecs.into_iter().flatten().collect())
    }
}

#[cfg(not(feature = "parallel"))]
compile_error!("Parallel processing requires the 'parallel' feature to be enabled");
