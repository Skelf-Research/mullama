//! Python bindings for Mullama LLM library
//!
//! This module provides PyO3-based Python bindings for the Mullama library,
//! enabling high-performance LLM inference from Python.

use mullama::{Context, ContextParams, Model, ModelParams, SamplerParams};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

/// Convert a MullamaError to a PyErr
fn to_py_err(e: mullama::MullamaError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

/// Model class for loading and managing LLM models
#[pyclass(name = "Model")]
pub struct PyModel {
    inner: Arc<Model>,
}

#[pymethods]
impl PyModel {
    /// Load a model from a GGUF file
    ///
    /// Args:
    ///     path: Path to the GGUF model file
    ///     n_gpu_layers: Number of layers to offload to GPU (0 = CPU only, -1 = all)
    ///     use_mmap: Use memory mapping for model loading
    ///     use_mlock: Lock model in memory
    ///     vocab_only: Only load vocabulary (for tokenization only)
    ///
    /// Returns:
    ///     Model: Loaded model instance
    ///
    /// Raises:
    ///     RuntimeError: If model loading fails
    #[staticmethod]
    #[pyo3(signature = (path, n_gpu_layers=0, use_mmap=true, use_mlock=false, vocab_only=false))]
    fn load(
        path: &str,
        n_gpu_layers: i32,
        use_mmap: bool,
        use_mlock: bool,
        vocab_only: bool,
    ) -> PyResult<Self> {
        let params = ModelParams {
            n_gpu_layers,
            use_mmap,
            use_mlock,
            vocab_only,
            ..Default::default()
        };

        let model = Model::load_with_params(path, params).map_err(to_py_err)?;
        Ok(PyModel {
            inner: Arc::new(model),
        })
    }

    /// Tokenize text into token IDs
    ///
    /// Args:
    ///     text: Text to tokenize
    ///     add_bos: Whether to add beginning-of-sequence token
    ///     special: Whether to parse special tokens
    ///
    /// Returns:
    ///     list[int]: List of token IDs
    #[pyo3(signature = (text, add_bos=true, special=false))]
    fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> PyResult<Vec<i32>> {
        self.inner
            .tokenize(text, add_bos, special)
            .map_err(to_py_err)
    }

    /// Detokenize token IDs back to text
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///     remove_special: Remove special tokens from output
    ///     unparse_special: Include special token text in output
    ///
    /// Returns:
    ///     str: Decoded text
    #[pyo3(signature = (tokens, remove_special=false, unparse_special=false))]
    fn detokenize(
        &self,
        tokens: Vec<i32>,
        remove_special: bool,
        unparse_special: bool,
    ) -> PyResult<String> {
        self.inner
            .detokenize(&tokens, remove_special, unparse_special)
            .map_err(to_py_err)
    }

    /// Get the model's training context size
    #[getter]
    fn n_ctx_train(&self) -> i32 {
        self.inner.n_ctx_train()
    }

    /// Get the model's embedding dimension
    #[getter]
    fn n_embd(&self) -> i32 {
        self.inner.n_embd()
    }

    /// Get the vocabulary size
    #[getter]
    fn n_vocab(&self) -> i32 {
        self.inner.vocab_size()
    }

    /// Get the number of layers
    #[getter]
    fn n_layer(&self) -> i32 {
        self.inner.n_layer()
    }

    /// Get the number of attention heads
    #[getter]
    fn n_head(&self) -> i32 {
        self.inner.n_head()
    }

    /// Get the BOS (beginning of sequence) token ID
    #[getter]
    fn token_bos(&self) -> i32 {
        self.inner.token_bos()
    }

    /// Get the EOS (end of sequence) token ID
    #[getter]
    fn token_eos(&self) -> i32 {
        self.inner.token_eos()
    }

    /// Get the model size in bytes
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }

    /// Get the number of parameters
    #[getter]
    fn n_params(&self) -> u64 {
        self.inner.n_params()
    }

    /// Get the model description
    #[getter]
    fn description(&self) -> String {
        self.inner.desc()
    }

    /// Get the model architecture
    #[getter]
    fn architecture(&self) -> Option<String> {
        self.inner.architecture()
    }

    /// Get the model name from metadata
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// Check if a token is end-of-generation
    fn token_is_eog(&self, token: i32) -> bool {
        self.inner.token_is_eog(token)
    }

    /// Get all metadata as a dictionary
    fn metadata(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            for (key, value) in self.inner.metadata() {
                dict.set_item(key, value)?;
            }
            Ok(dict.unbind())
        })
    }

    /// Apply chat template to format messages
    ///
    /// Args:
    ///     messages: List of (role, content) tuples
    ///     add_generation_prompt: Whether to add generation prompt
    ///
    /// Returns:
    ///     str: Formatted prompt
    #[pyo3(signature = (messages, add_generation_prompt=true))]
    fn apply_chat_template(
        &self,
        messages: Vec<(String, String)>,
        add_generation_prompt: bool,
    ) -> PyResult<String> {
        let msg_refs: Vec<(&str, &str)> = messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        self.inner
            .apply_chat_template(None, &msg_refs, add_generation_prompt)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(name={:?}, arch={:?}, params={}, size={}MB)",
            self.inner.name(),
            self.inner.architecture(),
            self.inner.n_params(),
            self.inner.size() / (1024 * 1024)
        )
    }
}

/// Sampler parameters for text generation
#[pyclass(name = "SamplerParams")]
#[derive(Clone)]
pub struct PySamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalty_last_n: i32,
    pub seed: u32,
}

#[pymethods]
impl PySamplerParams {
    /// Create new sampler parameters
    ///
    /// Args:
    ///     temperature: Randomness (0.0 = deterministic, higher = more random)
    ///     top_k: Top-k sampling (0 = disabled)
    ///     top_p: Top-p/nucleus sampling (1.0 = disabled)
    ///     min_p: Min-p sampling (0.0 = disabled)
    ///     typical_p: Typical sampling (1.0 = disabled)
    ///     penalty_repeat: Repeat penalty (1.0 = disabled)
    ///     penalty_freq: Frequency penalty (0.0 = disabled)
    ///     penalty_present: Presence penalty (0.0 = disabled)
    ///     penalty_last_n: Tokens to consider for penalties
    ///     seed: Random seed (0 = random)
    #[new]
    #[pyo3(signature = (
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        typical_p=1.0,
        penalty_repeat=1.1,
        penalty_freq=0.0,
        penalty_present=0.0,
        penalty_last_n=64,
        seed=0
    ))]
    fn new(
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        typical_p: f32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
        penalty_last_n: i32,
        seed: u32,
    ) -> Self {
        PySamplerParams {
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            penalty_last_n,
            seed,
        }
    }

    /// Create greedy (deterministic) sampler params
    #[staticmethod]
    fn greedy() -> Self {
        PySamplerParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 0,
            seed: 0,
        }
    }

    /// Create creative (high randomness) sampler params
    #[staticmethod]
    fn creative() -> Self {
        PySamplerParams {
            temperature: 1.2,
            top_k: 100,
            top_p: 0.95,
            min_p: 0.02,
            typical_p: 1.0,
            penalty_repeat: 1.15,
            penalty_freq: 0.1,
            penalty_present: 0.1,
            penalty_last_n: 128,
            seed: 0,
        }
    }

    /// Create precise (low randomness) sampler params
    #[staticmethod]
    fn precise() -> Self {
        PySamplerParams {
            temperature: 0.3,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.1,
            typical_p: 1.0,
            penalty_repeat: 1.05,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 32,
            seed: 0,
        }
    }

    #[getter]
    fn get_temperature(&self) -> f32 {
        self.temperature
    }

    #[setter]
    fn set_temperature(&mut self, value: f32) {
        self.temperature = value;
    }

    #[getter]
    fn get_top_k(&self) -> i32 {
        self.top_k
    }

    #[setter]
    fn set_top_k(&mut self, value: i32) {
        self.top_k = value;
    }

    #[getter]
    fn get_top_p(&self) -> f32 {
        self.top_p
    }

    #[setter]
    fn set_top_p(&mut self, value: f32) {
        self.top_p = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplerParams(temperature={}, top_k={}, top_p={})",
            self.temperature, self.top_k, self.top_p
        )
    }
}

impl From<&PySamplerParams> for SamplerParams {
    fn from(p: &PySamplerParams) -> Self {
        SamplerParams {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            typical_p: p.typical_p,
            penalty_repeat: p.penalty_repeat,
            penalty_freq: p.penalty_freq,
            penalty_present: p.penalty_present,
            penalty_last_n: p.penalty_last_n,
            seed: p.seed,
            ..Default::default()
        }
    }
}

/// Context for model inference
#[pyclass(name = "Context")]
pub struct PyContext {
    inner: Context,
    model: Arc<Model>,
}

#[pymethods]
impl PyContext {
    /// Create a new context from a model
    ///
    /// Args:
    ///     model: The model to create context for
    ///     n_ctx: Context size (0 = use model default)
    ///     n_batch: Batch size for prompt processing
    ///     n_threads: Number of threads (0 = auto)
    ///     embeddings: Enable embeddings mode
    ///
    /// Returns:
    ///     Context: New context instance
    #[new]
    #[pyo3(signature = (model, n_ctx=0, n_batch=2048, n_threads=0, embeddings=false))]
    fn new(
        model: &PyModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        embeddings: bool,
    ) -> PyResult<Self> {
        let n_threads = if n_threads <= 0 {
            num_cpus::get() as i32
        } else {
            n_threads
        };

        let params = ContextParams {
            n_ctx,
            n_batch,
            n_threads,
            n_threads_batch: n_threads,
            embeddings,
            ..Default::default()
        };

        let model_arc = model.inner.clone();
        let context = Context::new(Arc::new((*model_arc).clone()), params).map_err(to_py_err)?;

        Ok(PyContext {
            inner: context,
            model: model_arc,
        })
    }

    /// Generate text from a prompt
    ///
    /// Args:
    ///     prompt: Text prompt or list of token IDs
    ///     max_tokens: Maximum tokens to generate
    ///     params: Optional sampler parameters
    ///
    /// Returns:
    ///     str: Generated text
    #[pyo3(signature = (prompt, max_tokens=100, params=None))]
    fn generate(
        &mut self,
        prompt: &Bound<'_, PyAny>,
        max_tokens: usize,
        params: Option<&PySamplerParams>,
    ) -> PyResult<String> {
        // Handle either string or list of tokens
        let tokens: Vec<i32> = if let Ok(text) = prompt.extract::<String>() {
            self.model.tokenize(&text, true, false).map_err(to_py_err)?
        } else if let Ok(token_list) = prompt.extract::<Vec<i32>>() {
            token_list
        } else {
            return Err(PyValueError::new_err(
                "prompt must be a string or list of token IDs",
            ));
        };

        let sampler_params = params.map(SamplerParams::from).unwrap_or_default();

        self.inner
            .generate_with_params(&tokens, max_tokens, &sampler_params)
            .map_err(to_py_err)
    }

    /// Generate text with streaming (returns a generator)
    ///
    /// Args:
    ///     prompt: Text prompt or list of token IDs
    ///     max_tokens: Maximum tokens to generate
    ///     params: Optional sampler parameters
    ///
    /// Yields:
    ///     str: Generated tokens one at a time
    #[pyo3(signature = (prompt, max_tokens=100, params=None))]
    fn generate_stream(
        &mut self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        max_tokens: usize,
        params: Option<PySamplerParams>,
    ) -> PyResult<Py<PyList>> {
        // For simplicity, we'll collect all tokens and return as a list
        // A proper generator implementation would require more complex PyO3 patterns
        let tokens: Vec<i32> = if let Ok(text) = prompt.extract::<String>() {
            self.model.tokenize(&text, true, false).map_err(to_py_err)?
        } else if let Ok(token_list) = prompt.extract::<Vec<i32>>() {
            token_list
        } else {
            return Err(PyValueError::new_err(
                "prompt must be a string or list of token IDs",
            ));
        };

        let sampler_params = params.as_ref().map(SamplerParams::from).unwrap_or_default();

        let mut pieces: Vec<String> = Vec::new();

        self.inner
            .generate_streaming(&tokens, max_tokens, &sampler_params, |piece| {
                pieces.push(piece.to_string());
                true
            })
            .map_err(to_py_err)?;

        let list = PyList::new_bound(py, pieces);
        Ok(list.unbind())
    }

    /// Decode tokens (process through the model)
    ///
    /// Args:
    ///     tokens: List of token IDs to decode
    fn decode(&mut self, tokens: Vec<i32>) -> PyResult<()> {
        self.inner.decode(&tokens).map_err(to_py_err)
    }

    /// Clear the KV cache
    fn clear_cache(&mut self) {
        self.inner.kv_cache_clear();
    }

    /// Get the context size
    #[getter]
    fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    /// Get the batch size
    #[getter]
    fn n_batch(&self) -> u32 {
        self.inner.n_batch()
    }

    /// Get embeddings (if embeddings mode is enabled)
    fn get_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        match self.inner.get_embeddings() {
            Some(embeddings) => {
                let array = PyArray1::from_slice_bound(py, embeddings);
                Ok(Some(array))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Context(n_ctx={}, n_batch={})",
            self.n_ctx(),
            self.n_batch()
        )
    }
}

/// Embedding generator for creating text embeddings
#[pyclass(name = "EmbeddingGenerator")]
pub struct PyEmbeddingGenerator {
    context: Context,
    model: Arc<Model>,
    normalize: bool,
}

#[pymethods]
impl PyEmbeddingGenerator {
    /// Create a new embedding generator
    ///
    /// Args:
    ///     model: The model to use for embeddings
    ///     n_ctx: Context size (0 = model default)
    ///     normalize: Whether to normalize embeddings
    #[new]
    #[pyo3(signature = (model, n_ctx=512, normalize=true))]
    fn new(model: &PyModel, n_ctx: u32, normalize: bool) -> PyResult<Self> {
        let params = ContextParams {
            n_ctx,
            embeddings: true,
            pooling_type: mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            ..Default::default()
        };

        let model_arc = model.inner.clone();
        let context = Context::new(Arc::new((*model_arc).clone()), params).map_err(to_py_err)?;

        Ok(PyEmbeddingGenerator {
            context,
            model: model_arc,
            normalize,
        })
    }

    /// Generate embeddings for text
    ///
    /// Args:
    ///     text: Text to embed
    ///
    /// Returns:
    ///     numpy.ndarray: Embedding vector
    fn embed<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let tokens = self.model.tokenize(text, true, false).map_err(to_py_err)?;

        self.context.kv_cache_clear();
        self.context.decode(&tokens).map_err(to_py_err)?;

        match self.context.get_embeddings() {
            Some(embeddings) => {
                let mut vec: Vec<f32> = embeddings.to_vec();

                if self.normalize {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in vec.iter_mut() {
                            *v /= norm;
                        }
                    }
                }

                Ok(PyArray1::from_vec_bound(py, vec))
            }
            None => Err(PyRuntimeError::new_err("No embeddings available")),
        }
    }

    /// Generate embeddings for multiple texts
    ///
    /// Args:
    ///     texts: List of texts to embed
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of embedding vectors
    fn embed_batch<'py>(&mut self, py: Python<'py>, texts: Vec<String>) -> PyResult<Py<PyList>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let tokens = self.model.tokenize(&text, true, false).map_err(to_py_err)?;

            self.context.kv_cache_clear();
            self.context.decode(&tokens).map_err(to_py_err)?;

            match self.context.get_embeddings() {
                Some(emb) => {
                    let mut vec: Vec<f32> = emb.to_vec();

                    if self.normalize {
                        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for v in vec.iter_mut() {
                                *v /= norm;
                            }
                        }
                    }

                    embeddings.push(PyArray1::from_vec_bound(py, vec).to_object(py));
                }
                None => return Err(PyRuntimeError::new_err("No embeddings available")),
            }
        }

        let list = PyList::new_bound(py, embeddings);
        Ok(list.unbind())
    }

    /// Get the embedding dimension
    #[getter]
    fn n_embd(&self) -> i32 {
        self.model.n_embd()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingGenerator(n_embd={}, normalize={})",
            self.n_embd(),
            self.normalize
        )
    }
}

/// Compute cosine similarity between two vectors
#[pyfunction]
fn cosine_similarity(a: &Bound<'_, PyArray1<f32>>, b: &Bound<'_, PyArray1<f32>>) -> PyResult<f32> {
    let a_slice = unsafe { a.as_slice()? };
    let b_slice = unsafe { b.as_slice()? };

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("Vectors must have the same length"));
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a_slice.len() {
        dot += a_slice[i] * b_slice[i];
        norm_a += a_slice[i] * a_slice[i];
        norm_b += b_slice[i] * b_slice[i];
    }

    let norm = norm_a.sqrt() * norm_b.sqrt();
    if norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / norm)
    }
}

/// Initialize the mullama backend
#[pyfunction]
fn backend_init() {
    mullama::backend_init();
}

/// Free the mullama backend resources
#[pyfunction]
fn backend_free() {
    mullama::backend_free();
}

/// Check if GPU offloading is supported
#[pyfunction]
fn supports_gpu_offload() -> bool {
    mullama::supports_gpu_offload()
}

/// Get system information
#[pyfunction]
fn system_info() -> String {
    mullama::print_system_info()
}

/// Get the maximum number of supported devices
#[pyfunction]
fn max_devices() -> usize {
    mullama::max_devices()
}

/// Python module definition
#[pymodule]
fn _mullama(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyContext>()?;
    m.add_class::<PySamplerParams>()?;
    m.add_class::<PyEmbeddingGenerator>()?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(backend_init, m)?)?;
    m.add_function(wrap_pyfunction!(backend_free, m)?)?;
    m.add_function(wrap_pyfunction!(supports_gpu_offload, m)?)?;
    m.add_function(wrap_pyfunction!(system_info, m)?)?;
    m.add_function(wrap_pyfunction!(max_devices, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
