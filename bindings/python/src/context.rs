use mullama::{Context, ContextParams, SamplerParams};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::model::PyModel;
use crate::sampler::PySamplerParams;
use crate::to_py_err;

/// Context for model inference
#[pyclass(name = "Context")]
pub struct PyContext {
    inner: Context,
    model: Arc<mullama::Model>,
}

#[pymethods]
impl PyContext {
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

        Ok(Self {
            inner: context,
            model: model_arc,
        })
    }

    #[pyo3(signature = (prompt, max_tokens=100, params=None))]
    fn generate(
        &mut self,
        prompt: &Bound<'_, PyAny>,
        max_tokens: usize,
        params: Option<&PySamplerParams>,
    ) -> PyResult<String> {
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

    #[pyo3(signature = (prompt, max_tokens=100, params=None))]
    fn generate_stream(
        &mut self,
        prompt: &Bound<'_, PyAny>,
        max_tokens: usize,
        params: Option<PySamplerParams>,
    ) -> PyResult<PyTokenIterator> {
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

        Ok(PyTokenIterator {
            tokens: Mutex::new(pieces),
            index: Mutex::new(0),
        })
    }

    fn decode(&mut self, tokens: Vec<i32>) -> PyResult<()> {
        self.inner.decode(&tokens).map_err(to_py_err)
    }

    fn clear_cache(&mut self) {
        self.inner.kv_cache_clear();
    }

    #[getter]
    fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    #[getter]
    fn n_batch(&self) -> u32 {
        self.inner.n_batch()
    }

    fn get_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        match self.inner.get_embeddings() {
            Some(embeddings) => Ok(Some(PyArray1::from_slice_bound(py, embeddings))),
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

/// Token iterator for streaming generation (implements __iter__/__next__)
#[pyclass(name = "TokenIterator")]
pub struct PyTokenIterator {
    tokens: Mutex<Vec<String>>,
    index: Mutex<usize>,
}

#[pymethods]
impl PyTokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<String> {
        let tokens = self.tokens.lock().unwrap();
        let mut index = self.index.lock().unwrap();
        if *index < tokens.len() {
            let token = tokens[*index].clone();
            *index += 1;
            Some(token)
        } else {
            None
        }
    }

    fn __len__(&self) -> usize {
        self.tokens.lock().unwrap().len()
    }

    fn collect(&self) -> Vec<String> {
        self.tokens.lock().unwrap().clone()
    }

    fn text(&self) -> String {
        self.tokens.lock().unwrap().join("")
    }
}
