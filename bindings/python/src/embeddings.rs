use mullama::{Context, ContextParams};
use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

use crate::model::PyModel;
use crate::to_py_err;

/// Embedding generator for creating text embeddings
#[pyclass(name = "EmbeddingGenerator")]
pub struct PyEmbeddingGenerator {
    context: Context,
    model: Arc<mullama::Model>,
    normalize: bool,
}

#[pymethods]
impl PyEmbeddingGenerator {
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

        Ok(Self {
            context,
            model: model_arc,
            normalize,
        })
    }

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
                        for value in &mut vec {
                            *value /= norm;
                        }
                    }
                }

                Ok(PyArray1::from_vec_bound(py, vec))
            }
            None => Err(PyRuntimeError::new_err("No embeddings available")),
        }
    }

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
                            for value in &mut vec {
                                *value /= norm;
                            }
                        }
                    }

                    embeddings.push(PyArray1::from_vec_bound(py, vec).to_object(py));
                }
                None => return Err(PyRuntimeError::new_err("No embeddings available")),
            }
        }

        Ok(PyList::new_bound(py, embeddings).unbind())
    }

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
