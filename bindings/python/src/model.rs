use mullama::{Model, ModelParams};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use crate::to_py_err;

/// Model class for loading and managing LLM models
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Arc<Model>,
}

#[pymethods]
impl PyModel {
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
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    #[pyo3(signature = (text, add_bos=true, special=false))]
    fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> PyResult<Vec<i32>> {
        self.inner
            .tokenize(text, add_bos, special)
            .map_err(to_py_err)
    }

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

    #[getter]
    fn n_ctx_train(&self) -> i32 {
        self.inner.n_ctx_train()
    }

    #[getter]
    fn n_embd(&self) -> i32 {
        self.inner.n_embd()
    }

    #[getter]
    fn n_vocab(&self) -> i32 {
        self.inner.vocab_size()
    }

    #[getter]
    fn n_layer(&self) -> i32 {
        self.inner.n_layer()
    }

    #[getter]
    fn n_head(&self) -> i32 {
        self.inner.n_head()
    }

    #[getter]
    fn token_bos(&self) -> i32 {
        self.inner.token_bos()
    }

    #[getter]
    fn token_eos(&self) -> i32 {
        self.inner.token_eos()
    }

    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }

    #[getter]
    fn n_params(&self) -> u64 {
        self.inner.n_params()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.desc()
    }

    #[getter]
    fn architecture(&self) -> Option<String> {
        self.inner.architecture()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    fn token_is_eog(&self, token: i32) -> bool {
        self.inner.token_is_eog(token)
    }

    fn metadata(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            for (key, value) in self.inner.metadata() {
                dict.set_item(key, value)?;
            }
            Ok(dict.unbind())
        })
    }

    #[pyo3(signature = (messages, add_generation_prompt=true))]
    fn apply_chat_template(
        &self,
        messages: Vec<(String, String)>,
        add_generation_prompt: bool,
    ) -> PyResult<String> {
        let msg_refs: Vec<(&str, &str)> = messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
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
