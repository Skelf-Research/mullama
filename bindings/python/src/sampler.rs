use mullama::SamplerParams;
use pyo3::prelude::*;

/// Sampler parameters for text generation.
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
    pub penalize_nl: bool,
    pub ignore_eos: bool,
    pub seed: u32,
}

#[pymethods]
impl PySamplerParams {
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
        penalize_nl=true,
        ignore_eos=false,
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
        penalize_nl: bool,
        ignore_eos: bool,
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
            penalize_nl,
            ignore_eos,
            seed,
        }
    }

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
            penalize_nl: true,
            ignore_eos: false,
            seed: 0,
        }
    }

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
            penalize_nl: true,
            ignore_eos: false,
            seed: 0,
        }
    }

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
            penalize_nl: true,
            ignore_eos: false,
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
            penalize_nl: p.penalize_nl,
            ignore_eos: p.ignore_eos,
            seed: p.seed,
        }
    }
}
