//! Python bindings for Mullama LLM library
//!
//! This module provides PyO3-based Python bindings for the Mullama library,
//! enabling high-performance LLM inference from Python.

mod context;
mod embeddings;
mod model;
mod sampler;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub use context::{PyContext, PyTokenIterator};
pub use embeddings::PyEmbeddingGenerator;
pub use model::PyModel;
pub use sampler::PySamplerParams;

pub(crate) fn to_py_err(e: mullama::MullamaError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

/// Hardware preset for common deployment configurations
#[pyclass(name = "HardwarePreset")]
#[derive(Clone)]
pub struct PyHardwarePreset {
    inner: mullama::presets::HardwarePreset,
}

#[pymethods]
impl PyHardwarePreset {
    /// Create CPU low memory preset (4GB RAM)
    #[staticmethod]
    pub fn cpu_low_memory() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::CpuLowMemory,
        }
    }

    /// Create CPU standard preset (8-16GB RAM)
    #[staticmethod]
    pub fn cpu_standard() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::CpuStandard,
        }
    }

    /// Create GPU low VRAM preset (4GB)
    #[staticmethod]
    pub fn gpu_low_vram() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::GpuLowVram,
        }
    }

    /// Create GPU medium VRAM preset (8GB)
    #[staticmethod]
    pub fn gpu_medium_vram() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::GpuMediumVram,
        }
    }

    /// Create GPU high VRAM preset (16GB+)
    #[staticmethod]
    pub fn gpu_high_vram() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::GpuHighVram,
        }
    }

    /// Create Apple Silicon preset (M-series)
    #[staticmethod]
    pub fn apple_silicon() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::AppleSilicon,
        }
    }

    /// Create maximum performance preset
    #[staticmethod]
    pub fn max_performance() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::MaxPerformance,
        }
    }

    /// Auto-detect the best preset for the current hardware
    #[staticmethod]
    pub fn detect() -> Self {
        Self {
            inner: mullama::presets::HardwarePreset::detect(),
        }
    }

    /// Get a preset by name (e.g., "cpu", "gpu", "apple-silicon", "max", "auto")
    ///
    /// Returns None if name is not recognized.
    #[staticmethod]
    pub fn from_name(name: &str) -> Option<Self> {
        mullama::presets::HardwarePreset::from_name(name).map(|p| Self { inner: p })
    }

    /// Get the human-readable name of this preset
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get a short description of this preset
    pub fn description(&self) -> &str {
        self.inner.description()
    }

    /// Get the recommended quantization format (e.g., "Q4_K_M")
    pub fn recommended_quant(&self) -> &str {
        self.inner.recommended_quant()
    }

    /// Get the recommended number of GPU layers (-1 = all)
    pub fn gpu_layers(&self) -> i32 {
        self.inner.model_params().n_gpu_layers
    }

    /// Get the recommended context size
    pub fn context_size(&self) -> u32 {
        self.inner.context_params().n_ctx
    }

    /// Check if this preset enables flash attention
    pub fn flash_attn(&self) -> bool {
        self.inner.flash_attn()
    }

    fn __repr__(&self) -> String {
        format!(
            "HardwarePreset(name='{}', gpu_layers={}, context_size={})",
            self.inner.name(),
            self.inner.model_params().n_gpu_layers,
            self.inner.context_params().n_ctx,
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
    m.add_class::<PyTokenIterator>()?;
    m.add_class::<PyEmbeddingGenerator>()?;
    m.add_class::<PyHardwarePreset>()?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(backend_init, m)?)?;
    m.add_function(wrap_pyfunction!(backend_free, m)?)?;
    m.add_function(wrap_pyfunction!(supports_gpu_offload, m)?)?;
    m.add_function(wrap_pyfunction!(system_info, m)?)?;
    m.add_function(wrap_pyfunction!(max_devices, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
