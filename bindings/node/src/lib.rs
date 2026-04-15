//! Node.js bindings for Mullama LLM library
//!
//! This module provides napi-rs based Node.js bindings for the Mullama library,
//! enabling high-performance LLM inference from JavaScript/TypeScript.

mod context;
mod embeddings;
mod model;
mod sampler;

use napi::bindgen_prelude::*;
use napi_derive::napi;

pub use context::{JsContext, JsContextParams, StreamResult};
pub use embeddings::JsEmbeddingGenerator;
pub use model::{JsModel, JsModelParams};
pub use sampler::{
    sampler_params_creative, sampler_params_greedy, sampler_params_precise, JsSamplerParams,
};

pub(crate) fn napi_error(prefix: &str, err: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("{}: {}", prefix, err))
}

/// Compute cosine similarity between two vectors
#[napi]
pub fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::from_reason("Vectors must have the same length"));
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm = norm_a.sqrt() * norm_b.sqrt();
    if norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / norm)
    }
}

/// Initialize the mullama backend
#[napi]
pub fn backend_init() {
    mullama::backend_init();
}

/// Free the mullama backend resources
#[napi]
pub fn backend_free() {
    mullama::backend_free();
}

/// Check if GPU offloading is supported
#[napi]
pub fn supports_gpu_offload() -> bool {
    mullama::supports_gpu_offload()
}

/// Get system information
#[napi]
pub fn system_info() -> String {
    mullama::print_system_info()
}

/// Get the maximum number of supported devices
#[napi]
pub fn max_devices() -> u32 {
    mullama::max_devices() as u32
}

/// Get the library version
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Hardware preset information
#[napi(object)]
#[derive(Clone)]
pub struct JsHardwarePresetInfo {
    /// Human-readable name
    pub name: String,
    /// Short description
    pub description: String,
    /// Recommended quantization format
    pub recommended_quant: String,
    /// Recommended GPU layers (-1 = all)
    pub gpu_layers: i32,
    /// Recommended context size
    pub context_size: u32,
    /// Whether flash attention is enabled
    pub flash_attn: bool,
}

fn preset_to_info(p: &mullama::presets::HardwarePreset) -> JsHardwarePresetInfo {
    JsHardwarePresetInfo {
        name: p.name().to_string(),
        description: p.description().to_string(),
        recommended_quant: p.recommended_quant().to_string(),
        gpu_layers: p.model_params().n_gpu_layers,
        context_size: p.context_params().n_ctx,
        flash_attn: p.flash_attn(),
    }
}

/// Get all available hardware presets
#[napi]
pub fn get_hardware_presets() -> Vec<JsHardwarePresetInfo> {
    mullama::presets::HardwarePreset::all()
        .iter()
        .map(|p| preset_to_info(p))
        .collect()
}

/// Detect the best hardware preset for the current system
#[napi]
pub fn detect_hardware_preset() -> JsHardwarePresetInfo {
    let p = mullama::presets::HardwarePreset::detect();
    preset_to_info(&p)
}

/// Get a hardware preset by name (e.g., "cpu", "gpu", "apple-silicon", "max", "auto")
///
/// Returns null if the name is not recognized.
#[napi]
pub fn get_hardware_preset_by_name(name: String) -> Option<JsHardwarePresetInfo> {
    mullama::presets::HardwarePreset::from_name(&name).map(|p| preset_to_info(&p))
}
