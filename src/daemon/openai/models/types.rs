use serde::{Deserialize, Serialize};

/// Request to pull a model
#[derive(Debug, Deserialize)]
pub(in crate::daemon::openai) struct PullModelRequest {
    /// Model name or HuggingFace spec
    pub name: String,
}

/// Request to load a model into the daemon
#[derive(Debug, Deserialize)]
pub(in crate::daemon::openai) struct LoadModelRequest {
    pub name: String,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub use_mmap: Option<bool>,
    #[serde(default)]
    pub use_mlock: bool,
    #[serde(default)]
    pub rope_freq_base: Option<f32>,
    #[serde(default)]
    pub rope_freq_scale: Option<f32>,
    #[serde(default)]
    pub n_batch: Option<u32>,
    #[serde(default)]
    pub defrag_thold: Option<f32>,
    #[serde(default)]
    pub split_mode: Option<String>,
}

/// Response for model operations
#[derive(Debug, Serialize)]
pub(in crate::daemon::openai) struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<serde_json::Value>,
}

/// Detailed model information
#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub(in crate::daemon::openai) struct ModelDetails {
    pub name: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub size: u64,
    pub size_formatted: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded: Option<String>,
}
