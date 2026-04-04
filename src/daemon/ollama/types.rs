use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::daemon::ollama_template::ChatTemplate;
use crate::MullamaError;

/// Ollama manifest (OCI-like format)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "mediaType")]
    pub media_type: Option<String>,
    pub config: LayerRef,
    pub layers: Vec<Layer>,
}

/// Reference to a layer/blob
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LayerRef {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

/// Layer in the manifest
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

/// Parsed Ollama model reference
#[derive(Debug, Clone)]
pub struct OllamaModelRef {
    /// Namespace (e.g., "library" for official models, username for user models)
    pub namespace: String,
    /// Repository name (e.g., "llama3")
    pub repository: String,
    /// Tag (e.g., "1b", "latest")
    pub tag: String,
}

impl OllamaModelRef {
    /// Parse a model name into components
    ///
    /// Examples:
    /// - "llama3" -> library/llama3:latest
    /// - "llama3:1b" -> library/llama3:1b
    /// - "user/model:v1" -> user/model:v1
    pub fn parse(name: &str) -> Self {
        // Strip "ollama:" prefix if present
        let name = name.strip_prefix("ollama:").unwrap_or(name);

        // Split into name and tag
        let (name_part, tag) = name.split_once(':').unwrap_or((name, "latest"));

        // Split into namespace and repository
        let (namespace, repository) = if let Some((ns, repo)) = name_part.split_once('/') {
            (ns.to_string(), repo.to_string())
        } else {
            ("library".to_string(), name_part.to_string())
        };

        Self {
            namespace,
            repository,
            tag: tag.to_string(),
        }
    }

    /// Get the full display name (e.g., "llama3:1b")
    pub fn display_name(&self) -> String {
        if self.namespace == "library" {
            format!("{}:{}", self.repository, self.tag)
        } else {
            format!("{}/{}:{}", self.namespace, self.repository, self.tag)
        }
    }

    /// Get the registry path
    pub fn registry_path(&self) -> String {
        format!("{}/{}", self.namespace, self.repository)
    }
}

/// All parameters from Ollama's configuration layers
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OllamaParameters {
    // Sampling parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    // Mirostat parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,

    // Context parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_keep: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<u32>,

    // Hardware parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mlock: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_vram: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_only: Option<bool>,

    // Deprecated but handled gracefully
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f32>,
}

impl OllamaParameters {
    /// Convert to mullama SamplerParams
    pub fn to_sampler_params(&self) -> crate::SamplerParams {
        let mut p = crate::SamplerParams::default();

        if let Some(v) = self.temperature {
            p.temperature = v;
        }
        if let Some(v) = self.top_k {
            p.top_k = v;
        }
        if let Some(v) = self.top_p {
            p.top_p = v;
        }
        if let Some(v) = self.min_p {
            p.min_p = v;
        }
        if let Some(v) = self.typical_p {
            p.typical_p = v;
        }
        if let Some(v) = self.repeat_penalty {
            p.penalty_repeat = v;
        }
        if let Some(v) = self.repeat_last_n {
            p.penalty_last_n = v;
        }
        if let Some(v) = self.frequency_penalty {
            p.penalty_freq = v;
        }
        if let Some(v) = self.presence_penalty {
            p.penalty_present = v;
        }
        if let Some(v) = self.penalize_newline {
            p.penalize_nl = v;
        }
        if let Some(v) = self.seed {
            p.seed = v as u32;
        }

        p
    }

    /// Get stop sequences
    pub fn stop_sequences(&self) -> Vec<String> {
        self.stop.clone().unwrap_or_default()
    }

    /// Get max tokens (-1 or None means unlimited)
    pub fn max_tokens(&self) -> Option<u32> {
        self.num_predict
            .and_then(|n| if n < 0 { None } else { Some(n as u32) })
    }

    /// Get context size
    pub fn context_size(&self) -> Option<u32> {
        self.num_ctx
    }

    /// Get GPU layers
    pub fn gpu_layers(&self) -> Option<i32> {
        self.num_gpu
    }

    /// Check if mirostat is enabled
    pub fn mirostat_enabled(&self) -> bool {
        matches!(self.mirostat, Some(1) | Some(2))
    }
}

/// A chat message from Ollama's messages layer
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

/// Complete Ollama model with all configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    /// Model name (e.g., "llama3:1b")
    pub name: String,
    /// Tag
    pub tag: String,

    /// Path to the GGUF model file
    pub gguf_path: PathBuf,
    /// Path to the vision projector (for multimodal models)
    pub projector_path: Option<PathBuf>,

    /// Chat template (Go template format)
    pub template: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Model parameters
    pub parameters: OllamaParameters,
    /// Pre-defined conversation messages
    pub messages: Vec<OllamaMessage>,
    /// License text
    pub license: Option<String>,

    /// When this model was pulled
    pub pulled_at: String,
    /// Total size of all blobs
    pub total_size: u64,
}

impl OllamaModel {
    /// Get stop sequences from template analysis and parameters
    ///
    /// Combines stop sequences from:
    /// 1. Explicit parameters (parameters.stop)
    /// 2. Template analysis (end-of-turn tokens from Go template)
    pub fn get_stop_sequences(&self) -> Vec<String> {
        let mut stops = Vec::new();

        // From explicit parameters
        if let Some(ref param_stops) = self.parameters.stop {
            stops.extend(param_stops.clone());
        }

        // From template analysis
        if let Some(ref template) = self.template {
            let chat_template = ChatTemplate::from_ollama_template(template);
            stops.extend(chat_template.stop_sequences);
        }

        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        stops.retain(|s| seen.insert(s.clone()));

        stops
    }
}

/// Index of pulled Ollama models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaModelIndex {
    pub models: HashMap<String, OllamaModel>,
}

impl OllamaModelIndex {
    /// Load index from file
    pub fn load(path: &Path) -> Result<Self, MullamaError> {
        if path.exists() {
            let content = fs::read_to_string(path).map_err(|e| {
                MullamaError::OllamaError(format!("Failed to read Ollama index: {}", e))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                MullamaError::OllamaError(format!("Failed to parse Ollama index: {}", e))
            })
        } else {
            Ok(Self::default())
        }
    }

    /// Save index to file
    pub fn save(&self, path: &Path) -> Result<(), MullamaError> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            MullamaError::OllamaError(format!("Failed to serialize Ollama index: {}", e))
        })?;
        fs::write(path, content)
            .map_err(|e| MullamaError::OllamaError(format!("Failed to write Ollama index: {}", e)))
    }

    /// Get a model by name
    pub fn get(&self, name: &str) -> Option<&OllamaModel> {
        self.models.get(name)
    }

    /// Insert a model
    pub fn insert(&mut self, model: OllamaModel) {
        let key = format!("{}:{}", model.name, model.tag);
        self.models.insert(key, model);
    }

    /// List all models
    pub fn list(&self) -> Vec<&OllamaModel> {
        self.models.values().collect()
    }
}
