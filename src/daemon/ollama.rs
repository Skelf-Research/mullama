//! Ollama Registry Client
//!
//! Downloads and caches models from Ollama's registry (registry.ollama.ai).
//! Provides full compatibility with Ollama's model naming and configuration.
//!
//! ## Model Specification Formats
//!
//! ```text
//! # Official library models
//! llama3              -> registry.ollama.ai/library/llama3:latest
//! llama3:1b           -> registry.ollama.ai/library/llama3:1b
//! llama3:70b-instruct -> registry.ollama.ai/library/llama3:70b-instruct
//!
//! # User models
//! user/mymodel:v1     -> registry.ollama.ai/user/mymodel:v1
//!
//! # Explicit ollama: prefix
//! ollama:llama3:1b    -> registry.ollama.ai/library/llama3:1b
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ollama_template::ChatTemplate;
use crate::MullamaError;

/// Ollama registry URL
const OLLAMA_REGISTRY_URL: &str = "https://registry.ollama.ai";

/// Cache directory for Ollama models
const OLLAMA_CACHE_DIR: &str = "ollama";

/// Layer media types
pub const LAYER_MODEL: &str = "application/vnd.ollama.image.model";
pub const LAYER_TEMPLATE: &str = "application/vnd.ollama.image.template";
pub const LAYER_PARAMS: &str = "application/vnd.ollama.image.params";
pub const LAYER_SYSTEM: &str = "application/vnd.ollama.image.system";
pub const LAYER_PROJECTOR: &str = "application/vnd.ollama.image.projector";
pub const LAYER_LICENSE: &str = "application/vnd.ollama.image.license";
pub const LAYER_MESSAGES: &str = "application/vnd.ollama.image.messages";

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
        let (namespace, repository) = if name_part.contains('/') {
            let (ns, repo) = name_part.split_once('/').unwrap();
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

/// Ollama registry client
pub struct OllamaClient {
    client: Client,
    registry_url: String,
    storage_dir: PathBuf,
}

impl OllamaClient {
    /// Create a new Ollama client with default settings
    pub fn new() -> Result<Self, MullamaError> {
        let storage_dir = Self::default_storage_dir()?;
        Self::with_storage_dir(storage_dir)
    }

    /// Create a new Ollama client with custom storage directory
    pub fn with_storage_dir(storage_dir: PathBuf) -> Result<Self, MullamaError> {
        let client = Client::builder()
            .user_agent("mullama/1.0")
            .build()
            .map_err(|e| MullamaError::OllamaError(e.to_string()))?;

        // Create storage directories
        let manifests_dir = storage_dir.join("manifests");
        let blobs_dir = storage_dir.join("blobs");
        fs::create_dir_all(&manifests_dir).map_err(|e| {
            MullamaError::OllamaError(format!("Failed to create manifests dir: {}", e))
        })?;
        fs::create_dir_all(&blobs_dir)
            .map_err(|e| MullamaError::OllamaError(format!("Failed to create blobs dir: {}", e)))?;

        Ok(Self {
            client,
            registry_url: OLLAMA_REGISTRY_URL.to_string(),
            storage_dir,
        })
    }

    /// Get the default storage directory
    fn default_storage_dir() -> Result<PathBuf, MullamaError> {
        // Check environment variable first
        if let Ok(dir) = std::env::var("MULLAMA_CACHE_DIR") {
            return Ok(PathBuf::from(dir).join(OLLAMA_CACHE_DIR));
        }

        // Platform-specific cache directory
        #[cfg(target_os = "linux")]
        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".cache")
            })
            .join("mullama")
            .join(OLLAMA_CACHE_DIR);

        #[cfg(target_os = "macos")]
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Library/Caches/mullama")
            .join(OLLAMA_CACHE_DIR);

        #[cfg(target_os = "windows")]
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mullama")
            .join(OLLAMA_CACHE_DIR);

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mullama")
            .join(OLLAMA_CACHE_DIR);

        Ok(cache_dir)
    }

    /// Get the path to the model index file
    fn index_path(&self) -> PathBuf {
        self.storage_dir.join("models.json")
    }

    /// Get the path for a manifest
    fn manifest_path(&self, model_ref: &OllamaModelRef) -> PathBuf {
        self.storage_dir
            .join("manifests")
            .join("registry.ollama.ai")
            .join(&model_ref.namespace)
            .join(&model_ref.repository)
            .join(&model_ref.tag)
    }

    /// Get the path for a blob
    fn blob_path(&self, digest: &str) -> PathBuf {
        // Convert sha256:abc123 to sha256-abc123 for filesystem
        let filename = digest.replace(':', "-");
        self.storage_dir.join("blobs").join(filename)
    }

    /// Check if a model is cached
    pub fn is_cached(&self, name: &str) -> bool {
        let model_ref = OllamaModelRef::parse(name);
        let index_path = self.index_path();

        if let Ok(index) = OllamaModelIndex::load(&index_path) {
            let key = format!("{}:{}", model_ref.repository, model_ref.tag);
            if let Some(model) = index.get(&key) {
                // Verify the GGUF file exists
                return model.gguf_path.exists();
            }
        }
        false
    }

    /// Get cached model info
    pub fn get_cached(&self, name: &str) -> Option<OllamaModel> {
        let model_ref = OllamaModelRef::parse(name);
        let index_path = self.index_path();

        OllamaModelIndex::load(&index_path).ok().and_then(|index| {
            let key = format!("{}:{}", model_ref.repository, model_ref.tag);
            index.get(&key).cloned()
        })
    }

    /// List all cached Ollama models
    pub fn list_cached(&self) -> Vec<OllamaModel> {
        let index_path = self.index_path();
        OllamaModelIndex::load(&index_path)
            .map(|index| index.list().into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Fetch manifest from the registry
    pub async fn fetch_manifest(
        &self,
        model_ref: &OllamaModelRef,
    ) -> Result<OllamaManifest, MullamaError> {
        let url = format!(
            "{}/v2/{}/{}/manifests/{}",
            self.registry_url, model_ref.namespace, model_ref.repository, model_ref.tag
        );

        let response = self
            .client
            .get(&url)
            .header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            .send()
            .await
            .map_err(|e| MullamaError::OllamaError(format!("Failed to fetch manifest: {}", e)))?;

        if !response.status().is_success() {
            return Err(MullamaError::OllamaError(format!(
                "Registry returned {}: {}",
                response.status(),
                model_ref.display_name()
            )));
        }

        let manifest: OllamaManifest = response
            .json()
            .await
            .map_err(|e| MullamaError::OllamaError(format!("Failed to parse manifest: {}", e)))?;

        Ok(manifest)
    }

    /// Download a blob from the registry
    pub async fn download_blob(
        &self,
        model_ref: &OllamaModelRef,
        digest: &str,
        size: u64,
        show_progress: bool,
    ) -> Result<PathBuf, MullamaError> {
        let blob_path = self.blob_path(digest);

        // Check if already downloaded
        if blob_path.exists() {
            if let Ok(metadata) = fs::metadata(&blob_path) {
                if metadata.len() == size {
                    return Ok(blob_path);
                }
            }
        }

        let url = format!(
            "{}/v2/{}/{}/blobs/{}",
            self.registry_url, model_ref.namespace, model_ref.repository, digest
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| MullamaError::OllamaError(format!("Failed to fetch blob: {}", e)))?;

        if !response.status().is_success() {
            return Err(MullamaError::OllamaError(format!(
                "Failed to download blob {}: {}",
                digest,
                response.status()
            )));
        }

        // Create parent directory
        if let Some(parent) = blob_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                MullamaError::OllamaError(format!("Failed to create blob directory: {}", e))
            })?;
        }

        // Download to temp file
        let temp_path = blob_path.with_extension("part");
        let mut file = File::create(&temp_path)
            .map_err(|e| MullamaError::OllamaError(format!("Failed to create temp file: {}", e)))?;

        let total_size = response.content_length().unwrap_or(size);

        // Setup progress bar
        let progress = if show_progress {
            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Download with streaming and hash verification
        let mut hasher = Sha256::new();
        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| MullamaError::OllamaError(format!("Download error: {}", e)))?;

            file.write_all(&chunk)
                .map_err(|e| MullamaError::OllamaError(format!("Write error: {}", e)))?;

            hasher.update(&chunk);
            downloaded += chunk.len() as u64;

            if let Some(ref pb) = progress {
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = progress {
            pb.finish_with_message("Downloaded");
        }

        // Verify digest
        let computed_hash = format!("sha256:{:x}", hasher.finalize());
        if computed_hash != digest {
            fs::remove_file(&temp_path).ok();
            return Err(MullamaError::OllamaError(format!(
                "Digest mismatch: expected {}, got {}",
                digest, computed_hash
            )));
        }

        // Atomic rename
        fs::rename(&temp_path, &blob_path)
            .map_err(|e| MullamaError::OllamaError(format!("Failed to rename blob: {}", e)))?;

        Ok(blob_path)
    }

    /// Read a text blob (for template, system, params, etc.)
    fn read_text_blob(&self, digest: &str) -> Option<String> {
        let blob_path = self.blob_path(digest);
        fs::read_to_string(&blob_path).ok()
    }

    /// Read a JSON blob
    fn read_json_blob<T: for<'de> Deserialize<'de>>(&self, digest: &str) -> Option<T> {
        let content = self.read_text_blob(digest)?;
        serde_json::from_str(&content).ok()
    }

    /// Pull a complete model from the registry
    pub async fn pull(&self, name: &str, show_progress: bool) -> Result<OllamaModel, MullamaError> {
        let model_ref = OllamaModelRef::parse(name);

        if show_progress {
            println!("Pulling {}...", model_ref.display_name());
        }

        // Fetch manifest
        let manifest = self.fetch_manifest(&model_ref).await?;

        // Save manifest
        let manifest_path = self.manifest_path(&model_ref);
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            MullamaError::OllamaError(format!("Failed to serialize manifest: {}", e))
        })?;
        fs::write(&manifest_path, &manifest_json)
            .map_err(|e| MullamaError::OllamaError(format!("Failed to write manifest: {}", e)))?;

        // Download all layers
        let mut gguf_path: Option<PathBuf> = None;
        let mut projector_path: Option<PathBuf> = None;
        let mut template: Option<String> = None;
        let mut system_prompt: Option<String> = None;
        let mut parameters = OllamaParameters::default();
        let mut messages: Vec<OllamaMessage> = Vec::new();
        let mut license: Option<String> = None;
        let mut total_size: u64 = 0;

        // Download config blob
        if show_progress {
            println!("Downloading config...");
        }
        self.download_blob(
            &model_ref,
            &manifest.config.digest,
            manifest.config.size,
            false,
        )
        .await?;

        // Download each layer
        for (i, layer) in manifest.layers.iter().enumerate() {
            if show_progress {
                println!(
                    "Downloading layer {}/{} ({})...",
                    i + 1,
                    manifest.layers.len(),
                    Self::format_size(layer.size)
                );
            }

            let blob_path = self
                .download_blob(
                    &model_ref,
                    &layer.digest,
                    layer.size,
                    show_progress && layer.size > 1_000_000,
                )
                .await?;

            total_size += layer.size;

            // Process layer based on media type
            match layer.media_type.as_str() {
                LAYER_MODEL => {
                    gguf_path = Some(blob_path);
                }
                LAYER_PROJECTOR => {
                    projector_path = Some(blob_path);
                }
                LAYER_TEMPLATE => {
                    template = self.read_text_blob(&layer.digest);
                }
                LAYER_SYSTEM => {
                    system_prompt = self.read_text_blob(&layer.digest);
                }
                LAYER_PARAMS => {
                    if let Some(params) = self.read_json_blob::<OllamaParameters>(&layer.digest) {
                        parameters = params;
                    }
                }
                LAYER_MESSAGES => {
                    if let Some(msgs) = self.read_json_blob::<Vec<OllamaMessage>>(&layer.digest) {
                        messages = msgs;
                    }
                }
                LAYER_LICENSE => {
                    license = self.read_text_blob(&layer.digest);
                }
                _ => {
                    // Unknown layer type, skip
                }
            }
        }

        let gguf_path = gguf_path.ok_or_else(|| {
            MullamaError::OllamaError("Model manifest does not contain a GGUF layer".to_string())
        })?;

        // Create model entry
        let model = OllamaModel {
            name: model_ref.repository.clone(),
            tag: model_ref.tag.clone(),
            gguf_path,
            projector_path,
            template,
            system_prompt,
            parameters,
            messages,
            license,
            pulled_at: chrono::Utc::now().to_rfc3339(),
            total_size,
        };

        // Update index
        let mut index = OllamaModelIndex::load(&self.index_path()).unwrap_or_default();
        index.insert(model.clone());
        index.save(&self.index_path())?;

        if show_progress {
            println!(
                "Successfully pulled {} ({})",
                model_ref.display_name(),
                Self::format_size(total_size)
            );
        }

        Ok(model)
    }

    /// Format size in human-readable format
    fn format_size(bytes: u64) -> String {
        if bytes >= 1_073_741_824 {
            format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{} KB", bytes / 1024)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Check if a name looks like an Ollama model reference
    pub fn is_ollama_ref(name: &str) -> bool {
        let looks_like_windows_abs = name.len() >= 3
            && name.as_bytes()[0].is_ascii_alphabetic()
            && name.as_bytes()[1] == b':'
            && (name.as_bytes()[2] == b'\\' || name.as_bytes()[2] == b'/');

        // Explicit ollama: prefix
        if name.starts_with("ollama:") {
            return true;
        }

        // Don't match HuggingFace specs
        if name.starts_with("hf:") {
            return false;
        }

        // Don't match local paths
        if name.starts_with('/')
            || name.starts_with("./")
            || name.starts_with("../")
            || name.starts_with("~/")
            || looks_like_windows_abs
            || name.ends_with(".gguf")
            || name.contains('\\')
        {
            return false;
        }

        // Match Ollama-style names: model, model:tag, user/model, user/model:tag
        // These are typically short names without special characters
        let parts: Vec<&str> = name.split(':').collect();
        if parts.len() > 2 {
            return false;
        }

        let name_part = parts[0];

        // Check if it looks like an Ollama model name
        // - Contains at most one slash (for user/model format)
        // - No special characters except hyphen and underscore
        let slash_count = name_part.matches('/').count();
        if slash_count > 1 {
            return false;
        }

        // Should be alphanumeric with hyphens, underscores, and optional single slash
        name_part
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_name() {
        let model_ref = OllamaModelRef::parse("llama3");
        assert_eq!(model_ref.namespace, "library");
        assert_eq!(model_ref.repository, "llama3");
        assert_eq!(model_ref.tag, "latest");
    }

    #[test]
    fn test_parse_name_with_tag() {
        let model_ref = OllamaModelRef::parse("llama3:1b");
        assert_eq!(model_ref.namespace, "library");
        assert_eq!(model_ref.repository, "llama3");
        assert_eq!(model_ref.tag, "1b");
    }

    #[test]
    fn test_parse_user_model() {
        let model_ref = OllamaModelRef::parse("user/mymodel:v1");
        assert_eq!(model_ref.namespace, "user");
        assert_eq!(model_ref.repository, "mymodel");
        assert_eq!(model_ref.tag, "v1");
    }

    #[test]
    fn test_parse_with_ollama_prefix() {
        let model_ref = OllamaModelRef::parse("ollama:llama3:1b");
        assert_eq!(model_ref.namespace, "library");
        assert_eq!(model_ref.repository, "llama3");
        assert_eq!(model_ref.tag, "1b");
    }

    #[test]
    fn test_is_ollama_ref() {
        assert!(OllamaClient::is_ollama_ref("llama3"));
        assert!(OllamaClient::is_ollama_ref("llama3:1b"));
        assert!(OllamaClient::is_ollama_ref("ollama:llama3:1b"));
        assert!(OllamaClient::is_ollama_ref("user/model:tag"));

        assert!(!OllamaClient::is_ollama_ref("hf:owner/repo"));
        assert!(!OllamaClient::is_ollama_ref("/path/to/model.gguf"));
        assert!(!OllamaClient::is_ollama_ref("./model.gguf"));
        assert!(!OllamaClient::is_ollama_ref("model.gguf"));
    }

    #[test]
    fn test_display_name() {
        let model_ref = OllamaModelRef::parse("llama3:1b");
        assert_eq!(model_ref.display_name(), "llama3:1b");

        let user_ref = OllamaModelRef::parse("user/model:v1");
        assert_eq!(user_ref.display_name(), "user/model:v1");
    }
}
