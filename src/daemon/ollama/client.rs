use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    LAYER_LICENSE, LAYER_MESSAGES, LAYER_MODEL, LAYER_PARAMS, LAYER_PROJECTOR, LAYER_SYSTEM,
    LAYER_TEMPLATE, OLLAMA_CACHE_DIR, OLLAMA_REGISTRY_URL,
};
use crate::daemon::ollama::{
    OllamaManifest, OllamaMessage, OllamaModel, OllamaModelIndex, OllamaModelRef,
    OllamaParameters,
};
use crate::MullamaError;

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
                    .expect("static progress bar template")
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
                    crate::daemon::protocol::format_size(layer.size)
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
                crate::daemon::protocol::format_size(total_size)
            );
        }

        Ok(model)
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

/// Check if a name looks like an Ollama model reference (convenience function)
pub fn is_ollama_ref(name: &str) -> bool {
    OllamaClient::is_ollama_ref(name)
}

impl crate::daemon::provider::ModelProvider for OllamaClient {
    fn supports(&self, spec: &str) -> bool {
        Self::is_ollama_ref(spec)
    }

    fn resolve(
        &self,
        spec: &str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<crate::daemon::provider::ResolvedModelPath, MullamaError>,
                > + Send
                + '_,
        >,
    > {
        let spec = spec.to_string();
        Box::pin(async move {
            let model_ref = OllamaModelRef::parse(&spec);
            let alias = format!("{}-{}", model_ref.repository, model_ref.tag);

            // Check cache first
            if let Some(model) = self.get_cached(&spec) {
                return Ok(crate::daemon::provider::ResolvedModelPath {
                    path: model.gguf_path,
                    alias,
                    was_cached: true,
                });
            }

            // Download from registry
            let model = self.pull(&spec, false).await?;
            Ok(crate::daemon::provider::ResolvedModelPath {
                path: model.gguf_path,
                alias,
                was_cached: false,
            })
        })
    }

    fn is_cached(&self, spec: &str) -> bool {
        self.is_cached(spec)
    }

    fn name(&self) -> &str {
        "ollama"
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
