//! Unified Hugging Face Hub integration.
//!
//! This module owns the public Hugging Face API for Mullama.
//! Synchronous discovery/client types are available in all builds, while the
//! async downloader and daemon provider integration are enabled with the
//! `daemon` feature.

mod client;
mod types;

#[cfg(feature = "daemon")]
use crate::error::MullamaError;

#[cfg(feature = "daemon")]
use std::path::{Path, PathBuf};

/// Hugging Face Hub API base URL.
#[cfg(feature = "daemon")]
const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_MODELS_BASE: &str = "https://huggingface.co";

pub use client::HFClient;
pub use types::{
    DownloadProgress, GGUFFile, HFModelInfo, ModelSearchFilters, ModelTestResult, ProgressCallback,
    QuantizationType,
};

/// URL encoding helper shared by the sync client surface.
#[cfg(feature = "daemon")]
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut result = String::new();
        for c in s.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                    result.push(c);
                }
                ' ' => result.push_str("%20"),
                _ => {
                    for b in c.to_string().bytes() {
                        result.push_str(&format!("%{:02X}", b));
                    }
                }
            }
        }
        result
    }
}

/// Convenience functions for common Hugging Face workflows.
#[cfg(feature = "daemon")]
pub mod quick {
    use super::*;

    /// Download the best quantization of a model for a given VRAM budget.
    #[cfg(feature = "daemon")]
    pub fn download_best_for_vram(
        model_id: &str,
        vram_mb: u64,
        download_dir: Option<&Path>,
    ) -> Result<PathBuf, MullamaError> {
        let client = if let Some(dir) = download_dir {
            HFClient::with_download_dir(dir).with_token_from_env()
        } else {
            HFClient::new().with_token_from_env()
        };

        let gguf_files = client.list_gguf_files(model_id)?;

        let best_file = gguf_files
            .iter()
            .filter(|f| f.estimated_vram_mb() <= vram_mb)
            .max_by_key(|f| f.quantization.quality_rating())
            .ok_or_else(|| {
                MullamaError::HuggingFaceError(format!(
                    "No suitable quantization found for {} MB VRAM",
                    vram_mb
                ))
            })?;

        client.download_gguf(model_id, best_file, None)
    }

    /// Download the smallest quantization of a model.
    #[cfg(feature = "daemon")]
    pub fn download_smallest(
        model_id: &str,
        download_dir: Option<&Path>,
    ) -> Result<PathBuf, MullamaError> {
        let client = if let Some(dir) = download_dir {
            HFClient::with_download_dir(dir).with_token_from_env()
        } else {
            HFClient::new().with_token_from_env()
        };

        let gguf_files = client.list_gguf_files(model_id)?;

        let smallest = gguf_files
            .iter()
            .min_by_key(|f| f.size)
            .ok_or_else(|| MullamaError::HuggingFaceError("No GGUF files found".to_string()))?;

        client.download_gguf(model_id, smallest, None)
    }

    /// Search for GGUF models by name.
    #[cfg(feature = "daemon")]
    pub fn search_gguf(query: &str, limit: usize) -> Result<Vec<HFModelInfo>, MullamaError> {
        let client = HFClient::new();
        let filters = ModelSearchFilters::new()
            .with_query(query)
            .gguf_only()
            .sort_by_downloads()
            .with_limit(limit);

        client.search_models(&filters)
    }

    /// Get popular GGUF models.
    #[cfg(feature = "daemon")]
    pub fn popular_models(limit: usize) -> Result<Vec<HFModelInfo>, MullamaError> {
        let client = HFClient::new();
        client.get_popular_gguf_models(limit)
    }

    /// Download and test a model.
    #[cfg(feature = "daemon")]
    pub fn download_and_test(
        model_id: &str,
        quantization: Option<QuantizationType>,
    ) -> Result<(PathBuf, ModelTestResult), MullamaError> {
        let client = HFClient::new().with_token_from_env();
        let gguf_files = client.list_gguf_files(model_id)?;

        let file = if let Some(quant) = quantization {
            gguf_files
                .iter()
                .find(|f| f.quantization == quant)
                .or_else(|| gguf_files.first())
        } else {
            gguf_files
                .iter()
                .find(|f| matches!(f.quantization, QuantizationType::Q4_K_M))
                .or_else(|| {
                    gguf_files
                        .iter()
                        .find(|f| matches!(f.quantization, QuantizationType::Q4_0))
                })
                .or_else(|| gguf_files.first())
        };

        let file = file.ok_or_else(|| {
            MullamaError::HuggingFaceError("No suitable GGUF file found".to_string())
        })?;

        let path = client.download_gguf(model_id, file, None)?;
        let test_result = client.test_model(&path)?;

        Ok((path, test_result))
    }
}

#[cfg(feature = "daemon")]
pub use downloader::{
    resolve_model_path, CachedModel, GgufFileInfo, HfDownloader, HfFileInfo, HfModelSpec,
    HfRepoInfo, HfSearchResult,
};

#[cfg(feature = "daemon")]
mod downloader {
    use super::MullamaError;
    use futures::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};
    use reqwest::Client;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};
    use tokio::fs::{self, File};
    use tokio::io::AsyncWriteExt;

    /// Hugging Face API base URL for daemon download workflows.
    const HF_API_URL: &str = "https://huggingface.co/api";
    const HF_CDN_URL: &str = "https://huggingface.co";
    const CACHE_DIR: &str = "mullama";
    const MODELS_SUBDIR: &str = "models";

    #[derive(Debug, Clone, Deserialize)]
    pub struct HfFileInfo {
        #[serde(rename = "rfilename")]
        pub filename: String,
        pub size: Option<u64>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct HfRepoInfo {
        pub id: String,
        pub siblings: Option<Vec<HfFileInfo>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct HfSearchResult {
        #[serde(rename = "modelId")]
        pub model_id: Option<String>,
        pub id: String,
        pub author: Option<String>,
        #[serde(rename = "lastModified")]
        pub last_modified: Option<String>,
        pub downloads: Option<u64>,
        pub likes: Option<u64>,
        pub tags: Option<Vec<String>>,
        #[serde(rename = "pipeline_tag")]
        pub pipeline_tag: Option<String>,
        pub library_name: Option<String>,
    }

    impl HfSearchResult {
        pub fn is_gguf(&self) -> bool {
            if let Some(ref tags) = self.tags {
                tags.iter().any(|t| t.to_lowercase() == "gguf")
            } else {
                self.id.to_lowercase().contains("gguf")
            }
        }

        pub fn downloads_formatted(&self) -> String {
            match self.downloads {
                Some(d) if d >= 1_000_000 => format!("{:.1}M", d as f64 / 1_000_000.0),
                Some(d) if d >= 1_000 => format!("{:.1}K", d as f64 / 1_000.0),
                Some(d) => format!("{}", d),
                None => "?".to_string(),
            }
        }

        pub fn author_name(&self) -> &str {
            self.author
                .as_deref()
                .unwrap_or_else(|| self.id.split('/').next().unwrap_or("unknown"))
        }
    }

    #[derive(Debug, Clone)]
    pub struct GgufFileInfo {
        pub filename: String,
        pub size_bytes: Option<u64>,
        pub quantization: Option<String>,
    }

    impl GgufFileInfo {
        pub fn size_formatted(&self) -> String {
            match self.size_bytes {
                Some(s) if s >= 1_073_741_824 => format!("{:.2} GB", s as f64 / 1_073_741_824.0),
                Some(s) if s >= 1_048_576 => format!("{:.1} MB", s as f64 / 1_048_576.0),
                Some(s) => format!("{} KB", s / 1024),
                None => "? GB".to_string(),
            }
        }
    }

    fn extract_quantization(filename: &str) -> Option<String> {
        let quantizations = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0",
            "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32", "IQ1_S", "IQ1_M", "IQ2_XXS",
            "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
        ];

        let upper = filename.to_uppercase();
        for q in quantizations {
            if upper.contains(q) {
                return Some(q.to_string());
            }
        }
        None
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CachedModel {
        pub repo_id: String,
        pub filename: String,
        pub local_path: PathBuf,
        pub size_bytes: u64,
        pub downloaded_at: String,
        pub etag: Option<String>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct CacheIndex {
        models: Vec<CachedModel>,
    }

    #[derive(Debug, Clone)]
    pub struct HfModelSpec {
        pub alias: Option<String>,
        pub repo_id: String,
        pub filename: Option<String>,
        pub revision: String,
    }

    impl HfModelSpec {
        pub fn parse(spec: &str) -> Option<Self> {
            let (alias, rest) = if spec.contains(":hf:") {
                let parts: Vec<&str> = spec.splitn(2, ":hf:").collect();
                (Some(parts[0].to_string()), parts.get(1).copied()?)
            } else if spec.starts_with("hf:") {
                (None, spec.strip_prefix("hf:")?)
            } else {
                return None;
            };

            let (repo_id, filename) = if let Some(pos) = rest.rfind(':') {
                let before = &rest[..pos];
                let after = &rest[pos + 1..];

                if after.contains('.') && (after.ends_with(".gguf") || after.ends_with(".bin")) {
                    (before.to_string(), Some(after.to_string()))
                } else {
                    (rest.to_string(), None)
                }
            } else {
                (rest.to_string(), None)
            };

            Some(Self {
                alias,
                repo_id,
                filename,
                revision: "main".to_string(),
            })
        }

        pub fn is_hf_spec(spec: &str) -> bool {
            spec.starts_with("hf:") || spec.contains(":hf:")
        }

        pub fn get_alias(&self) -> String {
            self.alias.clone().unwrap_or_else(|| {
                self.repo_id
                    .split('/')
                    .next_back()
                    .unwrap_or("model")
                    .to_lowercase()
                    .replace("-gguf", "")
                    .replace("_gguf", "")
            })
        }
    }

    pub struct HfDownloader {
        client: Client,
        cache_dir: PathBuf,
        hf_token: Option<String>,
    }

    impl HfDownloader {
        pub fn new() -> Result<Self, MullamaError> {
            let cache_dir = Self::default_cache_dir()?;
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to create cache dir: {}", e))
            })?;

            let hf_token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();

            Ok(Self {
                client: Client::new(),
                cache_dir,
                hf_token,
            })
        }

        pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, MullamaError> {
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to create cache dir: {}", e))
            })?;

            let hf_token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();

            Ok(Self {
                client: Client::new(),
                cache_dir,
                hf_token,
            })
        }

        pub fn default_cache_dir() -> Result<PathBuf, MullamaError> {
            if let Ok(custom_dir) = std::env::var("MULLAMA_CACHE_DIR") {
                return Ok(PathBuf::from(custom_dir));
            }
            if let Some(cache) = dirs::cache_dir() {
                return Ok(cache.join(CACHE_DIR).join(MODELS_SUBDIR));
            }
            if let Some(data_local) = dirs::data_local_dir() {
                return Ok(data_local.join(CACHE_DIR).join(MODELS_SUBDIR));
            }
            if let Some(home) = dirs::home_dir() {
                return Ok(home.join(format!(".{}", CACHE_DIR)).join(MODELS_SUBDIR));
            }
            Ok(PathBuf::from(".").join(CACHE_DIR).join(MODELS_SUBDIR))
        }

        pub fn cache_dir_info() -> String {
            let dir = Self::default_cache_dir().unwrap_or_else(|_| PathBuf::from("(unknown)"));

            #[cfg(target_os = "linux")]
            let platform = "Linux: $XDG_CACHE_HOME/mullama/models or ~/.cache/mullama/models";
            #[cfg(target_os = "macos")]
            let platform = "macOS: ~/Library/Caches/mullama/models";
            #[cfg(target_os = "windows")]
            let platform = "Windows: %LOCALAPPDATA%\\mullama\\models";
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            let platform = "Other: ~/.mullama/models";

            format!("Current: {}\nDefault for {}", dir.display(), platform)
        }

        pub fn cache_dir(&self) -> &Path {
            &self.cache_dir
        }

        fn load_index(&self) -> CacheIndex {
            let index_path = self.cache_dir.join("index.json");
            if index_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&index_path) {
                    if let Ok(index) = serde_json::from_str(&content) {
                        return index;
                    }
                }
            }
            CacheIndex::default()
        }

        fn save_index(&self, index: &CacheIndex) -> Result<(), MullamaError> {
            let index_path = self.cache_dir.join("index.json");
            let content = serde_json::to_string_pretty(index).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to serialize index: {}", e))
            })?;
            std::fs::write(&index_path, content).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to write index: {}", e))
            })?;
            Ok(())
        }

        async fn load_index_async(&self) -> CacheIndex {
            let index_path = self.cache_dir.join("index.json");
            if index_path.exists() {
                if let Ok(content) = tokio::fs::read_to_string(&index_path).await {
                    if let Ok(index) = serde_json::from_str(&content) {
                        return index;
                    }
                }
            }
            CacheIndex::default()
        }

        async fn save_index_async(&self, index: &CacheIndex) -> Result<(), MullamaError> {
            let index_path = self.cache_dir.join("index.json");
            let content = serde_json::to_string_pretty(index).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to serialize index: {}", e))
            })?;
            tokio::fs::write(&index_path, content).await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to write index: {}", e))
            })?;
            Ok(())
        }

        pub fn get_cached(&self, repo_id: &str, filename: &str) -> Option<CachedModel> {
            let index = self.load_index();
            index
                .models
                .into_iter()
                .find(|m| m.repo_id == repo_id && m.filename == filename && m.local_path.exists())
        }

        pub fn list_cached(&self) -> Vec<CachedModel> {
            let index = self.load_index();
            index
                .models
                .into_iter()
                .filter(|m| m.local_path.exists())
                .collect()
        }

        pub async fn get_repo_info(&self, repo_id: &str) -> Result<HfRepoInfo, MullamaError> {
            let url = format!("{}/models/{}", HF_API_URL, repo_id);
            let mut req = self.client.get(&url);
            if let Some(ref token) = self.hf_token {
                req = req.header("Authorization", format!("Bearer {}", token));
            }

            let resp = req.send().await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to fetch repo info: {}", e))
            })?;

            if !resp.status().is_success() {
                return Err(MullamaError::OperationFailed(format!(
                    "HF API error: {} - {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                )));
            }

            resp.json().await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to parse repo info: {}", e))
            })
        }

        pub async fn search(
            &self,
            query: &str,
            gguf_only: bool,
            limit: usize,
        ) -> Result<Vec<HfSearchResult>, MullamaError> {
            let limit = limit.clamp(1, 100);
            let mut url = format!(
                "{}/models?search={}&sort=downloads&direction=-1&limit={}",
                HF_API_URL,
                super::urlencoding::encode(query),
                if gguf_only { limit * 2 } else { limit }
            );

            if gguf_only {
                url.push_str("&filter=gguf");
            }

            let mut req = self.client.get(&url);
            if let Some(ref token) = self.hf_token {
                req = req.header("Authorization", format!("Bearer {}", token));
            }

            let resp = req
                .send()
                .await
                .map_err(|e| MullamaError::OperationFailed(format!("Search failed: {}", e)))?;

            if !resp.status().is_success() {
                return Err(MullamaError::OperationFailed(format!(
                    "HF API error: {}",
                    resp.status()
                )));
            }

            let mut results: Vec<HfSearchResult> = resp.json().await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to parse search results: {}", e))
            })?;

            if gguf_only {
                results.retain(|r| r.is_gguf());
                results.truncate(limit);
            }

            Ok(results)
        }

        pub async fn search_gguf(
            &self,
            query: &str,
            limit: usize,
        ) -> Result<Vec<HfSearchResult>, MullamaError> {
            self.search(query, true, limit).await
        }

        pub async fn list_gguf_files(
            &self,
            repo_id: &str,
        ) -> Result<Vec<GgufFileInfo>, MullamaError> {
            let info = self.get_repo_info(repo_id).await?;

            let siblings = info.siblings.ok_or_else(|| {
                MullamaError::OperationFailed("No files found in repository".into())
            })?;

            let gguf_files: Vec<GgufFileInfo> = siblings
                .into_iter()
                .filter(|f| f.filename.ends_with(".gguf"))
                .map(|f| GgufFileInfo {
                    quantization: extract_quantization(&f.filename),
                    filename: f.filename,
                    size_bytes: f.size,
                })
                .collect();

            if gguf_files.is_empty() {
                return Err(MullamaError::OperationFailed(
                    "No GGUF files found in repository".into(),
                ));
            }

            Ok(gguf_files)
        }

        pub async fn find_best_gguf(&self, repo_id: &str) -> Result<String, MullamaError> {
            let info = self.get_repo_info(repo_id).await?;

            let siblings = info.siblings.ok_or_else(|| {
                MullamaError::OperationFailed("No files found in repository".into())
            })?;

            let mut gguf_files: Vec<_> = siblings
                .into_iter()
                .filter(|f| f.filename.ends_with(".gguf"))
                .collect();

            if gguf_files.is_empty() {
                return Err(MullamaError::OperationFailed(
                    "No GGUF files found in repository".into(),
                ));
            }

            let preference_order = [
                "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q4_1", "Q8_0", "Q6_K", "Q3_K_M",
                "Q3_K_S", "Q2_K",
            ];

            gguf_files.sort_by(|a, b| {
                let a_score = preference_order
                    .iter()
                    .position(|q| a.filename.contains(q))
                    .unwrap_or(100);
                let b_score = preference_order
                    .iter()
                    .position(|q| b.filename.contains(q))
                    .unwrap_or(100);
                a_score.cmp(&b_score)
            });

            Ok(gguf_files[0].filename.clone())
        }

        pub async fn download(
            &self,
            repo_id: &str,
            filename: &str,
            show_progress: bool,
        ) -> Result<PathBuf, MullamaError> {
            if let Some(cached) = self.get_cached(repo_id, filename) {
                if show_progress {
                    println!("Using cached model: {}", cached.local_path.display());
                }
                return Ok(cached.local_path);
            }

            let repo_dir = self.cache_dir.join(repo_id.replace('/', "--"));
            fs::create_dir_all(&repo_dir).await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to create repo dir: {}", e))
            })?;

            let local_path = repo_dir.join(filename);
            let url = format!("{}/{}/resolve/main/{}", HF_CDN_URL, repo_id, filename);

            if show_progress {
                println!("Downloading {} from {}", filename, repo_id);
            }

            let mut req = self.client.get(&url);
            if let Some(ref token) = self.hf_token {
                req = req.header("Authorization", format!("Bearer {}", token));
            }

            let resp = req
                .send()
                .await
                .map_err(|e| MullamaError::OperationFailed(format!("Download failed: {}", e)))?;

            if !resp.status().is_success() {
                return Err(MullamaError::OperationFailed(format!(
                    "Download failed: {} - {}",
                    resp.status(),
                    if resp.status().as_u16() == 401 {
                        "Unauthorized. Set HF_TOKEN for gated models."
                    } else {
                        "Check repo and filename"
                    }
                )));
            }

            let total_size = resp.content_length().unwrap_or(0);
            let etag = resp
                .headers()
                .get("etag")
                .and_then(|v| v.to_str().ok())
                .map(String::from);

            let progress = if show_progress && total_size > 0 {
                let pb = ProgressBar::new(total_size);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                        .expect("static progress bar template")
                        .progress_chars("#>-"),
                );
                Some(pb)
            } else {
                None
            };

            let temp_path = local_path.with_extension("part");
            let mut file = File::create(&temp_path).await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to create file: {}", e))
            })?;

            let mut downloaded: u64 = 0;
            let mut stream = resp.bytes_stream();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk
                    .map_err(|e| MullamaError::OperationFailed(format!("Download error: {}", e)))?;

                file.write_all(&chunk)
                    .await
                    .map_err(|e| MullamaError::OperationFailed(format!("Write error: {}", e)))?;

                downloaded += chunk.len() as u64;
                if let Some(ref pb) = progress {
                    pb.set_position(downloaded);
                }
            }

            file.flush().await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to flush file: {}", e))
            })?;
            drop(file);

            if let Some(pb) = progress {
                pb.finish_with_message("Download complete");
            }

            fs::rename(&temp_path, &local_path).await.map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to finalize download: {}", e))
            })?;

            let mut index = self.load_index_async().await;
            index
                .models
                .retain(|m| !(m.repo_id == repo_id && m.filename == filename));
            index.models.push(CachedModel {
                repo_id: repo_id.to_string(),
                filename: filename.to_string(),
                local_path: local_path.clone(),
                size_bytes: downloaded,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                etag,
            });
            self.save_index_async(&index).await?;

            if show_progress {
                println!("Saved to: {}", local_path.display());
            }

            Ok(local_path)
        }

        pub async fn download_spec(
            &self,
            spec: &HfModelSpec,
            show_progress: bool,
        ) -> Result<PathBuf, MullamaError> {
            let filename = match &spec.filename {
                Some(f) => f.clone(),
                None => {
                    if show_progress {
                        println!("Finding best GGUF file in {}...", spec.repo_id);
                    }
                    self.find_best_gguf(&spec.repo_id).await?
                }
            };

            self.download(&spec.repo_id, &filename, show_progress).await
        }

        pub fn remove_cached(&self, repo_id: &str, filename: &str) -> Result<(), MullamaError> {
            let mut index = self.load_index();

            if let Some(pos) = index
                .models
                .iter()
                .position(|m| m.repo_id == repo_id && m.filename == filename)
            {
                let model = index.models.remove(pos);
                if model.local_path.exists() {
                    std::fs::remove_file(&model.local_path).map_err(|e| {
                        MullamaError::OperationFailed(format!("Failed to remove file: {}", e))
                    })?;
                }
                self.save_index(&index)?;
            }

            Ok(())
        }

        pub fn clear_cache(&self) -> Result<(), MullamaError> {
            if self.cache_dir.exists() {
                std::fs::remove_dir_all(&self.cache_dir).map_err(|e| {
                    MullamaError::OperationFailed(format!("Failed to clear cache: {}", e))
                })?;
                std::fs::create_dir_all(&self.cache_dir).map_err(|e| {
                    MullamaError::OperationFailed(format!("Failed to recreate cache dir: {}", e))
                })?;
            }
            Ok(())
        }

        pub fn cache_size(&self) -> u64 {
            self.list_cached().iter().map(|m| m.size_bytes).sum()
        }
    }

    impl Default for HfDownloader {
        fn default() -> Self {
            match Self::new() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[WARN] Failed to create HfDownloader: {}", e);
                    Self {
                        cache_dir: dirs::cache_dir()
                            .unwrap_or_else(|| PathBuf::from("/tmp"))
                            .join("mullama")
                            .join("models"),
                        client: reqwest::Client::new(),
                        hf_token: None,
                    }
                }
            }
        }
    }

    pub async fn resolve_model_path(
        spec: &str,
        show_progress: bool,
    ) -> Result<(String, PathBuf), MullamaError> {
        if let Some(hf_spec) = HfModelSpec::parse(spec) {
            let downloader = HfDownloader::new()?;
            let path = downloader.download_spec(&hf_spec, show_progress).await?;
            let alias = hf_spec.get_alias();
            Ok((alias, path))
        } else {
            let path = PathBuf::from(spec);

            let (alias, path) = if let Some(pos) = spec.find(':') {
                let alias = &spec[..pos];
                let path_str = &spec[pos + 1..];
                if alias.len() == 1 && path_str.starts_with('\\') {
                    let p = PathBuf::from(spec);
                    let alias = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "model".to_string());
                    (alias, p)
                } else {
                    (alias.to_string(), PathBuf::from(path_str))
                }
            } else {
                let alias = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                (alias, path)
            };

            if !path.exists() {
                return Err(MullamaError::OperationFailed(format!(
                    "Model file not found: {}",
                    path.display()
                )));
            }

            Ok((alias, path))
        }
    }

    impl crate::daemon::provider::ModelProvider for HfDownloader {
        fn supports(&self, spec: &str) -> bool {
            HfModelSpec::is_hf_spec(spec)
                || (spec.contains('/') && !spec.starts_with('/') && !spec.starts_with('.'))
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
                let hf_spec = HfModelSpec::parse(&spec).ok_or_else(|| {
                    MullamaError::OperationFailed(format!("Cannot parse HF spec: {}", spec))
                });

                let hf_spec = match hf_spec {
                    Ok(s) => s,
                    Err(_) => {
                        let synthetic = format!("hf:{}", spec);
                        HfModelSpec::parse(&synthetic).ok_or_else(|| {
                            MullamaError::OperationFailed(format!(
                                "Cannot parse as HuggingFace spec: {}",
                                spec
                            ))
                        })?
                    }
                };

                let alias = hf_spec.get_alias();

                if let Some(ref filename) = hf_spec.filename {
                    if let Some(cached) = self.get_cached(&hf_spec.repo_id, filename) {
                        return Ok(crate::daemon::provider::ResolvedModelPath {
                            path: cached.local_path,
                            alias,
                            was_cached: true,
                        });
                    }
                }

                let path = self.download_spec(&hf_spec, false).await?;
                Ok(crate::daemon::provider::ResolvedModelPath {
                    path,
                    alias,
                    was_cached: false,
                })
            })
        }

        fn is_cached(&self, spec: &str) -> bool {
            if let Some(hf_spec) = HfModelSpec::parse(spec) {
                let cached = self.list_cached();
                cached.iter().any(|m| m.repo_id == hf_spec.repo_id)
            } else {
                false
            }
        }

        fn name(&self) -> &str {
            "huggingface"
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_parse_hf_spec() {
            let spec = HfModelSpec::parse("hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf");
            assert!(spec.is_some());
            let spec = spec.unwrap();
            assert_eq!(spec.repo_id, "TheBloke/Llama-2-7B-GGUF");
            assert_eq!(spec.filename, Some("llama-2-7b.Q4_K_M.gguf".to_string()));
            assert!(spec.alias.is_none());

            let spec = HfModelSpec::parse("llama:hf:TheBloke/Llama-2-7B-GGUF:model.gguf");
            assert!(spec.is_some());
            let spec = spec.unwrap();
            assert_eq!(spec.alias, Some("llama".to_string()));
            assert_eq!(spec.repo_id, "TheBloke/Llama-2-7B-GGUF");

            let spec = HfModelSpec::parse("hf:TheBloke/Llama-2-7B-GGUF");
            assert!(spec.is_some());
            let spec = spec.unwrap();
            assert!(spec.filename.is_none());

            assert!(HfModelSpec::parse("./model.gguf").is_none());
            assert!(HfModelSpec::parse("model:./path.gguf").is_none());
        }

        #[test]
        fn test_is_hf_spec() {
            assert!(HfModelSpec::is_hf_spec("hf:owner/repo"));
            assert!(HfModelSpec::is_hf_spec("alias:hf:owner/repo"));
            assert!(!HfModelSpec::is_hf_spec("./local/path.gguf"));
            assert!(!HfModelSpec::is_hf_spec("alias:./local/path.gguf"));
        }

        #[test]
        fn test_get_alias() {
            let spec = HfModelSpec {
                alias: Some("custom".to_string()),
                repo_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
                filename: None,
                revision: "main".to_string(),
            };
            assert_eq!(spec.get_alias(), "custom");

            let spec = HfModelSpec {
                alias: None,
                repo_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
                filename: None,
                revision: "main".to_string(),
            };
            assert_eq!(spec.get_alias(), "llama-2-7b");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_from_filename() {
        assert_eq!(
            QuantizationType::from_filename("model-q4_k_m.gguf"),
            QuantizationType::Q4_K_M
        );
        assert_eq!(
            QuantizationType::from_filename("llama-7b-Q8_0.gguf"),
            QuantizationType::Q8_0
        );
        assert_eq!(
            QuantizationType::from_filename("model-f16.gguf"),
            QuantizationType::F16
        );
    }

    #[test]
    fn test_search_filters_builder() {
        let filters = ModelSearchFilters::new()
            .with_query("llama")
            .gguf_only()
            .sort_by_downloads()
            .with_limit(10);

        assert_eq!(filters.query, Some("llama".to_string()));
        assert!(filters.gguf_only);
        assert_eq!(filters.limit, Some(10));
    }

    #[test]
    #[cfg(feature = "daemon")]
    fn test_url_encoding() {
        assert_eq!(super::urlencoding::encode("hello world"), "hello%20world");
        assert_eq!(super::urlencoding::encode("test-123"), "test-123");
    }
}
