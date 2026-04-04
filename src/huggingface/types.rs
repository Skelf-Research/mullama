//! Type definitions for Hugging Face Hub integration

use super::HF_MODELS_BASE;

/// GGUF quantization types commonly available
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantizationType {
    /// Full precision (F32)
    F32,
    /// Half precision (F16)
    F16,
    /// Brain float 16
    BF16,
    /// 8-bit quantization
    Q8_0,
    /// 6-bit quantization
    Q6_K,
    /// 5-bit quantization (medium)
    Q5_K_M,
    /// 5-bit quantization (small)
    Q5_K_S,
    /// 5-bit quantization
    Q5_0,
    /// 5-bit quantization variant 1
    Q5_1,
    /// 4-bit quantization (medium)
    Q4_K_M,
    /// 4-bit quantization (small)
    Q4_K_S,
    /// 4-bit quantization
    Q4_0,
    /// 4-bit quantization variant 1
    Q4_1,
    /// 3-bit quantization
    Q3_K_M,
    /// 3-bit quantization (small)
    Q3_K_S,
    /// 3-bit quantization (large)
    Q3_K_L,
    /// 2-bit quantization
    Q2_K,
    /// IQ quantization variants
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_XS,
    IQ3_S,
    IQ4_XS,
    IQ4_NL,
    /// Unknown/other quantization
    Other(String),
}

impl QuantizationType {
    /// Parse quantization type from filename
    pub fn from_filename(filename: &str) -> Self {
        let lower = filename.to_lowercase();

        // Check for specific quantization patterns
        // Note: bf16 must be checked before f16 since "bf16" contains "f16"
        if lower.contains("f32") {
            return Self::F32;
        }
        if lower.contains("bf16") {
            return Self::BF16;
        }
        if lower.contains("f16") {
            return Self::F16;
        }
        if lower.contains("q8_0") {
            return Self::Q8_0;
        }
        if lower.contains("q6_k") {
            return Self::Q6_K;
        }
        if lower.contains("q5_k_m") {
            return Self::Q5_K_M;
        }
        if lower.contains("q5_k_s") {
            return Self::Q5_K_S;
        }
        if lower.contains("q5_0") {
            return Self::Q5_0;
        }
        if lower.contains("q5_1") {
            return Self::Q5_1;
        }
        if lower.contains("q4_k_m") {
            return Self::Q4_K_M;
        }
        if lower.contains("q4_k_s") {
            return Self::Q4_K_S;
        }
        if lower.contains("q4_0") {
            return Self::Q4_0;
        }
        if lower.contains("q4_1") {
            return Self::Q4_1;
        }
        if lower.contains("q3_k_m") {
            return Self::Q3_K_M;
        }
        if lower.contains("q3_k_s") {
            return Self::Q3_K_S;
        }
        if lower.contains("q3_k_l") {
            return Self::Q3_K_L;
        }
        if lower.contains("q2_k") {
            return Self::Q2_K;
        }
        if lower.contains("iq2_xxs") {
            return Self::IQ2_XXS;
        }
        if lower.contains("iq2_xs") {
            return Self::IQ2_XS;
        }
        if lower.contains("iq2_s") {
            return Self::IQ2_S;
        }
        if lower.contains("iq3_xxs") {
            return Self::IQ3_XXS;
        }
        if lower.contains("iq3_xs") {
            return Self::IQ3_XS;
        }
        if lower.contains("iq3_s") {
            return Self::IQ3_S;
        }
        if lower.contains("iq4_xs") {
            return Self::IQ4_XS;
        }
        if lower.contains("iq4_nl") {
            return Self::IQ4_NL;
        }

        Self::Other(filename.to_string())
    }

    /// Get approximate bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q6_K => 6.5,
            Self::Q5_K_M | Self::Q5_K_S | Self::Q5_0 | Self::Q5_1 => 5.5,
            Self::Q4_K_M | Self::Q4_K_S | Self::Q4_0 | Self::Q4_1 => 4.5,
            Self::Q3_K_M | Self::Q3_K_S | Self::Q3_K_L => 3.5,
            Self::Q2_K => 2.5,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => 2.5,
            Self::IQ3_XXS | Self::IQ3_XS | Self::IQ3_S => 3.5,
            Self::IQ4_XS | Self::IQ4_NL => 4.5,
            Self::Other(_) => 4.0, // Assume 4-bit as default
        }
    }

    /// Get quality rating (1-10)
    pub fn quality_rating(&self) -> u8 {
        match self {
            Self::F32 => 10,
            Self::F16 | Self::BF16 => 10,
            Self::Q8_0 => 9,
            Self::Q6_K => 8,
            Self::Q5_K_M => 7,
            Self::Q5_K_S | Self::Q5_0 | Self::Q5_1 => 7,
            Self::Q4_K_M => 6,
            Self::Q4_K_S | Self::Q4_0 | Self::Q4_1 => 5,
            Self::Q3_K_M | Self::Q3_K_L => 4,
            Self::Q3_K_S => 3,
            Self::Q2_K => 2,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => 2,
            Self::IQ3_XXS | Self::IQ3_XS | Self::IQ3_S => 4,
            Self::IQ4_XS | Self::IQ4_NL => 5,
            Self::Other(_) => 5,
        }
    }
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q6_K => write!(f, "Q6_K"),
            Self::Q5_K_M => write!(f, "Q5_K_M"),
            Self::Q5_K_S => write!(f, "Q5_K_S"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q4_K_M => write!(f, "Q4_K_M"),
            Self::Q4_K_S => write!(f, "Q4_K_S"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q3_K_M => write!(f, "Q3_K_M"),
            Self::Q3_K_S => write!(f, "Q3_K_S"),
            Self::Q3_K_L => write!(f, "Q3_K_L"),
            Self::Q2_K => write!(f, "Q2_K"),
            Self::IQ2_XXS => write!(f, "IQ2_XXS"),
            Self::IQ2_XS => write!(f, "IQ2_XS"),
            Self::IQ2_S => write!(f, "IQ2_S"),
            Self::IQ3_XXS => write!(f, "IQ3_XXS"),
            Self::IQ3_XS => write!(f, "IQ3_XS"),
            Self::IQ3_S => write!(f, "IQ3_S"),
            Self::IQ4_XS => write!(f, "IQ4_XS"),
            Self::IQ4_NL => write!(f, "IQ4_NL"),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Information about a GGUF file available for download
#[derive(Debug, Clone)]
pub struct GGUFFile {
    /// Filename
    pub filename: String,
    /// File size in bytes
    pub size: u64,
    /// Quantization type
    pub quantization: QuantizationType,
    /// Download URL
    pub download_url: String,
    /// SHA256 hash if available
    pub sha256: Option<String>,
}

impl GGUFFile {
    /// Get human-readable file size
    pub fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.size >= GB {
            format!("{:.2} GB", self.size as f64 / GB as f64)
        } else if self.size >= MB {
            format!("{:.2} MB", self.size as f64 / MB as f64)
        } else if self.size >= KB {
            format!("{:.2} KB", self.size as f64 / KB as f64)
        } else {
            format!("{} bytes", self.size)
        }
    }

    /// Estimate VRAM required to load this model (rough estimate)
    pub fn estimated_vram_mb(&self) -> u64 {
        // GGUF files are already quantized, VRAM usage is approximately file size + overhead
        (self.size / (1024 * 1024)) + 512 // Add 512MB overhead
    }
}

/// Information about a model on Hugging Face Hub
#[derive(Debug, Clone)]
pub struct HFModelInfo {
    /// Model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
    pub model_id: String,
    /// Author/organization
    pub author: String,
    /// Model name
    pub name: String,
    /// Description/model card excerpt
    pub description: Option<String>,
    /// Number of downloads
    pub downloads: u64,
    /// Number of likes
    pub likes: u64,
    /// Tags
    pub tags: Vec<String>,
    /// Last modified date
    pub last_modified: Option<String>,
    /// Available GGUF files
    pub gguf_files: Vec<GGUFFile>,
    /// Pipeline tag (e.g., "text-generation")
    pub pipeline_tag: Option<String>,
    /// License
    pub license: Option<String>,
}

impl HFModelInfo {
    /// Get the model URL on Hugging Face
    pub fn url(&self) -> String {
        format!("{}/{}", HF_MODELS_BASE, self.model_id)
    }

    /// Check if this is a GGUF model repository
    pub fn is_gguf(&self) -> bool {
        self.tags.iter().any(|t| t.to_lowercase() == "gguf")
            || self.model_id.to_lowercase().contains("gguf")
            || !self.gguf_files.is_empty()
    }

    /// Get the best quantization for a given VRAM budget (in MB)
    pub fn best_quantization_for_vram(&self, vram_mb: u64) -> Option<&GGUFFile> {
        let mut suitable: Vec<&GGUFFile> = self
            .gguf_files
            .iter()
            .filter(|f| f.estimated_vram_mb() <= vram_mb)
            .collect();

        // Sort by quality rating (descending)
        suitable.sort_by(|a, b| {
            b.quantization
                .quality_rating()
                .cmp(&a.quantization.quality_rating())
        });

        suitable.first().copied()
    }

    /// Get the smallest available quantization
    pub fn smallest_quantization(&self) -> Option<&GGUFFile> {
        self.gguf_files.iter().min_by_key(|f| f.size)
    }

    /// Get the highest quality quantization
    pub fn highest_quality(&self) -> Option<&GGUFFile> {
        self.gguf_files
            .iter()
            .max_by_key(|f| f.quantization.quality_rating())
    }
}

/// Search filters for finding models
#[derive(Debug, Clone, Default)]
pub struct ModelSearchFilters {
    /// Search query
    pub query: Option<String>,
    /// Filter by author
    pub author: Option<String>,
    /// Filter by tags
    pub tags: Vec<String>,
    /// Only GGUF models
    pub gguf_only: bool,
    /// Minimum downloads
    pub min_downloads: Option<u64>,
    /// Sort by (downloads, likes, lastModified)
    pub sort: Option<String>,
    /// Limit results
    pub limit: Option<usize>,
}

impl ModelSearchFilters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_query(mut self, query: &str) -> Self {
        self.query = Some(query.to_string());
        self
    }

    pub fn with_author(mut self, author: &str) -> Self {
        self.author = Some(author.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn gguf_only(mut self) -> Self {
        self.gguf_only = true;
        self.tags.push("gguf".to_string());
        self
    }

    pub fn with_min_downloads(mut self, min: u64) -> Self {
        self.min_downloads = Some(min);
        self
    }

    pub fn sort_by_downloads(mut self) -> Self {
        self.sort = Some("downloads".to_string());
        self
    }

    pub fn sort_by_likes(mut self) -> Self {
        self.sort = Some("likes".to_string());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Progress callback for downloads
pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Bytes downloaded so far
    pub downloaded: u64,
    /// Total bytes to download
    pub total: u64,
    /// Download speed in bytes per second
    pub speed_bps: u64,
    /// Estimated time remaining in seconds
    pub eta_seconds: u64,
    /// Current filename being downloaded
    pub filename: String,
}

impl DownloadProgress {
    /// Get progress as percentage (0-100)
    pub fn percentage(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.downloaded as f64 / self.total as f64 * 100.0) as f32
        }
    }

    /// Get human-readable speed
    pub fn speed_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;

        if self.speed_bps >= MB {
            format!("{:.2} MB/s", self.speed_bps as f64 / MB as f64)
        } else if self.speed_bps >= KB {
            format!("{:.2} KB/s", self.speed_bps as f64 / KB as f64)
        } else {
            format!("{} B/s", self.speed_bps)
        }
    }

    /// Get human-readable ETA
    pub fn eta_human(&self) -> String {
        if self.eta_seconds >= 3600 {
            format!(
                "{}h {}m",
                self.eta_seconds / 3600,
                (self.eta_seconds % 3600) / 60
            )
        } else if self.eta_seconds >= 60 {
            format!("{}m {}s", self.eta_seconds / 60, self.eta_seconds % 60)
        } else {
            format!("{}s", self.eta_seconds)
        }
    }
}

/// Model test result
#[derive(Debug, Clone)]
pub struct ModelTestResult {
    /// Whether the model loaded successfully
    pub load_success: bool,
    /// Model load time in milliseconds
    pub load_time_ms: u64,
    /// Whether tokenization works
    pub tokenization_works: bool,
    /// Whether generation works
    pub generation_works: bool,
    /// Sample generated text (if generation works)
    pub sample_output: Option<String>,
    /// Model parameters detected
    pub n_params: u64,
    /// Context size
    pub n_ctx: u32,
    /// Embedding dimension
    pub n_embd: u32,
    /// Number of layers
    pub n_layers: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Error message if any test failed
    pub error: Option<String>,
}
