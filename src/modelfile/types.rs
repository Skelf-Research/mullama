use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A parsed Modelfile or Mullamafile
#[derive(Debug, Clone, Default)]
pub struct Modelfile {
    /// The base model (FROM directive)
    /// Can be: local path, HF spec (hf:owner/repo:file), or model alias (llama3.2:1b)
    pub from: String,

    /// Model parameters (PARAMETER directives)
    pub parameters: HashMap<String, ParameterValue>,

    /// System prompt (SYSTEM directive)
    pub system: Option<String>,

    /// Template for formatting prompts (TEMPLATE directive)
    pub template: Option<String>,

    /// Pre-defined messages (MESSAGE directives)
    pub messages: Vec<Message>,

    /// LoRA adapter path (ADAPTER directive) - Mullama extension
    pub adapter: Option<PathBuf>,

    /// Number of GPU layers to offload (GPU_LAYERS directive) - Mullama extension
    pub gpu_layers: Option<i32>,

    /// Enable flash attention (FLASH_ATTENTION directive) - Mullama extension
    pub flash_attention: Option<bool>,

    /// Vision projector path for multimodal models (VISION_PROJECTOR directive) - Mullama extension
    pub vision_projector: Option<PathBuf>,

    /// License information (LICENSE directive) - Metadata
    pub license: Option<String>,

    /// Author information (AUTHOR directive) - Metadata
    pub author: Option<String>,

    /// The original file path this was parsed from
    pub source_path: Option<PathBuf>,

    /// Thinking token configuration (THINKING directive) - Mullama extension
    pub thinking: Option<ThinkingConfig>,

    /// Tool calling format (TOOLFORMAT directive) - Mullama extension
    pub tool_format: Option<ToolFormat>,

    /// Model capabilities (CAPABILITY directive) - Mullama extension
    pub capabilities: Capabilities,

    /// Stop sequences (accumulated from PARAMETER stop lines)
    pub stop_sequences: Vec<String>,

    /// Expected SHA256 digest for model verification (DIGEST directive)
    pub digest: Option<String>,

    /// Revision/commit pin parsed from FROM (e.g., hf:org/repo@commit)
    pub revision: Option<String>,
}

/// Thinking token configuration for reasoning models
#[derive(Debug, Clone, Default)]
pub struct ThinkingConfig {
    /// Start token for thinking content (e.g., "<think>")
    pub start_token: String,
    /// End token for thinking content (e.g., "</think>")
    pub end_token: String,
    /// Whether thinking is enabled
    pub enabled: bool,
}

/// Tool calling format specification
#[derive(Debug, Clone, Default)]
pub struct ToolFormat {
    /// Style name (e.g., "qwen", "llama", "custom")
    pub style: String,
    /// Start token for tool calls
    pub call_start: String,
    /// End token for tool calls
    pub call_end: String,
    /// Start token for tool results
    pub result_start: String,
    /// End token for tool results
    pub result_end: String,
}

/// Model capabilities flags
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    /// Model can output JSON
    pub json: bool,
    /// Model supports tool/function calling
    pub tools: bool,
    /// Model supports reasoning/thinking mode
    pub thinking: bool,
    /// Model supports vision/image input
    pub vision: bool,
}

/// A parameter value that can be a string, number, or boolean
#[derive(Debug, Clone)]
pub enum ParameterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

impl ParameterValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParameterValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParameterValue::Integer(i) => Some(*i),
            ParameterValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(f) => Some(*f),
            ParameterValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterValue::String(s) => write!(f, "{}", s),
            ParameterValue::Integer(i) => write!(f, "{}", i),
            ParameterValue::Float(fl) => write!(f, "{}", fl),
            ParameterValue::Bool(b) => write!(f, "{}", b),
        }
    }
}

/// A pre-defined conversation message
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Execution record for audit logging
/// Captures what ran, with what config, for reproducibility
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionRecord {
    /// Unique execution ID
    pub id: String,
    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: u64,
    /// Model file digest (sha256:...)
    pub model_digest: String,
    /// Model reference (e.g., "hf:org/repo@commit")
    pub model_ref: String,
    /// Revision/commit if pinned
    pub revision: Option<String>,
    /// Hash of the resolved config (sampling params, etc.)
    pub config_hash: String,
    /// Backend version (llama.cpp build info)
    pub backend_version: String,
    /// GPU info if used
    pub gpu_info: Option<String>,
    /// Context size used
    pub context_size: u32,
    /// GPU layers offloaded
    pub gpu_layers: i32,
    /// Temperature setting
    pub temperature: f32,
    /// Number of tokens in prompt
    pub prompt_tokens: u32,
    /// Number of tokens generated
    pub completion_tokens: u32,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ExecutionRecord {
    /// Generate a unique execution ID
    pub fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("exec_{:x}", ts)
    }

    /// Get current timestamp in milliseconds
    pub fn now_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Compute hash of config for deduplication/comparison
    pub fn hash_config(
        temperature: f32,
        top_p: f32,
        top_k: i32,
        context_size: u32,
        stop_sequences: &[String],
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(temperature.to_le_bytes());
        hasher.update(top_p.to_le_bytes());
        hasher.update(top_k.to_le_bytes());
        hasher.update(context_size.to_le_bytes());
        for stop in stop_sequences {
            hasher.update(stop.as_bytes());
        }
        format!("{:x}", hasher.finalize())[..16].to_string() // Short hash
    }

    /// Format as a single log line (JSON-lines compatible)
    pub fn to_log_line(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"ts\":{},\"model\":\"{}\",\"digest\":\"{}\",\"tokens\":{},\"duration_ms\":{},\"success\":{}}}",
            self.id,
            self.timestamp,
            self.model_ref,
            &self.model_digest[..std::cmp::min(20, self.model_digest.len())],
            self.prompt_tokens + self.completion_tokens,
            self.duration_ms,
            self.success
        )
    }
}

/// Errors that can occur while parsing a Modelfile
#[derive(Debug)]
pub enum ModelfileError {
    /// File not found
    FileNotFound(PathBuf),
    /// IO error reading file
    IoError(std::io::Error),
    /// Missing required FROM directive
    MissingFrom,
    /// Invalid directive syntax
    InvalidSyntax { line: usize, message: String },
    /// Unknown directive
    UnknownDirective { line: usize, directive: String },
    /// Invalid parameter value
    InvalidParameter {
        line: usize,
        name: String,
        message: String,
    },
    /// Digest verification failed
    DigestMismatch { expected: String, computed: String },
}

impl std::fmt::Display for ModelfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelfileError::FileNotFound(path) => {
                write!(f, "Modelfile not found: {}", path.display())
            }
            ModelfileError::IoError(e) => write!(f, "IO error: {}", e),
            ModelfileError::MissingFrom => write!(f, "Missing required FROM directive"),
            ModelfileError::InvalidSyntax { line, message } => {
                write!(f, "Syntax error at line {}: {}", line, message)
            }
            ModelfileError::UnknownDirective { line, directive } => {
                write!(f, "Unknown directive '{}' at line {}", directive, line)
            }
            ModelfileError::InvalidParameter {
                line,
                name,
                message,
            } => {
                write!(
                    f,
                    "Invalid parameter '{}' at line {}: {}",
                    name, line, message
                )
            }
            ModelfileError::DigestMismatch { expected, computed } => {
                write!(
                    f,
                    "Digest mismatch: expected {}, computed {}",
                    expected, computed
                )
            }
        }
    }
}

impl std::error::Error for ModelfileError {}

impl From<std::io::Error> for ModelfileError {
    fn from(e: std::io::Error) -> Self {
        ModelfileError::IoError(e)
    }
}

impl Modelfile {
    /// Create a new empty Modelfile
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a Modelfile with a FROM directive
    pub fn from_model(model: impl Into<String>) -> Self {
        Self {
            from: model.into(),
            ..Default::default()
        }
    }

    /// Set a parameter
    pub fn set_parameter(&mut self, name: impl Into<String>, value: ParameterValue) {
        self.parameters.insert(name.into().to_lowercase(), value);
    }

    /// Get a parameter
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(&name.to_lowercase())
    }

    /// Get temperature parameter
    pub fn temperature(&self) -> Option<f64> {
        self.get_parameter("temperature").and_then(|v| v.as_f64())
    }

    /// Get top_p parameter
    pub fn top_p(&self) -> Option<f64> {
        self.get_parameter("top_p").and_then(|v| v.as_f64())
    }

    /// Get top_k parameter
    pub fn top_k(&self) -> Option<i64> {
        self.get_parameter("top_k").and_then(|v| v.as_i64())
    }

    /// Get num_ctx (context size) parameter
    pub fn num_ctx(&self) -> Option<i64> {
        self.get_parameter("num_ctx").and_then(|v| v.as_i64())
    }

    /// Get repeat_penalty parameter
    pub fn repeat_penalty(&self) -> Option<f64> {
        self.get_parameter("repeat_penalty")
            .and_then(|v| v.as_f64())
    }

    /// Get seed parameter
    pub fn seed(&self) -> Option<i64> {
        self.get_parameter("seed").and_then(|v| v.as_i64())
    }

    /// Get stop sequences
    pub fn stop(&self) -> Option<Vec<String>> {
        self.get_parameter("stop").and_then(|v| {
            v.as_str()
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        })
    }

    /// Check if this is a HuggingFace model spec
    pub fn is_huggingface(&self) -> bool {
        self.from.starts_with("hf:")
    }

    /// Check if this is a local file path
    pub fn is_local_path(&self) -> bool {
        self.from.starts_with('/')
            || self.from.starts_with('.')
            || self.from.contains('/')
            || self.from.ends_with(".gguf")
    }

    /// Check if this is a model alias (like llama3.2:1b)
    pub fn is_alias(&self) -> bool {
        !self.is_huggingface() && !self.is_local_path()
    }

    /// Get the full model reference including revision
    pub fn full_model_ref(&self) -> String {
        if let Some(ref rev) = self.revision {
            format!("{}@{}", self.from, rev)
        } else {
            self.from.clone()
        }
    }

    /// Verify a file against the expected digest
    /// Returns Ok(computed_digest) if verification passes or no digest specified
    /// Returns Err if digest doesn't match
    pub fn verify_digest(&self, file_path: &Path) -> Result<String, ModelfileError> {
        use std::io::Read;

        let mut file = std::fs::File::open(file_path).map_err(ModelfileError::IoError)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer).map_err(ModelfileError::IoError)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let computed = format!("sha256:{:x}", hasher.finalize());

        if let Some(ref expected) = self.digest {
            if &computed != expected {
                return Err(ModelfileError::DigestMismatch {
                    expected: expected.clone(),
                    computed: computed.clone(),
                });
            }
        }

        Ok(computed)
    }

    /// Serialize to Modelfile format
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut output = String::new();

        // FROM (with optional revision)
        if let Some(ref rev) = self.revision {
            output.push_str(&format!("FROM {}@{}\n", self.from, rev));
        } else {
            output.push_str(&format!("FROM {}\n", self.from));
        }

        // DIGEST
        if let Some(ref digest) = self.digest {
            output.push_str(&format!("DIGEST {}\n", digest));
        }
        output.push('\n');

        // Parameters
        if !self.parameters.is_empty() {
            for (name, value) in &self.parameters {
                output.push_str(&format!("PARAMETER {} {}\n", name, value));
            }
            output.push('\n');
        }

        // System
        if let Some(ref system) = self.system {
            if system.contains('\n') {
                output.push_str(&format!("SYSTEM \"\"\"\n{}\n\"\"\"\n", system));
            } else {
                output.push_str(&format!("SYSTEM \"{}\"\n", system));
            }
            output.push('\n');
        }

        // Template
        if let Some(ref template) = self.template {
            if template.contains('\n') {
                output.push_str(&format!("TEMPLATE \"\"\"\n{}\n\"\"\"\n", template));
            } else {
                output.push_str(&format!("TEMPLATE \"{}\"\n", template));
            }
            output.push('\n');
        }

        // Messages
        for msg in &self.messages {
            output.push_str(&format!("MESSAGE {} \"{}\"\n", msg.role, msg.content));
        }
        if !self.messages.is_empty() {
            output.push('\n');
        }

        // Stop sequences
        for stop in &self.stop_sequences {
            output.push_str(&format!("PARAMETER stop \"{}\"\n", stop));
        }
        if !self.stop_sequences.is_empty() {
            output.push('\n');
        }

        // Mullama extensions
        let has_extensions = self.adapter.is_some()
            || self.gpu_layers.is_some()
            || self.flash_attention.is_some()
            || self.vision_projector.is_some()
            || self.thinking.is_some()
            || self.tool_format.is_some()
            || self.capabilities.json
            || self.capabilities.tools
            || self.capabilities.thinking
            || self.capabilities.vision;

        if has_extensions {
            output.push_str("# Mullama extensions\n");

            if let Some(ref adapter) = self.adapter {
                output.push_str(&format!("ADAPTER {}\n", adapter.display()));
            }
            if let Some(layers) = self.gpu_layers {
                output.push_str(&format!("GPU_LAYERS {}\n", layers));
            }
            if let Some(flash) = self.flash_attention {
                output.push_str(&format!("FLASH_ATTENTION {}\n", flash));
            }
            if let Some(ref mmproj) = self.vision_projector {
                output.push_str(&format!("VISION_PROJECTOR {}\n", mmproj.display()));
            }

            // Thinking configuration
            if let Some(ref thinking) = self.thinking {
                output.push_str(&format!("THINKING start \"{}\"\n", thinking.start_token));
                output.push_str(&format!("THINKING end \"{}\"\n", thinking.end_token));
                output.push_str(&format!("THINKING enabled {}\n", thinking.enabled));
            }

            // Tool format
            if let Some(ref tool_format) = self.tool_format {
                output.push_str(&format!("TOOLFORMAT style \"{}\"\n", tool_format.style));
                output.push_str(&format!(
                    "TOOLFORMAT call_start \"{}\"\n",
                    tool_format.call_start
                ));
                output.push_str(&format!(
                    "TOOLFORMAT call_end \"{}\"\n",
                    tool_format.call_end
                ));
                output.push_str(&format!(
                    "TOOLFORMAT result_start \"{}\"\n",
                    tool_format.result_start
                ));
                output.push_str(&format!(
                    "TOOLFORMAT result_end \"{}\"\n",
                    tool_format.result_end
                ));
            }

            // Capabilities
            if self.capabilities.json {
                output.push_str("CAPABILITY json true\n");
            }
            if self.capabilities.tools {
                output.push_str("CAPABILITY tools true\n");
            }
            if self.capabilities.thinking {
                output.push_str("CAPABILITY thinking true\n");
            }
            if self.capabilities.vision {
                output.push_str("CAPABILITY vision true\n");
            }

            output.push('\n');
        }

        // Metadata
        if self.license.is_some() || self.author.is_some() {
            output.push_str("# Metadata\n");

            if let Some(ref license) = self.license {
                if license.contains('\n') {
                    output.push_str(&format!("LICENSE \"\"\"\n{}\n\"\"\"\n", license));
                } else {
                    output.push_str(&format!("LICENSE {}\n", license));
                }
            }
            if let Some(ref author) = self.author {
                output.push_str(&format!("AUTHOR \"{}\"\n", author));
            }
        }

        output
    }

    /// Save to a file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_string())
    }
}
