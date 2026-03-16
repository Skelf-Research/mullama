//! Modelfile Parser for Mullama
//!
//! Supports both Ollama-compatible Modelfile syntax and extended Mullamafile features.
//!
//! ## Ollama-Compatible Keys (Modelfile)
//!
//! ```dockerfile
//! FROM llama3.2:1b
//! PARAMETER temperature 0.7
//! PARAMETER num_ctx 8192
//! SYSTEM """
//! You are a helpful assistant.
//! """
//! TEMPLATE """
//! {{ .System }}
//! {{ .Prompt }}
//! """
//! MESSAGE user Hello!
//! MESSAGE assistant Hi there!
//! ```
//!
//! ## Mullama Extensions (Mullamafile)
//!
//! ```dockerfile
//! FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
//!
//! # Standard Modelfile keys
//! PARAMETER temperature 0.7
//! PARAMETER num_ctx 8192
//!
//! SYSTEM """
//! You are a helpful assistant.
//! """
//!
//! # Mullama-specific extensions
//! ADAPTER ./my-lora-adapter.safetensors
//! GPU_LAYERS 32
//! FLASH_ATTENTION true
//! VISION_PROJECTOR ./mmproj.gguf
//!
//! # Metadata
//! LICENSE MIT
//! AUTHOR "Your Name"
//! ```

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

/// Parser for Modelfile/Mullamafile
pub struct ModelfileParser {
    /// Whether to allow Mullama extensions
    allow_extensions: bool,
    /// Base directory for resolving relative paths
    base_dir: Option<PathBuf>,
}

impl Default for ModelfileParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelfileParser {
    /// Create a new parser with extensions enabled
    pub fn new() -> Self {
        Self {
            allow_extensions: true,
            base_dir: None,
        }
    }

    /// Create a parser that only allows Ollama-compatible syntax
    pub fn ollama_compatible() -> Self {
        Self {
            allow_extensions: false,
            base_dir: None,
        }
    }

    /// Set the base directory for resolving relative paths
    pub fn with_base_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.base_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Parse a Modelfile from a file path
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Modelfile, ModelfileError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ModelfileError::FileNotFound(path.to_path_buf()));
        }

        let content = std::fs::read_to_string(path)?;
        let base_dir = path.parent().map(|p| p.to_path_buf());

        let parser = Self {
            allow_extensions: self.allow_extensions,
            base_dir: base_dir.or_else(|| self.base_dir.clone()),
        };

        let mut modelfile = parser.parse_str(&content)?;
        modelfile.source_path = Some(path.to_path_buf());
        Ok(modelfile)
    }

    /// Parse a Modelfile from a string
    pub fn parse_str(&self, content: &str) -> Result<Modelfile, ModelfileError> {
        let mut modelfile = Modelfile::default();
        let mut current_multiline: Option<(String, String, usize)> = None; // (directive, content, start_line)
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let line_num = i + 1;

            // Handle multiline content
            if let Some((ref directive, ref mut content, _start_line)) = current_multiline {
                if let Some(close_idx) = line.find("\"\"\"") {
                    let before_close = &line[..close_idx];
                    let after_close = &line[close_idx + 3..];
                    if !after_close.trim().is_empty() {
                        return Err(ModelfileError::InvalidSyntax {
                            line: line_num,
                            message: format!(
                                "{} has trailing content after closing triple quotes",
                                directive
                            ),
                        });
                    }

                    if !before_close.is_empty() {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(before_close);
                    }

                    // End of multiline
                    let directive = directive.clone();
                    let final_content = content.clone();
                    current_multiline = None;

                    match directive.as_str() {
                        "SYSTEM" => modelfile.system = Some(final_content),
                        "TEMPLATE" => modelfile.template = Some(final_content),
                        "LICENSE" => modelfile.license = Some(final_content),
                        _ => {}
                    }
                } else {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(line);
                }
                i += 1;
                continue;
            }

            // Skip empty lines and comments
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                i += 1;
                continue;
            }

            // Parse directive
            let (directive, value) = Self::parse_directive(trimmed, line_num)?;
            let directive_upper = directive.to_uppercase();

            match directive_upper.as_str() {
                "FROM" => {
                    // Parse revision from FROM (e.g., hf:org/repo@commit)
                    let (model_ref, revision) = Self::parse_model_ref(value);
                    modelfile.from = model_ref;
                    if revision.is_some() {
                        modelfile.revision = revision;
                    }
                }

                "DIGEST" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let digest = Self::unquote(value);
                    // Validate digest format (sha256:... or just hex)
                    let normalized = if digest.starts_with("sha256:") {
                        digest
                    } else if digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit()) {
                        format!("sha256:{}", digest)
                    } else {
                        return Err(ModelfileError::InvalidParameter {
                            line: line_num,
                            name: "DIGEST".to_string(),
                            message: "Expected sha256:... or 64-character hex string".to_string(),
                        });
                    };
                    modelfile.digest = Some(normalized);
                }

                "PARAMETER" => {
                    let (name, param_value) = Self::parse_parameter(value, line_num)?;
                    // Handle multiple stop sequences specially
                    if name == "stop" {
                        if let Some(s) = param_value.as_str() {
                            modelfile.stop_sequences.push(s.to_string());
                        }
                    }
                    modelfile.parameters.insert(name, param_value);
                }

                "SYSTEM" => {
                    if value.starts_with("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "SYSTEM")? {
                            Some(inline) => modelfile.system = Some(inline),
                            None => {
                                let initial = &value[3..];
                                current_multiline =
                                    Some(("SYSTEM".to_string(), initial.to_string(), line_num));
                            }
                        }
                    } else {
                        modelfile.system = Some(Self::unquote(value));
                    }
                }

                "TEMPLATE" => {
                    if value.starts_with("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "TEMPLATE")? {
                            Some(inline) => modelfile.template = Some(inline),
                            None => {
                                let initial = &value[3..];
                                current_multiline =
                                    Some(("TEMPLATE".to_string(), initial.to_string(), line_num));
                            }
                        }
                    } else {
                        modelfile.template = Some(Self::unquote(value));
                    }
                }

                "MESSAGE" => {
                    let (role, content) = Self::parse_message(value, line_num)?;
                    modelfile.messages.push(Message { role, content });
                }

                "LICENSE" => {
                    if value.starts_with("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "LICENSE")? {
                            Some(inline) => modelfile.license = Some(inline),
                            None => {
                                let initial = &value[3..];
                                current_multiline =
                                    Some(("LICENSE".to_string(), initial.to_string(), line_num));
                            }
                        }
                    } else {
                        modelfile.license = Some(Self::unquote(value));
                    }
                }

                // Mullama extensions
                "ADAPTER" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let path = self.resolve_path(Self::unquote(value));
                    modelfile.adapter = Some(path);
                }

                "GPU_LAYERS" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let layers: i32 =
                        value
                            .parse()
                            .map_err(|_| ModelfileError::InvalidParameter {
                                line: line_num,
                                name: "GPU_LAYERS".to_string(),
                                message: "Expected integer".to_string(),
                            })?;
                    modelfile.gpu_layers = Some(layers);
                }

                "FLASH_ATTENTION" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let enabled = Self::parse_bool(value, line_num, "FLASH_ATTENTION")?;
                    modelfile.flash_attention = Some(enabled);
                }

                "VISION_PROJECTOR" | "MMPROJ" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let path = self.resolve_path(Self::unquote(value));
                    modelfile.vision_projector = Some(path);
                }

                "AUTHOR" => {
                    modelfile.author = Some(Self::unquote(value));
                }

                // THINKING directive for reasoning models
                "THINKING" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let (key, val) = Self::parse_key_value(value, line_num, "THINKING")?;
                    if modelfile.thinking.is_none() {
                        modelfile.thinking = Some(ThinkingConfig::default());
                    }
                    let thinking = modelfile.thinking.as_mut().unwrap();
                    match key.as_str() {
                        "start" => thinking.start_token = Self::unquote(&val),
                        "end" => thinking.end_token = Self::unquote(&val),
                        "enabled" => {
                            thinking.enabled = Self::parse_bool(&val, line_num, "THINKING enabled")?
                        }
                        _ => {
                            return Err(ModelfileError::InvalidParameter {
                                line: line_num,
                                name: format!("THINKING {}", key),
                                message: "Unknown thinking key. Expected: start, end, enabled"
                                    .to_string(),
                            })
                        }
                    }
                }

                // TOOLFORMAT directive for tool calling
                "TOOLFORMAT" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let (key, val) = Self::parse_key_value(value, line_num, "TOOLFORMAT")?;
                    if modelfile.tool_format.is_none() {
                        modelfile.tool_format = Some(ToolFormat::default());
                    }
                    let tool_format = modelfile.tool_format.as_mut().unwrap();
                    match key.as_str() {
                        "style" => tool_format.style = Self::unquote(&val),
                        "call_start" => tool_format.call_start = Self::unquote(&val),
                        "call_end" => tool_format.call_end = Self::unquote(&val),
                        "result_start" => tool_format.result_start = Self::unquote(&val),
                        "result_end" => tool_format.result_end = Self::unquote(&val),
                        _ => return Err(ModelfileError::InvalidParameter {
                            line: line_num,
                            name: format!("TOOLFORMAT {}", key),
                            message: "Unknown toolformat key. Expected: style, call_start, call_end, result_start, result_end".to_string(),
                        }),
                    }
                }

                // CAPABILITY directive for model capabilities
                "CAPABILITY" => {
                    if !self.allow_extensions {
                        return Err(ModelfileError::UnknownDirective {
                            line: line_num,
                            directive: directive.to_string(),
                        });
                    }
                    let (key, val) = Self::parse_key_value(value, line_num, "CAPABILITY")?;
                    let enabled = Self::parse_bool(&val, line_num, &format!("CAPABILITY {}", key))?;
                    match key.as_str() {
                        "json" => modelfile.capabilities.json = enabled,
                        "tools" => modelfile.capabilities.tools = enabled,
                        "thinking" => modelfile.capabilities.thinking = enabled,
                        "vision" => modelfile.capabilities.vision = enabled,
                        _ => {
                            return Err(ModelfileError::InvalidParameter {
                                line: line_num,
                                name: format!("CAPABILITY {}", key),
                                message:
                                    "Unknown capability. Expected: json, tools, thinking, vision"
                                        .to_string(),
                            })
                        }
                    }
                }

                _ => {
                    return Err(ModelfileError::UnknownDirective {
                        line: line_num,
                        directive: directive.to_string(),
                    });
                }
            }

            i += 1;
        }

        // Check for unclosed multiline
        if let Some((directive, _, start_line)) = current_multiline {
            return Err(ModelfileError::InvalidSyntax {
                line: start_line,
                message: format!("Unclosed multiline {} directive", directive),
            });
        }

        // Validate required fields
        if modelfile.from.is_empty() {
            return Err(ModelfileError::MissingFrom);
        }

        Ok(modelfile)
    }

    /// Parse model reference, extracting optional revision
    /// Examples:
    ///   "hf:org/repo@abc123" -> ("hf:org/repo", Some("abc123"))
    ///   "hf:org/repo" -> ("hf:org/repo", None)
    ///   "./model.gguf" -> ("./model.gguf", None)
    fn parse_model_ref(value: &str) -> (String, Option<String>) {
        // Check for @revision suffix (but not in local paths with @ in filename)
        if value.starts_with("hf:") || value.starts_with("https://") || value.starts_with("s3://") {
            if let Some(at_pos) = value.rfind('@') {
                let model_ref = value[..at_pos].to_string();
                let revision = value[at_pos + 1..].to_string();
                if !revision.is_empty() {
                    return (model_ref, Some(revision));
                }
            }
        }
        (value.to_string(), None)
    }

    /// Parse a directive line into (directive, value)
    fn parse_directive(line: &str, line_num: usize) -> Result<(&str, &str), ModelfileError> {
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();

        if parts.is_empty() {
            return Err(ModelfileError::InvalidSyntax {
                line: line_num,
                message: "Empty directive".to_string(),
            });
        }

        let directive = parts[0];
        let value = if parts.len() > 1 { parts[1].trim() } else { "" };

        Ok((directive, value))
    }

    /// Parse a PARAMETER directive value
    fn parse_parameter(
        value: &str,
        line_num: usize,
    ) -> Result<(String, ParameterValue), ModelfileError> {
        let parts: Vec<&str> = value.splitn(2, char::is_whitespace).collect();

        if parts.len() < 2 {
            return Err(ModelfileError::InvalidSyntax {
                line: line_num,
                message: "PARAMETER requires name and value".to_string(),
            });
        }

        let name = parts[0].to_lowercase();
        let value_str = parts[1].trim();

        // Try to parse as different types
        let param_value = if value_str == "true" || value_str == "false" {
            ParameterValue::Bool(value_str == "true")
        } else if let Ok(i) = value_str.parse::<i64>() {
            ParameterValue::Integer(i)
        } else if let Ok(f) = value_str.parse::<f64>() {
            ParameterValue::Float(f)
        } else {
            ParameterValue::String(Self::unquote(value_str))
        };

        Ok((name, param_value))
    }

    /// Parse a MESSAGE directive value
    fn parse_message(value: &str, line_num: usize) -> Result<(String, String), ModelfileError> {
        let parts: Vec<&str> = value.splitn(2, char::is_whitespace).collect();

        if parts.len() < 2 {
            return Err(ModelfileError::InvalidSyntax {
                line: line_num,
                message: "MESSAGE requires role and content".to_string(),
            });
        }

        let role = parts[0].to_lowercase();
        let content = Self::unquote(parts[1].trim());

        // Validate role
        if !["system", "user", "assistant"].contains(&role.as_str()) {
            return Err(ModelfileError::InvalidParameter {
                line: line_num,
                name: "MESSAGE role".to_string(),
                message: format!("Invalid role '{}'. Expected: system, user, assistant", role),
            });
        }

        Ok((role, content))
    }

    /// Parse a key-value pair (e.g., "start \"<think>\"")
    fn parse_key_value(
        value: &str,
        line_num: usize,
        directive: &str,
    ) -> Result<(String, String), ModelfileError> {
        let parts: Vec<&str> = value.splitn(2, char::is_whitespace).collect();

        if parts.len() < 2 {
            return Err(ModelfileError::InvalidSyntax {
                line: line_num,
                message: format!("{} requires key and value", directive),
            });
        }

        let key = parts[0].to_lowercase();
        let val = parts[1].trim().to_string();

        Ok((key, val))
    }

    /// Parse a boolean value
    fn parse_bool(value: &str, line_num: usize, name: &str) -> Result<bool, ModelfileError> {
        match value.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => Err(ModelfileError::InvalidParameter {
                line: line_num,
                name: name.to_string(),
                message: "Expected boolean (true/false)".to_string(),
            }),
        }
    }

    /// Parse an inline triple-quoted value (`"""..."""`) if closing quotes are on the same line.
    /// Returns `None` when the directive starts a multiline block.
    fn parse_triple_quoted_inline(
        value: &str,
        line_num: usize,
        directive: &str,
    ) -> Result<Option<String>, ModelfileError> {
        let rest = &value[3..];
        let Some(end_idx) = rest.find("\"\"\"") else {
            return Ok(None);
        };

        let after = &rest[end_idx + 3..];
        if !after.trim().is_empty() {
            return Err(ModelfileError::InvalidSyntax {
                line: line_num,
                message: format!(
                    "{} has trailing content after closing triple quotes",
                    directive
                ),
            });
        }

        Ok(Some(rest[..end_idx].to_string()))
    }

    /// Remove surrounding quotes from a string
    fn unquote(s: &str) -> String {
        let s = s.trim();
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    }

    /// Resolve a potentially relative path
    fn resolve_path(&self, path: String) -> PathBuf {
        let p = PathBuf::from(&path);
        if p.is_absolute() {
            p
        } else if let Some(ref base) = self.base_dir {
            base.join(p)
        } else {
            p
        }
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

        let mut file = std::fs::File::open(file_path).map_err(|e| ModelfileError::IoError(e))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file
                .read(&mut buffer)
                .map_err(|e| ModelfileError::IoError(e))?;
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

/// Find a Modelfile in a directory (checks both Modelfile and Mullamafile)
pub fn find_modelfile(dir: impl AsRef<Path>) -> Option<PathBuf> {
    let dir = dir.as_ref();

    // Check in order of preference
    let candidates = [
        "Mullamafile",
        "Modelfile",
        "mullamafile",
        "modelfile",
        ".mullamafile",
        ".modelfile",
    ];

    for name in candidates {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_modelfile() {
        let content = r#"
FROM llama3.2:1b
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are a helpful assistant."
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.from, "llama3.2:1b");
        assert_eq!(modelfile.temperature(), Some(0.7));
        assert_eq!(modelfile.num_ctx(), Some(4096));
        assert_eq!(
            modelfile.system,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_parse_multiline_system() {
        let content = r#"
FROM llama3.2:1b
SYSTEM """
You are a helpful assistant.
You answer questions clearly.
"""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.system.is_some());
        assert!(modelfile
            .system
            .as_ref()
            .unwrap()
            .contains("You are a helpful assistant."));
        assert!(modelfile
            .system
            .as_ref()
            .unwrap()
            .contains("You answer questions clearly."));
    }

    #[test]
    fn test_parse_single_line_triple_quoted_system() {
        let content = r#"
FROM llama3.2:1b
SYSTEM """You are a helpful assistant."""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(
            modelfile.system,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_parse_multiline_template_with_inline_close() {
        let content = r#"
FROM llama3.2:1b
TEMPLATE """Line one
Line two"""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.template, Some("Line one\nLine two".to_string()));
    }

    #[test]
    fn test_parse_mullama_extensions() {
        let content = r#"
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
GPU_LAYERS 32
FLASH_ATTENTION true
VISION_PROJECTOR ./mmproj.gguf
ADAPTER ./lora.safetensors
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.is_huggingface());
        assert_eq!(modelfile.gpu_layers, Some(32));
        assert_eq!(modelfile.flash_attention, Some(true));
        assert!(modelfile.vision_projector.is_some());
        assert!(modelfile.adapter.is_some());
    }

    #[test]
    fn test_ollama_compat_rejects_extensions() {
        let content = r#"
FROM llama3.2:1b
GPU_LAYERS 32
"#;

        let parser = ModelfileParser::ollama_compatible();
        let result = parser.parse_str(content);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_messages() {
        let content = r#"
FROM llama3.2:1b
MESSAGE user "Hello!"
MESSAGE assistant "Hi there! How can I help?"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.messages.len(), 2);
        assert_eq!(modelfile.messages[0].role, "user");
        assert_eq!(modelfile.messages[0].content, "Hello!");
        assert_eq!(modelfile.messages[1].role, "assistant");
    }

    #[test]
    fn test_missing_from() {
        let content = r#"
PARAMETER temperature 0.7
"#;

        let parser = ModelfileParser::new();
        let result = parser.parse_str(content);

        assert!(matches!(result, Err(ModelfileError::MissingFrom)));
    }

    #[test]
    fn test_serialize_modelfile() {
        let mut modelfile = Modelfile::from_model("llama3.2:1b");
        modelfile.set_parameter("temperature", ParameterValue::Float(0.7));
        modelfile.system = Some("You are helpful.".to_string());
        modelfile.gpu_layers = Some(32);

        let output = modelfile.to_string();

        assert!(output.contains("FROM llama3.2:1b"));
        assert!(output.contains("PARAMETER temperature 0.7"));
        assert!(output.contains("SYSTEM"));
        assert!(output.contains("GPU_LAYERS 32"));
    }

    #[test]
    fn test_is_alias() {
        let m1 = Modelfile::from_model("llama3.2:1b");
        assert!(m1.is_alias());

        let m2 = Modelfile::from_model("hf:TheBloke/Llama-2-7B-GGUF");
        assert!(m2.is_huggingface());
        assert!(!m2.is_alias());

        let m3 = Modelfile::from_model("./model.gguf");
        assert!(m3.is_local_path());
        assert!(!m3.is_alias());
    }

    #[test]
    fn test_parse_thinking_config() {
        let content = r#"
FROM qwq:32b
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true
CAPABILITY thinking true
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.thinking.is_some());
        let thinking = modelfile.thinking.as_ref().unwrap();
        assert_eq!(thinking.start_token, "<think>");
        assert_eq!(thinking.end_token, "</think>");
        assert!(thinking.enabled);
        assert!(modelfile.capabilities.thinking);
    }

    #[test]
    fn test_parse_multiple_stop_sequences() {
        let content = r#"
FROM qwen2.5:7b
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.stop_sequences.len(), 2);
        assert!(modelfile.stop_sequences.contains(&"<|im_end|>".to_string()));
        assert!(modelfile
            .stop_sequences
            .contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn test_parse_capabilities() {
        let content = r#"
FROM llama3.2:1b
CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking false
CAPABILITY vision true
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.capabilities.json);
        assert!(modelfile.capabilities.tools);
        assert!(!modelfile.capabilities.thinking);
        assert!(modelfile.capabilities.vision);
    }

    #[test]
    fn test_parse_toolformat() {
        let content = r#"
FROM qwen2.5:7b
TOOLFORMAT style "qwen"
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.tool_format.is_some());
        let tf = modelfile.tool_format.as_ref().unwrap();
        assert_eq!(tf.style, "qwen");
        assert_eq!(tf.call_start, "<tool_call>");
        assert_eq!(tf.call_end, "</tool_call>");
        assert_eq!(tf.result_start, "<tool_response>");
        assert_eq!(tf.result_end, "</tool_response>");
    }

    #[test]
    fn test_serialize_thinking_and_capabilities() {
        let mut modelfile = Modelfile::from_model("qwq:32b");
        modelfile.thinking = Some(ThinkingConfig {
            start_token: "<think>".to_string(),
            end_token: "</think>".to_string(),
            enabled: true,
        });
        modelfile.capabilities.thinking = true;
        modelfile.stop_sequences.push("<|im_end|>".to_string());

        let output = modelfile.to_string();

        assert!(output.contains("THINKING start \"<think>\""));
        assert!(output.contains("THINKING end \"</think>\""));
        assert!(output.contains("THINKING enabled true"));
        assert!(output.contains("CAPABILITY thinking true"));
        assert!(output.contains("PARAMETER stop \"<|im_end|>\""));
    }

    #[test]
    fn test_parse_revision_from_hf_spec() {
        let content = r#"
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF@abc123def
PARAMETER temperature 0.7
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.from, "hf:meta-llama/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(modelfile.revision, Some("abc123def".to_string()));
        assert_eq!(
            modelfile.full_model_ref(),
            "hf:meta-llama/Llama-3.2-1B-Instruct-GGUF@abc123def"
        );
    }

    #[test]
    fn test_parse_digest() {
        let content = r#"
FROM hf:org/model
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(
            modelfile.digest,
            Some(
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_parse_digest_without_prefix() {
        let content = r#"
FROM hf:org/model
DIGEST e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        // Should auto-prefix with sha256:
        assert_eq!(
            modelfile.digest,
            Some(
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_serialize_with_revision_and_digest() {
        let mut modelfile = Modelfile::from_model("hf:org/repo");
        modelfile.revision = Some("main".to_string());
        modelfile.digest = Some("sha256:abc123".to_string());

        let output = modelfile.to_string();

        assert!(output.contains("FROM hf:org/repo@main"));
        assert!(output.contains("DIGEST sha256:abc123"));
    }

    #[test]
    fn test_execution_record_id_generation() {
        let id1 = ExecutionRecord::generate_id();
        let id2 = ExecutionRecord::generate_id();

        assert!(id1.starts_with("exec_"));
        assert!(id2.starts_with("exec_"));
        // IDs should be unique (technically could collide but extremely unlikely)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_execution_record_config_hash() {
        let hash1 = ExecutionRecord::hash_config(0.7, 0.9, 40, 4096, &["<|im_end|>".to_string()]);
        let hash2 = ExecutionRecord::hash_config(0.7, 0.9, 40, 4096, &["<|im_end|>".to_string()]);
        let hash3 = ExecutionRecord::hash_config(0.8, 0.9, 40, 4096, &["<|im_end|>".to_string()]);

        // Same config = same hash
        assert_eq!(hash1, hash2);
        // Different config = different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_execution_record_log_line() {
        let record = ExecutionRecord {
            id: "exec_test123".to_string(),
            timestamp: 1700000000000,
            model_digest: "sha256:abcdef123456".to_string(),
            model_ref: "hf:org/model".to_string(),
            revision: None,
            config_hash: "abc123".to_string(),
            backend_version: "1.0.0".to_string(),
            gpu_info: None,
            context_size: 4096,
            gpu_layers: 32,
            temperature: 0.7,
            prompt_tokens: 100,
            completion_tokens: 50,
            duration_ms: 1500,
            success: true,
            error: None,
        };

        let log_line = record.to_log_line();

        assert!(log_line.contains("\"id\":\"exec_test123\""));
        assert!(log_line.contains("\"success\":true"));
        assert!(log_line.contains("\"tokens\":150"));
    }

    #[test]
    fn test_local_path_no_revision_extraction() {
        // Local paths with @ should NOT have revision extracted
        let content = r#"
FROM ./models/my@special-model.gguf
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        // Should keep the full path, no revision extraction
        assert_eq!(modelfile.from, "./models/my@special-model.gguf");
        assert_eq!(modelfile.revision, None);
    }
}
