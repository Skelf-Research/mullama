use std::path::{Path, PathBuf};

use super::types::{
    Message, Modelfile, ModelfileError, ParameterValue, ThinkingConfig, ToolFormat,
};

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
                    if let Some(initial) = value.strip_prefix("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "SYSTEM")? {
                            Some(inline) => modelfile.system = Some(inline),
                            None => {
                                current_multiline =
                                    Some(("SYSTEM".to_string(), initial.to_string(), line_num));
                            }
                        }
                    } else {
                        modelfile.system = Some(Self::unquote(value));
                    }
                }

                "TEMPLATE" => {
                    if let Some(initial) = value.strip_prefix("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "TEMPLATE")? {
                            Some(inline) => modelfile.template = Some(inline),
                            None => {
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
                    if let Some(initial) = value.strip_prefix("\"\"\"") {
                        match Self::parse_triple_quoted_inline(value, line_num, "LICENSE")? {
                            Some(inline) => modelfile.license = Some(inline),
                            None => {
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
