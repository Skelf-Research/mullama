//! Model Registry for Mullama
//!
//! Maps short model names (like "llama3.2:1b") to HuggingFace repositories.
//! Supports quantization suffixes and automatic file selection.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// A model alias entry in the registry
#[derive(Debug, Clone, Deserialize)]
pub struct ModelAlias {
    /// HuggingFace repository (e.g., "bartowski/Llama-3.2-1B-Instruct-GGUF")
    pub repo: String,

    /// Default GGUF file to use
    #[serde(default)]
    pub default_file: Option<String>,

    /// Vision projector file for multimodal models
    #[serde(default)]
    pub mmproj: Option<String>,

    /// Model family (llama, qwen, mistral, etc.)
    #[serde(default)]
    pub family: Option<String>,

    /// Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Capability tags
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Quantization configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    /// Maps quantization suffixes to file patterns
    #[serde(flatten)]
    pub mappings: HashMap<String, Vec<String>>,

    /// Default quantization preference order
    #[serde(default)]
    pub default_order: Vec<String>,
}

/// Registry metadata
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RegistryMeta {
    pub version: Option<String>,
    pub updated: Option<String>,
}

/// The model registry
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    /// Model aliases
    pub aliases: HashMap<String, ModelAlias>,

    /// Quantization configuration
    pub quantizations: QuantizationConfig,

    /// Registry metadata
    pub meta: RegistryMeta,
}

/// Parsed model specification
#[derive(Debug, Clone)]
pub struct ParsedModelSpec {
    /// The original input string
    pub original: String,

    /// Base model name (without quantization suffix)
    pub name: String,

    /// Requested quantization (e.g., "q4", "q8", "f16")
    pub quantization: Option<String>,

    /// Whether this is a HuggingFace spec (hf:...)
    pub is_hf_spec: bool,

    /// Whether this is a local path
    pub is_local_path: bool,

    /// Resolved alias (if found in registry)
    pub alias: Option<ModelAlias>,
}

impl ModelRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Load registry from the embedded TOML file
    pub fn load_embedded() -> Result<Self, RegistryError> {
        let toml_content = include_str!("../../configs/models.toml");
        Self::from_toml(toml_content)
    }

    /// Load registry from a TOML string
    pub fn from_toml(content: &str) -> Result<Self, RegistryError> {
        #[derive(Deserialize)]
        struct RawRegistry {
            #[serde(default)]
            meta: RegistryMeta,
            #[serde(default)]
            aliases: HashMap<String, ModelAlias>,
            #[serde(default)]
            quantizations: QuantizationConfig,
        }

        let raw: RawRegistry = toml::from_str(content)
            .map_err(|e| RegistryError::ParseError(e.to_string()))?;

        Ok(Self {
            aliases: raw.aliases,
            quantizations: raw.quantizations,
            meta: raw.meta,
        })
    }

    /// Load registry from a file path
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, RegistryError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| RegistryError::IoError(e.to_string()))?;
        Self::from_toml(&content)
    }

    /// Look up a model alias
    pub fn get(&self, name: &str) -> Option<&ModelAlias> {
        self.aliases.get(name)
    }

    /// List all available aliases
    pub fn list_aliases(&self) -> Vec<&str> {
        self.aliases.keys().map(|s| s.as_str()).collect()
    }

    /// Search for aliases matching a pattern
    pub fn search(&self, query: &str) -> Vec<(&str, &ModelAlias)> {
        let query_lower = query.to_lowercase();
        self.aliases
            .iter()
            .filter(|(name, alias)| {
                name.to_lowercase().contains(&query_lower)
                    || alias.description.as_ref()
                        .map(|d| d.to_lowercase().contains(&query_lower))
                        .unwrap_or(false)
                    || alias.family.as_ref()
                        .map(|f| f.to_lowercase().contains(&query_lower))
                        .unwrap_or(false)
                    || alias.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .map(|(name, alias)| (name.as_str(), alias))
            .collect()
    }

    /// Parse a model specification
    pub fn parse_spec(&self, input: &str) -> ParsedModelSpec {
        let input = input.trim();

        // Check if it's a HuggingFace spec
        if input.starts_with("hf:") {
            return ParsedModelSpec {
                original: input.to_string(),
                name: input.to_string(),
                quantization: None,
                is_hf_spec: true,
                is_local_path: false,
                alias: None,
            };
        }

        // Check if it's a local path
        if input.starts_with('/')
            || input.starts_with('.')
            || input.contains(std::path::MAIN_SEPARATOR)
            || input.ends_with(".gguf")
        {
            return ParsedModelSpec {
                original: input.to_string(),
                name: input.to_string(),
                quantization: None,
                is_hf_spec: false,
                is_local_path: true,
                alias: None,
            };
        }

        // Try to parse as alias with optional quantization suffix
        // Format: name:variant-quant or name:variant
        // Examples: llama3.2:1b, llama3.2:1b-q8, qwen2.5:7b-instruct-q4

        let (base_name, quantization) = self.extract_quantization(input);

        // Look up the alias
        let alias = self.get(&base_name).cloned();

        ParsedModelSpec {
            original: input.to_string(),
            name: base_name,
            quantization,
            is_hf_spec: false,
            is_local_path: false,
            alias,
        }
    }

    /// Extract quantization suffix from a model name
    fn extract_quantization(&self, input: &str) -> (String, Option<String>) {
        // Check for quantization suffix at the end
        // Patterns: -q2, -q3, -q4, -q5, -q6, -q8, -f16, -f32
        let quant_suffixes = ["q2", "q3", "q4", "q5", "q6", "q8", "f16", "f32"];

        for suffix in quant_suffixes {
            let pattern = format!("-{}", suffix);
            if input.ends_with(&pattern) {
                let base = input[..input.len() - pattern.len()].to_string();
                return (base, Some(suffix.to_string()));
            }
        }

        (input.to_string(), None)
    }

    /// Get the preferred quantization file patterns for a given suffix
    pub fn get_quant_patterns(&self, suffix: &str) -> Vec<String> {
        self.quantizations.mappings
            .get(suffix)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the default quantization order
    pub fn default_quant_order(&self) -> Vec<String> {
        if self.quantizations.default_order.is_empty() {
            vec![
                "Q4_K_M".to_string(),
                "Q4_K_S".to_string(),
                "Q5_K_M".to_string(),
                "Q4_0".to_string(),
                "Q8_0".to_string(),
                "F16".to_string(),
            ]
        } else {
            self.quantizations.default_order.clone()
        }
    }

    /// Convert a parsed spec to an HF model spec string
    pub fn to_hf_spec(&self, spec: &ParsedModelSpec) -> Option<String> {
        if spec.is_hf_spec {
            return Some(spec.original.clone());
        }

        if spec.is_local_path {
            return None;
        }

        let alias = spec.alias.as_ref()?;

        let mut hf_spec = format!("hf:{}", alias.repo);

        // Add filename if specified
        if let Some(ref file) = alias.default_file {
            hf_spec.push(':');
            hf_spec.push_str(file);
        }

        Some(hf_spec)
    }
}

/// Registry errors
#[derive(Debug)]
pub enum RegistryError {
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::IoError(e) => write!(f, "IO error: {}", e),
            RegistryError::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Global registry instance
static REGISTRY: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

/// Get the global registry instance
pub fn registry() -> &'static ModelRegistry {
    REGISTRY.get_or_init(|| {
        ModelRegistry::load_embedded().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to load model registry: {}", e);
            ModelRegistry::new()
        })
    })
}

/// Resolve a model name to an HF spec
///
/// This is the main entry point for model resolution.
/// Handles aliases, HF specs, and local paths.
pub fn resolve_model_name(name: &str) -> ResolvedModel {
    let reg = registry();
    let spec = reg.parse_spec(name);

    if spec.is_local_path {
        return ResolvedModel::LocalPath(PathBuf::from(&spec.name));
    }

    if spec.is_hf_spec {
        return ResolvedModel::HuggingFace {
            spec: spec.original,
            mmproj: None,
        };
    }

    if let Some(ref alias) = spec.alias {
        let hf_spec = reg.to_hf_spec(&spec).unwrap_or_else(|| {
            format!("hf:{}", alias.repo)
        });

        return ResolvedModel::HuggingFace {
            spec: hf_spec,
            mmproj: alias.mmproj.clone(),
        };
    }

    // Unknown alias - treat as potential HF repo
    ResolvedModel::Unknown(name.to_string())
}

/// Result of resolving a model name
#[derive(Debug, Clone)]
pub enum ResolvedModel {
    /// A local file path
    LocalPath(PathBuf),

    /// A HuggingFace model spec
    HuggingFace {
        spec: String,
        mmproj: Option<String>,
    },

    /// Unknown model name (not in registry)
    Unknown(String),
}

impl ResolvedModel {
    /// Check if this is a local path
    pub fn is_local(&self) -> bool {
        matches!(self, ResolvedModel::LocalPath(_))
    }

    /// Check if this is a HuggingFace model
    pub fn is_hf(&self) -> bool {
        matches!(self, ResolvedModel::HuggingFace { .. })
    }

    /// Check if this is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, ResolvedModel::Unknown(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_embedded_registry() {
        let reg = ModelRegistry::load_embedded().unwrap();
        assert!(!reg.aliases.is_empty());
    }

    #[test]
    fn test_parse_alias() {
        let reg = ModelRegistry::load_embedded().unwrap();
        let spec = reg.parse_spec("llama3.2:1b");

        assert_eq!(spec.name, "llama3.2:1b");
        assert!(spec.alias.is_some());
        assert!(!spec.is_hf_spec);
        assert!(!spec.is_local_path);
    }

    #[test]
    fn test_parse_with_quantization() {
        let reg = ModelRegistry::load_embedded().unwrap();
        let spec = reg.parse_spec("llama3.2:1b-q8");

        assert_eq!(spec.name, "llama3.2:1b");
        assert_eq!(spec.quantization, Some("q8".to_string()));
    }

    #[test]
    fn test_parse_hf_spec() {
        let reg = ModelRegistry::load_embedded().unwrap();
        let spec = reg.parse_spec("hf:TheBloke/Llama-2-7B-GGUF");

        assert!(spec.is_hf_spec);
        assert!(!spec.is_local_path);
        assert!(spec.alias.is_none());
    }

    #[test]
    fn test_parse_local_path() {
        let reg = ModelRegistry::load_embedded().unwrap();

        let spec1 = reg.parse_spec("./model.gguf");
        assert!(spec1.is_local_path);

        let spec2 = reg.parse_spec("/home/user/model.gguf");
        assert!(spec2.is_local_path);

        let spec3 = reg.parse_spec("model.gguf");
        assert!(spec3.is_local_path);
    }

    #[test]
    fn test_search() {
        let reg = ModelRegistry::load_embedded().unwrap();

        let results = reg.search("llama");
        assert!(!results.is_empty());

        let results = reg.search("coding");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_resolve_alias() {
        let resolved = resolve_model_name("llama3.2:1b");
        assert!(resolved.is_hf());
    }

    #[test]
    fn test_resolve_local() {
        let resolved = resolve_model_name("./model.gguf");
        assert!(resolved.is_local());
    }
}
