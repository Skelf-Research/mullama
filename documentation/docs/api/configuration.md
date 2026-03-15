---
title: Configuration API
description: Serde-based configuration with JSON/YAML support, environment overrides, and validation
---

# Configuration API

The configuration module provides a structured, type-safe system for managing Mullama settings. All configuration structs implement `Serialize` and `Deserialize` for JSON and YAML file support, with environment variable overrides and comprehensive validation.

## MullamaConfig

The top-level configuration struct encompassing all settings.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MullamaConfig {
    pub model: ModelConfig,
    pub context: ContextConfig,
    pub sampler: SamplerConfig,
    pub server: Option<ServerConfig>,
    pub logging: Option<LoggingConfig>,
}
```

### Fields

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model` | `ModelConfig` | Yes | Model loading and resource configuration |
| `context` | `ContextConfig` | Yes | Context parameters for inference |
| `sampler` | `SamplerConfig` | Yes | Sampling strategy configuration |
| `server` | `Option<ServerConfig>` | No | Web server configuration (when using `web` feature) |
| `logging` | `Option<LoggingConfig>` | No | Logging configuration |

### Loading Configuration

```rust
use mullama::config::MullamaConfig;

// From JSON file
let config = MullamaConfig::from_file("config.json")?;

// From YAML file
let config = MullamaConfig::from_file("config.yaml")?;

// From string
let json = r#"{"model": {"path": "model.gguf"}, ...}"#;
let config: MullamaConfig = serde_json::from_str(json)?;

// From environment variables (prefix: MULLAMA_)
let config = MullamaConfig::from_env()?;

// With validation
let config = MullamaConfig::from_file("config.json")?.validate()?;
```

### Methods

```rust
impl MullamaConfig {
    /// Load configuration from a JSON or YAML file (auto-detected by extension)
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, MullamaError>;

    /// Load from environment variables with MULLAMA_ prefix
    pub fn from_env() -> Result<Self, MullamaError>;

    /// Merge with another config (other takes precedence)
    pub fn merge(&self, other: &Self) -> Self;

    /// Validate all fields and return validated config
    pub fn validate(self) -> Result<Self, MullamaError>;

    /// Get default configuration
    pub fn default() -> Self;

    /// Write configuration to a file
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), MullamaError>;
}
```

## ModelConfig

Configuration for model loading behavior, corresponding to `ModelParams`.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub gpu_layers: Option<i32>,
    pub split_mode: Option<String>,
    pub main_gpu: Option<i32>,
    pub tensor_split: Option<Vec<f32>>,
    pub vocab_only: Option<bool>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub check_tensors: Option<bool>,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `String` | (required) | Path to the GGUF model file |
| `gpu_layers` | `Option<i32>` | `0` | Number of layers to offload to GPU |
| `split_mode` | `Option<String>` | `"none"` | GPU split mode: "none", "layer", "row" |
| `main_gpu` | `Option<i32>` | `0` | Primary GPU device index |
| `tensor_split` | `Option<Vec<f32>>` | `None` | GPU tensor split proportions |
| `vocab_only` | `Option<bool>` | `false` | Load vocabulary only |
| `use_mmap` | `Option<bool>` | `true` | Enable memory-mapped loading |
| `use_mlock` | `Option<bool>` | `false` | Lock model in RAM |
| `check_tensors` | `Option<bool>` | `true` | Validate tensor integrity on load |

### Conversion

```rust
impl From<ModelConfig> for ModelParams {
    fn from(config: ModelConfig) -> Self;
}

// Use in code:
let config = ModelConfig { path: "model.gguf".to_string(), ..Default::default() };
let params: ModelParams = config.into();
let model = Model::load_with_params(&config.path, params)?;
```

## ContextConfig

Configuration for inference context parameters.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub flash_attention: Option<bool>,
    pub embeddings: Option<bool>,
    pub offload_kqv: Option<bool>,
    pub kv_cache_type: Option<String>,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_ctx` | `Option<u32>` | `0` (model default) | Context window size |
| `n_batch` | `Option<u32>` | `2048` | Maximum batch size |
| `n_threads` | `Option<i32>` | CPU count | Generation threads |
| `n_threads_batch` | `Option<i32>` | CPU count | Batch processing threads |
| `rope_freq_base` | `Option<f32>` | `0.0` | RoPE base frequency |
| `rope_freq_scale` | `Option<f32>` | `0.0` | RoPE scale factor |
| `flash_attention` | `Option<bool>` | `false` | Enable flash attention |
| `embeddings` | `Option<bool>` | `false` | Enable embedding output |
| `offload_kqv` | `Option<bool>` | `true` | Offload KQV to GPU |
| `kv_cache_type` | `Option<String>` | `"f16"` | Cache type: "f32", "f16", "bf16", "q8_0", "q4_0" |

### Conversion

```rust
impl From<ContextConfig> for ContextParams {
    fn from(config: ContextConfig) -> Self;
}
```

## SamplerConfig

Configuration for sampling strategies.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub penalty_last_n: Option<i32>,
    pub seed: Option<u32>,
    pub mirostat: Option<MirostatConfig>,
    pub grammar: Option<String>,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `temperature` | `Option<f32>` | `0.8` | Sampling temperature |
| `top_k` | `Option<i32>` | `40` | Top-K sampling limit |
| `top_p` | `Option<f32>` | `0.95` | Nucleus sampling threshold |
| `min_p` | `Option<f32>` | `0.05` | Minimum probability threshold |
| `typical_p` | `Option<f32>` | `1.0` | Typical sampling threshold |
| `repeat_penalty` | `Option<f32>` | `1.1` | Repetition penalty |
| `frequency_penalty` | `Option<f32>` | `0.0` | Frequency penalty |
| `presence_penalty` | `Option<f32>` | `0.0` | Presence penalty |
| `penalty_last_n` | `Option<i32>` | `64` | Penalty window size |
| `seed` | `Option<u32>` | Random | Random seed |
| `mirostat` | `Option<MirostatConfig>` | `None` | Mirostat sampling config |
| `grammar` | `Option<String>` | `None` | GBNF grammar for constrained output |

### MirostatConfig

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirostatConfig {
    pub version: u8,       // 1 or 2
    pub tau: f32,          // Target entropy
    pub eta: f32,          // Learning rate
}
```

### Conversion

```rust
impl From<SamplerConfig> for SamplerParams {
    fn from(config: SamplerConfig) -> Self;
}
```

## Environment Variable Overrides

Every configuration field can be overridden via environment variables with the `MULLAMA_` prefix. Nested fields use double underscores:

| Environment Variable | Config Field | Example |
|---------------------|--------------|---------|
| `MULLAMA_MODEL__PATH` | `model.path` | `/path/to/model.gguf` |
| `MULLAMA_MODEL__GPU_LAYERS` | `model.gpu_layers` | `32` |
| `MULLAMA_MODEL__USE_MMAP` | `model.use_mmap` | `true` |
| `MULLAMA_CONTEXT__N_CTX` | `context.n_ctx` | `4096` |
| `MULLAMA_CONTEXT__N_BATCH` | `context.n_batch` | `512` |
| `MULLAMA_CONTEXT__N_THREADS` | `context.n_threads` | `8` |
| `MULLAMA_CONTEXT__FLASH_ATTENTION` | `context.flash_attention` | `true` |
| `MULLAMA_CONTEXT__KV_CACHE_TYPE` | `context.kv_cache_type` | `q8_0` |
| `MULLAMA_SAMPLER__TEMPERATURE` | `sampler.temperature` | `0.7` |
| `MULLAMA_SAMPLER__TOP_K` | `sampler.top_k` | `50` |
| `MULLAMA_SAMPLER__TOP_P` | `sampler.top_p` | `0.9` |
| `MULLAMA_SAMPLER__SEED` | `sampler.seed` | `42` |

**Priority order (highest to lowest):**

1. Environment variables
2. CLI arguments (if using the daemon)
3. Configuration file values
4. Default values

**Example:**

```bash
# Override model path and GPU layers via environment
export MULLAMA_MODEL__PATH="/models/llama-7b-q4.gguf"
export MULLAMA_MODEL__GPU_LAYERS=32
export MULLAMA_CONTEXT__N_CTX=4096
export MULLAMA_SAMPLER__TEMPERATURE=0.7

# Application uses these overrides automatically
cargo run --example simple
```

## Validation

The `validate()` method checks for:

- Model path exists and is readable
- GPU layers does not exceed model layer count (when known)
- Context size is reasonable (0 < n_ctx <= 1048576)
- Batch size is positive and not larger than context size
- Thread count is positive
- Temperature is non-negative
- Top-K is non-negative (0 = disabled)
- Top-P is in range [0.0, 1.0]
- Penalties are in reasonable ranges
- KV cache type string is valid

```rust
use mullama::config::MullamaConfig;

let config = MullamaConfig::from_file("config.json")?;

match config.validate() {
    Ok(valid_config) => {
        // Use valid_config safely
    }
    Err(MullamaError::ConfigError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

## Example JSON Configuration

```json
{
  "model": {
    "path": "models/llama-7b-q4_0.gguf",
    "gpu_layers": 32,
    "use_mmap": true,
    "use_mlock": false,
    "check_tensors": true
  },
  "context": {
    "n_ctx": 4096,
    "n_batch": 512,
    "n_threads": 8,
    "n_threads_batch": 8,
    "flash_attention": true,
    "kv_cache_type": "q8_0",
    "embeddings": false,
    "offload_kqv": true
  },
  "sampler": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "penalty_last_n": 64,
    "seed": 42
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "cors_origins": ["*"],
    "max_connections": 100
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
```

## Example YAML Configuration

```yaml
model:
  path: models/llama-7b-q4_0.gguf
  gpu_layers: 32
  use_mmap: true
  use_mlock: false

context:
  n_ctx: 4096
  n_batch: 512
  n_threads: 8
  flash_attention: true
  kv_cache_type: q8_0

sampler:
  temperature: 0.7
  top_k: 40
  top_p: 0.9
  min_p: 0.05
  repeat_penalty: 1.1
  penalty_last_n: 64

server:
  host: 127.0.0.1
  port: 8080
  cors_origins:
    - "*"

logging:
  level: info
  format: json
```

## ServerConfig

Configuration for the web server (requires `web` feature).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
}
```

## LoggingConfig

Configuration for logging output.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,      // "trace", "debug", "info", "warn", "error"
    pub format: String,     // "text", "json"
    pub file: Option<String>,  // Optional log file path
}
```

## Complete Example

```rust
use mullama::{Model, Context, SamplerParams};
use mullama::config::MullamaConfig;
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load and validate configuration
    let config = MullamaConfig::from_file("mullama.json")?
        .validate()?;

    // Convert config sections to API types
    let model_params = config.model.clone().into();
    let model = Arc::new(Model::load_with_params(&config.model.path, model_params)?);

    let ctx_params = config.context.clone().into();
    let mut ctx = Context::new(model.clone(), ctx_params)?;

    let sampler_params: SamplerParams = config.sampler.clone().into();

    // Use model, context, and sampler...
    let tokens = model.tokenize("Hello, world!", true, false)?;
    let output = ctx.generate_with_params(&tokens, 100, &sampler_params)?;
    println!("{}", output);

    Ok(())
}
```
