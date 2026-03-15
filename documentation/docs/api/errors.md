---
title: Error Handling
description: MullamaError enum, error variants, recovery strategies, and best practices
---

# Error Handling

Mullama uses a unified error type `MullamaError` for all fallible operations. The error system provides detailed context for each failure mode, conversion from common error types, and clear recovery strategies.

## MullamaError

The central error enum covering all possible failure modes in the library.

```rust
#[derive(Debug)]
pub enum MullamaError {
    // Core errors
    ModelLoadError(String),
    ContextError(String),
    SamplingError(String),
    TokenizationError(String),
    GenerationError(String),
    EmbeddingError(String),

    // Resource errors
    IoError(std::io::Error),
    InvalidParameter(String),
    InvalidInput(String),
    OutOfMemory(String),
    GpuError(String),

    // Feature-gated errors
    StreamingError(String),
    WebError(String),
    WebSocketError(String),
    AudioError(String),
    ImageError(String),
    FormatConversionError(String),

    // Advanced feature errors
    SessionError(String),
    LoraError(String),
    GrammarError(String),
    ConfigError(String),
    DaemonError(String),

    // General
    OperationFailed(String),
    FeatureDisabled(String),
}
```

## Error Variants

### Core Errors

#### `ModelLoadError`

Model loading failed due to file issues, format problems, or resource constraints.

**Causes:**

- Model file not found or not accessible
- Invalid or corrupted GGUF file format
- Insufficient RAM for model loading
- Unsupported model architecture

**Recovery:**

- Verify file path exists and is readable
- Check file integrity (re-download if corrupted)
- Reduce `n_gpu_layers` or use a smaller quantization
- Ensure sufficient available RAM

```rust
match Model::load("model.gguf") {
    Ok(model) => { /* success */ }
    Err(MullamaError::ModelLoadError(msg)) => {
        if msg.contains("not found") {
            eprintln!("Model file missing: {}", msg);
        } else if msg.contains("memory") {
            eprintln!("Not enough RAM. Try a smaller model.");
        } else {
            eprintln!("Load failed: {}", msg);
        }
    }
    Err(e) => eprintln!("Unexpected: {}", e),
}
```

#### `ContextError`

Context creation or operation failed.

**Causes:**

- Insufficient memory for KV-cache allocation
- Invalid context parameters (e.g., n_ctx too large)
- Internal llama.cpp context failure

**Recovery:**

- Reduce `n_ctx` to use less memory
- Use quantized KV-cache (`KvCacheType::Q8_0` or `Q4_0`)
- Reduce `n_batch` size
- Ensure GPU VRAM is available if `offload_kqv` is true

```rust
match Context::new(model, params) {
    Ok(ctx) => { /* success */ }
    Err(MullamaError::ContextError(msg)) => {
        eprintln!("Context creation failed: {}", msg);
        // Retry with smaller context
        let fallback_params = ContextParams {
            n_ctx: 1024,
            type_k: KvCacheType::Q4_0,
            type_v: KvCacheType::Q4_0,
            ..Default::default()
        };
        let ctx = Context::new(model, fallback_params)?;
    }
    Err(e) => return Err(e),
}
```

#### `SamplingError`

Token sampling operation failed.

**Causes:**

- Invalid sampler configuration (e.g., temperature < 0)
- Grammar sampler received invalid GBNF
- Sampler chain is empty (no distribution sampler)
- All tokens filtered out by overly aggressive sampling

**Recovery:**

- Validate sampler parameters before building chain
- Check grammar syntax
- Ensure at least one token survives filtering (increase `min_keep`)
- Use `SamplerParams::default()` as a known-good baseline

#### `TokenizationError`

Text tokenization or detokenization failed.

**Causes:**

- Empty input text
- Input text too long for internal buffer
- Invalid UTF-8 in input
- Token ID out of vocabulary range (for detokenization)

**Recovery:**

- Validate input text is non-empty and valid UTF-8
- Split very long texts into chunks
- Check token IDs are within `0..model.n_vocab()`

#### `GenerationError`

Token generation or decoding failed during inference.

**Causes:**

- Batch decode returned an error code
- Context window overflow (tokens exceed `n_ctx`)
- Empty prompt tokens

**Recovery:**

- Clear the KV-cache and retry with shorter prompt
- Increase `n_ctx` in context parameters
- Ensure prompt is not empty

#### `EmbeddingError`

Embedding generation failed.

**Causes:**

- Context not configured for embedding mode (`embeddings: false`)
- Model does not support embeddings
- Empty input for embedding

**Recovery:**

- Set `embeddings: true` in `ContextParams`
- Use an embedding-specific model (nomic-embed, BGE, etc.)
- Validate input is non-empty

### Resource Errors

#### `IoError`

Filesystem or I/O operation failed. Wraps `std::io::Error`.

**Causes:**

- File not found
- Permission denied
- Disk full (for save operations)
- Network error (for URL-based loading)

**Recovery:**

- Check file permissions
- Verify path exists
- Check available disk space
- Retry transient network errors

#### `InvalidParameter`

A function received an invalid parameter value.

**Causes:**

- Parameter out of valid range
- Conflicting parameter combination
- Required parameter missing

**Recovery:**

- Check parameter documentation for valid ranges
- Use default values as starting point

#### `InvalidInput`

Invalid input data provided to a function.

**Causes:**

- Empty or malformed input
- Wrong data format
- Dimension mismatch

**Recovery:**

- Validate input before calling API functions

#### `OutOfMemory`

System ran out of available memory.

**Causes:**

- Model too large for available RAM
- KV-cache allocation exceeds memory
- Too many contexts created simultaneously

**Recovery:**

- Use smaller model or higher quantization
- Reduce context size (`n_ctx`)
- Use quantized KV-cache
- Close unused contexts

#### `GpuError`

GPU operation failed.

**Causes:**

- Insufficient VRAM
- GPU driver error
- CUDA/Metal/ROCm initialization failure
- GPU device not found

**Recovery:**

- Reduce `n_gpu_layers`
- Update GPU drivers
- Verify GPU is present and accessible
- Fall back to CPU (set `n_gpu_layers: 0`)

### Feature-Gated Errors

#### `StreamingError`

Error during token streaming. Requires `streaming` feature.

**Causes:**

- Stream timeout exceeded
- Internal channel closed
- Backpressure buffer overflow

**Recovery:**

- Increase `timeout_ms` in `StreamConfig`
- Consume tokens faster or increase `buffer_size`
- Check if cancellation was triggered

#### `WebError`

Web server error. Requires `web` feature.

**Causes:**

- Port already in use
- TLS certificate invalid
- Request parsing failure
- Server shutdown error

**Recovery:**

- Use a different port
- Check TLS certificate and key files
- Validate request format

#### `WebSocketError`

WebSocket connection error. Requires `websockets` feature.

**Causes:**

- Connection refused or dropped
- Protocol error
- Message too large
- Authentication failure

**Recovery:**

- Reconnect with backoff
- Check message size limits
- Verify authentication credentials

#### `AudioError`

Audio processing error. Requires `multimodal` or `streaming-audio` feature.

**Causes:**

- Unsupported audio format
- Audio device not available
- Sample rate conversion failure
- Audio buffer underrun

**Recovery:**

- Convert audio to a supported format (WAV, FLAC)
- Check system audio device availability
- Adjust buffer sizes

#### `ImageError`

Image processing error. Requires `multimodal` feature.

**Causes:**

- Unsupported image format
- Image too large for processing
- Corrupt image data
- Decoding failure

**Recovery:**

- Convert to supported format (JPEG, PNG, WebP)
- Resize image before processing
- Validate image file integrity

#### `FormatConversionError`

Format conversion between audio/image types failed. Requires `format-conversion` feature.

**Causes:**

- Unsupported conversion path
- Data corruption during conversion
- Missing codec

**Recovery:**

- Check supported conversion pairs
- Validate source data
- Install required system codecs

### Advanced Feature Errors

#### `SessionError`

Session management error.

**Causes:**

- Session state file corrupt
- Session ID not found
- State version mismatch

**Recovery:**

- Create a new session
- Delete corrupt state file
- Ensure version compatibility

#### `LoraError`

LoRA adapter error.

**Causes:**

- LoRA file format invalid
- Dimension mismatch with base model
- LoRA scaling factor invalid

**Recovery:**

- Verify LoRA file matches base model architecture
- Check adapter dimensions
- Use valid scaling factor

#### `GrammarError`

Grammar parsing or application error.

**Causes:**

- Invalid GBNF syntax
- Undefined rule reference
- Grammar cannot generate any valid output
- Infinite recursion in grammar rules

**Recovery:**

- Validate GBNF syntax
- Check all rule references are defined
- Simplify grammar structure
- Test grammar with known inputs

#### `ConfigError`

Configuration loading or validation error.

**Causes:**

- JSON/YAML parse error
- Missing required field
- Invalid field value
- File not found

**Recovery:**

- Fix JSON/YAML syntax
- Provide all required fields
- Check field value ranges
- Verify config file path

#### `DaemonError`

Daemon process error.

**Causes:**

- Daemon already running
- Port conflict
- Model management failure
- IPC communication error

**Recovery:**

- Stop existing daemon instance
- Use a different port
- Check model availability
- Restart daemon

#### `FeatureDisabled`

Operation requires a feature that is not enabled.

**Causes:**

- Calling feature-gated API without the corresponding feature enabled

**Recovery:**

- Add the required feature to your `Cargo.toml`
- Check the error message for the specific feature needed

```rust
Err(MullamaError::FeatureDisabled(msg)) => {
    eprintln!("Feature not available: {}", msg);
    // The error message tells you which feature to enable
}
```

## Error Conversions

`MullamaError` implements `From` for common error types:

```rust
impl From<std::io::Error> for MullamaError { ... }        // -> IoError
impl From<serde_json::Error> for MullamaError { ... }     // -> ConfigError
impl From<serde_yaml::Error> for MullamaError { ... }     // -> ConfigError
impl From<std::string::FromUtf8Error> for MullamaError { ... } // -> TokenizationError
```

This enables the `?` operator for automatic conversion:

```rust
fn load_and_process(path: &str) -> Result<String, MullamaError> {
    let content = std::fs::read_to_string(path)?; // IoError auto-converted
    let config: MullamaConfig = serde_json::from_str(&content)?; // ConfigError auto-converted
    // ...
    Ok("done".to_string())
}
```

## Display and Debug

`MullamaError` implements both `Display` and `Debug`:

```rust
impl std::fmt::Display for MullamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            Self::ContextError(msg) => write!(f, "Context error: {}", msg),
            Self::GpuError(msg) => write!(f, "GPU error: {}", msg),
            // ... all variants
        }
    }
}

impl std::error::Error for MullamaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}
```

## Best Practices

### 1. Use the `?` Operator

```rust
fn process(model_path: &str) -> Result<String, MullamaError> {
    let model = Model::load(model_path)?;
    let ctx = Context::new(Arc::new(model.clone()), ContextParams::default())?;
    let tokens = model.tokenize("Hello", true, false)?;
    // All errors propagate naturally
    Ok("success".to_string())
}
```

### 2. Match on Specific Variants for Recovery

```rust
fn load_with_fallback(path: &str) -> Result<Model, MullamaError> {
    match Model::load(path) {
        Ok(model) => Ok(model),
        Err(MullamaError::OutOfMemory(_)) => {
            // Try with less GPU offloading
            let params = ModelParams { n_gpu_layers: 0, ..Default::default() };
            Model::load_with_params(path, params)
        }
        Err(e) => Err(e), // Propagate other errors
    }
}
```

### 3. Provide Context with `map_err`

```rust
fn load_model(config: &Config) -> Result<Model, MullamaError> {
    Model::load(&config.model_path)
        .map_err(|e| MullamaError::ModelLoadError(
            format!("Failed to load model '{}': {}", config.model_path, e)
        ))
}
```

### 4. Use Result Combinators

```rust
let output = model.tokenize(text, true, false)
    .and_then(|tokens| ctx.generate(&tokens, 100))
    .unwrap_or_else(|e| {
        eprintln!("Generation failed: {}", e);
        String::new()
    });
```

### 5. Never Panic in Library Code

```rust
// WRONG: panic on error
let model = Model::load(path).unwrap();

// CORRECT: propagate errors
let model = Model::load(path)?;

// CORRECT: handle gracefully
let model = match Model::load(path) {
    Ok(m) => m,
    Err(e) => {
        log::error!("Model load failed: {}", e);
        return Err(e);
    }
};
```

## Error Recovery Matrix

| Error Type | Retry? | Fallback Strategy |
|------------|--------|-------------------|
| `ModelLoadError` | No | Try smaller model, check path |
| `ContextError` | Yes (with smaller params) | Reduce n_ctx, use Q4 KV-cache |
| `SamplingError` | No | Reset to default params |
| `TokenizationError` | No | Validate input text |
| `GenerationError` | Yes | Clear cache, shorter prompt |
| `IoError` | Maybe | Check permissions, path |
| `OutOfMemory` | Yes (with less resources) | Reduce model/context size |
| `GpuError` | Yes | Fall back to CPU |
| `StreamingError` | Yes | Increase timeout/buffer |
| `ConfigError` | No | Fix configuration file |
