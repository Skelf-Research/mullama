---
title: Model API
description: Model loading, management, tokenization, and metadata access
---

# Model API

The `Model` struct is the foundational type in Mullama, representing a loaded GGUF language model. Models are reference-counted and thread-safe, designed to be shared across multiple contexts and threads.

## Model Struct

```rust
/// Represents a loaded LLM model.
///
/// Models are reference-counted via Arc and can be safely cloned and shared
/// across threads. The underlying C++ model is freed when the last reference
/// is dropped.
#[derive(Debug, Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}
```

**Thread Safety:** `Model` implements `Send + Sync` and uses `Arc` internally. Cloning a `Model` is a cheap reference count increment -- the underlying model data is shared.

!!! note "Node.js / Python Equivalents"
    - **Node.js:** `const model = await Model.load("model.gguf")`
    - **Python:** `model = Model.load("model.gguf")`

## Loading Models

### `Model::load`

Load a model from a GGUF file with default parameters.

```rust
pub fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `impl AsRef<Path>` | -- | Path to the GGUF model file |

**Returns:** `Result<Model, MullamaError>`

**Errors:**

- `MullamaError::ModelLoadError` -- File not found, invalid GGUF format, or insufficient memory
- `MullamaError::IoError` -- Filesystem access failure

**Example:**

```rust
use mullama::Model;

let model = Model::load("models/llama-7b.gguf")?;
println!("Loaded model with {} parameters", model.n_params());
println!("Vocabulary size: {}", model.n_vocab());
println!("Embedding dimension: {}", model.n_embd());
```

### `Model::load_with_params`

Load a model with custom parameters for fine-grained control over GPU offloading, memory mapping, and other options.

```rust
pub fn load_with_params(
    path: impl AsRef<Path>,
    params: ModelParams,
) -> Result<Self, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `impl AsRef<Path>` | -- | Path to the GGUF model file |
| `params` | `ModelParams` | -- | Custom loading parameters |

**Returns:** `Result<Model, MullamaError>`

**Errors:**

- `MullamaError::ModelLoadError` -- Loading failure due to parameters or file issues
- `MullamaError::GpuError` -- GPU layer offloading failure
- `MullamaError::IoError` -- Filesystem access failure

**Example:**

```rust
use mullama::{Model, ModelParams};

let params = ModelParams {
    n_gpu_layers: 32,
    use_mmap: true,
    use_mlock: false,
    ..Default::default()
};

let model = Model::load_with_params("models/llama-7b.gguf", params)?;
```

## ModelParams

Configuration for model loading behavior.

```rust
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub n_gpu_layers: i32,
    pub split_mode: llama_split_mode,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub kv_overrides: Vec<ModelKvOverride>,
    pub progress_callback: Option<fn(f32) -> bool>,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_gpu_layers` | `i32` | `0` | Number of layers to offload to GPU. Set to `999` to offload all layers. |
| `split_mode` | `llama_split_mode` | `NONE` | How to split the model across multiple GPUs (NONE, LAYER, ROW) |
| `main_gpu` | `i32` | `0` | Main GPU device index for single-GPU or primary device |
| `tensor_split` | `Vec<f32>` | `[]` | Proportion of model to assign to each GPU (must sum to 1.0) |
| `vocab_only` | `bool` | `false` | Load only vocabulary (for tokenization-only use cases) |
| `use_mmap` | `bool` | `true` | Use memory-mapped file I/O for faster loading and lower RAM usage |
| `use_mlock` | `bool` | `false` | Lock model memory to prevent swapping to disk |
| `check_tensors` | `bool` | `true` | Validate tensor data integrity on load (slight overhead) |
| `use_extra_bufts` | `bool` | `false` | Use extra buffer types for specialized backends |
| `kv_overrides` | `Vec<ModelKvOverride>` | `[]` | Override model metadata key-value pairs |
| `progress_callback` | `Option<fn(f32) -> bool>` | `None` | Progress callback during loading (return `false` to cancel) |

!!! note "GPU Layer Offloading"
    When `n_gpu_layers` is 0, the `split_mode` and `main_gpu` fields are ignored to avoid "invalid value for main_gpu" errors when no GPU is available. Set `n_gpu_layers` to a positive number to enable GPU acceleration.

### ModelKvOverride

Override model metadata values at load time. This allows customizing model behavior without modifying the GGUF file.

```rust
#[derive(Debug, Clone)]
pub struct ModelKvOverride {
    pub key: String,
    pub value: ModelKvOverrideValue,
}

#[derive(Debug, Clone)]
pub enum ModelKvOverrideValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}
```

**Example:**

```rust
use mullama::{Model, ModelParams, ModelKvOverride, ModelKvOverrideValue};

let params = ModelParams {
    kv_overrides: vec![
        ModelKvOverride {
            key: "tokenizer.chat_template".to_string(),
            value: ModelKvOverrideValue::Str("custom template".to_string()),
        },
    ],
    ..Default::default()
};

let model = Model::load_with_params("model.gguf", params)?;
```

## ModelBuilder

Fluent API for model configuration with validation and sensible defaults.

```rust
use mullama::builder::ModelBuilder;

let model = ModelBuilder::new()
    .path("models/llama-7b.gguf")
    .gpu_layers(32)
    .context_size(4096)
    .memory_mapping(true)
    .build()
    .await?;
```

### Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `new()` | -- | Create a new builder |
| `path(p)` | `impl Into<String>` | Set model file path (required) |
| `gpu_layers(n)` | `i32` | Set GPU layer offload count |
| `context_size(n)` | `u32` | Set context size in tokens |
| `memory_mapping(b)` | `bool` | Enable/disable mmap |
| `memory_lock(b)` | `bool` | Enable/disable mlock |
| `vocab_only(b)` | `bool` | Load vocabulary only |
| `check_tensors(b)` | `bool` | Enable tensor validation |
| `progress_callback(f)` | `fn(f32) -> bool` | Set loading progress callback |
| `build()` | -- | Build and load the model |

## Model Metadata Methods

Access model architecture and configuration information. These methods are safe to call from any thread.

### `n_params`

```rust
pub fn n_params(&self) -> u64
```

Returns the total number of parameters in the model.

### `n_vocab` / `vocab_size`

```rust
pub fn n_vocab(&self) -> usize
```

Returns the vocabulary size (number of unique tokens). Aliased as `vocab_size()`.

### `n_embd`

```rust
pub fn n_embd(&self) -> u32
```

Returns the embedding dimension (hidden size).

### `n_layer`

```rust
pub fn n_layer(&self) -> u32
```

Returns the number of transformer layers.

### `n_head`

```rust
pub fn n_head(&self) -> usize
```

Returns the number of attention heads.

### `n_ctx_train`

```rust
pub fn n_ctx_train(&self) -> u32
```

Returns the training context length (maximum sequence length the model was trained with).

### `model_type`

```rust
pub fn model_type(&self) -> String
```

Returns the model architecture type (e.g., "llama", "mistral", "phi").

### `model_desc`

```rust
pub fn model_desc(&self) -> String
```

Returns a human-readable model description including parameter count and quantization.

**Example:**

```rust
use mullama::Model;

let model = Model::load("model.gguf")?;
println!("Parameters: {}", model.n_params());
println!("Vocab size: {}", model.n_vocab());
println!("Embedding dim: {}", model.n_embd());
println!("Layers: {}", model.n_layer());
println!("Heads: {}", model.n_head());
println!("Training ctx: {}", model.n_ctx_train());
println!("Type: {}", model.model_type());
println!("Description: {}", model.model_desc());
```

## Tokenization

### `tokenize`

Convert text to a sequence of token IDs.

```rust
pub fn tokenize(
    &self,
    text: &str,
    add_bos: bool,
    special: bool,
) -> Result<Vec<TokenId>, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `&str` | -- | Input text to tokenize |
| `add_bos` | `bool` | -- | Prepend beginning-of-sequence token |
| `special` | `bool` | -- | Parse special tokens (e.g., `<|system|>`) in the text |

**Returns:** `Result<Vec<TokenId>, MullamaError>`

**Errors:** `MullamaError::TokenizationError` -- Invalid text or internal tokenizer failure.

**Example:**

```rust
let tokens = model.tokenize("Hello, world!", true, false)?;
println!("Token count: {}", tokens.len());
println!("Tokens: {:?}", tokens);
```

### `detokenize`

Convert a sequence of token IDs back to text.

```rust
pub fn detokenize(
    &self,
    tokens: &[TokenId],
    remove_special: bool,
    unparse_special: bool,
) -> Result<String, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `&[TokenId]` | -- | Token IDs to convert back to text |
| `remove_special` | `bool` | -- | Skip control/special tokens in output |
| `unparse_special` | `bool` | -- | Render special tokens as their text markers |

**Returns:** `Result<String, MullamaError>`

!!! note "SentencePiece Leading Space"
    SentencePiece-based tokenizers include a leading space marker on the first token. If you need exact round-trip fidelity for text that did not start with a space, you may need to strip the leading space from the result.

**Example:**

```rust
let tokens = model.tokenize("Hello, world!", true, false)?;
let text = model.detokenize(&tokens, true, false)?;
assert_eq!(text.trim(), "Hello, world!");
```

### `token_to_str`

Convert a single token ID to its text representation.

```rust
pub fn token_to_str(
    &self,
    token: TokenId,
    lstrip: i32,
    special: bool,
) -> Result<String, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `token` | `TokenId` | -- | Token ID to convert |
| `lstrip` | `i32` | -- | Number of leading spaces to strip (for SentencePiece) |
| `special` | `bool` | -- | Render special tokens as text |

**Returns:** `Result<String, MullamaError>`

## Vocabulary Access

### `vocab_info`

Get information about the model's vocabulary.

```rust
pub fn vocab_info(&self) -> VocabInfo
```

### `vocab_get_text`

Get the text representation of a token by ID.

```rust
pub fn vocab_get_text(&self, token: TokenId) -> Option<&str>
```

### `vocab_get_score`

Get the score/probability of a token in the vocabulary.

```rust
pub fn vocab_get_score(&self, token: TokenId) -> f32
```

## Special Tokens

Access the model's special token IDs.

```rust
/// Beginning-of-sequence token
pub fn bos_token(&self) -> TokenId

/// End-of-sequence token
pub fn eos_token(&self) -> TokenId

/// Check if a token is end-of-generation
pub fn token_is_eog(&self, token: TokenId) -> bool

/// Check if a token is a control token
pub fn token_is_control(&self, token: TokenId) -> bool
```

**Example:**

```rust
let bos = model.bos_token();
let eos = model.eos_token();
println!("BOS token ID: {}", bos);
println!("EOS token ID: {}", eos);

// Check during generation
if model.token_is_eog(generated_token) {
    println!("Generation complete");
}
```

## Chat Templates

### `apply_chat_template`

Apply the model's built-in chat template to format messages for instruction-tuned models.

```rust
pub fn apply_chat_template(
    &self,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `messages` | `&[ChatMessage]` | -- | Conversation messages with roles |
| `add_generation_prompt` | `bool` | -- | Append the assistant generation prompt marker |

**Example:**

```rust
use mullama::{Model, ChatMessage};

let model = Model::load("model.gguf")?;

let messages = vec![
    ChatMessage { role: "system".into(), content: "You are helpful.".into() },
    ChatMessage { role: "user".into(), content: "Hello!".into() },
];

let prompt = model.apply_chat_template(&messages, true)?;
println!("{}", prompt);
```

## Context Creation

### `create_context`

Convenience method to create an inference context from this model.

```rust
pub fn create_context(&self, params: ContextParams) -> Result<Context, MullamaError>
```

**Example:**

```rust
use mullama::{Model, ContextParams};

let model = Model::load("model.gguf")?;
let ctx = model.create_context(ContextParams::default())?;
```

## Thread Safety

`Model` is designed for safe concurrent access:

```rust
use mullama::Model;
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::load("model.gguf")?);

// Share across threads safely
let model_clone = model.clone(); // Cheap Arc clone
thread::spawn(move || {
    let tokens = model_clone.tokenize("Hello", true, false).unwrap();
    println!("Tokens: {:?}", tokens);
});
```

- **Clone** -- Cheap reference count increment (Arc-based)
- **Send** -- Can be moved between threads
- **Sync** -- Can be shared between threads via `&Model`

## Memory Considerations

| Aspect | Behavior |
|--------|----------|
| **Model size** | Determined by GGUF file size and quantization level |
| **mmap mode** | Model stays on disk, pages loaded on demand (lower RSS) |
| **mlock mode** | Entire model locked in RAM (prevents swapping, higher RSS) |
| **GPU offload** | Layers on GPU consume VRAM, reduce RAM usage proportionally |
| **Drop behavior** | Model freed when last `Arc` reference is dropped |

!!! tip "Memory-Mapped Loading"
    With `use_mmap: true` (the default), the OS memory-maps the model file. This means the model loads almost instantly and the OS manages which pages are resident in RAM. For production deployments with limited RAM, this is the recommended approach.
