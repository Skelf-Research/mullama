# Model API

The `Model` struct represents a loaded GGUF model.

## Creating Models

### `Model::load`

Load a model from a file path.

```rust
pub fn load(path: &str) -> Result<Self, MullamaError>
```

**Parameters:**
- `path` - Path to the GGUF model file

**Returns:** `Result<Model, MullamaError>`

**Example:**
```rust
let model = Model::load("llama-7b-q4.gguf")?;
```

---

### `ModelBuilder`

Builder pattern for advanced model configuration.

```rust
let model = ModelBuilder::new("model.gguf")
    .with_n_gpu_layers(35)
    .with_use_mmap(true)
    .with_use_mlock(false)
    .build()?;
```

#### Builder Methods

| Method | Type | Description |
|--------|------|-------------|
| `with_n_gpu_layers(n)` | `i32` | Layers to offload to GPU |
| `with_use_mmap(enable)` | `bool` | Memory-map the model file |
| `with_use_mlock(enable)` | `bool` | Lock model in RAM |
| `with_vocab_only(enable)` | `bool` | Load vocabulary only |

---

## Model Properties

### `vocab_size`

Get the vocabulary size.

```rust
pub fn vocab_size(&self) -> i32
```

**Example:**
```rust
println!("Vocabulary: {} tokens", model.vocab_size());
```

---

### `n_ctx_train`

Get the training context length.

```rust
pub fn n_ctx_train(&self) -> u32
```

---

### `n_embd`

Get the embedding dimension.

```rust
pub fn n_embd(&self) -> i32
```

---

### `n_layer`

Get the number of transformer layers.

```rust
pub fn n_layer(&self) -> i32
```

---

### `description`

Get the model description from metadata.

```rust
pub fn description(&self) -> Option<String>
```

---

## Tokenization

### `tokenize`

Convert text to tokens.

```rust
pub fn tokenize(
    &self,
    text: &str,
    add_bos: bool,
    parse_special: bool
) -> Result<Vec<i32>, MullamaError>
```

**Parameters:**
- `text` - Input text
- `add_bos` - Add beginning-of-sequence token
- `parse_special` - Parse special tokens in text

**Example:**
```rust
let tokens = model.tokenize("Hello, world!", true, false)?;
println!("Tokens: {:?}", tokens);
```

---

### `token_to_str`

Convert a token to its string representation.

```rust
pub fn token_to_str(
    &self,
    token: i32,
    lstrip: i32,
    special: bool
) -> Result<String, MullamaError>
```

**Parameters:**
- `token` - Token ID
- `lstrip` - Strip leading whitespace (0 = none)
- `special` - Include special token text

**Example:**
```rust
let text = model.token_to_str(token_id, 0, false)?;
```

---

## Special Tokens

### `bos_token`

Get the beginning-of-sequence token ID.

```rust
pub fn bos_token(&self) -> Option<i32>
```

---

### `eos_token`

Get the end-of-sequence token ID.

```rust
pub fn eos_token(&self) -> Option<i32>
```

---

### `token_is_eog`

Check if a token is end-of-generation.

```rust
pub fn token_is_eog(&self, token: i32) -> bool
```

**Example:**
```rust
if model.token_is_eog(token) {
    println!("Generation complete");
}
```

---

## Chat Templates

### `apply_chat_template`

Format messages using the model's chat template.

```rust
pub fn apply_chat_template(
    &self,
    messages: &[ChatMessage]
) -> Result<String, MullamaError>
```

**Example:**
```rust
use mullama::ChatMessage;

let messages = vec![
    ChatMessage::system("You are helpful."),
    ChatMessage::user("Hello!"),
];

let prompt = model.apply_chat_template(&messages)?;
```

---

## Thread Safety

`Model` is `Send + Sync` and can be safely shared between threads when wrapped in `Arc`:

```rust
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);

// Clone for use in multiple threads
let model2 = Arc::clone(&model);
```

---

## Memory Management

Models are automatically freed when dropped. The `Drop` implementation calls the underlying `llama_free_model`.

For manual control, wrap in `Arc` and manage lifetime explicitly.
