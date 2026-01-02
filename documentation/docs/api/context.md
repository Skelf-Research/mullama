# Context API

The `Context` struct manages inference state for text generation.

## Creating Contexts

### `Context::new`

Create a new inference context.

```rust
pub fn new(
    model: Arc<Model>,
    params: ContextParams
) -> Result<Self, MullamaError>
```

**Parameters:**
- `model` - Shared reference to a loaded model
- `params` - Context configuration

**Example:**
```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);
let ctx = Context::new(model, ContextParams::default())?;
```

---

## ContextParams

Configuration for context creation.

```rust
pub struct ContextParams {
    pub n_ctx: u32,           // Context window size
    pub n_batch: u32,         // Batch size for prompt processing
    pub n_ubatch: u32,        // Micro-batch size
    pub n_threads: u32,       // CPU threads for generation
    pub n_threads_batch: u32, // CPU threads for batching
    pub flash_attn: bool,     // Enable flash attention
    pub embeddings: bool,     // Enable embedding mode
    pub rope_freq_base: f32,  // RoPE frequency base
    pub rope_freq_scale: f32, // RoPE frequency scale
}
```

**Defaults:**
```rust
ContextParams {
    n_ctx: 0,              // Use model default
    n_batch: 2048,
    n_ubatch: 512,
    n_threads: num_cpus,
    n_threads_batch: num_cpus,
    flash_attn: true,
    embeddings: false,
    rope_freq_base: 0.0,   // Auto
    rope_freq_scale: 0.0,  // Auto
}
```

---

## Text Generation

### `generate`

Generate text from a prompt.

```rust
pub fn generate(
    &mut self,
    prompt: &str,
    max_tokens: usize
) -> Result<String, MullamaError>
```

**Parameters:**
- `prompt` - Input text
- `max_tokens` - Maximum tokens to generate

**Example:**
```rust
let response = ctx.generate("Hello, ", 100)?;
```

---

### `generate_with_params`

Generate with custom sampling parameters.

```rust
pub fn generate_with_params(
    &mut self,
    prompt: &str,
    max_tokens: usize,
    params: SamplingParams
) -> Result<String, MullamaError>
```

**Example:**
```rust
let params = SamplingParams {
    temperature: 0.8,
    top_p: 0.9,
    ..Default::default()
};

let response = ctx.generate_with_params("Hello", 100, params)?;
```

---

### `generate_streaming`

Stream tokens as they're generated.

```rust
pub fn generate_streaming<F>(
    &mut self,
    prompt: &str,
    max_tokens: usize,
    callback: F
) -> Result<(), MullamaError>
where
    F: FnMut(&str) -> bool
```

**Parameters:**
- `prompt` - Input text
- `max_tokens` - Maximum tokens
- `callback` - Called for each token, return `false` to stop

**Example:**
```rust
ctx.generate_streaming("Hello", 100, |token| {
    print!("{}", token);
    true
})?;
```

---

### `generate_continue`

Continue generation from current position.

```rust
pub fn generate_continue(
    &mut self,
    n_past: i32,
    max_tokens: usize
) -> Result<String, MullamaError>
```

---

## Token Operations

### `decode`

Process tokens through the model.

```rust
pub fn decode(&mut self, tokens: &[i32]) -> Result<(), MullamaError>
```

**Example:**
```rust
let tokens = model.tokenize("Hello", true, false)?;
ctx.decode(&tokens)?;
```

---

### `get_logits`

Get logits from the last decode operation.

```rust
pub fn get_logits(&self) -> &[f32]
```

**Returns:** Slice of logits for the last token position.

---

### `get_logits_ith`

Get logits for a specific position.

```rust
pub fn get_logits_ith(&self, i: i32) -> &[f32]
```

---

## Embeddings

### `get_embeddings`

Get text embeddings (requires `embeddings: true` in params).

```rust
pub fn get_embeddings(&self, text: &str) -> Result<Vec<f32>, MullamaError>
```

**Example:**
```rust
let params = ContextParams { embeddings: true, ..Default::default() };
let ctx = Context::new(model, params)?;
let embedding = ctx.get_embeddings("Hello")?;
```

---

## State Management

### `clear`

Clear the context state for a new conversation.

```rust
pub fn clear(&mut self) -> Result<(), MullamaError>
```

---

### `n_past`

Get the current position in the context.

```rust
pub fn n_past(&self) -> i32
```

---

### `n_ctx`

Get the context window size.

```rust
pub fn n_ctx(&self) -> u32
```

---

## KV Cache

### `kv_cache_clear`

Clear the key-value cache.

```rust
pub fn kv_cache_clear(&mut self)
```

---

### `kv_cache_seq_rm`

Remove a sequence from the cache.

```rust
pub fn kv_cache_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32)
```

---

### `kv_cache_seq_cp`

Copy a sequence in the cache.

```rust
pub fn kv_cache_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32)
```

---

## FFI Access

### `as_ptr`

Get the raw context pointer for FFI.

```rust
pub fn as_ptr(&self) -> *mut llama_context
```

!!! warning
    Use with caution. Incorrect use can cause memory unsafety.

---

## Thread Safety

`Context` is `Send` but **not** `Sync`. Use `Mutex` or `RwLock` for shared access:

```rust
use std::sync::{Arc, Mutex};

let ctx = Arc::new(Mutex::new(Context::new(model, params)?));

// In another thread
let mut guard = ctx.lock().unwrap();
guard.generate("Hello", 100)?;
```

---

## Memory Management

Contexts are automatically cleaned up when dropped. The destructor calls `llama_free`.
