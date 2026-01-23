---
title: Context API
description: Inference context management, generation, and KV-cache control
---

# Context API

The `Context` struct manages the inference state for text generation. It holds the KV-cache, processes token batches, and coordinates sampling. Each context is bound to a single model and maintains its own generation state.

## Context Struct

```rust
/// Represents a model context for inference.
///
/// Holds the KV-cache and internal state for token processing.
/// Not thread-safe -- each thread should create its own context.
pub struct Context {
    pub model: Arc<Model>,
    pub ctx_ptr: *mut llama_context,
}
```

!!! warning "Thread Safety"
    `Context` is **not** `Send` or `Sync`. It must not be shared across threads or moved between threads. Each thread should create its own context from a shared `Arc<Model>`. For async use, wrap operations with `AsyncContext` which uses `spawn_blocking` internally.

## Creating Contexts

### `Context::new`

Create a new inference context for a model with specified parameters.

```rust
pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Arc<Model>` | -- | Shared reference to the loaded model |
| `params` | `ContextParams` | -- | Context configuration parameters |

**Returns:** `Result<Context, MullamaError>`

**Errors:**

- `MullamaError::ContextError` -- Failed to allocate context (usually insufficient memory for KV-cache)
- `MullamaError::InvalidInput` -- Invalid parameter combination

**Example:**

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);

let mut params = ContextParams::default();
params.n_ctx = 4096;
params.n_batch = 512;
params.n_threads = 8;

let mut ctx = Context::new(model, params)?;
```

## ContextParams

Full configuration for context behavior including threading, RoPE, flash attention, and KV-cache quantization.

```rust
#[derive(Debug, Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub flash_attn_type: llama_flash_attn_type,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub type_k: KvCacheType,
    pub type_v: KvCacheType,
}
```

### Core Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_ctx` | `u32` | `0` (model default) | Context window size in tokens. Set to 0 to use model's training context length. |
| `n_batch` | `u32` | `2048` | Maximum batch size for prompt processing. Larger values use more memory but process prompts faster. |
| `n_ubatch` | `u32` | `512` | Physical batch size (micro-batch). Internal chunking size for processing. |
| `n_seq_max` | `u32` | `1` | Maximum number of parallel sequences (for beam search or parallel generation). |
| `n_threads` | `i32` | CPU count | Number of threads for single-token generation. |
| `n_threads_batch` | `i32` | CPU count | Number of threads for batch/prompt processing. Can differ from `n_threads`. |

### Attention and RoPE Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `flash_attn_type` | `llama_flash_attn_type` | `AUTO` | Flash attention mode (AUTO, DISABLED, ENABLED). |
| `rope_freq_base` | `f32` | `0.0` (model default) | RoPE base frequency for positional encoding. |
| `rope_freq_scale` | `f32` | `0.0` (model default) | RoPE frequency scaling factor. Values < 1.0 extend context. |
| `rope_scaling_type` | `llama_rope_scaling_type` | `UNSPECIFIED` | RoPE scaling strategy (NONE, LINEAR, YARN). |
| `yarn_ext_factor` | `f32` | `-1.0` | YaRN extension factor. -1.0 uses model default. |
| `yarn_attn_factor` | `f32` | `1.0` | YaRN attention scaling factor. |
| `yarn_beta_fast` | `f32` | `32.0` | YaRN fast interpolation beta parameter. |
| `yarn_beta_slow` | `f32` | `1.0` | YaRN slow interpolation beta parameter. |
| `yarn_orig_ctx` | `u32` | `0` | YaRN original context size (0 = use model default). |
| `pooling_type` | `llama_pooling_type` | `UNSPECIFIED` | Pooling type for embedding models. |
| `attention_type` | `llama_attention_type` | `UNSPECIFIED` | Attention type (CAUSAL, NON_CAUSAL). |

### Memory and Performance Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `embeddings` | `bool` | `false` | Enable embedding output mode. Required for generating embeddings. |
| `offload_kqv` | `bool` | `true` | Offload KQV operations to GPU when GPU layers are used. |
| `type_k` | `KvCacheType` | `F16` | Key cache quantization type. Lower precision saves memory. |
| `type_v` | `KvCacheType` | `F16` | Value cache quantization type. Lower precision saves memory. |
| `defrag_thold` | `f32` | `-1.0` | KV-cache defragmentation threshold (-1.0 = disabled). |
| `no_perf` | `bool` | `false` | Disable performance counters for slightly faster inference. |
| `op_offload` | `bool` | `false` | Enable operation offloading to GPU. |
| `swa_full` | `bool` | `true` | Use full sliding window attention. |
| `kv_unified` | `bool` | `false` | Use unified KV-cache layout. |

## KvCacheType

Controls KV-cache quantization precision for memory optimization. Lower quantization significantly reduces memory usage at the cost of some quality.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheType {
    F32,     // 2x memory vs F16 (highest precision)
    #[default]
    F16,     // 1x memory (default, best balance)
    BF16,    // 1x memory (alternative 16-bit, better for some hardware)
    Q8_0,    // 0.5x memory (~50% savings, minimal quality loss)
    Q4_0,    // 0.25x memory (~75% savings, may affect quality)
}
```

### Memory Factors

| Type | Memory Factor | Quality Impact | Use Case |
|------|--------------|----------------|----------|
| `F32` | 2.0x | None | Maximum precision, debugging |
| `F16` | 1.0x (baseline) | None | Default, recommended for most uses |
| `BF16` | 1.0x | Negligible | Alternative for hardware with BF16 support |
| `Q8_0` | 0.5x | Minimal | Large context windows with limited memory |
| `Q4_0` | 0.25x | Noticeable | Extreme memory constraints, shorter contexts |

**Example with KV-cache quantization:**

```rust
use mullama::{Context, ContextParams, KvCacheType};

let params = ContextParams {
    n_ctx: 8192,
    type_k: KvCacheType::Q8_0,  // 50% KV-cache memory savings
    type_v: KvCacheType::Q8_0,
    ..Default::default()
};
// For a 7B model with 8192 context, Q8_0 saves ~1GB of KV-cache memory
```

### Flash Attention Types

```rust
pub enum llama_flash_attn_type {
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1,     // Auto-detect best setting
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,  // Disable flash attention
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1,   // Enable flash attention
}
```

## ContextBuilder

Fluent API for context creation with validation.

```rust
use mullama::builder::ContextBuilder;

let ctx = ContextBuilder::new(model.clone())
    .context_size(4096)
    .batch_size(512)
    .threads(8)
    .embeddings(true)
    .flash_attention(true)
    .kv_cache_type(KvCacheType::Q8_0)
    .build()
    .await?;
```

### Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `new(model)` | `Arc<Model>` | Create builder with model reference |
| `context_size(n)` | `u32` | Set context window size |
| `batch_size(n)` | `u32` | Set batch processing size |
| `threads(n)` | `i32` | Set thread count for generation |
| `threads_batch(n)` | `i32` | Set thread count for batch ops |
| `embeddings(b)` | `bool` | Enable embedding output |
| `flash_attention(b)` | `bool` | Enable flash attention |
| `kv_cache_type(t)` | `KvCacheType` | Set KV-cache quantization for both K and V |
| `rope_freq_base(f)` | `f32` | Set RoPE base frequency |
| `rope_freq_scale(f)` | `f32` | Set RoPE frequency scale |
| `build()` | -- | Build the context |

## Generation

### `generate`

Generate text from prompt tokens using default sampling parameters.

```rust
pub fn generate(
    &mut self,
    prompt_tokens: &[TokenId],
    max_tokens: usize,
) -> Result<String, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt_tokens` | `&[TokenId]` | -- | Tokenized prompt (use `model.tokenize()`) |
| `max_tokens` | `usize` | -- | Maximum number of tokens to generate |

**Returns:** `Result<String, MullamaError>` -- The generated text (decoded from tokens).

**Errors:**

- `MullamaError::GenerationError` -- Decode failure or empty prompt
- `MullamaError::ContextError` -- Context overflow (prompt + generation > n_ctx)

**Example:**

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);
let mut ctx = Context::new(model.clone(), ContextParams::default())?;

let tokens = model.tokenize("Once upon a time", true, false)?;
let output = ctx.generate(&tokens, 100)?;
println!("{}", output);
```

### `generate_with_params`

Generate text with custom sampling parameters for control over temperature, top-k, top-p, and penalties.

```rust
pub fn generate_with_params(
    &mut self,
    prompt_tokens: &[TokenId],
    max_tokens: usize,
    sampler_params: &SamplerParams,
) -> Result<String, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt_tokens` | `&[TokenId]` | -- | Tokenized prompt |
| `max_tokens` | `usize` | -- | Maximum tokens to generate |
| `sampler_params` | `&SamplerParams` | -- | Custom sampling configuration |

**Example:**

```rust
use mullama::SamplerParams;

let params = SamplerParams {
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9,
    penalty_repeat: 1.1,
    ..Default::default()
};

let output = ctx.generate_with_params(&tokens, 100, &params)?;
```

## Token-Level Operations

### `decode`

Process a batch of tokens through the model. Automatically handles chunking if the token count exceeds `n_batch`.

```rust
pub fn decode(&mut self, tokens: &[TokenId]) -> Result<(), MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `&[TokenId]` | -- | Tokens to process through the model |

**Errors:** `MullamaError::GenerationError` -- Decode failure.

**Example:**

```rust
let tokens = model.tokenize("Hello, world!", true, false)?;
ctx.decode(&tokens)?;
// Logits are now available for sampling
```

### `decode_single`

Optimized single-token decode that avoids heap allocation. Used in generation loops for maximum performance.

```rust
pub fn decode_single(&mut self, token: TokenId) -> Result<(), MullamaError>
```

**Example:**

```rust
// During generation loop, decode one token at a time
ctx.decode_single(next_token)?;
```

### `eval`

Lower-level token evaluation with explicit position tracking.

```rust
pub fn eval(&mut self, tokens: &[TokenId], n_past: i32) -> Result<(), MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `&[TokenId]` | -- | Tokens to evaluate |
| `n_past` | `i32` | -- | Number of previously processed tokens (KV-cache position) |

## KV-Cache Management

### `clear_cache`

Clear the entire KV-cache, resetting the context state. Use this to start a new conversation or prompt.

```rust
pub fn clear_cache(&mut self)
```

### `kv_cache_seq_rm`

Remove tokens from the KV-cache for a specific sequence. Useful for implementing sliding window or trimming old context.

```rust
pub fn kv_cache_seq_rm(
    &mut self,
    seq_id: i32,
    p0: i32,
    p1: i32,
) -> bool
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `seq_id` | `i32` | -- | Sequence ID (-1 for all sequences) |
| `p0` | `i32` | -- | Start position inclusive (-1 for beginning) |
| `p1` | `i32` | -- | End position exclusive (-1 for end) |

**Returns:** `bool` -- Whether the operation succeeded.

### `kv_cache_seq_cp`

Copy a sequence in the KV-cache. Useful for beam search or branching conversations.

```rust
pub fn kv_cache_seq_cp(
    &mut self,
    seq_id_src: i32,
    seq_id_dst: i32,
    p0: i32,
    p1: i32,
)
```

### `kv_cache_seq_shift`

Shift token positions in the KV-cache. Used for implementing context window sliding.

```rust
pub fn kv_cache_seq_shift(
    &mut self,
    seq_id: i32,
    p0: i32,
    p1: i32,
    delta: i32,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `seq_id` | `i32` | -- | Sequence ID |
| `p0` | `i32` | -- | Start position |
| `p1` | `i32` | -- | End position |
| `delta` | `i32` | -- | Position shift amount (negative to shift left) |

## Logits Access

### `get_logits`

Get the logits array for all tokens in the last decoded batch.

```rust
pub fn get_logits(&self) -> &[f32]
```

**Returns:** Slice of logit values with length `n_vocab`.

### `get_logits_ith`

Get logits for a specific token position in the batch.

```rust
pub fn get_logits_ith(&self, i: i32) -> &[f32]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `i` | `i32` | -- | Token index in the batch (-1 for last token) |

## Properties

### `n_ctx`

Get the context window size (number of tokens this context can hold).

```rust
pub fn n_ctx(&self) -> u32
```

### `n_batch`

Get the configured batch size.

```rust
pub fn n_batch(&self) -> u32
```

## Thread Safety Notes

!!! warning "Context is NOT Send"
    `Context` holds a raw pointer to the llama.cpp context and is not safe to send between threads. For multi-threaded applications:

    - Create one `Context` per thread from a shared `Arc<Model>`
    - Use `AsyncContext` (feature: `async`) for non-blocking operations
    - Never share a `Context` reference across thread boundaries

```rust
// CORRECT: Each thread creates its own context
let model = Arc::new(Model::load("model.gguf")?);

let handle = std::thread::spawn({
    let model = model.clone();
    move || {
        let mut ctx = Context::new(model, ContextParams::default()).unwrap();
        ctx.generate(&[1, 2, 3], 50)
    }
});

// INCORRECT: This will not compile
// let ctx = Context::new(model.clone(), ContextParams::default())?;
// std::thread::spawn(move || ctx.generate(&[1], 10)); // Error: Context is !Send
```

## Complete Generation Example

```rust
use mullama::{Model, Context, ContextParams, SamplerParams, SamplerChain};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load model
    let model = Arc::new(Model::load("model.gguf")?);

    // Create context with custom parameters
    let params = ContextParams {
        n_ctx: 2048,
        n_batch: 512,
        n_threads: 8,
        type_k: KvCacheType::Q8_0,
        type_v: KvCacheType::Q8_0,
        ..Default::default()
    };
    let mut ctx = Context::new(model.clone(), params)?;

    // Configure sampling
    let sampler_params = SamplerParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        penalty_repeat: 1.1,
        ..Default::default()
    };
    let mut sampler = sampler_params.build_chain(model.clone())?;

    // Tokenize prompt
    let prompt = "The meaning of life is";
    let tokens = model.tokenize(prompt, true, false)?;

    // Process prompt
    ctx.decode(&tokens)?;

    // Generate tokens one at a time
    print!("{}", prompt);
    for _ in 0..100 {
        let next_token = sampler.sample(&mut ctx, -1);
        sampler.accept(next_token);

        if model.token_is_eog(next_token) {
            break;
        }

        let text = model.token_to_str(next_token, 0, false)?;
        print!("{}", text);

        ctx.decode_single(next_token)?;
    }
    println!();

    Ok(())
}
```
