---
title: Batch API
description: Efficient multi-token batch processing with SmallVec optimization
---

# Batch API

The `Batch` struct provides efficient multi-token processing for prompt evaluation and parallel sequence handling. It uses SmallVec optimization to avoid heap allocation for small batches (32 tokens or fewer).

## Batch Struct

```rust
/// Represents a batch of tokens for processing.
///
/// Uses SmallVec (via TokenBuffer) for token storage, providing:
/// - Stack allocation for batches up to 32 tokens (no heap allocation)
/// - Transparent heap fallback for larger batches
/// - 5-10% faster for typical small batch operations
pub struct Batch {
    inner: Option<llama_batch>,
    tokens_storage: Option<TokenBuffer>,
    needs_free: bool,
}
```

!!! info "SmallVec Optimization"
    For batches of 32 tokens or fewer, `Batch` uses stack-allocated storage via `SmallVec<[TokenId; 32]>`, completely avoiding heap allocation. This is a Rust-exclusive optimization that provides 5-10% faster processing for typical interactive generation where single tokens are decoded one at a time.

## Creating Batches

### `Batch::new`

Create a new batch with pre-allocated memory for a maximum number of tokens.

```rust
pub fn new(max_tokens: usize, embd: i32, max_seq: usize) -> Self
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_tokens` | `usize` | -- | Maximum number of tokens the batch can hold |
| `embd` | `i32` | -- | Embedding dimension (0 for token-based batches) |
| `max_seq` | `usize` | -- | Maximum number of parallel sequences |

**Example:**

```rust
use mullama::batch::Batch;

// Create batch for up to 512 tokens, 1 sequence
let batch = Batch::new(512, 0, 1);

// Create batch supporting 4 parallel sequences
let batch = Batch::new(2048, 0, 4);
```

### `Batch::from_tokens`

Create a batch from a token slice. Uses stack allocation for 32 tokens or fewer.

```rust
pub fn from_tokens(tokens: &[TokenId]) -> Self
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `&[TokenId]` | -- | Slice of token IDs to process |

**Example:**

```rust
use mullama::batch::Batch;

let tokens = vec![1, 2, 3, 4, 5];
let batch = Batch::from_tokens(&tokens);
assert_eq!(batch.len(), 5);
// Stack-allocated since len <= 32
```

### `Batch::from_tokens_owned`

Create a batch from an owned `Vec<TokenId>`, avoiding a copy when you already have ownership.

```rust
pub fn from_tokens_owned(tokens: Vec<TokenId>) -> Self
```

**Example:**

```rust
let tokens = model.tokenize("Hello, world!", true, false)?;
let batch = Batch::from_tokens_owned(tokens);
```

### `Batch::from_token_buffer`

Create a batch from a `TokenBuffer` (SmallVec-based storage).

```rust
pub fn from_token_buffer(tokens: TokenBuffer) -> Self
```

## Methods

### `len`

Get the number of tokens currently in the batch.

```rust
pub fn len(&self) -> usize
```

### `is_empty`

Check if the batch contains no tokens.

```rust
pub fn is_empty(&self) -> bool
```

### `get_llama_batch`

Get a reference to the internal `llama_batch` struct for advanced/FFI use.

```rust
pub fn get_llama_batch(&self) -> Option<&llama_batch>
```

## Default

The default batch is created with 512 max tokens, 0 embedding dimensions, and 1 sequence:

```rust
impl Default for Batch {
    fn default() -> Self {
        Self::new(512, 0, 1)
    }
}
```

## Multi-Sequence Support

Batches can handle multiple parallel sequences for use cases like beam search, speculative decoding, or parallel generation:

```rust
use mullama::batch::Batch;

// Create batch supporting up to 4 parallel sequences
let batch = Batch::new(2048, 0, 4);
```

When using multi-sequence batches, each token can be assigned to one or more sequence IDs, enabling independent generation streams within a single context.

## Usage with Context::decode

The primary use of `Batch` is with `Context::decode()` for processing tokens:

```rust
use mullama::{Model, Context, ContextParams, batch::Batch};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);
let mut ctx = Context::new(model.clone(), ContextParams::default())?;

// Tokenize prompt
let tokens = model.tokenize("Hello, world!", true, false)?;

// Method 1: Direct decode (creates batch internally)
ctx.decode(&tokens)?;

// Method 2: Manual batch for more control
let batch = Batch::from_tokens(&tokens);
// The batch is used internally by Context operations
```

## Memory Management

`Batch` implements `Drop` for automatic cleanup:

- Batches created with `Batch::new()` (via `llama_batch_init`) allocate internal memory that is freed on drop
- Batches created with `from_tokens` / `from_token_buffer` (via `llama_batch_get_one`) only manage the `TokenBuffer` storage

```rust
{
    let batch = Batch::new(1024, 0, 1); // Allocates internal C memory
    // ... use batch ...
} // Automatically freed here via llama_batch_free

{
    let batch = Batch::from_tokens(&[1, 2, 3]); // Stack-allocated for small sizes
    // ... use batch ...
} // TokenBuffer dropped, no llama_batch_free needed
```

### Memory Behavior

| Creation Method | Storage | Allocation | Drop Behavior |
|----------------|---------|------------|---------------|
| `Batch::new(n, 0, s)` | C heap | Always heap | Calls `llama_batch_free` |
| `from_tokens` (<=32) | Rust stack | No allocation | Stack unwind only |
| `from_tokens` (>32) | Rust heap | Vec allocation | Vec dropped |
| `from_tokens_owned` | Rust heap | Uses existing Vec | Vec dropped |

## Performance Considerations

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Single token decode | Use `Context::decode_single()` | Avoids all allocation |
| Small prompt (1-32 tokens) | Use `Batch::from_tokens()` | Stays entirely on stack |
| Medium prompt (33-512 tokens) | Use `Batch::from_tokens()` | Heap fallback is fast |
| Large prompt (>512 tokens) | Use `Context::decode()` | Handles chunking automatically |
| Multiple sequences | Use `Batch::new(n, 0, seq_count)` | Explicit capacity for sequences |

!!! tip "Generation Loop Performance"
    During token-by-token generation, always use `Context::decode_single()` rather than creating a `Batch` for each token. The single-token path avoids all allocation overhead and is the fastest way to process generated tokens.

## Complete Example

```rust
use mullama::{Model, Context, ContextParams, SamplerParams, batch::Batch};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;

    // Tokenize and process prompt
    let prompt_tokens = model.tokenize("The capital of France is", true, false)?;

    // For typical use, just call decode directly:
    ctx.decode(&prompt_tokens)?;

    // For generation, decode tokens one at a time using the optimized path:
    let mut sampler = SamplerParams::default().build_chain(model.clone())?;

    for _ in 0..50 {
        let token = sampler.sample(&mut ctx, -1);
        sampler.accept(token);

        if model.token_is_eog(token) {
            break;
        }

        // Single-token decode avoids batch allocation entirely
        ctx.decode_single(token)?;

        let text = model.token_to_str(token, 0, false)?;
        print!("{}", text);
    }
    println!();

    Ok(())
}
```
