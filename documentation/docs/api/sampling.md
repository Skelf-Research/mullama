---
title: Sampling API
description: Sampling strategies, sampler chains, and token selection control
---

# Sampling API

The sampling module provides comprehensive control over how tokens are selected during generation. It supports all llama.cpp sampling strategies, composable sampler chains, and high-level parameter presets.

## SamplerParams

High-level configuration struct that encapsulates common sampling parameters. Use `build_chain()` to convert into a ready-to-use `SamplerChain`.

```rust
#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalty_last_n: i32,
    pub penalize_nl: bool,
    pub ignore_eos: bool,
    pub seed: u32,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `temperature` | `f32` | `0.8` | Controls randomness. 0.0 = deterministic (greedy), 1.0 = neutral, 2.0 = very random. |
| `top_k` | `i32` | `40` | Keep only the K most probable tokens. 0 = disabled. |
| `top_p` | `f32` | `0.95` | Nucleus sampling -- keep tokens with cumulative probability up to P. 1.0 = disabled. |
| `min_p` | `f32` | `0.05` | Remove tokens with probability below `min_p * max_probability`. 0.0 = disabled. |
| `typical_p` | `f32` | `1.0` | Typical sampling threshold. Keeps tokens near expected entropy. 1.0 = disabled. |
| `penalty_repeat` | `f32` | `1.1` | Repetition penalty multiplier. 1.0 = no penalty. Values > 1.0 penalize repetition. |
| `penalty_freq` | `f32` | `0.0` | Frequency-based penalty. Penalizes tokens proportional to their count in context. |
| `penalty_present` | `f32` | `0.0` | Presence-based penalty. Flat penalty for any token that appears in context. |
| `penalty_last_n` | `i32` | `64` | Number of recent tokens to consider for penalties. -1 = entire context. |
| `penalize_nl` | `bool` | `true` | Whether to apply repetition penalties to newline tokens. |
| `ignore_eos` | `bool` | `false` | Ignore end-of-sequence token (force generation to continue). |
| `seed` | `u32` | `LLAMA_DEFAULT_SEED` | Random seed for reproducibility. Use a fixed value for deterministic output. |

### Presets

Quick configurations for common use cases:

```rust
// Greedy: always pick the highest probability token (deterministic)
let params = SamplerParams::greedy();
// temperature=0.0, top_k=1

// Creative: higher temperature, wider sampling for diverse output
let params = SamplerParams::creative();
// temperature=1.2, top_k=60, top_p=0.95, penalty_repeat=1.15

// Precise: lower temperature, focused sampling for factual/code output
let params = SamplerParams::precise();
// temperature=0.2, top_k=10, top_p=0.85, penalty_repeat=1.0

// Default: balanced settings for general use
let params = SamplerParams::default();
// temperature=0.8, top_k=40, top_p=0.95
```

### `build_chain`

Build a `SamplerChain` from these parameters. The chain is constructed in this order:

1. Penalties (repeat, frequency, presence)
2. Top-K filtering
3. Typical sampling
4. Top-P (nucleus) filtering
5. Min-P filtering
6. Temperature scaling
7. Distribution sampling (final selection)

```rust
pub fn build_chain(&self, model: Arc<Model>) -> Result<SamplerChain, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Arc<Model>` | -- | Model reference (needed for vocabulary size) |

**Returns:** `Result<SamplerChain, MullamaError>`

**Example:**

```rust
use mullama::{Model, SamplerParams};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);

let params = SamplerParams {
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9,
    min_p: 0.05,
    penalty_repeat: 1.1,
    ..Default::default()
};

let mut chain = params.build_chain(model)?;
```

## Sampler

Low-level individual sampler types. Each represents a single sampling strategy that can be composed into chains.

```rust
pub struct Sampler {
    sampler_ptr: *mut llama_sampler,
    _model: Option<Arc<Model>>,
}
```

**Thread Safety:** `Sampler` implements `Send + Sync`. Can be moved between threads.

### Factory Methods

#### `Sampler::greedy`

Always selects the highest probability token. Deterministic output.

```rust
pub fn greedy() -> Result<Self, MullamaError>
```

#### `Sampler::dist`

Random sampling from the probability distribution with a given seed.

```rust
pub fn dist(seed: u32) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `seed` | `u32` | -- | Random seed for reproducibility |

#### `Sampler::top_k`

Keeps only the top K most probable tokens, setting all others to zero probability.

```rust
pub fn top_k(k: i32) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `k` | `i32` | -- | Number of top tokens to keep |

#### `Sampler::top_p`

Nucleus sampling -- keeps tokens whose cumulative probability reaches P.

```rust
pub fn top_p(p: f32, min_keep: usize) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p` | `f32` | -- | Cumulative probability threshold (0.0 - 1.0) |
| `min_keep` | `usize` | -- | Minimum number of tokens to keep regardless of threshold |

#### `Sampler::min_p`

Removes tokens with probability less than `p * max_probability`.

```rust
pub fn min_p(p: f32, min_keep: usize) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p` | `f32` | -- | Minimum probability ratio relative to the top token |
| `min_keep` | `usize` | -- | Minimum tokens to keep |

#### `Sampler::typical`

Typical sampling -- selects tokens close to the expected information content, filtering out both too-predictable and too-surprising tokens.

```rust
pub fn typical(p: f32, min_keep: usize) -> Result<Self, MullamaError>
```

#### `Sampler::temperature`

Scales logits by `1/temperature` before softmax. Higher temperature = more random.

```rust
pub fn temperature(temperature: f32) -> Result<Self, MullamaError>
```

#### `Sampler::temperature_ext`

Extended temperature with dynamic range control for more nuanced temperature behavior.

```rust
pub fn temperature_ext(
    temperature: f32,
    delta: f32,
    exponent: f32,
) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `temperature` | `f32` | -- | Base temperature |
| `delta` | `f32` | -- | Range around temperature for dynamic adjustment |
| `exponent` | `f32` | -- | Exponent for temperature curve |

#### `Sampler::tail_free`

Tail-free sampling (TFS) -- removes low-probability tail tokens based on second derivative analysis.

```rust
pub fn tail_free(z: f32, min_keep: usize) -> Result<Self, MullamaError>
```

#### `Sampler::mirostat`

Mirostat v1 -- maintains a target entropy during generation for consistent output quality.

```rust
pub fn mirostat(
    model: Arc<Model>,
    seed: u32,
    tau: f32,
    eta: f32,
    m: i32,
) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Arc<Model>` | -- | Model reference |
| `seed` | `u32` | -- | Random seed |
| `tau` | `f32` | -- | Target entropy (5.0 is a good starting point) |
| `eta` | `f32` | -- | Learning rate (0.1 is typical) |
| `m` | `i32` | -- | Number of candidates to consider |

#### `Sampler::mirostat_v2`

Mirostat v2 -- improved entropy targeting without vocabulary size dependency.

```rust
pub fn mirostat_v2(
    seed: u32,
    tau: f32,
    eta: f32,
) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `seed` | `u32` | -- | Random seed |
| `tau` | `f32` | -- | Target entropy |
| `eta` | `f32` | -- | Learning rate |

#### `Sampler::penalties`

Repetition, frequency, and presence penalties applied to recent tokens.

```rust
pub fn penalties(
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `penalty_last_n` | `i32` | -- | Window of recent tokens to check |
| `penalty_repeat` | `f32` | -- | Repetition penalty (1.0 = disabled) |
| `penalty_freq` | `f32` | -- | Frequency penalty (0.0 = disabled) |
| `penalty_present` | `f32` | -- | Presence penalty (0.0 = disabled) |

#### `Sampler::grammar`

Grammar-constrained sampling using GBNF (Generalized Backus-Naur Form) format.

```rust
pub fn grammar(
    model: Arc<Model>,
    grammar_str: &str,
    grammar_root: &str,
) -> Result<Self, MullamaError>
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Arc<Model>` | -- | Model reference |
| `grammar_str` | `&str` | -- | GBNF grammar definition |
| `grammar_root` | `&str` | -- | Root rule name |

**Example:**

```rust
let json_grammar = r#"
root   ::= object
object ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= string ":" ws value
value  ::= string | number | "true" | "false" | "null"
string ::= "\"" [^"]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
"#;

let sampler = Sampler::grammar(model.clone(), json_grammar, "root")?;
```

#### `Sampler::logit_bias`

Bias specific tokens up or down in the logit space.

```rust
pub fn logit_bias(
    n_vocab: i32,
    logit_biases: &[LogitBias],
) -> Result<Self, MullamaError>
```

#### `Sampler::dry`

DRY (Don't Repeat Yourself) -- penalizes n-gram repetition patterns.

```rust
pub fn dry(
    model: Arc<Model>,
    n_ctx_train: i32,
    multiplier: f32,
    base: f32,
    allowed_length: i32,
    penalty_last_n: i32,
    seq_breakers: &[&str],
) -> Result<Self, MullamaError>
```

#### `Sampler::xtc`

XTC (Exclude Top Choices) -- randomly excludes top tokens to encourage diversity.

```rust
pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Result<Self, MullamaError>
```

#### `Sampler::infill`

Code infill sampler for fill-in-the-middle completion tasks.

```rust
pub fn infill(model: Arc<Model>) -> Result<Self, MullamaError>
```

#### `Sampler::softmax`

Normalizes logits to probabilities via softmax.

```rust
pub fn softmax() -> Result<Self, MullamaError>
```

### Instance Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `sample` | `(&mut self, ctx: &mut Context, idx: i32) -> TokenId` | Sample a token at position `idx` (-1 for last) |
| `accept` | `(&mut self, token: TokenId)` | Accept token (updates stateful samplers like penalties) |
| `apply` | `(&mut self, candidates: &mut TokenDataArray)` | Apply sampler to candidate token array |
| `reset` | `(&mut self)` | Reset sampler state |
| `try_clone` | `(&self) -> Result<Self, MullamaError>` | Clone the sampler |
| `name` | `(&self) -> String` | Get sampler type name |
| `perf_data` | `(&self) -> SamplerPerfData` | Get performance statistics |

## SamplerChain

Combines multiple samplers into a processing pipeline. Samplers are applied in order during token selection.

```rust
pub struct SamplerChain {
    chain_ptr: *mut llama_sampler,
}
```

**Thread Safety:** `SamplerChain` implements `Send + Sync`.

### Creating Chains

```rust
// From SamplerParams (recommended for most use cases)
let chain = sampler_params.build_chain(model)?;

// Empty chain with default settings
let chain = SamplerChain::with_defaults();

// Empty chain with custom settings
let chain = SamplerChain::new(SamplerChainParams { no_perf: false });
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(&mut self, sampler: Sampler)` | Add a sampler to the chain (takes ownership) |
| `get` | `(&self, index: i32) -> Option<*mut llama_sampler>` | Get raw sampler pointer at index |
| `len` | `(&self) -> i32` | Number of samplers in chain |
| `is_empty` | `(&self) -> bool` | Check if chain has no samplers |
| `remove` | `(&mut self, index: i32) -> Option<Sampler>` | Remove sampler at index |
| `sample` | `(&mut self, ctx: &mut Context, idx: i32) -> TokenId` | Sample using full chain |
| `accept` | `(&mut self, token: TokenId)` | Accept token for all samplers in chain |
| `reset` | `(&mut self)` | Reset all samplers in chain |
| `get_seed` | `(&self) -> u32` | Get distribution sampler seed |

**Example -- Custom Chain:**

```rust
use mullama::sampling::{SamplerChain, Sampler};

let mut chain = SamplerChain::with_defaults();

// Add samplers in processing order
chain.add(Sampler::penalties(64, 1.1, 0.0, 0.0)?);
chain.add(Sampler::top_k(40)?);
chain.add(Sampler::top_p(0.9, 1)?);
chain.add(Sampler::min_p(0.05, 1)?);
chain.add(Sampler::temperature(0.7)?);
chain.add(Sampler::dist(42)?);

// Use in generation loop
let token = chain.sample(&mut ctx, -1);
chain.accept(token);
```

## SamplerChainParams

```rust
#[derive(Debug, Clone, Default)]
pub struct SamplerChainParams {
    pub no_perf: bool,  // Disable performance counters for faster sampling
}
```

## Supporting Types

### TokenData

Represents a single token candidate with its logit and probability values.

```rust
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TokenData {
    pub id: TokenId,     // Token ID in vocabulary
    pub logit: f32,      // Raw logit value (before softmax)
    pub p: f32,          // Probability (after softmax)
}
```

### AlignedTokenData

Cache-line aligned variant for parallel sampling without false sharing between CPU cores.

```rust
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedTokenData {
    pub id: TokenId,
    pub logit: f32,
    pub p: f32,
}
```

!!! info "SIMD Acceleration"
    `AlignedTokenData` is aligned to 64-byte cache lines to prevent false sharing in parallel sampling. This provides 5-10% speedup in multi-threaded scenarios where multiple contexts are sampling simultaneously.

### LogitBias

Bias specific token probabilities up or down.

```rust
#[derive(Debug, Clone)]
pub struct LogitBias {
    pub token: TokenId,
    pub bias: f32,       // Positive = more likely, negative = less likely
                         // -inf effectively bans the token
}
```

### SamplerPerfData

Performance statistics for a sampler.

```rust
#[derive(Debug, Clone)]
pub struct SamplerPerfData {
    pub t_sample_ms: f64,   // Total sampling time in milliseconds
    pub n_sample: i32,      // Number of samples taken
}
```

## SamplerBuilder

Fluent API for constructing sampling configurations with method chaining.

```rust
use mullama::builder::SamplerBuilder;

let sampler = SamplerBuilder::new()
    .temperature(0.8)
    .top_k(50)
    .nucleus(0.95)
    .penalties(|p| p
        .repetition(1.1)
        .frequency(0.1)
        .presence(0.1)
    )
    .build(model.clone())?;
```

### Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `new()` | -- | Create a new builder |
| `temperature(t)` | `f32` | Set temperature |
| `top_k(k)` | `i32` | Set top-k value |
| `nucleus(p)` | `f32` | Set top-p (nucleus) value |
| `min_p(p)` | `f32` | Set min-p threshold |
| `typical(p)` | `f32` | Set typical sampling threshold |
| `seed(s)` | `u32` | Set random seed |
| `penalties(f)` | closure | Configure penalty settings |
| `build(model)` | `Arc<Model>` | Build the sampler chain |

## Grammar-Constrained Generation

Grammar sampling ensures output conforms to a formal grammar in GBNF format:

```rust
use mullama::sampling::{SamplerChain, Sampler};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);

let json_grammar = r#"
root   ::= object
object ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= string ":" ws value
value  ::= string | number | object | array | "true" | "false" | "null"
array  ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" [^"]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
"#;

let mut chain = SamplerChain::with_defaults();
chain.add(Sampler::grammar(model.clone(), json_grammar, "root")?);
chain.add(Sampler::temperature(0.7)?);
chain.add(Sampler::dist(42)?);

// All generated output will be valid JSON
```

## Thread Safety

Both `Sampler` and `SamplerChain` implement `Send + Sync`:

```rust
// Move sampler chain to another thread
let handle = std::thread::spawn(move || {
    let token = chain.sample(&mut ctx, -1);
    chain.accept(token);
    token
});
```
