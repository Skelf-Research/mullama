---
title: API Reference
description: Complete API reference for the Mullama Rust library
---

# API Reference

Complete API documentation for all Mullama types, traits, and functions. This reference covers the full public API surface of the library, including core types, feature-gated modules, and utility functions.

## How to Read This Reference

Each API page follows a consistent structure:

- **Type signatures** are shown as Rust code blocks with full generic bounds
- **Parameter tables** use the format: Name | Type | Default | Description
- **Error conditions** are listed under each method with the specific `MullamaError` variant returned
- **Feature gates** are indicated with admonitions at the top of each section
- **Examples** demonstrate typical usage patterns with compilable code

### Conventions

- `impl AsRef<Path>` indicates any type convertible to a filesystem path (e.g., `&str`, `String`, `PathBuf`)
- `Arc<Model>` indicates shared ownership -- the caller retains a reference
- `&mut self` indicates the method mutates internal state
- `Result<T, MullamaError>` is the standard return type for fallible operations

## Core Types Diagram

```
                    +------------------+
                    |   MullamaConfig  |
                    | (ModelConfig,    |
                    |  ContextConfig,  |
                    |  SamplerConfig)  |
                    +--------+---------+
                             |
                             v
+------------+       +-------+--------+       +----------------+
| ModelParams| ----> |     Model      | <---- | ModelBuilder    |
+------------+       | (Arc<Inner>)   |       +----------------+
                     | Send + Sync    |
                     +---+-----+------+
                         |     |
            tokenize()   |     |  create_context()
                         v     v
              +----------+     +------------+       +---------------+
              | Vec<Token>     |   Context   | <---- | ContextParams |
              +----------+     | (!Send)     |       +---------------+
                               +---+----+---+
                                   |    |
                        decode()   |    |  generate()
                                   v    v
                    +-----------+  +----+------+
                    |   Batch   |  |  String   |
                    | (SmallVec)|  +-----------+
                    +-----------+
                                   +---------------+
                                   | SamplerChain  |
                                   | (Send + Sync) |
                                   +---+-----------+
                                       |
                                       v
                              +--------+--------+
                              |    Sampler      |
                              | (Send + Sync)   |
                              +-----------------+
```

## Module Organization

| Module | Description | Feature Gate |
|--------|-------------|--------------|
| [`model`](model.md) | Model loading, metadata, and tokenization | Core |
| [`context`](context.md) | Inference context, generation, and KV-cache | Core |
| [`sampling`](sampling.md) | Sampling strategies and composable chains | Core |
| [`batch`](batch.md) | Efficient multi-token batch processing | Core |
| [`embeddings`](embeddings.md) | Text embedding generation and similarity | Core |
| [`config`](configuration.md) | Configuration management with serde | Core |
| [`error`](errors.md) | Error types and handling patterns | Core |
| [`multimodal`](multimodal.md) | Text, image, and audio processing | `multimodal` |
| [`async_support`](async.md) | Async model, context, and runtime | `async` |
| [`streaming`](streaming.md) | Real-time token streaming | `streaming` |

## Core Types Quick Reference

| Type | Module | Description |
|------|--------|-------------|
| `Model` | `model` | Loaded LLM model (Arc-based, Clone, Send+Sync) |
| `ModelParams` | `model` | Parameters for model loading |
| `ModelBuilder` | `model` | Fluent API for model configuration |
| `Context` | `context` | Inference context with KV-cache (!Send) |
| `ContextParams` | `context` | Context configuration parameters |
| `KvCacheType` | `context` | KV-cache quantization level |
| `SamplerParams` | `sampling` | High-level sampling configuration |
| `SamplerChain` | `sampling` | Composable chain of samplers |
| `Sampler` | `sampling` | Individual sampling strategy |
| `Batch` | `batch` | Token batch with SmallVec optimization |
| `Embeddings` | `embedding` | Generated embedding vectors |
| `EmbeddingGenerator` | `embedding` | Embedding generation utility |
| `EmbeddingConfig` | `embedding` | Embedding configuration |
| `PoolingStrategy` | `embedding` | Embedding pooling method |
| `MullamaConfig` | `config` | Top-level configuration struct |
| `MullamaError` | `error` | Unified error type |

## Feature-Gated Types

Types that require specific Cargo features to be enabled:

| Type | Required Feature | Description |
|------|-----------------|-------------|
| `AsyncModel` | `async` | Non-blocking model wrapper |
| `AsyncContext` | `async` | Non-blocking context wrapper |
| `MullamaRuntime` | `tokio-runtime` | Tokio runtime manager |
| `ModelPool` | `tokio-runtime` | Connection pool for models |
| `TaskManager` | `tokio-runtime` | Async task coordinator |
| `TokenStream` | `streaming` | Real-time token stream |
| `StreamConfig` | `streaming` | Streaming configuration |
| `MultimodalProcessor` | `multimodal` | Cross-modal processing |
| `VisionEncoder` | `multimodal` | Image encoding pipeline |
| `ImageInput` | `multimodal` | Image data container |
| `AudioInput` | `multimodal` | Audio data container |
| `StreamingAudioProcessor` | `streaming-audio` | Real-time audio capture |
| `AudioStreamConfig` | `streaming-audio` | Audio stream settings |
| `AppState` | `web` | Axum application state |
| `RouterBuilder` | `web` | Web route configuration |
| `WebSocketServer` | `websockets` | WebSocket server |
| `WebSocketConfig` | `websockets` | WebSocket settings |

## Thread Safety Overview

Mullama types are designed with clear thread safety semantics:

| Type | Send | Sync | Clone | Notes |
|------|------|------|-------|-------|
| `Model` | Yes | Yes | Yes (cheap) | Arc-based sharing, reference count increment |
| `Context` | **No** | **No** | No | Bound to creating thread; use per-thread instances |
| `SamplerChain` | Yes | Yes | No | Can be moved between threads |
| `Sampler` | Yes | Yes | No | Can be moved between threads |
| `Batch` | Yes | No | Yes | Move between threads, not shared references |
| `Embeddings` | Yes | Yes | Yes | Plain data, freely shareable |
| `AsyncModel` | Yes | Yes | Yes (cheap) | Arc-based, designed for concurrent use |

!!! warning "Context Thread Safety"
    `Context` is intentionally **not** `Send`. It holds mutable state (KV-cache, position counters) that is not safe for concurrent access. Each thread must create its own `Context` from a shared `Arc<Model>`. For async contexts, use `AsyncContext` which handles thread-safety internally via `spawn_blocking`.

### Typical Multi-Threaded Pattern

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::load("model.gguf")?);

let handles: Vec<_> = (0..4).map(|i| {
    let model = model.clone(); // Cheap Arc clone
    thread::spawn(move || {
        // Each thread creates its own context
        let mut ctx = Context::new(model, ContextParams::default()).unwrap();
        ctx.generate(&[1, 2, 3], 50).unwrap()
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap();
    println!("{}", result);
}
```

## Versioning and Stability Guarantees

Mullama follows [Semantic Versioning](https://semver.org/):

- **Major version (0.x)**: The library is in pre-1.0 development. Breaking changes may occur between minor versions.
- **Minor version**: May include new features and non-breaking API additions.
- **Patch version**: Bug fixes and documentation improvements only.

### Stability Tiers

| Tier | Guarantee | Types |
|------|-----------|-------|
| **Stable** | No breaking changes without major bump | `Model`, `Context`, `SamplerParams`, `Batch`, `Embeddings`, `MullamaError` |
| **Unstable** | May change between minor versions | Feature-gated types (`AsyncModel`, `TokenStream`, `MultimodalProcessor`) |
| **Internal** | No stability guarantee | Types in `sys` module, FFI bindings |

## Feature Flags

```toml
[dependencies.mullama]
version = "0.1"
features = [
    "async",            # AsyncModel, AsyncContext
    "streaming",        # TokenStream, StreamConfig (requires "async")
    "multimodal",       # MultimodalProcessor, ImageInput, AudioInput
    "streaming-audio",  # StreamingAudioProcessor (requires "multimodal")
    "format-conversion",# AudioConverter, ImageConverter (requires "multimodal")
    "web",              # Axum integration (requires "async")
    "websockets",       # WebSocket server (requires "async")
    "parallel",         # ParallelProcessor, batch operations
    "tokio-runtime",    # MullamaRuntime, TaskManager
    "late-interaction", # ColBERT-style multi-vector embeddings
    "daemon",           # Background daemon mode
    "full",             # All features enabled
]
```

## Dependency Chains

When enabling features, these dependencies are automatically resolved:

- `streaming` requires `async`
- `streaming-audio` requires `multimodal`
- `format-conversion` requires `multimodal`
- `web` requires `async`
- `websockets` requires `async`
- `full` enables all features

## Cross-Language Equivalents

Where applicable, API pages note equivalent APIs in other language bindings:

| Rust | Node.js | Python |
|------|---------|--------|
| `Model::load(path)` | `Model.load(path)` | `Model.load(path)` |
| `model.tokenize(text, true, false)` | `model.tokenize(text)` | `model.tokenize(text)` |
| `ctx.generate(&tokens, max)` | `ctx.generate(tokens, max)` | `ctx.generate(tokens, max)` |
| `SamplerParams::default()` | `new SamplerParams()` | `SamplerParams()` |

See the [Bindings documentation](../bindings/index.md) for complete language-specific guides.
