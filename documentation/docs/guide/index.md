# Library Guide

This guide covers **library-first** LLM inference with Mullama. Embed a high-performance inference engine directly in your application -- no daemon, no HTTP, no separate process.

!!! info "Library vs Daemon"
    This guide is for **library usage** (direct function calls in your code). If you need a server with OpenAI-compatible APIs, see the [Daemon & CLI](../daemon/index.md) docs.

    **Library advantages:** Zero HTTP overhead, in-process inference, microsecond latency.

The APIs documented here work across all 6 supported languages: **Node.js, Python, Rust, Go, PHP, and C/C++**.

## Architecture Overview

Mullama is organized in three layers, each building on the one below:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Integration Layer                            │
│  Async  │  Streaming  │  Web/WS  │  Multimodal  │  Parallel    │
├─────────────────────────────────────────────────────────────────┤
│                       Core API Layer                             │
│    Model    │   Context   │   Sampler   │   Batch   │ Embedding │
├─────────────────────────────────────────────────────────────────┤
│                     Foundation Layer                              │
│          sys.rs (FFI bindings)  │  build.rs (platform config)    │
│          llama.cpp C++ library  │  GPU acceleration backends     │
└─────────────────────────────────────────────────────────────────┘
```

**Foundation Layer** (`sys.rs`, `build.rs`): Low-level FFI bindings to the llama.cpp C++ library. Handles platform-specific build configuration, GPU acceleration detection, and memory-safe wrappers around unsafe C operations. You never interact with this layer directly.

**Core API Layer**: The primary API surface that applications interact with. Provides safe, ergonomic abstractions over the foundation layer. Available in all supported languages: Node.js, Python, Rust, Go, PHP, and C/C++.

**Integration Layer**: Optional, feature-gated modules that extend core functionality for specific use cases like async runtimes, streaming, web services, and multimodal processing.

## Core Concepts

### Model

The `Model` represents a loaded GGUF model file. It provides access to the model's vocabulary, architecture parameters, and tokenization capabilities. Models are designed for shared ownership -- a single loaded model can serve many concurrent inference contexts.

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    const model = await Model.load('./model.gguf');
    console.log(`Vocabulary size: ${model.vocabSize()}`);
    console.log(`Embedding dimension: ${model.embeddingDim()}`);
    ```

=== "Python"

    ```python
    from mullama import Model

    model = Model.load("./model.gguf")
    print(f"Vocabulary size: {model.vocab_size()}")
    print(f"Embedding dimension: {model.embedding_dim()}")
    ```

=== "Rust"

    ```rust
    use mullama::Model;
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);
    println!("Vocabulary size: {}", model.vocab_size());
    println!("Embedding dimension: {}", model.n_embd());
    ```

=== "CLI"

    ```bash
    # Inspect model metadata
    mullama show llama3.2:1b --modelfile
    ```

### Context

The `Context` holds the inference state, including the KV cache and generation position. You create a context from a model and use it to run inference.

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model, { nCtx: 4096 });
    const response = await context.generate("Hello!", 100);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")
    context = Context(model, ContextParams(n_ctx=4096))
    response = context.generate("Hello!", max_tokens=100)
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams};

    let mut context = Context::new(model.clone(), ContextParams::default())?;
    let response = context.generate("Hello!", 100)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!"
    ```

### Sampler

The `Sampler` controls how the next token is selected from the model's probability distribution. Mullama supports all llama.cpp sampling strategies including temperature, top-k, top-p, min-p, mirostat, and grammar-constrained sampling.

=== "Node.js"

    ```javascript
    import { SamplerChain } from 'mullama';

    const sampler = new SamplerChain()
      .addTemperature(0.7)
      .addTopK(40)
      .addTopP(0.9);
    ```

=== "Python"

    ```python
    from mullama import SamplerChain

    sampler = (SamplerChain()
        .add_temperature(0.7)
        .add_top_k(40)
        .add_top_p(0.9))
    ```

=== "Rust"

    ```rust
    use mullama::sampling::{Sampler, SamplerChain};

    let chain = SamplerChain::new()
        .add(Sampler::temperature(0.7)?)
        .add(Sampler::top_k(40)?)
        .add(Sampler::top_p(0.9, 1)?);
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" --temperature 0.7 --top-k 40 --top-p 0.9
    ```

### Batch

The `Batch` enables efficient processing of multiple token sequences simultaneously, which is essential for high-throughput applications.

=== "Node.js"

    ```javascript
    import { Batch } from 'mullama';

    const batch = new Batch(512, 3);
    batch.addSequence(0, tokensA);
    batch.addSequence(1, tokensB);
    await context.decodeBatch(batch);
    ```

=== "Python"

    ```python
    from mullama import Batch

    batch = Batch(max_tokens=512, n_sequences=3)
    batch.add_sequence(0, tokens_a)
    batch.add_sequence(1, tokens_b)
    context.decode_batch(batch)
    ```

=== "Rust"

    ```rust
    use mullama::Batch;

    let mut batch = Batch::new(512, 3);
    batch.add_sequence(0, &tokens_a)?;
    batch.add_sequence(1, &tokens_b)?;
    context.decode_batch(&batch)?;
    ```

=== "CLI"

    ```bash
    # Batch processing via daemon REST API
    curl -X POST http://localhost:8080/v1/batch \
      -H "Content-Type: application/json" \
      -d '{"prompts": ["Hello", "World"]}'
    ```

### Embedding

The `Embedding` module generates vector representations of text, useful for semantic search, RAG pipelines, and similarity calculations.

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./embedding-model.gguf');
    const context = new Context(model, { embeddings: true });
    const embedding = await context.getEmbedding("Hello world");
    console.log(`Dimension: ${embedding.dimension}`);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./embedding-model.gguf")
    context = Context(model, ContextParams(embeddings=True))
    embedding = context.get_embedding("Hello world")
    print(f"Dimension: {embedding.dimension}")
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams};

    let params = ContextParams { embeddings: true, ..Default::default() };
    let mut context = Context::new(model, params)?;
    let embedding = context.get_embedding("Hello world")?;
    println!("Dimension: {}", embedding.dimension);
    ```

=== "CLI"

    ```bash
    mullama embed llama3.2:1b "Hello world"
    ```

## Design Patterns

### Builder Pattern

Mullama uses the builder pattern extensively for complex configuration. All builders follow the same `.build()` completion pattern:

=== "Node.js"

    ```javascript
    const model = await Model.builder('./model.gguf')
      .gpuLayers(35)
      .useMmap(true)
      .build();
    ```

=== "Python"

    ```python
    model = (Model.builder("./model.gguf")
        .gpu_layers(35)
        .use_mmap(True)
        .build())
    ```

=== "Rust"

    ```rust
    let model = ModelBuilder::new("model.gguf")
        .with_n_gpu_layers(35)
        .with_use_mmap(true)
        .build()?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" --gpu-layers 35
    ```

### RAII (Resource Acquisition Is Initialization)

All resources (models, contexts, samplers, LoRA adapters) are automatically cleaned up when they go out of scope. In Node.js and Python, the garbage collector handles this. In Rust, the `Drop` trait ensures deterministic cleanup.

### Result-Based Error Handling

All fallible operations return errors in a language-appropriate way: exceptions in Node.js/Python, `Result<T, MullamaError>` in Rust.

### Arc for Shared Ownership

Models can be shared across multiple contexts and threads. In Rust, wrap models in `Arc`. In Node.js and Python, reference counting is handled automatically.

## Feature Flags

Mullama uses Cargo feature flags (Rust) to keep the core library lightweight. In Node.js and Python, all features are bundled in the native binary.

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `async` | AsyncModel, AsyncContext, Tokio integration | `tokio` |
| `streaming` | TokenStream, StreamConfig | `async` |
| `web` | Axum REST API integration | `async` |
| `websockets` | Real-time bidirectional communication | `async` |
| `multimodal` | Image and audio processing | -- |
| `streaming-audio` | Real-time audio capture | `multimodal` |
| `format-conversion` | Audio/image format conversion | `multimodal` |
| `parallel` | Rayon-based batch processing | -- |
| `tokio-runtime` | MullamaRuntime, TaskManager | `async` |
| `full` | All features enabled | all |

```toml
# Rust: enable specific features in Cargo.toml
[dependencies]
mullama = { version = "0.1", features = ["async", "streaming"] }
```

!!! info "Node.js and Python"
    The Node.js (`@mullama/node`) and Python (`mullama`) packages include all features by default. No additional configuration is needed.

## How to Read This Guide

The guide is organized from fundamental to advanced topics:

1. **[Loading Models](models.md)** -- Start here. Learn how to load and configure GGUF models.
2. **[Text Generation](generation.md)** -- Core inference: context parameters, sampling basics, and chat templates.
3. **[Streaming](streaming.md)** -- Real-time token output for responsive applications.
4. **[Async Support](async.md)** -- Non-blocking and concurrent inference.
5. **[Embeddings](embeddings.md)** -- Vector representations for search and RAG.
6. **[Sampling Strategies](sampling.md)** -- Deep dive into all sampling methods and configurations.
7. **[Structured Output](structured-output.md)** -- JSON Schema-constrained generation.
8. **[Grammar Constraints](grammar.md)** -- GBNF grammars for arbitrary output formats.
9. **[LoRA Adapters](lora.md)** -- Fine-tuned adapter loading and hot-swapping.
10. **[Multimodal](multimodal.md)** -- Vision-language and audio-language model support.
11. **[Sessions & State](sessions.md)** -- Saving and restoring inference state.
12. **[Memory Management](memory.md)** -- KV cache management, monitoring, and optimization.

!!! tip "Quick Start"
    If you are new to Mullama, start with [Loading Models](models.md) and [Text Generation](generation.md) to get a working example running quickly.

## See Also

- [Getting Started](../getting-started/index.md) -- Installation and platform setup
- [API Reference](../api/index.md) -- Complete API documentation
- [Tutorials & Examples](../examples/index.md) -- End-to-end application examples
- [Language Bindings](../bindings/index.md) -- Language-specific binding documentation
