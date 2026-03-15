# Mullama

**Comprehensive Rust bindings for llama.cpp with advanced integration features**

[![Crates.io](https://img.shields.io/crates/v/mullama)](https://crates.io/crates/mullama)
[![Documentation](https://docs.rs/mullama/badge.svg)](https://docs.rs/mullama)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Mullama provides memory-safe Rust bindings for llama.cpp with production-ready features including async/await support, real-time streaming, multimodal processing, and web framework integration.

## Why Mullama?

Most llama.cpp Rust bindings expose low-level C APIs directly. Mullama provides an **idiomatic Rust experience**:

```rust
// Other wrappers: manual memory management, raw pointers, verbose setup
let params = llama_context_default_params();
let ctx = unsafe { llama_new_context_with_model(model, params) };
let tokens = unsafe { llama_tokenize(model, text.as_ptr(), ...) };
// Don't forget to free everything...

// Mullama: builder patterns, async/await, automatic resource management
let model = ModelBuilder::new()
    .path("model.gguf")
    .gpu_layers(35)
    .build().await?;

let response = model.generate("Hello", 100).await?;
```

**Developer experience improvements:**

| Feature | Other Wrappers | Mullama |
|---------|---------------|---------|
| API Style | Raw FFI / C-like | Builder patterns, fluent API |
| Async Support | Manual threading | Native async/await with Tokio |
| Error Handling | Error codes / panics | `Result<T, MullamaError>` with context |
| Memory Management | Manual free/cleanup | Automatic RAII |
| Streaming | Callbacks | `Stream` trait, async iterators |
| Configuration | Struct fields | Type-safe builders with validation |
| Web Integration | DIY | Built-in Axum routes |

## Key Features

- **Async/Await Native** - Full Tokio integration for non-blocking operations
- **Real-time Streaming** - Token-by-token generation with backpressure handling
- **Multimodal Processing** - Text, image, and audio in a single pipeline
- **Late Interaction / ColBERT** - Multi-vector embeddings with MaxSim scoring for retrieval
- **Web Framework Ready** - Direct Axum integration with REST APIs
- **WebSocket Support** - Real-time bidirectional communication
- **Parallel Processing** - Work-stealing parallelism for batch operations
- **GPU Acceleration** - CUDA, Metal, ROCm, and OpenCL support
- **Memory Safe** - Zero unsafe operations in public API

## Quick Start

### Installation

```toml
[dependencies]
mullama = "0.1.1"

# With all features
mullama = { version = "0.1.1", features = ["full"] }
```

### Prerequisites

**Linux (Ubuntu/Debian):**
```bash
sudo apt install -y build-essential cmake pkg-config libasound2-dev libpulse-dev
```

**macOS:**
```bash
brew install cmake pkg-config portaudio
```

**Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) and [CMake](https://cmake.org/download/).

See [Platform Setup Guide](./docs/PLATFORM_SETUP.md) for detailed instructions.

### Basic Example

```rust
use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    let model = ModelBuilder::new()
        .path("model.gguf")
        .context_size(4096)
        .build().await?;

    let response = model.generate("The future of AI is", 100).await?;
    println!("{}", response);

    Ok(())
}
```

## Feature Flags

```toml
[dependencies.mullama]
version = "0.1.1"
features = [
    "async",              # Async/await support
    "streaming",          # Token streaming
    "web",                # Axum web framework
    "websockets",         # WebSocket support
    "multimodal",         # Image and audio processing
    "streaming-audio",    # Real-time audio capture
    "format-conversion",  # Audio/image format conversion
    "parallel",           # Rayon parallel processing
    "late-interaction",   # ColBERT-style multi-vector embeddings
    "daemon",             # Daemon mode with TUI client
    "full"                # All features
]
```

### Common Combinations

```toml
# Web applications
features = ["web", "websockets", "async", "streaming"]

# Multimodal AI
features = ["multimodal", "streaming-audio", "format-conversion"]

# High-performance batch processing
features = ["parallel", "async"]

# Semantic search / RAG with ColBERT-style retrieval
features = ["late-interaction", "parallel"]

# Daemon with TUI chat interface
features = ["daemon"]
```

## Daemon Mode

Mullama includes a multi-model daemon with OpenAI-compatible HTTP API and TUI client:

```bash
# Build the CLI
cargo build --release --features daemon

# Start daemon with local model
mullama serve --model llama:./llama.gguf

# Start with HuggingFace model (auto-downloads and caches)
mullama serve --model hf:TheBloke/Llama-2-7B-GGUF

# Multiple models with custom aliases
mullama serve \
  --model llama:hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf \
  --model mistral:hf:TheBloke/Mistral-7B-v0.1-GGUF

# Interactive TUI chat
mullama chat

# One-shot generation
mullama run "What is the meaning of life?"

# Model management
mullama models            # List loaded models
mullama load phi:./phi.gguf  # Load a model
mullama unload phi        # Unload a model
mullama default llama     # Set default model

# Search for models on HuggingFace
mullama search "llama 7b"          # Search GGUF models
mullama search "mistral" --files   # Show available files
mullama search "phi" --all         # Include non-GGUF models
mullama info TheBloke/Llama-2-7B-GGUF  # Show repo details

# Cache management
mullama pull hf:TheBloke/Llama-2-7B-GGUF  # Pre-download model
mullama cache list        # List cached models
mullama cache size        # Show cache size
mullama cache clear       # Clear cache

# Use OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### HuggingFace Model Format

```
hf:<owner>/<repo>:<filename>   # Specific file
hf:<owner>/<repo>              # Auto-detect best GGUF
<alias>:hf:<owner>/<repo>      # With custom alias
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for gated/private models |
| `MULLAMA_CACHE_DIR` | Override default cache directory |

### Cache Locations (Cross-Platform)

| Platform | Default Location |
|----------|-----------------|
| Linux | `$XDG_CACHE_HOME/mullama/models` or `~/.cache/mullama/models` |
| macOS | `~/Library/Caches/mullama/models` |
| Windows | `%LOCALAPPDATA%\mullama\models` |

Architecture:
```
                                   ┌──────────────────────────────────┐
                                   │           Daemon                 │
┌─────────────┐                    │  ┌────────────────────────────┐  │
│  TUI Client │◄── nng (IPC) ─────►│  │     Model Manager          │  │
└─────────────┘                    │  │  ┌───────┐  ┌───────┐      │  │
                                   │  │  │Model 1│  │Model 2│ ...  │  │
┌─────────────┐                    │  │  └───────┘  └───────┘      │  │
│   curl/app  │◄── HTTP/REST ─────►│  └────────────────────────────┘  │
└─────────────┘   (OpenAI API)     │                                  │
                                   │  Endpoints:                      │
┌─────────────┐                    │  • /v1/chat/completions          │
│ Other Client│◄── nng (IPC) ─────►│  • /v1/completions               │
└─────────────┘                    │  • /v1/models                    │
                                   │  • /v1/embeddings                │
                                   └──────────────────────────────────┘
```

Programmatic usage:
```rust
use mullama::daemon::{DaemonClient, DaemonBuilder};

// Connect as client
let client = DaemonClient::connect_default()?;
let result = client.chat("Hello, AI!", None, 100, 0.7)?;
println!("{} ({:.1} tok/s)", result.text, result.tokens_per_second());

// List models
for model in client.list_models()? {
    println!("{}: {}M params", model.alias, model.info.parameters / 1_000_000);
}
```

## Late Interaction / ColBERT

Mullama supports ColBERT-style late interaction retrieval with multi-vector embeddings. Unlike traditional embeddings that pool all tokens into a single vector, late interaction preserves per-token embeddings for fine-grained matching using MaxSim scoring.

```rust
use mullama::late_interaction::{
    MultiVectorGenerator, MultiVectorConfig, LateInteractionScorer
};
use std::sync::Arc;

// Create generator (works with any embedding model)
let model = Arc::new(Model::load("model.gguf")?);
let config = MultiVectorConfig::default()
    .normalize(true)
    .skip_special_tokens(true);
let mut generator = MultiVectorGenerator::new(model, config)?;

// Generate multi-vector embeddings
let query = generator.embed_text("What is machine learning?")?;
let doc = generator.embed_text("Machine learning is a branch of AI...")?;

// Score with MaxSim
let score = LateInteractionScorer::max_sim(&query, &doc);

// Top-k retrieval
let documents: Vec<_> = texts.iter()
    .map(|t| generator.embed_text(t))
    .collect::<Result<Vec<_>, _>>()?;
let top_k = LateInteractionScorer::find_top_k(&query, &documents, 10);
```

**With parallel processing:**
```rust
// Enable both features: ["late-interaction", "parallel"]
let top_k = LateInteractionScorer::find_top_k_parallel(&query, &documents, 10);
let scores = LateInteractionScorer::batch_score_parallel(&queries, &documents);
```

**Recommended models:**
- `LiquidAI/LFM2-ColBERT-350M-GGUF` - Purpose-trained ColBERT model
- Any GGUF embedding model (works but suboptimal for retrieval)

## GPU Acceleration

```bash
# NVIDIA CUDA
export LLAMA_CUDA=1

# Apple Metal (macOS)
export LLAMA_METAL=1

# AMD ROCm (Linux)
export LLAMA_HIPBLAS=1

# Intel OpenCL
export LLAMA_CLBLAST=1
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./docs/GETTING_STARTED.md) | Installation and first application |
| [Platform Setup](./docs/PLATFORM_SETUP.md) | OS-specific setup instructions |
| [Features Guide](./docs/FEATURES.md) | Integration features overview |
| [Use Cases](./docs/USE_CASES.md) | Real-world application examples |
| [API Reference](./docs/API_REFERENCE.md) | Complete API documentation |
| [Sampling Guide](./docs/sampling.md) | Sampling strategies and configuration |
| [GPU Guide](./docs/gpu.md) | GPU acceleration setup |
| [Feature Status](./docs/FEATURE_STATUS.md) | Implementation status and roadmap |

## Examples

```bash
# Basic text generation
cargo run --example simple --features async

# Streaming responses
cargo run --example streaming_generation --features "async,streaming"

# Web service
cargo run --example web_service --features "web,websockets"

# Audio processing
cargo run --example streaming_audio_demo --features "streaming-audio,multimodal"

# Late interaction / ColBERT retrieval
cargo run --example late_interaction --features late-interaction
cargo run --example late_interaction --features late-interaction -- model.gguf
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

```bash
git clone --recurse-submodules https://github.com/neul-labs/mullama.git
cd mullama
cargo test --all-features
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## llama.cpp Compatibility

Mullama tracks upstream llama.cpp releases:

| Mullama Version | llama.cpp Version | Release Date |
|-----------------|-------------------|--------------|
| 0.1.x           | b7542             | Dec 2025     |

### Supported Model Architectures

All architectures supported by llama.cpp b7542, including:
- LLaMA 1/2/3, Mistral, Mixtral, Phi-1/2/3/4
- Qwen, Qwen2, DeepSeek, Yi, Gemma
- And [many more](https://github.com/ggml-org/llama.cpp#supported-models)

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying inference engine
- [ggml](https://github.com/ggerganov/ggml) - Tensor operations library
