# Feature Matrix

This page provides a comprehensive view of Mullama's feature status across all dimensions: implementation maturity, platform support, language binding parity, GPU backends, and quantization formats.

!!! info "Current Version"
    **Mullama v0.1.1** | Built on **llama.cpp b7542** | Last updated: January 2026

## Feature Status Overview

### Core Features

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Model loading (GGUF) | :material-check-circle:{ .green } Stable | v0.1.0 | All GGUF versions supported |
| Tokenization | :material-check-circle:{ .green } Stable | v0.1.0 | SPM, BPE, WPM, UGM, RWKV, PLaMo-2 |
| Text generation | :material-check-circle:{ .green } Stable | v0.1.0 | Full pipeline with callbacks |
| Context management | :material-check-circle:{ .green } Stable | v0.1.0 | Create, evaluate, reset |
| KV cache operations | :material-check-circle:{ .green } Stable | v0.1.0 | Save, load, clear, defragment |
| Session persistence | :material-check-circle:{ .green } Stable | v0.1.0 | State save/restore to file |
| Batch processing | :material-check-circle:{ .green } Stable | v0.1.0 | Multi-sequence evaluation |
| Memory management | :material-check-circle:{ .green } Stable | v0.1.0 | RAII, mmap, custom allocators |
| Model metadata access | :material-check-circle:{ .green } Stable | v0.1.0 | All GGUF metadata fields |
| Automatic batch chunking | :material-check-circle:{ .green } Stable | v0.1.0 | Long prompt handling |

### Streaming

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Token-by-token callbacks | :material-check-circle:{ .green } Stable | v0.1.0 | Synchronous callbacks |
| Channel-based streaming | :material-check-circle:{ .green } Stable | v0.1.0 | Tokio mpsc channels |
| Backpressure handling | :material-check-circle:{ .green } Stable | v0.1.0 | Configurable buffer sizes |
| Stream cancellation | :material-check-circle:{ .green } Stable | v0.1.0 | Graceful stop |
| SSE (Server-Sent Events) | :material-check-circle:{ .green } Stable | v0.1.0 | OpenAI-compatible format |
| WebSocket streaming | :material-check-circle:{ .green } Stable | v0.1.0 | Bidirectional |
| Streaming configuration | :material-check-circle:{ .green } Stable | v0.1.0 | StreamConfig builder |

### Async Support

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Tokio runtime integration | :material-check-circle:{ .green } Stable | v0.1.0 | Multi-threaded runtime |
| Async model operations | :material-check-circle:{ .green } Stable | v0.1.0 | Non-blocking load/unload |
| Async generation | :material-check-circle:{ .green } Stable | v0.1.0 | Spawn on blocking pool |
| Async streaming | :material-check-circle:{ .green } Stable | v0.1.0 | Stream trait implementation |
| Task cancellation | :material-check-circle:{ .green } Stable | v0.1.0 | CancellationToken support |

### Sampling Strategies

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Temperature | :material-check-circle:{ .green } Stable | v0.1.0 | Range: 0.0 - 2.0 |
| Top-K | :material-check-circle:{ .green } Stable | v0.1.0 | Token count filtering |
| Top-P (Nucleus) | :material-check-circle:{ .green } Stable | v0.1.0 | Cumulative probability |
| Min-P | :material-check-circle:{ .green } Stable | v0.1.0 | Minimum probability threshold |
| Typical-P | :material-check-circle:{ .green } Stable | v0.1.0 | Information-theoretic |
| Mirostat v1 | :material-check-circle:{ .green } Stable | v0.1.0 | Perplexity-controlled |
| Mirostat v2 | :material-check-circle:{ .green } Stable | v0.1.0 | Improved perplexity control |
| Tail-Free Sampling | :material-check-circle:{ .green } Stable | v0.1.0 | Second derivative filtering |
| Frequency penalty | :material-check-circle:{ .green } Stable | v0.1.0 | Token frequency reduction |
| Presence penalty | :material-check-circle:{ .green } Stable | v0.1.0 | Token presence penalty |
| Repeat penalty | :material-check-circle:{ .green } Stable | v0.1.0 | N-gram repetition control |
| Logit bias | :material-check-circle:{ .green } Stable | v0.1.0 | Per-token logit adjustment |
| Dry sampling | :material-check-circle:{ .green } Stable | v0.1.0 | Diversity-promoting |
| Sampler chains | :material-check-circle:{ .green } Stable | v0.1.0 | Composable pipelines |
| Custom stopping criteria | :material-check-circle:{ .green } Stable | v0.1.0 | User-defined stop conditions |

### Web & API

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Axum HTTP server | :material-check-circle:{ .green } Stable | v0.1.0 | High-performance async |
| OpenAI API (`/v1/chat/completions`) | :material-check-circle:{ .green } Stable | v0.1.0 | Full compatibility |
| OpenAI API (`/v1/completions`) | :material-check-circle:{ .green } Stable | v0.1.0 | Text completions |
| OpenAI API (`/v1/embeddings`) | :material-check-circle:{ .green } Stable | v0.1.0 | Embedding generation |
| OpenAI API (`/v1/models`) | :material-check-circle:{ .green } Stable | v0.1.0 | Model listing |
| Anthropic API (`/v1/messages`) | :material-check-circle:{ .green } Stable | v0.1.1 | Messages format |
| WebSocket support | :material-check-circle:{ .green } Stable | v0.1.0 | Real-time bidirectional |
| Prometheus metrics | :material-check-circle:{ .green } Stable | v0.1.1 | `/metrics` endpoint |
| Health check | :material-check-circle:{ .green } Stable | v0.1.0 | `/health` endpoint |
| CORS configuration | :material-check-circle:{ .green } Stable | v0.1.0 | Configurable origins |

### Multimodal

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| MultimodalProcessor | :material-check-circle:{ .green } Stable | v0.1.0 | Unified pipeline |
| Image input (JPEG) | :material-check-circle:{ .green } Stable | v0.1.0 | Via image crate |
| Image input (PNG) | :material-check-circle:{ .green } Stable | v0.1.0 | Via image crate |
| Image input (WebP) | :material-check-circle:{ .green } Stable | v0.1.0 | Via image crate |
| Image format conversion | :material-check-circle:{ .green } Stable | v0.1.0 | Between supported formats |
| Vision projector support | :material-check-circle:{ .green } Stable | v0.1.0 | mmproj models |
| Text + image combined | :material-check-circle:{ .green } Stable | v0.1.0 | Interleaved processing |

### Audio

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| StreamingAudioProcessor | :material-check-circle:{ .green } Stable | v0.1.0 | Real-time capture |
| Voice Activity Detection | :material-check-circle:{ .green } Stable | v0.1.0 | Energy + zero-crossing |
| Noise reduction | :material-check-circle:{ .green } Stable | v0.1.0 | Spectral subtraction |
| Ring buffer architecture | :material-check-circle:{ .green } Stable | v0.1.0 | Low-latency processing |
| WAV format | :material-check-circle:{ .green } Stable | v0.1.0 | Read/write |
| MP3 format | :material-check-circle:{ .green } Stable | v0.1.0 | Read (decode) |
| FLAC format | :material-check-circle:{ .green } Stable | v0.1.0 | Read/write |
| Opus format | :material-check-circle:{ .green } Stable | v0.1.0 | Read (decode) |
| Audio format conversion | :material-check-circle:{ .green } Stable | v0.1.0 | Between supported formats |
| AudioStreamConfig builder | :material-check-circle:{ .green } Stable | v0.1.0 | Configurable pipeline |

### Embeddings & Retrieval

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Single embedding generation | :material-check-circle:{ .green } Stable | v0.1.0 | Per-text embedding |
| Batch embedding | :material-check-circle:{ .green } Stable | v0.1.0 | Multiple texts |
| Cosine similarity | :material-check-circle:{ .green } Stable | v0.1.0 | Standard metric |
| Dot product similarity | :material-check-circle:{ .green } Stable | v0.1.0 | Standard metric |
| ColBERT multi-vector | :material-check-circle:{ .green } Stable | v0.1.1 | Per-token embeddings |
| MaxSim scoring | :material-check-circle:{ .green } Stable | v0.1.1 | Late interaction |
| Top-k retrieval | :material-check-circle:{ .green } Stable | v0.1.1 | Parallel ranking |
| Token-level analysis | :material-check-circle:{ .green } Stable | v0.1.1 | Token similarity maps |
| Normalized scoring | :material-check-circle:{ .green } Stable | v0.1.1 | Unit-normalized vectors |
| Symmetric scoring | :material-check-circle:{ .green } Stable | v0.1.1 | Bidirectional similarity |

### Daemon & CLI

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| `mullama run` | :material-check-circle:{ .green } Stable | v0.1.0 | One-shot generation |
| `mullama serve` | :material-check-circle:{ .green } Stable | v0.1.0 | Start HTTP server |
| `mullama pull` | :material-check-circle:{ .green } Stable | v0.1.1 | Download from registry |
| `mullama list` | :material-check-circle:{ .green } Stable | v0.1.1 | List available models |
| `mullama show` | :material-check-circle:{ .green } Stable | v0.1.1 | Model details |
| `mullama create` | :material-check-circle:{ .green } Stable | v0.1.1 | Create from Modelfile |
| `mullama rm` | :material-check-circle:{ .green } Stable | v0.1.1 | Remove model |
| `mullama cp` | :material-check-circle:{ .green } Stable | v0.1.1 | Copy/alias model |
| `mullama ps` | :material-check-circle:{ .green } Stable | v0.1.1 | Show running models |
| `mullama chat` | :material-check-circle:{ .green } Stable | v0.1.1 | TUI chat interface |
| `mullama daemon start/stop/status` | :material-check-circle:{ .green } Stable | v0.1.1 | Lifecycle management |
| `mullama daemon logs` | :material-check-circle:{ .green } Stable | v0.1.1 | Log viewing |
| Auto-spawn | :material-check-circle:{ .green } Stable | v0.1.1 | Daemon on demand |
| Model aliases (40+) | :material-check-circle:{ .green } Stable | v0.1.1 | Pre-configured |
| Modelfile support | :material-check-circle:{ .green } Stable | v0.1.1 | Ollama-compatible + extensions |
| Embedded Web UI | :material-check-circle:{ .green } Stable | v0.1.1 | Vue.js frontend |

### Model Adaptation

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| LoRA adapter loading | :material-check-circle:{ .green } Stable | v0.1.0 | Single adapter |
| Multiple LoRA adapters | :material-check-circle:{ .green } Stable | v0.1.0 | Simultaneous |
| Dynamic LoRA scale | :material-check-circle:{ .green } Stable | v0.1.0 | Runtime weight adjustment |
| LoRA metadata access | :material-check-circle:{ .green } Stable | v0.1.0 | Parameter info |
| Control vectors (basic) | :material-progress-clock:{ .amber } Beta | v0.1.1 | Data structures + FFI |
| Control vectors (API) | :material-progress-clock:{ .amber } Beta | -- | High-level wrapper |
| Control vector loading | :material-progress-clock:{ .amber } Beta | -- | File format support |

### Structured Output

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| JSON mode | :material-check-circle:{ .green } Stable | v0.1.0 | Force JSON output |
| GBNF grammar parsing | :material-check-circle:{ .green } Stable | v0.1.0 | Full parser |
| Grammar-constrained sampling | :material-check-circle:{ .green } Stable | v0.1.0 | Token-level constraints |
| Simple pattern grammars | :material-check-circle:{ .green } Stable | v0.1.0 | Regex-like patterns |
| Complex grammar validation | :material-progress-clock:{ .amber } Beta | -- | Edge case testing |
| Grammar composition | :material-calendar:{ .red } Planned | v0.2.0 | Combine grammars |
| Dynamic grammar modification | :material-calendar:{ .red } Planned | v0.2.0 | Runtime changes |

### Advanced Features

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Parallel processing (Rayon) | :material-check-circle:{ .green } Stable | v0.1.0 | Work-stealing scheduler |
| SIMD optimizations | :material-check-circle:{ .green } Stable | v0.1.0 | AVX2, AVX-512, NEON |
| Flash Attention | :material-check-circle:{ .green } Stable | v0.1.0 | Auto/enabled/disabled |
| Memory-mapped I/O | :material-check-circle:{ .green } Stable | v0.1.0 | Efficient model loading |
| Multi-GPU layer splitting | :material-check-circle:{ .green } Stable | v0.1.0 | Across devices |
| Performance timing | :material-check-circle:{ .green } Stable | v0.1.0 | Per-operation metrics |
| Speculative decoding | :material-calendar:{ .red } Planned | v0.2.0 | Draft model acceleration |
| Runtime quantization | :material-calendar:{ .red } Planned | v0.2.0 | Dynamic precision |
| Distributed inference | :material-calendar:{ .red } Planned | v0.3.0 | Multi-node |

## Platform Support Matrix

### Core Library

| Feature | Linux (x86_64) | Linux (ARM64) | macOS (Apple Silicon) | macOS (Intel) | Windows (x86_64) |
|---------|:---:|:---:|:---:|:---:|:---:|
| Model loading | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Text generation | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Streaming | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Async (Tokio) | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Embeddings | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| ColBERT | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| LoRA adapters | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Grammar constraints | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Batch processing | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |

### Multimodal & Audio

| Feature | Linux (x86_64) | Linux (ARM64) | macOS (Apple Silicon) | macOS (Intel) | Windows (x86_64) |
|---------|:---:|:---:|:---:|:---:|:---:|
| Image processing | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Audio capture | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Voice activity detection | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Audio format conversion | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Streaming audio | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |

### Web & Daemon

| Feature | Linux (x86_64) | Linux (ARM64) | macOS (Apple Silicon) | macOS (Intel) | Windows (x86_64) |
|---------|:---:|:---:|:---:|:---:|:---:|
| HTTP server | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| WebSocket | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Daemon mode | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Web UI | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| TUI chat | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Prometheus metrics | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Auto-spawn daemon | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |

### Performance Optimizations

| Feature | Linux (x86_64) | Linux (ARM64) | macOS (Apple Silicon) | macOS (Intel) | Windows (x86_64) |
|---------|:---:|:---:|:---:|:---:|:---:|
| AVX2 | :material-check-circle:{ .green } | -- | -- | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| AVX-512 | :material-check-circle:{ .green } | -- | -- | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| ARM NEON | -- | :material-check-circle:{ .green } | :material-check-circle:{ .green } | -- | -- |
| CUDA | :material-check-circle:{ .green } | :material-check-circle:{ .green } | -- | -- | :material-check-circle:{ .green } |
| Metal | -- | -- | :material-check-circle:{ .green } | :material-check-circle:{ .green } | -- |
| ROCm | :material-check-circle:{ .green } | -- | -- | -- | :material-check-circle:{ .green } |
| OpenCL | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Rayon parallelism | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Memory mapping | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |

## Language Binding Feature Parity

This table shows which features are available in each language binding. The Rust core always has full feature access.

| Feature | Rust | Node.js | Python | Go | PHP | C/C++ |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Model loading | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Text generation | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Tokenization | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Streaming | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Async generation | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-minus-circle:{ .amber } |
| Embeddings | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Batch embedding | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| ColBERT scoring | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| Sampling config | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Grammar constraints | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| LoRA adapters | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Image input | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| Audio input | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| Session save/load | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| GPU configuration | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Performance metrics | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } | :material-check-circle:{ .green } |
| Control vectors | :material-progress-clock:{ .amber } | :material-calendar:{ .red } | :material-calendar:{ .red } | :material-calendar:{ .red } | :material-calendar:{ .red } | :material-progress-clock:{ .amber } |

**Legend**: :material-check-circle:{ .green } Full support | :material-minus-circle:{ .amber } Partial/limited | :material-progress-clock:{ .amber } In development | :material-calendar:{ .red } Planned

### Binding Technology Details

| Language | Technology | Thread Safety | Async Model | Package |
|----------|-----------|:---:|-------------|---------|
| Rust | Native | :material-check-circle:{ .green } Send+Sync | Tokio | `mullama` (crate) |
| Node.js | NAPI-RS | :material-check-circle:{ .green } ThreadsafeFunction | Promises/async-await | `mullama` (npm) |
| Python | PyO3 | :material-check-circle:{ .green } GIL-aware | asyncio compatible | `mullama` (pip) |
| Go | cgo | :material-check-circle:{ .green } goroutine-safe | Goroutines | `mullama-go` (module) |
| PHP | FFI | :material-minus-circle:{ .amber } Single-thread | N/A | `mullama-php` |
| C/C++ | Direct FFI | :material-check-circle:{ .green } User-managed | User-managed | `libmullama.h` |

## GPU Backend Support

### Backend Availability

| Backend | Linux | macOS | Windows | Min Driver | Notes |
|---------|:---:|:---:|:---:|-----------|-------|
| CUDA | :material-check-circle:{ .green } | -- | :material-check-circle:{ .green } | 525.60+ | NVIDIA GPUs |
| Metal | -- | :material-check-circle:{ .green } | -- | macOS 13+ | Apple Silicon + Intel |
| ROCm | :material-check-circle:{ .green } | -- | :material-check-circle:{ .green } | 5.5+ | AMD GPUs |
| OpenCL | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | 1.2+ | Cross-vendor |
| Vulkan | :material-calendar:{ .red } | :material-calendar:{ .red } | :material-calendar:{ .red } | -- | Planned |

### GPU Feature Support by Backend

| Feature | CUDA | Metal | ROCm | OpenCL |
|---------|:---:|:---:|:---:|:---:|
| Full layer offloading | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Partial layer offloading | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| Multi-GPU | :material-check-circle:{ .green } | -- | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } |
| Flash Attention | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } |
| FP16 compute | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } |
| BF16 compute | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-check-circle:{ .green } | :material-minus-circle:{ .amber } |
| INT8 tensor cores | :material-check-circle:{ .green } | -- | :material-minus-circle:{ .amber } | -- |

### Environment Variables for GPU

```bash
# Enable specific backends (set before building)
export LLAMA_CUDA=1        # NVIDIA CUDA
export LLAMA_METAL=1       # Apple Metal
export LLAMA_HIPBLAS=1     # AMD ROCm (HIP)
export LLAMA_CLBLAST=1     # OpenCL (CLBlast)

# CUDA-specific
export CUDA_VISIBLE_DEVICES=0,1    # Select GPUs
export LLAMA_CUDA_F16=1            # Force FP16

# Metal-specific (auto-detected on macOS)
# No additional configuration needed

# ROCm-specific
export HIP_VISIBLE_DEVICES=0,1     # Select GPUs
```

## Quantization Format Support

Mullama supports all GGUF quantization types provided by llama.cpp:

### Integer Quantization

| Format | Bits/Weight | Quality | Speed | Memory (7B) | Notes |
|--------|:---:|---------|-------|-------------|-------|
| Q2_K | 2.63 | Low | Fastest | ~2.6 GB | Extreme compression |
| Q3_K_S | 3.44 | Low-Med | Fast | ~3.0 GB | Small K-quant |
| Q3_K_M | 3.91 | Medium | Fast | ~3.3 GB | Medium K-quant |
| Q3_K_L | 4.27 | Med-High | Fast | ~3.6 GB | Large K-quant |
| Q4_0 | 4.50 | Medium | Fast | ~3.8 GB | Legacy, uniform |
| Q4_1 | 5.00 | Medium | Fast | ~4.2 GB | Legacy, with offset |
| Q4_K_S | 4.58 | Med-High | Fast | ~3.9 GB | Small K-quant |
| Q4_K_M | 4.85 | High | Balanced | ~4.1 GB | **Recommended default** |
| Q5_0 | 5.50 | High | Balanced | ~4.6 GB | Legacy, uniform |
| Q5_1 | 6.00 | High | Balanced | ~5.0 GB | Legacy, with offset |
| Q5_K_S | 5.54 | High | Balanced | ~4.7 GB | Small K-quant |
| Q5_K_M | 5.69 | Very High | Balanced | ~4.8 GB | Medium K-quant |
| Q6_K | 6.56 | Very High | Slower | ~5.5 GB | Highest integer quality |
| Q8_0 | 8.50 | Near-FP16 | Slower | ~7.1 GB | High quality |

### Floating Point Formats

| Format | Bits/Weight | Quality | Speed | Memory (7B) | Notes |
|--------|:---:|---------|-------|-------------|-------|
| F16 | 16.00 | Reference | Slow | ~13.4 GB | Half precision |
| F32 | 32.00 | Maximum | Slowest | ~26.8 GB | Full precision |
| BF16 | 16.00 | Reference | Slow | ~13.4 GB | Brain floating point |

### I-Quant Formats

| Format | Bits/Weight | Quality | Speed | Memory (7B) | Notes |
|--------|:---:|---------|-------|-------------|-------|
| IQ1_S | 1.56 | Very Low | Fastest | ~1.6 GB | Extreme compression |
| IQ1_M | 1.75 | Low | Fastest | ~1.8 GB | Improved 1-bit |
| IQ2_XXS | 2.06 | Low | Very Fast | ~2.0 GB | Ultra-small |
| IQ2_XS | 2.31 | Low | Very Fast | ~2.2 GB | Extra-small |
| IQ2_S | 2.50 | Low-Med | Very Fast | ~2.4 GB | Small |
| IQ2_M | 2.70 | Medium | Fast | ~2.5 GB | Medium |
| IQ3_XXS | 3.06 | Medium | Fast | ~2.8 GB | Ultra-small 3-bit |
| IQ3_XS | 3.30 | Medium | Fast | ~3.0 GB | Extra-small 3-bit |
| IQ3_S | 3.44 | Med-High | Fast | ~3.2 GB | Small 3-bit |
| IQ3_M | 3.70 | Med-High | Fast | ~3.4 GB | Medium 3-bit |
| IQ4_NL | 4.50 | High | Balanced | ~3.8 GB | Non-linear 4-bit |
| IQ4_XS | 4.25 | High | Balanced | ~3.6 GB | Extra-small 4-bit |

!!! tip "Choosing a Quantization"
    For most use cases, **Q4_K_M** provides the best balance of quality, speed, and memory usage. Use Q5_K_M or Q6_K when quality is paramount. Use Q2_K or IQ2 variants for extremely memory-constrained environments (mobile, edge devices).

### Quantization Selection Guide

```
Quality vs Memory Tradeoff:

Quality
  |
  |  F16/F32 ●
  |
  |          Q8_0 ●
  |       Q6_K ●
  |     Q5_K_M ●
  |    Q5_K_S ●
  |   Q4_K_M ●  ← Recommended
  |  Q4_K_S ●
  | Q3_K_M ●
  |Q3_K_S ●
  |Q2_K ●
  +──────────────────────── Memory Usage
  Low                      High
```

## Feature Flags (Cargo)

Mullama uses Cargo feature flags to control compilation. Features can be combined:

| Feature | Description | Dependencies |
|---------|-------------|-------------|
| `async` | Tokio async runtime support | tokio |
| `streaming` | Real-time token streaming | -- |
| `multimodal` | Image processing pipeline | image crate |
| `streaming-audio` | Real-time audio capture + VAD | multimodal, audio libs |
| `format-conversion` | Audio/image format conversion | multimodal |
| `web` | Axum HTTP server framework | async, axum |
| `websockets` | WebSocket support | web, tokio-tungstenite |
| `parallel` | Rayon batch processing | rayon |
| `daemon` | CLI daemon mode | web, websockets |
| `embedded-ui` | Embedded Vue.js Web UI | daemon |
| `full` | All features enabled | all of the above |

### Feature Dependency Chain

```
full
├── daemon
│   ├── embedded-ui
│   ├── web
│   │   ├── async
│   │   └── websockets
│   └── parallel
├── streaming-audio
│   └── multimodal
├── format-conversion
│   └── multimodal
└── streaming
```

### Build Examples

```bash
# Minimal (core inference only)
cargo build --release --no-default-features

# Library with async and streaming
cargo build --release --features "async,streaming"

# Multimodal with audio
cargo build --release --features "multimodal,streaming-audio"

# Web server
cargo build --release --features "web,websockets"

# Full daemon with Web UI
cargo build --release --features "daemon,embedded-ui"

# Everything
cargo build --release --features full
```
