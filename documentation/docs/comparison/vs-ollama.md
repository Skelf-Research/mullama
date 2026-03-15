# Mullama vs Ollama

Both Mullama and Ollama are tools for running large language models locally. They both use llama.cpp as the underlying inference engine and support the same GGUF model format. The fundamental difference lies in their architecture: Mullama is a **library-first** tool that can also serve as a daemon, while Ollama is a **server-first** tool that exposes models exclusively through HTTP APIs.

This page provides a detailed technical comparison to help you make an informed decision.

## Philosophy Comparison

| Aspect | Mullama | Ollama |
|--------|---------|--------|
| **Primary design** | Embeddable library | Standalone server |
| **Integration model** | Native function calls | HTTP REST API |
| **Language support** | 6 native bindings | REST wrappers |
| **Deployment** | In-process or daemon | Daemon only |
| **API surface** | Rust API + REST + WebSocket | REST only |
| **Target audience** | Application developers | Model operators |

## Architecture Diagrams

### Mullama: In-Process Architecture

When used as a library, Mullama runs entirely within your application's process space. There is no inter-process communication, no serialization, and no network stack involved.

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Native Binding Layer                   │   │
│   │  (NAPI-RS / PyO3 / cgo / FFI - language specific)  │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          │ Direct function call              │
│   ┌──────────────────────▼──────────────────────────────┐   │
│   │              Mullama Rust Core                       │   │
│   │                                                     │   │
│   │  Model ──► Context ──► Sampling ──► Token Output    │   │
│   │    │                                                │   │
│   │    ├── Batch Processing (Rayon)                     │   │
│   │    ├── Embedding Generation                         │   │
│   │    ├── Multimodal Pipeline                          │   │
│   │    └── Audio Processing (VAD)                       │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          │ Safe FFI                          │
│   ┌──────────────────────▼──────────────────────────────┐   │
│   │              llama.cpp Engine                        │   │
│   │  (GGUF loading, GPU offload, SIMD, mmap)           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Ollama: Out-of-Process Architecture

Ollama operates as a separate server process. Every interaction traverses the network stack, even on localhost.

```
┌──────────────────┐          ┌──────────────────────────────────────┐
│   Your App       │          │         Ollama Server                │
│                  │          │                                      │
│  ┌────────────┐  │  HTTP    │  ┌──────────────────────────────┐   │
│  │ HTTP Client├──┼─────────►│  │  HTTP Router (Go net/http)   │   │
│  │ (fetch/    │  │  JSON    │  └──────────────┬───────────────┘   │
│  │  requests) │  │          │                 │                    │
│  └────────────┘  │          │  ┌──────────────▼───────────────┐   │
│                  │          │  │  Request Handler              │   │
└──────────────────┘          │  │  (JSON parse, validate)      │   │
                              │  └──────────────┬───────────────┘   │
┌──────────────────┐          │                 │                    │
│  Another Client  │  HTTP    │  ┌──────────────▼───────────────┐   │
│  (curl, browser) ├──────────►  │  Go Runtime                  │   │
└──────────────────┘          │  │  (GC pauses, goroutines)     │   │
                              │  └──────────────┬───────────────┘   │
                              │                 │ cgo                │
                              │  ┌──────────────▼───────────────┐   │
                              │  │  llama.cpp Engine             │   │
                              │  └──────────────────────────────┘   │
                              │                                      │
                              └──────────────────────────────────────┘
```

### The Overhead Path

Every HTTP-based inference call follows this path:

```
App ──► Serialize JSON ──► TCP connect ──► HTTP headers ──► Send body
                                                                │
Result ◄── Parse JSON ◄── Read response ◄── Process request ◄──┘
```

A native binding call follows this path:

```
App ──► Function call ──► Result
```

## Comprehensive Feature Comparison

### Core Inference

| Feature | Mullama | Ollama |
|---------|---------|--------|
| GGUF model loading | Yes | Yes |
| Text generation | Yes | Yes |
| Tokenization API | Yes (6 vocab types) | Limited |
| Detokenization | Yes | Yes |
| Token probabilities | Yes (logits access) | No |
| Context window configuration | Yes (per-context) | Yes (per-model) |
| KV cache management | Yes (save/load/clear) | Internal only |
| Session persistence | Yes (state save/restore) | No |
| Batch processing (multi-sequence) | Yes | No |

### API & Protocol Support

| Feature | Mullama | Ollama |
|---------|---------|--------|
| REST API | Yes | Yes |
| OpenAI-compatible API | Yes (`/v1/chat/completions`) | Yes (`/api/chat`) |
| Anthropic-compatible API | Yes (`/v1/messages`) | No |
| WebSocket streaming | Yes (bidirectional) | No |
| Server-Sent Events (SSE) | Yes | No (newline-delimited JSON) |
| gRPC | Planned | No |
| Native function call API | Yes (6 languages) | No |
| IPC socket | Yes | No |

### Streaming & Async

| Feature | Mullama | Ollama |
|---------|---------|--------|
| Token-by-token streaming | Yes (callback + channel) | Yes (HTTP stream) |
| Async/await support | Yes (Tokio) | N/A (server-side) |
| Backpressure handling | Yes (configurable) | No |
| Stream cancellation | Yes (graceful) | Yes (connection close) |
| Multiple concurrent streams | Yes | Yes |
| WebSocket streaming | Yes | No |

### Embeddings & Retrieval

| Feature | Mullama | Ollama |
|---------|---------|--------|
| Embedding generation | Yes | Yes |
| Batch embedding | Yes (parallel) | Sequential |
| ColBERT (late interaction) | Yes (MaxSim scoring) | No |
| Multi-vector embeddings | Yes (per-token) | No |
| Similarity scoring | Yes (cosine, dot, MaxSim) | No |
| Top-k retrieval | Yes (parallel) | No |

### Multimodal

| Feature | Mullama | Ollama |
|---------|---------|--------|
| Vision (image input) | Yes | Yes |
| Image format conversion | Yes (JPEG/PNG/WebP) | Limited |
| Audio input | Yes (WAV/MP3/FLAC/Opus) | No |
| Streaming audio capture | Yes (real-time) | No |
| Voice Activity Detection | Yes (built-in) | No |
| Noise reduction | Yes | No |
| Combined text+image+audio | Yes (unified pipeline) | Text+image only |

### Structured Output & Constraints

| Feature | Mullama | Ollama |
|---------|---------|--------|
| JSON mode | Yes | Yes |
| Grammar constraints (GBNF) | Yes | Yes |
| Structured output schemas | Yes | Yes |
| Grammar composition | In progress | No |
| Custom stopping criteria | Yes | Limited |

### Sampling & Generation Control

| Feature | Mullama | Ollama |
|---------|---------|--------|
| Temperature | Yes | Yes |
| Top-K | Yes | Yes |
| Top-P (Nucleus) | Yes | Yes |
| Min-P | Yes | Yes |
| Typical-P | Yes | No |
| Mirostat v1/v2 | Yes | Yes |
| Tail-Free Sampling | Yes | No |
| Frequency penalty | Yes | Yes |
| Presence penalty | Yes | Yes |
| Repeat penalty | Yes | Yes |
| Logit bias | Yes | No |
| Sampler chains (composable) | Yes | No |
| Dry sampling | Yes | No |

### Model Adaptation

| Feature | Mullama | Ollama |
|---------|---------|--------|
| LoRA adapter loading | Yes | Yes |
| Multiple simultaneous LoRA | Yes | No |
| Dynamic LoRA weight adjustment | Yes (runtime scale) | No |
| Control vectors | In progress | No |
| Speculative decoding | Planned (v0.2.0) | No |

### Performance & Optimization

| Feature | Mullama | Ollama |
|---------|---------|--------|
| GPU acceleration (CUDA) | Yes | Yes |
| GPU acceleration (Metal) | Yes | Yes |
| GPU acceleration (ROCm) | Yes | Yes |
| GPU acceleration (OpenCL) | Yes | No |
| SIMD (AVX2) | Yes | Via llama.cpp |
| SIMD (AVX-512) | Yes | Via llama.cpp |
| SIMD (ARM NEON) | Yes | Via llama.cpp |
| Flash Attention | Yes (auto/enabled/disabled) | Yes |
| Memory mapping (mmap) | Yes | Yes |
| Multi-GPU layer splitting | Yes | Yes |
| Parallel processing (Rayon) | Yes (work-stealing) | No |
| Batch memory optimization | Yes | No |

### Deployment & Operations

| Feature | Mullama | Ollama |
|---------|---------|--------|
| Daemon mode | Yes (optional) | Yes (required) |
| Embedded library mode | Yes | No |
| Auto-spawn on demand | Yes | No |
| Embedded Web UI | Yes (Vue.js) | No |
| TUI chat interface | Yes | No |
| Prometheus metrics | Yes (`/metrics`) | No |
| Health check endpoint | Yes | Yes |
| Model aliases | Yes (40+ built-in) | Yes |
| Modelfile support | Yes (extended format) | Yes |
| Model pull from registry | Yes (HuggingFace) | Yes (ollama.com) |
| Single-binary deployment | Yes | Yes |
| Docker support | Yes | Yes |

### Language Bindings

| Language | Mullama | Ollama |
|----------|---------|--------|
| Rust | Native (core library) | N/A |
| Node.js | Native (NAPI-RS) | REST wrapper (ollama-js) |
| Python | Native (PyO3) | REST wrapper (ollama-python) |
| Go | Native (cgo) | REST wrapper / Go native |
| PHP | Native (FFI) | REST wrapper |
| C/C++ | Native (direct FFI) | N/A |
| Ruby | Planned | REST wrapper |
| Java/Kotlin | Planned | REST wrapper |

## Performance: Native vs HTTP

### Call Overhead Analysis

The fundamental performance difference is in call overhead -- the cost of initiating an inference operation, separate from the inference computation itself.

!!! warning "Understanding Overhead"
    Call overhead does not affect how fast the model generates tokens. It affects how quickly your application can start and stop inference operations, which matters for latency-sensitive applications, batch processing, and high-frequency API calls.

| Metric | Native Binding | HTTP (localhost) | Difference |
|--------|---------------|-----------------|------------|
| Call initiation | 1-5 microseconds | 1-5 milliseconds | 100-1000x |
| Data transfer | Memory pointer | JSON serialize + TCP | No serialization |
| Connection setup | None | TCP handshake | No connection |
| Per-token streaming | Channel/callback | SSE parsing | Lower per-token cost |
| Batch of 100 calls | ~0.5ms total overhead | ~200ms total overhead | 400x |

### Where It Matters

=== "High-Throughput Embedding"

    ```
    Generating 10,000 embeddings:

    Native binding:
      Call overhead: 10,000 x 3us = 30ms
      Total time dominated by computation

    HTTP REST:
      Call overhead: 10,000 x 3ms = 30,000ms (30 seconds!)
      Overhead alone exceeds many workloads
    ```

=== "Real-Time Streaming"

    ```
    Streaming 500 tokens at 50 tok/s:

    Native binding:
      Per-token overhead: ~1us (callback)
      Total streaming overhead: 0.5ms

    HTTP REST:
      Per-token overhead: ~0.5ms (SSE parse)
      Total streaming overhead: 250ms
      (Plus connection setup time)
    ```

=== "Batch Processing"

    ```
    Processing 1,000 short prompts:

    Native binding:
      Batch submitted as single call
      Rayon parallel processing across cores
      Total overhead: ~5ms

    HTTP REST:
      1,000 sequential HTTP requests
      Each with connection + serialize + parse
      Total overhead: ~3,000ms minimum
    ```

!!! note "When Overhead Does Not Matter"
    For a chatbot generating a single response of several hundred tokens, the actual inference time (seconds) vastly exceeds the call overhead (milliseconds). In this common case, both approaches perform similarly from the user's perspective. The difference becomes significant in programmatic, high-frequency, or batch scenarios.

## Use Case Recommendations

### Choose Mullama

!!! success "Embedding LLM in a product"
    If you are building a desktop application, mobile app, CLI tool, or edge device that needs built-in LLM capabilities, Mullama's library architecture is the natural fit. No daemon to manage, no network to configure, no process to monitor.

!!! success "Node.js / Python / Go application"
    When your application is written in a specific language and you want LLM inference to feel like calling any other library function, Mullama's native bindings provide that experience with minimal overhead.

!!! success "Advanced features (ColBERT, audio, WebSocket)"
    If your use case requires semantic search with late interaction, real-time audio processing, or WebSocket-based streaming, these capabilities are unique to Mullama.

!!! success "Multi-API server (OpenAI + Anthropic)"
    If you need to serve both OpenAI and Anthropic-compatible APIs from the same local server, Mullama's daemon mode provides this out of the box.

!!! success "High-throughput batch workloads"
    For embedding generation pipelines, batch inference, or any workload that makes many inference calls, native bindings eliminate the cumulative HTTP overhead.

### Choose Either

!!! info "Quick local experimentation"
    Both tools make it easy to download and run models locally. Ollama's simpler setup may be slightly faster for first-time use; Mullama's CLI is intentionally compatible and nearly identical in usage.

!!! info "Running models via OpenAI-compatible API"
    Both provide OpenAI-compatible endpoints. If your primary need is serving models through a standard API, either tool works well.

!!! info "Basic chat applications"
    For straightforward chat applications where latency is dominated by inference time rather than call overhead, both tools deliver similar user-facing performance.

### Choose Ollama

!!! note "Ecosystem integrations"
    If your workflow depends on tools with built-in Ollama support (LangChain, Open WebUI, Continue, etc.), the existing ecosystem integration may be the deciding factor.

!!! note "Simplest possible setup"
    If you want the absolute minimum friction to run a model locally and do not need library-level integration, Ollama's one-command install and run experience is optimized for this.

## Model Compatibility

Both Mullama and Ollama use the same underlying model format and inference engine:

| Aspect | Mullama | Ollama |
|--------|---------|--------|
| Model format | GGUF | GGUF |
| Inference engine | llama.cpp | llama.cpp |
| Quantization support | All GGUF types | All GGUF types |
| Model sources | HuggingFace, local files | ollama.com, local files |
| Modelfile format | Ollama-compatible + extensions | Standard |

!!! tip "Model Portability"
    Any GGUF model file works with both tools. If you have downloaded models through Ollama, you can point Mullama at the same GGUF files. No conversion is needed.

## Summary

| Dimension | Mullama | Ollama |
|-----------|---------|--------|
| **Architecture** | Library-first, optionally a server | Server-first, always a server |
| **Integration** | Native bindings (6 languages) | HTTP REST wrappers |
| **Call overhead** | Microseconds (function call) | Milliseconds (HTTP round-trip) |
| **API compatibility** | OpenAI + Anthropic + WebSocket | OpenAI |
| **Multimodal** | Text + Image + Audio (streaming) | Text + Image |
| **Advanced search** | ColBERT, MaxSim, parallel scoring | Not available |
| **Deployment** | Embedded, daemon, or hybrid | Daemon only |
| **Monitoring** | Prometheus, structured logs | Basic logging |
| **Maturity** | Newer, feature-rich | Established, large community |
| **Best for** | Product integration, advanced use | Simple serving, ecosystem |

Both tools share the same foundation (llama.cpp, GGUF models) and produce identical inference results for the same model and parameters. The choice comes down to how you want to integrate LLM inference into your workflow: as a library call or as an API request.
