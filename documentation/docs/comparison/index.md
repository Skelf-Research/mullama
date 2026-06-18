# Why Mullama

!!! tip "TL;DR"
    **Mullama = Ollama's features + native language bindings + zero HTTP overhead**

    - **Drop-in Ollama replacement:** Same CLI commands, same Modelfile format, same GGUF models
    - **Native bindings:** Python, Node.js, Go, PHP, Rust, C/C++ -- direct function calls, not HTTP
    - **All-in-one toolkit:** Library + daemon + CLI + Web UI + TUI + OpenAI/Anthropic APIs

Mullama takes a fundamentally different approach to local LLM inference. Where traditional tools like Ollama operate as standalone servers that applications communicate with over HTTP, Mullama is designed from the ground up as an **embeddable library** that integrates directly into your application -- with native bindings for multiple languages and an optional daemon when server functionality is needed.

## Library-First vs Server-First

The core architectural distinction is simple: Mullama treats inference as a **library call**, not a **network request**.

```
Server-First (Ollama)                    Library-First (Mullama)
=====================                    =======================

┌──────────────┐                        ┌──────────────────────────────┐
│  Your App    │                        │         Your App             │
│              │     Network            │                              │
│  HTTP Client ├────────────┐           │  ┌────────────────────────┐  │
│              │            │           │  │    Mullama Library     │  │
└──────────────┘            │           │  │                        │  │
                            ▼           │  │  Native Binding Layer  │  │
              ┌──────────────────┐      │  │  Rust Core (14k+ LoC) │  │
              │   LLM Server     │      │  │  llama.cpp Engine      │  │
              │  (Go runtime)    │      │  └────────────────────────┘  │
              │  (HTTP parsing)  │      │                              │
              │  (JSON ser/de)   │      └──────────────────────────────┘
              │  (llama.cpp)     │
              └──────────────────┘       In-process. Zero IPC. Zero overhead.
```

!!! info "The Core Insight"
    When your application needs LLM inference, why pay the cost of HTTP connections, JSON serialization, and inter-process communication on every single call? Mullama gives you direct, in-process access to inference -- the same way you would use any other library in your language of choice.

## Key Differentiators

<div class="grid cards" markdown>

-   **Native Language Bindings**

    ---

    First-class bindings for 6 languages -- not REST wrappers, but compiled native extensions with near-zero call overhead.

    | Language | Technology | Overhead |
    |----------|-----------|----------|
    | Rust | Native core | Zero |
    | Node.js | NAPI-RS | ~microseconds |
    | Python | PyO3 | ~microseconds |
    | Go | cgo | ~microseconds |
    | PHP | FFI | ~microseconds |
    | C/C++ | Direct FFI | Zero |

-   **Embeddable Architecture**

    ---

    Deploy LLM inference as part of your application binary. No external processes, no daemon management, no network configuration. Ship a single binary with embedded inference.

    - Desktop applications with local AI
    - CLI tools with built-in intelligence
    - Edge devices with offline inference
    - Embedded systems and IoT

-   **Multi-API Compatibility**

    ---

    When running in daemon mode, Mullama serves both OpenAI and Anthropic-compatible APIs simultaneously. Existing client libraries work without modification.

    - `POST /v1/chat/completions` (OpenAI)
    - `POST /v1/messages` (Anthropic)
    - `POST /v1/embeddings` (OpenAI)
    - `WS /ws/chat` (WebSocket)

-   **Advanced Multimodal Pipeline**

    ---

    A unified processing pipeline for text, images, and audio -- including real-time streaming audio with voice activity detection, noise reduction, and format conversion.

    - Text: 23+ sampling strategies
    - Image: JPEG, PNG, WebP processing
    - Audio: WAV, MP3, FLAC, Opus with VAD
    - Combined multimodal inference

-   **Production-Ready Infrastructure**

    ---

    Built for real deployments with monitoring, metrics, and operational tooling out of the box.

    - Prometheus metrics at `/metrics`
    - Embedded Web UI for management
    - TUI chat interface
    - Health checks and status endpoints
    - Configurable resource limits

-   **Advanced Retrieval & Search**

    ---

    ColBERT-style late interaction with MaxSim scoring, parallel batch processing with Rayon work-stealing, and SIMD-optimized vector operations.

    - Multi-vector embeddings (per-token)
    - Top-k document retrieval
    - Parallel scoring across CPU cores
    - AVX2/AVX-512/ARM NEON acceleration

</div>

## Choose Mullama When...

!!! success "Mullama is the right choice when you need:"

    **Embedded inference in your application**
    :   You are building a desktop app, CLI tool, or service that needs LLM capabilities without depending on an external server process.

    **Native SDK for your language**
    :   You want to call inference functions directly from Node.js, Python, Go, PHP, or C/C++ without HTTP overhead or JSON serialization.

    **Real-time audio and multimodal**
    :   Your application processes streaming audio with voice activity detection, or combines text, image, and audio in a single pipeline.

    **High-throughput batch processing**
    :   You are generating embeddings at scale, running semantic search, or processing many inference requests with maximum throughput.

    **Anthropic API compatibility**
    :   You have existing code using the Anthropic SDK and want to run it against local models without code changes.

    **WebSocket streaming**
    :   You need bidirectional real-time communication for chat interfaces, collaborative applications, or streaming pipelines.

    **Production monitoring**
    :   You require Prometheus metrics, structured logging, and operational visibility for deployed inference services.

    **Single-binary deployment**
    :   You want to ship one executable with all inference capabilities included -- no runtime dependencies, no separate processes.

## When Ollama Might Be Better

!!! note "Ollama remains a solid choice for:"

    **Quick local experimentation**
    :   If you just want to download a model and chat with it immediately, Ollama's one-command setup is hard to beat for simplicity.

    **Large existing ecosystem**
    :   Many third-party tools (LangChain, Continue, Open WebUI, etc.) have built-in Ollama integrations. If your workflow depends on these, the ecosystem matters.

    **Minimal configuration**
    :   Ollama's defaults work well for casual use. If you do not need advanced features, its simplicity is an advantage.

    **Model discovery hub**
    :   Ollama's model hub at ollama.com provides convenient browsing and one-click downloads for a curated model collection.

## Getting Started

The CLI experience is intentionally familiar if you are coming from Ollama:

=== "Daemon Mode (CLI)"

    ```bash
    # Pull and run a model (auto-spawns daemon)
    mullama run llama3.2:1b "What is Rust?"

    # Interactive chat
    mullama chat --model llama3.2:1b

    # Model management
    mullama pull qwen2.5:7b-instruct
    mullama list
    ```

=== "Library Mode (Rust)"

    ```rust
    use mullama::{Model, Context, ContextParams, SamplerParams};

    let model = Model::from_file("model.gguf")?;
    let mut context = Context::new(&model, ContextParams::default())?;

    let tokens = model.tokenize("Explain quantum computing.", true)?;
    let output = context.generate(&tokens, 256)?;
    println!("{}", model.detokenize(&output)?);
    ```

=== "Library Mode (Python)"

    ```python
    import mullama

    model = mullama.Model("model.gguf")
    context = model.create_context(n_ctx=4096)

    result = context.generate("Explain quantum computing.", max_tokens=256)
    print(result)
    ```

=== "Library Mode (Node.js)"

    ```javascript
    const { Model } = require('mullama');

    const model = new Model('model.gguf');
    const context = model.createContext({ nCtx: 4096 });

    const result = await context.generate('Explain quantum computing.', {
      maxTokens: 256
    });
    console.log(result);
    ```

## Learn More

<div class="grid cards" markdown>

-   **[Mullama vs Ollama](vs-ollama.md)**

    ---

    Detailed architecture comparison with feature-by-feature breakdown, performance analysis, and use case recommendations.

-   **[Benchmarks](benchmarks.md)**

    ---

    Quantitative performance data: inference throughput, binding overhead, memory efficiency, and batch processing.

-   **[Migration from Ollama](migration-from-ollama.md)**

    ---

    Step-by-step guide for moving from Ollama to Mullama -- CLI mapping, API endpoints, Modelfile compatibility, and code examples.

-   **[Feature Matrix](feature-matrix.md)**

    ---

    Comprehensive feature status across platforms, language bindings, GPU backends, and quantization formats.

</div>
