# Migration from Ollama

This guide provides a complete walkthrough for migrating from Ollama to Mullama. The transition is designed to be straightforward -- Mullama intentionally provides CLI and API compatibility with Ollama while extending functionality significantly.

!!! success "Key Migration Facts"
    - Both use the **GGUF** model format -- your models work without conversion
    - CLI commands use the same syntax and semantics
    - Modelfile format is fully backward-compatible
    - Migration can be done incrementally -- run both side by side

## CLI Command Mapping

Mullama provides equivalent commands for all Ollama operations, with the same syntax and behavior:

### Core Commands

| Ollama Command | Mullama Command | Behavior |
|----------------|-----------------|----------|
| `ollama run <model> "prompt"` | `mullama run <model> "prompt"` | Identical syntax, auto-spawns daemon |
| `ollama serve` | `mullama serve` | Extended options (port, model, GPU layers) |
| `ollama pull <model>` | `mullama pull <model>` | Same alias resolution, downloads from registry |
| `ollama list` | `mullama list` | Same output format |
| `ollama show <model>` | `mullama show <model>` | Additional flags available |
| `ollama show --modelfile` | `mullama show --modelfile` | Compatible format output |
| `ollama create -f Modelfile` | `mullama create -f Modelfile` | Extended Modelfile support |
| `ollama rm <model>` | `mullama rm <model>` | Identical syntax |
| `ollama cp <src> <dst>` | `mullama cp <src> <dst>` | Identical syntax |
| `ollama ps` | `mullama ps` | Shows running models |

### Additional Mullama Commands

Commands not available in Ollama:

```bash
# Daemon lifecycle management
mullama daemon start              # Start daemon in background
mullama daemon stop               # Stop running daemon
mullama daemon restart            # Restart daemon
mullama daemon status             # Show daemon health, loaded models, memory
mullama daemon logs               # View daemon logs
mullama daemon logs -f            # Follow logs in real-time

# Interactive TUI chat
mullama chat                      # Full TUI chat interface
mullama chat --model llama3.2:1b  # Specify model for TUI

# Extended serve options
mullama serve --model llama3.2:1b --http-port 9090 --gpu-layers 35
mullama serve --flash-attention auto --context-size 8192
```

### Command Examples Side by Side

=== "Model Management"

    ```bash
    # Pull a model
    ollama pull llama3.2:1b          # Ollama
    mullama pull llama3.2:1b         # Mullama (identical)

    # List models
    ollama list                      # Ollama
    mullama list                     # Mullama (identical)

    # Show model info
    ollama show llama3.2:1b          # Ollama
    mullama show llama3.2:1b         # Mullama (identical)

    # Remove a model
    ollama rm llama3.2:1b            # Ollama
    mullama rm llama3.2:1b           # Mullama (identical)

    # Copy/alias a model
    ollama cp llama3.2:1b my-model   # Ollama
    mullama cp llama3.2:1b my-model  # Mullama (identical)
    ```

=== "Running Models"

    ```bash
    # One-shot generation
    ollama run llama3.2:1b "Hello!"          # Ollama
    mullama run llama3.2:1b "Hello!"         # Mullama (identical)

    # Start server
    ollama serve                              # Ollama
    mullama serve --model llama3.2:1b         # Mullama (model specified)

    # Create from Modelfile
    ollama create my-model -f Modelfile       # Ollama
    mullama create my-model -f Modelfile      # Mullama (identical)
    ```

=== "Process Management"

    ```bash
    # Show running models
    ollama ps                         # Ollama
    mullama ps                        # Mullama (identical)

    # Background daemon (Mullama-only)
    mullama daemon start
    mullama daemon status
    mullama daemon stop
    ```

## API Endpoint Mapping

### Ollama API to Mullama API

Mullama provides both Ollama-compatible and OpenAI-compatible endpoints:

| Ollama Endpoint | Mullama Equivalent | Format | Notes |
|----------------|-------------------|--------|-------|
| `POST /api/generate` | `POST /v1/completions` | OpenAI | Text completions |
| `POST /api/chat` | `POST /v1/chat/completions` | OpenAI | Chat completions |
| `POST /api/embeddings` | `POST /v1/embeddings` | OpenAI | Embedding generation |
| `GET /api/tags` | `GET /v1/models` | OpenAI | List models |
| `POST /api/pull` | `POST /api/models/pull` | Mullama | Download model |
| `POST /api/show` | `GET /api/models/<name>` | Mullama | Model details |
| `DELETE /api/delete` | `DELETE /api/models/<name>` | Mullama | Remove model |
| -- | `POST /v1/messages` | Anthropic | Anthropic-compatible |
| -- | `GET /metrics` | Prometheus | Monitoring metrics |
| -- | `WS /ws/chat` | WebSocket | Bidirectional streaming |
| -- | `GET /api/system/status` | Mullama | System health |

### Mullama Management API

Additional management endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all available models |
| `/api/models/pull` | POST | Download a model by alias |
| `/api/models/<name>` | GET | Get model details and metadata |
| `/api/models/<name>` | DELETE | Remove a model from cache |
| `/api/system/status` | GET | System status, memory, GPU info |
| `/metrics` | GET | Prometheus-format metrics |

### Request Format Comparison

=== "Chat Completion"

    ```bash
    # Ollama format
    curl http://localhost:11434/api/chat -d '{
      "model": "llama3.2",
      "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
      ]
    }'

    # Mullama format (OpenAI-compatible)
    curl http://localhost:8080/v1/chat/completions -d '{
      "model": "llama3.2:1b",
      "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
      ],
      "max_tokens": 256,
      "temperature": 0.7
    }'
    ```

=== "Text Completion"

    ```bash
    # Ollama format
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3.2",
      "prompt": "Once upon a time",
      "stream": false
    }'

    # Mullama format (OpenAI-compatible)
    curl http://localhost:8080/v1/completions -d '{
      "model": "llama3.2:1b",
      "prompt": "Once upon a time",
      "max_tokens": 256,
      "stream": false
    }'
    ```

=== "Embeddings"

    ```bash
    # Ollama format
    curl http://localhost:11434/api/embeddings -d '{
      "model": "nomic-embed-text",
      "prompt": "Hello world"
    }'

    # Mullama format (OpenAI-compatible)
    curl http://localhost:8080/v1/embeddings -d '{
      "model": "nomic-embed-text",
      "input": "Hello world"
    }'
    ```

=== "Streaming"

    ```bash
    # Ollama format (newline-delimited JSON)
    curl http://localhost:11434/api/chat -d '{
      "model": "llama3.2",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }'
    # Response: {"message":{"content":"Hi"},"done":false}\n

    # Mullama format (Server-Sent Events)
    curl http://localhost:8080/v1/chat/completions -d '{
      "model": "llama3.2:1b",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }'
    # Response: data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n
    ```

=== "Anthropic Format (Mullama-only)"

    ```bash
    # Mullama Anthropic-compatible endpoint
    curl http://localhost:8080/v1/messages -d '{
      "model": "llama3.2:1b",
      "max_tokens": 1024,
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }'
    # Response follows Anthropic message format
    ```

## Modelfile Compatibility

!!! info "Full Backward Compatibility"
    Mullama reads Ollama Modelfile format without any modification. Your existing Modelfiles work as-is. Mullama also supports an extended format with additional directives.

### Standard Modelfile (Works in Both)

```dockerfile
FROM llama3.2:1b

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1

SYSTEM """
You are a helpful coding assistant. You provide clear,
concise explanations and working code examples.
"""

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

LICENSE """MIT License..."""
```

### Mullama Extended Modelfile (Mullamafile)

Mullama extends the format with additional directives that are ignored by Ollama:

```dockerfile
FROM llama3.2:1b

# === Standard Directives (Ollama-compatible) ===
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
SYSTEM """You are a helpful assistant."""
TEMPLATE """{{ .System }}\n{{ .Prompt }}"""

# === Mullama Extensions ===

# GPU configuration
GPU_LAYERS 35
FLASH_ATTENTION auto

# Model adaptation
ADAPTER ./my-lora-adapter.gguf

# Revision pinning for reproducibility
# FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF@abc123def456

# Content verification
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Thinking token configuration (for reasoning models)
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true

# Tool calling format
TOOLFORMAT style qwen
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"

# Capability declarations
CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking true
CAPABILITY vision false

# Vision projector (multimodal models)
VISION_PROJECTOR ./mmproj-model.gguf

# Metadata
AUTHOR "Your Name"
```

### Extended Directives Reference

| Directive | Purpose | Ollama Equivalent |
|-----------|---------|-------------------|
| `GPU_LAYERS <n>` | Number of layers to offload to GPU | Not available |
| `FLASH_ATTENTION <auto/true/false>` | Flash attention mode | Not available |
| `ADAPTER <path>` | LoRA adapter file path | `ADAPTER` (limited) |
| `VISION_PROJECTOR <path>` | Multimodal vision projector path | Not available |
| `DIGEST <sha256:hash>` | Content integrity verification | Not available |
| `THINKING start/end/enabled` | Reasoning token boundaries | Not available |
| `TOOLFORMAT style/call_start/...` | Tool calling format specification | Not available |
| `CAPABILITY <name> <bool>` | Model capability flags | Not available |
| `AUTHOR <name>` | Author metadata | Not available |

## Model File Reuse

### Same Format, Same Files

Both tools use GGUF (GPT-Generated Unified Format) model files. Any GGUF file works with both:

```bash
# Use an Ollama-downloaded model with Mullama
# Find where Ollama stores models:
ls ~/.ollama/models/blobs/

# Point Mullama at the file:
mullama run --model-path ~/.ollama/models/blobs/sha256-<hash>

# Or set cache directory:
export MULLAMA_CACHE_DIR=~/.ollama/models
mullama run llama3.2:1b "Hello"
```

### Model Storage Locations

| Tool | Platform | Default Path |
|------|----------|--------------|
| Ollama | Linux | `~/.ollama/models/` |
| Ollama | macOS | `~/.ollama/models/` |
| Ollama | Windows | `%USERPROFILE%\.ollama\models\` |
| Mullama | Linux | `~/.cache/mullama/models/` |
| Mullama | macOS | `~/Library/Caches/mullama/models/` |
| Mullama | Windows | `%LOCALAPPDATA%\mullama\models\` |

### Same Download Sources

Both support downloading quantized GGUF models. Mullama downloads from HuggingFace:

```bash
# Ollama pulls from ollama.com registry
ollama pull llama3.2:1b

# Mullama pulls from HuggingFace (same model, same quantization)
mullama pull llama3.2:1b
# Resolves to: hf:meta-llama/Llama-3.2-1B-Instruct-GGUF
```

## Code Migration Examples

### Python: REST Client to Native Binding

=== "Before (Ollama REST)"

    ```python
    # Using ollama-python (REST wrapper)
    import ollama

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': 'What is Python?'}
    ])
    print(response['message']['content'])

    # Streaming
    stream = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': 'Explain recursion'}
    ], stream=True)
    for chunk in stream:
        print(chunk['message']['content'], end='')
    ```

=== "After (Mullama Native)"

    ```python
    # Using mullama (native PyO3 binding - no HTTP)
    import mullama

    model = mullama.Model("llama3.2-1b.gguf")
    context = model.create_context(n_ctx=4096)

    result = context.generate("What is Python?", max_tokens=256)
    print(result)

    # Streaming
    for token in context.generate_stream("Explain recursion", max_tokens=256):
        print(token, end='', flush=True)
    ```

=== "After (Mullama via OpenAI SDK)"

    ```python
    # Using standard OpenAI SDK pointed at Mullama daemon
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="unused"  # No auth required for local
    )

    response = client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "What is Python?"}],
        max_tokens=256
    )
    print(response.choices[0].message.content)

    # Streaming
    stream = client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "Explain recursion"}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='')
    ```

### Node.js: REST Client to Native Binding

=== "Before (Ollama REST)"

    ```javascript
    // Using ollama-js (REST wrapper)
    import { Ollama } from 'ollama';

    const ollama = new Ollama();
    const response = await ollama.chat({
      model: 'llama3.2',
      messages: [{ role: 'user', content: 'Hello!' }]
    });
    console.log(response.message.content);

    // Streaming
    const stream = await ollama.chat({
      model: 'llama3.2',
      messages: [{ role: 'user', content: 'Tell me a story' }],
      stream: true
    });
    for await (const chunk of stream) {
      process.stdout.write(chunk.message.content);
    }
    ```

=== "After (Mullama Native)"

    ```javascript
    // Using mullama (native NAPI-RS binding - no HTTP)
    const { Model } = require('mullama');

    const model = new Model('llama3.2-1b.gguf');
    const context = model.createContext({ nCtx: 4096 });

    const result = await context.generate('Hello!', { maxTokens: 256 });
    console.log(result);

    // Streaming
    const stream = context.generateStream('Tell me a story', { maxTokens: 256 });
    stream.on('token', (token) => process.stdout.write(token));
    await stream.done();
    ```

=== "After (Mullama via OpenAI SDK)"

    ```javascript
    // Using standard OpenAI SDK pointed at Mullama daemon
    import OpenAI from 'openai';

    const client = new OpenAI({
      baseURL: 'http://localhost:8080/v1',
      apiKey: 'unused'
    });

    const response = await client.chat.completions.create({
      model: 'llama3.2:1b',
      messages: [{ role: 'user', content: 'Hello!' }],
      max_tokens: 256
    });
    console.log(response.choices[0].message.content);

    // Streaming
    const stream = await client.chat.completions.create({
      model: 'llama3.2:1b',
      messages: [{ role: 'user', content: 'Tell me a story' }],
      stream: true
    });
    for await (const chunk of stream) {
      process.stdout.write(chunk.choices[0]?.delta?.content || '');
    }
    ```

### Go: REST Client to Native Binding

=== "Before (Ollama REST)"

    ```go
    // Using ollama-go (REST wrapper)
    package main

    import (
        "context"
        "fmt"
        "github.com/ollama/ollama/api"
    )

    func main() {
        client, _ := api.ClientFromEnvironment()
        req := &api.ChatRequest{
            Model:    "llama3.2",
            Messages: []api.Message{{Role: "user", Content: "Hello!"}},
        }
        resp, _ := client.Chat(context.Background(), req)
        fmt.Println(resp.Message.Content)
    }
    ```

=== "After (Mullama Native)"

    ```go
    // Using mullama-go (native cgo binding - no HTTP)
    package main

    import (
        "fmt"
        "github.com/neul-labs/mullama-go"
    )

    func main() {
        model, _ := mullama.LoadModel("llama3.2-1b.gguf")
        defer model.Close()

        ctx, _ := model.CreateContext(mullama.ContextParams{NCtx: 4096})
        defer ctx.Close()

        result, _ := ctx.Generate("Hello!", mullama.GenerateParams{MaxTokens: 256})
        fmt.Println(result)
    }
    ```

## Environment Variable Differences

| Purpose | Ollama | Mullama | Notes |
|---------|--------|---------|-------|
| Server address | `OLLAMA_HOST` | `--http-addr` / `--http-port` | CLI flags preferred |
| Model storage | `OLLAMA_MODELS` | `MULLAMA_CACHE_DIR` | Cache directory |
| GPU layers | `OLLAMA_NUM_GPU` | `--gpu-layers` / `GPU_LAYERS` | CLI flag or Modelfile |
| HuggingFace auth | -- | `HF_TOKEN` | For gated model downloads |
| Binary path | -- | `MULLAMA_BIN` | For auto-spawn feature |
| Context size | `OLLAMA_NUM_CTX` | `--context-size` / `num_ctx` | CLI flag or param |
| Debug logging | `OLLAMA_DEBUG` | `RUST_LOG=debug` | Standard Rust logging |

### Default Port Difference

| Setting | Ollama | Mullama | Notes |
|---------|--------|---------|-------|
| HTTP port | 11434 | 8080 | No conflict when running both |
| Bind address | 127.0.0.1 | 0.0.0.0 | Mullama accessible from network by default |

!!! warning "Network Binding"
    Mullama binds to `0.0.0.0` by default, making it accessible from the network. For local-only access, use `--http-addr 127.0.0.1` or configure in your daemon config file.

## What You Gain by Migrating

After migrating from Ollama to Mullama, you gain access to:

### Native Library Integration

- **Zero-overhead inference** from Rust, Node.js, Python, Go, PHP, and C/C++
- **No daemon required** for library usage -- embed inference directly in your application
- **Single-binary deployment** with inference capabilities baked in

### Advanced API Features

- **Anthropic-compatible API** (`/v1/messages`) for Claude SDK compatibility
- **WebSocket streaming** for bidirectional real-time communication
- **Prometheus metrics** for production monitoring at `/metrics`
- **Embedded Web UI** for model management and chat

### Extended Inference Capabilities

- **ColBERT / late interaction** with MaxSim scoring for advanced retrieval
- **Streaming audio** with voice activity detection and noise reduction
- **Parallel batch processing** with Rayon work-stealing scheduler
- **Composable sampler chains** for advanced generation control
- **Multiple simultaneous LoRA** adapters with dynamic weight adjustment

### Operational Improvements

- **Auto-spawn daemon** -- starts automatically on first command
- **TUI chat interface** -- rich terminal UI for interactive sessions
- **Extended Modelfile** -- thinking tokens, tool format, capabilities, digest verification
- **Model aliases** -- 40+ pre-configured aliases for popular models

## Step-by-Step Migration

### Step 1: Install Mullama

```bash
# Clone and build
git clone https://github.com/neul-labs/mullama.git
cd mullama
git submodule update --init --recursive

# Install system dependencies (Linux)
sudo apt install -y libasound2-dev libpulse-dev libflac-dev \
    libvorbis-dev libopus-dev libpng-dev libjpeg-dev \
    libtiff-dev libwebp-dev

# Build with full daemon support
cargo build --release --features daemon

# Optionally build with Web UI
cd ui && npm install && npm run build && cd ..
cargo build --release --features daemon,embedded-ui

# Add to PATH
export PATH="$PWD/target/release:$PATH"
```

### Step 2: Verify Installation

```bash
mullama --version
mullama daemon status
```

### Step 3: Pull Your Models

```bash
# Same model aliases work
mullama pull llama3.2:1b
mullama pull qwen2.5:7b-instruct
mullama list
```

### Step 4: Test Basic Operations

```bash
# One-shot generation
mullama run llama3.2:1b "What is the capital of France?"

# Interactive chat
mullama chat --model llama3.2:1b
```

### Step 5: Migrate Modelfiles

```bash
# Your existing Modelfiles work without changes
mullama create my-model -f ./Modelfile
mullama show my-model --modelfile
```

### Step 6: Update Application Code

Choose your integration approach:

- **Native binding**: Rewrite to use `mullama` package (maximum performance)
- **OpenAI SDK**: Point existing OpenAI client at `http://localhost:8080/v1` (minimal changes)
- **Anthropic SDK**: Point existing Anthropic client at `http://localhost:8080` (minimal changes)

### Step 7: Explore Advanced Features

```bash
# Web UI
open http://localhost:8080/ui/

# Prometheus metrics
curl http://localhost:8080/metrics

# Anthropic-compatible API
curl http://localhost:8080/v1/messages -H "Content-Type: application/json" -d '{
  "model": "llama3.2:1b",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Hello from Anthropic SDK!"}]
}'
```

## Frequently Asked Questions

??? question "Can I run Ollama and Mullama side by side?"
    Yes. They use different default ports (Ollama: 11434, Mullama: 8080) and different storage directories. Both can run simultaneously without conflicts.

??? question "Do I need to re-download my models?"
    Not necessarily. If you have GGUF files from Ollama, you can point Mullama directly at them using `--model-path` or by setting `MULLAMA_CACHE_DIR`. Alternatively, `mullama pull` downloads models from HuggingFace into its own cache.

??? question "Is the Modelfile format exactly the same?"
    Mullama supports all standard Ollama Modelfile directives. It also adds extensions (GPU_LAYERS, FLASH_ATTENTION, THINKING, TOOLFORMAT, CAPABILITY, etc.) that are Mullama-specific. An Ollama Modelfile works in Mullama without changes.

??? question "Can I use OpenAI client libraries with Mullama?"
    Yes. Mullama's daemon provides a fully OpenAI-compatible API. Any OpenAI SDK (Python, Node.js, Go, etc.) works by setting `base_url` to `http://localhost:8080/v1` and `api_key` to any string.

??? question "Can I use Anthropic client libraries with Mullama?"
    Yes. Mullama provides an Anthropic-compatible endpoint at `/v1/messages`. Configure the Anthropic SDK's base URL to point to your Mullama instance.

??? question "What about model aliases like 'llama3.2:1b'?"
    Mullama ships with 40+ pre-configured model aliases that resolve to HuggingFace repositories. Custom aliases can be defined in `configs/models.toml`.

??? question "Is there a performance difference in daemon mode?"
    When both are running as daemons serving HTTP APIs, performance is comparable -- both use llama.cpp for inference. Mullama's advantage emerges when using native bindings (eliminating HTTP overhead entirely) or when leveraging features like parallel batch processing.

??? question "Can I use the same GPU settings?"
    Yes. Mullama supports CUDA, Metal, and ROCm, configured via `--gpu-layers` CLI flag or `GPU_LAYERS` in Modelfile, equivalent to Ollama's `OLLAMA_NUM_GPU`.

??? question "Does Mullama support chat templates?"
    Yes. Modelfile `TEMPLATE` directives work identically. Mullama also auto-detects chat templates from GGUF metadata.

??? question "What if a feature I need is missing?"
    Check the [Feature Matrix](feature-matrix.md) for current status. File an issue on GitHub for feature requests -- community feedback directly influences the roadmap.
