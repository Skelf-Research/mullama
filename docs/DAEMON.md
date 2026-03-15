# Mullama Daemon Guide

The Mullama daemon provides a high-performance LLM server with multiple API compatibility layers, auto-spawn capability, and an embedded web UI.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Aliases](#model-aliases)
- [Modelfile / Mullamafile](#modelfile--mullamafile)
- [CLI Commands](#cli-commands)
- [API Reference](#api-reference)
- [Web UI](#web-ui)
- [Auto-Spawn](#auto-spawn)
- [Configuration](#configuration)
- [Architecture](#architecture)

## Quick Start

```bash
# Build with daemon feature
cargo build --release --features daemon

# Run a model (daemon auto-starts)
mullama run llama3.2:1b "Hello, world!"

# Or start daemon explicitly
mullama serve --model llama3.2:1b --gpu-layers 35

# Open web UI
open http://localhost:8080/ui/
```

## Model Aliases

Mullama supports short model aliases that automatically resolve to HuggingFace repositories. This provides an Ollama-like experience.

### Available Aliases

See `configs/models.toml` for the full list. Some examples:

| Alias | HuggingFace Repository |
|-------|------------------------|
| `llama3.2:1b` | `meta-llama/Llama-3.2-1B-Instruct-GGUF` |
| `llama3.2:3b` | `meta-llama/Llama-3.2-3B-Instruct-GGUF` |
| `qwen2.5:7b-instruct` | `Qwen/Qwen2.5-7B-Instruct-GGUF` |
| `deepseek-r1:7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF` |
| `mistral:7b` | `mistralai/Mistral-7B-Instruct-v0.3-GGUF` |
| `phi3:mini` | `microsoft/Phi-3-mini-4k-instruct-gguf` |
| `gemma2:9b` | `google/gemma-2-9b-it-GGUF` |

### Usage

```bash
# Using alias
mullama run llama3.2:1b "Hello!"

# Using full HuggingFace path
mullama run hf:meta-llama/Llama-3.2-1B-Instruct-GGUF "Hello!"

# With specific quantization
mullama run hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M "Hello!"
```

### Adding Custom Aliases

Edit `configs/models.toml`:

```toml
[models.my-model]
repo = "owner/repo-name-GGUF"
default_quant = "Q4_K_M"
description = "My custom model"
```

## Modelfile / Mullamafile

Create custom model configurations using Modelfile (Ollama-compatible) or Mullamafile (extended).

### Modelfile (Ollama-compatible)

```dockerfile
FROM llama3.2:1b

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM """
You are a helpful coding assistant specialized in Rust.
"""

TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
```

### Mullamafile (Extended)

Mullamafile supports all Modelfile directives plus Mullama-specific extensions:

```dockerfile
# Pin to specific commit for reproducibility
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d

# Content verification
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Standard parameters
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER top_p 0.9

# Multiple stop sequences
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM """
You are a helpful assistant.
"""

# Mullama extensions
GPU_LAYERS 32
FLASH_ATTENTION true
ADAPTER ./my-lora-adapter.safetensors
VISION_PROJECTOR ./mmproj.gguf

# Thinking/reasoning configuration
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true

# Tool calling format
TOOLFORMAT style qwen
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"

# Capability flags
CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking true
CAPABILITY vision false

# Metadata
LICENSE MIT
AUTHOR "Your Name"
```

### Supported Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `FROM` | Base model (required) | `FROM llama3.2:1b` |
| `PARAMETER` | Set parameter | `PARAMETER temperature 0.7` |
| `SYSTEM` | System prompt | `SYSTEM """..."""` |
| `TEMPLATE` | Chat template | `TEMPLATE """..."""` |
| `MESSAGE` | Pre-defined message | `MESSAGE user "Hello"` |
| `ADAPTER` | LoRA adapter path | `ADAPTER ./adapter.safetensors` |
| `GPU_LAYERS` | GPU layers to offload | `GPU_LAYERS 35` |
| `FLASH_ATTENTION` | Enable flash attention | `FLASH_ATTENTION true` |
| `VISION_PROJECTOR` | Vision model projector | `VISION_PROJECTOR ./mmproj.gguf` |
| `LICENSE` | License metadata | `LICENSE MIT` |
| `AUTHOR` | Author metadata | `AUTHOR "Name"` |
| `DIGEST` | SHA256 verification | `DIGEST sha256:abc123...` |
| `THINKING` | Thinking token config | `THINKING start "<think>"` |
| `TOOLFORMAT` | Tool calling format | `TOOLFORMAT style qwen` |
| `CAPABILITY` | Model capabilities | `CAPABILITY thinking true` |

### Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `temperature` | float | Sampling temperature (0-2) |
| `top_p` | float | Nucleus sampling threshold |
| `top_k` | int | Top-k sampling |
| `num_ctx` | int | Context window size |
| `num_predict` | int | Max tokens to generate |
| `repeat_penalty` | float | Repetition penalty |
| `presence_penalty` | float | Presence penalty |
| `frequency_penalty` | float | Frequency penalty |
| `seed` | int | Random seed |
| `stop` | string | Stop sequence (multiple allowed) |

### Revision Pinning

Pin models to specific HuggingFace commits for reproducibility:

```dockerfile
# Pin to specific commit hash
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d

# With quantization
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M@abc123def
```

### Content-Addressed Verification

Verify model integrity with SHA256 digests:

```dockerfile
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# Verification happens automatically when loading
# Error raised if digest doesn't match
```

### Thinking Token Configuration

Configure thinking/reasoning output separation for models like DeepSeek-R1 or QwQ:

```dockerfile
FROM deepseek-r1:7b

# Configure thinking tokens
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true
```

When enabled, streaming responses include separate `thinking` content in the delta, allowing UIs to display reasoning in a collapsible block.

### Tool Format Configuration

Configure tool calling format for function-calling models:

```dockerfile
FROM qwen2.5:7b-instruct

TOOLFORMAT style qwen
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"
```

### Capability Flags

Declare model capabilities for API compatibility:

```dockerfile
FROM llama3.2:1b

CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking false
CAPABILITY vision false
```

These flags are exposed via the `/v1/models` endpoint for client introspection.

### Multiple Stop Sequences

Models can have multiple stop sequences:

```dockerfile
FROM qwen2.5:7b-instruct

# ChatML format stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

### Commands

```bash
# Create model from Modelfile
mullama create my-assistant -f ./Modelfile

# Create from Mullamafile
mullama create my-assistant -f ./Mullamafile

# Show model details
mullama show my-assistant

# Show Modelfile
mullama show my-assistant --modelfile

# Copy/rename model
mullama cp my-assistant my-assistant-v2

# Remove model
mullama rm my-assistant
```

## CLI Commands

### Model Management

```bash
mullama list              # List all local models
mullama list --verbose    # Show detailed info
mullama list --json       # Output as JSON

mullama pull llama3.2:1b  # Download model
mullama rm model-name     # Remove model
mullama cp src dst        # Copy model

mullama show model-name           # Show model info
mullama show model-name --modelfile  # Show Modelfile
```

### Running Models

```bash
mullama run llama3.2:1b "prompt"     # One-shot generation
mullama run llama3.2:1b -n 512       # Max 512 tokens
mullama run llama3.2:1b -t 0.9       # Temperature 0.9

mullama chat                          # Interactive TUI
mullama chat --model llama3.2:1b     # Specify model
```

### Daemon Management

```bash
mullama serve                        # Start daemon (foreground)
mullama serve --model llama3.2:1b   # With initial model
mullama serve -p 8080 -g 35         # Custom port and GPU layers

mullama daemon start                 # Start in background
mullama daemon stop                  # Stop daemon
mullama daemon restart               # Restart daemon
mullama daemon status                # Show status
mullama daemon status --json         # JSON output
mullama daemon logs                  # View logs
mullama daemon logs -f               # Follow logs
```

### Process Status

```bash
mullama ps              # Show running models
mullama ps --json       # JSON output
```

## API Reference

### OpenAI-Compatible API

#### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

#### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### Embeddings

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "input": "Hello, world!"
  }'
```

#### List Models

```bash
curl http://localhost:8080/v1/models
```

### Anthropic-Compatible API

The `/v1/messages` endpoint provides Anthropic Claude API compatibility:

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

#### Streaming

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Management API

#### List Models

```bash
curl http://localhost:8080/api/models
```

#### Pull Model

```bash
curl -X POST http://localhost:8080/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:1b"}'
```

#### Delete Model

```bash
curl -X DELETE http://localhost:8080/api/models/my-model
```

#### Get Model Details

```bash
curl http://localhost:8080/api/models/llama3.2:1b
```

#### System Status

```bash
curl http://localhost:8080/api/system/status
```

Response:
```json
{
  "version": "0.1.1",
  "uptime_secs": 3600,
  "models_loaded": 2,
  "http_endpoint": "http://0.0.0.0:8080"
}
```

### Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

Available metrics:
- `mullama_info` - Version information
- `mullama_uptime_seconds` - Daemon uptime
- `mullama_models_loaded` - Number of loaded models
- `mullama_requests_total` - Total requests processed
- `mullama_requests_active` - Currently active requests
- `mullama_tokens_generated_total` - Total tokens generated
- `mullama_model_parameters{model}` - Model parameter count
- `mullama_model_context_size{model}` - Model context size
- `mullama_model_gpu_layers{model}` - Model GPU layers

## Web UI

Access the embedded web UI at `http://localhost:8080/ui/`

### Building the UI

```bash
cd ui
npm install
npm run build
```

Then build Mullama with the embedded UI:

```bash
cargo build --release --features daemon,embedded-ui
```

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | System status, metrics, quick actions |
| **Models** | Model management (pull, list, delete) |
| **Chat** | Interactive multi-conversation chat |
| **Playground** | API testing with cURL generation |
| **Settings** | Theme, generation defaults |

## Auto-Spawn

The daemon automatically starts when you run CLI commands that require it:

```bash
# First run - daemon starts automatically
$ mullama run llama3.2:1b "Hello"
Daemon not running, starting it automatically...
Daemon started successfully, connecting...
Hello! How can I assist you today?

# Subsequent runs - instant connection
$ mullama chat
Connected to Mullama daemon v0.1.1 (uptime: 42s)
```

### Configuration

Auto-spawn uses default settings:
- HTTP port: 8080
- IPC socket: `/tmp/mullama.sock`
- Log file: `/tmp/mullamad.log`
- Background mode: enabled

Override with environment variables:
```bash
export MULLAMA_BIN=/path/to/mullama
```

## Configuration

### Command Line Options

```bash
mullama serve \
  --model llama3.2:1b \
  --socket ipc:///tmp/mullama.sock \
  --http-port 8080 \
  --http-addr 0.0.0.0 \
  --gpu-layers 35 \
  --context-size 4096 \
  --threads 8 \
  --verbose
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | - |
| `MULLAMA_CACHE_DIR` | Model cache directory | Platform-specific |
| `MULLAMA_BIN` | Path to mullama binary | Auto-detect |

### Cache Locations

| Platform | Default Path |
|----------|--------------|
| Linux | `~/.cache/mullama/models` |
| macOS | `~/Library/Caches/mullama/models` |
| Windows | `%LOCALAPPDATA%\mullama\models` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Mullama Daemon                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Model Manager                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │ Model 1 │  │ Model 2 │  │ Model 3 │  │   ...   │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐   │
│  │   HTTP Server   │  │   IPC Server    │  │  Web UI       │   │
│  │                 │  │                 │  │               │   │
│  │ /v1/chat/...   │  │  nng socket     │  │  /ui/*        │   │
│  │ /v1/messages   │  │                 │  │               │   │
│  │ /v1/embeddings │  │                 │  │  Dashboard    │   │
│  │ /api/*         │  │                 │  │  Models       │   │
│  │ /metrics       │  │                 │  │  Chat         │   │
│  └─────────────────┘  └─────────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         ▲                      ▲                    ▲
         │                      │                    │
    ┌────┴────┐            ┌────┴────┐          ┌────┴────┐
    │  curl   │            │   TUI   │          │ Browser │
    │  apps   │            │  Client │          │         │
    └─────────┘            └─────────┘          └─────────┘
```

### Components

- **Model Manager**: Loads, unloads, and manages multiple models
- **HTTP Server**: OpenAI, Anthropic, and management APIs
- **IPC Server**: High-performance local communication via nng
- **Web UI**: Vue.js embedded management interface
- **Auto-Spawn**: Automatic daemon lifecycle management
