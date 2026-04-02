# Mullama

**Run any LLM locally. Use it from any language. Deploy anywhere.**

[![Crates.io](https://img.shields.io/crates/v/mullama)](https://crates.io/crates/mullama)
[![PyPI](https://img.shields.io/pypi/v/mullama)](https://pypi.org/project/mullama/)
[![npm](https://img.shields.io/npm/v/mullama)](https://www.npmjs.com/package/mullama)
[![CI](https://img.shields.io/github/actions/workflow/status/skelf-research/mullama/ci.yml?branch=main&label=CI)](https://github.com/skelf-research/mullama/actions)
[![Documentation](https://img.shields.io/badge/docs-skelfresearch.com-blue)](https://docs.skelfresearch.com/mullama/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&color=5865F2)](https://discord.gg/skelfresearch)

Mullama is a local LLM server and library that works just like Ollama — same CLI commands, same model format, same Modelfile syntax — but with native language bindings for Python, Node.js, Go, PHP, Rust, and C/C++. Embed inference directly in your app with zero HTTP overhead, or run it as a server with OpenAI and Anthropic-compatible APIs.

## Install

```bash
# One-liner (Linux/macOS)
curl -fsSL https://skelfresearch.com/mullama/install.sh | sh

# Windows (PowerShell)
iwr -useb https://skelfresearch.com/mullama/install.ps1 | iex

# Or via package managers
pip install mullama          # Python
npm install mullama          # Node.js
cargo add mullama            # Rust
go get github.com/skelf-research/mullama-go   # Go
composer require skelf-research/mullama        # PHP
```

## Quick Start

```bash
# Run a model (daemon auto-starts)
mullama run llama3.2:1b "What is the capital of France?"

# Interactive chat
mullama chat

# Start an OpenAI-compatible server
mullama serve --model llama3.2:1b
```

**Coming from Ollama?** Your commands work unchanged — `run`, `pull`, `serve`, `list`, `ps`, `create`, `show`, `rm`, `cp`.

## Use as a Library

Embed LLM inference directly in your application — no server, no HTTP overhead, no separate process.

**Python:**

```python
from mullama import Model, Context

model = Model.load('llama3.2-1b.gguf', n_gpu_layers=32)
ctx = Context(model, n_ctx=4096)
response = ctx.generate('Hello, AI!')
print(response)
```

**Node.js:**

```javascript
const { Model, Context } = require('mullama');

const model = await Model.load('llama3.2-1b.gguf', { gpuLayers: 32 });
const ctx = new Context(model, { contextSize: 4096 });
const response = await ctx.generate('Hello, AI!');
console.log(response);
```

**Rust:**

```rust
use mullama::{Model, Context, ContextParams};

let model = Model::load("llama3.2-1b.gguf")?;
let mut ctx = Context::new(&model, ContextParams::default())?;
let response = ctx.generate("Hello, AI!", 256)?;
println!("{}", response);
```

**Go:**

```go
import mullama "github.com/skelf-research/mullama-go"

model, _ := mullama.LoadModel("llama3.2-1b.gguf", mullama.WithGPULayers(32))
ctx, _ := model.NewContext(mullama.ContextConfig{ContextSize: 4096})
response, _ := ctx.Generate("Hello, AI!")
fmt.Println(response)
```

**PHP:**

```php
use Mullama\Model;
use Mullama\Context;

$model = Model::load('llama3.2-1b.gguf', ['gpu_layers' => 32]);
$ctx = new Context($model, ['n_ctx' => 4096]);
$response = $ctx->generate('Hello, AI!');
echo $response;
```

See the [bindings documentation](https://docs.skelfresearch.com/mullama/bindings/) for full API details.

## Why Mullama?

| | |
|:--|:--|
| **Native bindings for 6 languages** | Python, Node.js, Go, PHP, Rust, C/C++ — call models directly, no HTTP roundtrips |
| **Drop-in Ollama replacement** | Same CLI commands, same Modelfile format, same model registry |
| **OpenAI + Anthropic API compatible** | Use your existing SDKs and tools without changes |
| **Embed in any app** | Run inference in-process — no separate daemon required |
| **7 GPU backends** | CUDA, Metal, ROCm, OpenCL, Vulkan, SYCL, RPC |
| **Multimodal** | Text, image, and real-time audio with voice activity detection |
| **Built-in Web UI and TUI** | Chat interface, model management, and API playground |

## What You Can Build

- **Chatbots and assistants** — Streaming responses, multi-turn context, and custom system prompts
- **RAG pipelines** — Embeddings, ColBERT-style semantic search, and grammar-constrained generation
- **Voice assistants** — Real-time audio capture with VAD, speech-to-text, and streaming LLM responses
- **API servers** — Production-ready OpenAI-compatible endpoints with streaming SSE
- **Edge deployments** — Embed a model directly in your app with no network dependency
- **Batch processing** — Parallel inference across documents with work-stealing scheduling

## Ollama Compatibility

| Feature | Mullama | Ollama |
|---------|:-------:|:------:|
| CLI commands (`run`, `pull`, `serve`, etc.) | Same syntax | — |
| Modelfile format | Compatible | — |
| GGUF models | Yes | Yes |
| OpenAI API | Yes | Yes |
| Anthropic API | Yes | No |
| Native language bindings | 6 languages | HTTP only |
| Embed in your app (no daemon) | Yes | No |
| Built-in Web UI | Yes | No |
| Built-in TUI chat | Yes | No |

[Full comparison](https://docs.skelfresearch.com/mullama/comparison/vs-ollama/) | [Migration guide](https://docs.skelfresearch.com/mullama/comparison/migration-from-ollama/)

## GPU Acceleration

Set the environment variable for your hardware before building:

```bash
export LLAMA_CUDA=1      # NVIDIA CUDA
export LLAMA_METAL=1     # Apple Silicon
export LLAMA_HIPBLAS=1   # AMD ROCm
export LLAMA_CLBLAST=1   # Intel/AMD OpenCL
export LLAMA_VULKAN=1    # Vulkan (cross-platform)
export LLAMA_SYCL=1      # Intel SYCL/oneAPI
export LLAMA_RPC=1       # Distributed RPC backend
```

## Roadmap

- **v0.4** — Speculative decoding, prompt caching, improved quantization support
- **v0.5** — Distributed inference across multiple nodes, model sharding
- **v0.6** — Built-in fine-tuning (LoRA/QLoRA), training data pipelines
- **v1.0** — Stable API, LTS release, comprehensive benchmarks

## Documentation

Full documentation is available at **[docs.skelfresearch.com/mullama](https://docs.skelfresearch.com/mullama/)**.

Guides cover installation, library usage, daemon configuration, language bindings, advanced features, API reference, and tutorials.

## Contributing

```bash
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama
cargo test --all-features
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
