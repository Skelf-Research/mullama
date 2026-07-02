# Mullama — local LLM inference in Rust, with native bindings for Python, Node.js, Go, PHP, and C

[![Crates.io](https://img.shields.io/crates/v/mullama)](https://crates.io/crates/mullama)
[![PyPI](https://img.shields.io/pypi/v/mullama)](https://pypi.org/project/mullama/)
[![npm](https://img.shields.io/npm/v/mullama)](https://www.npmjs.com/package/mullama)
[![CI](https://img.shields.io/github/actions/workflow/status/cognisoc/mullama/ci.yml?branch=main&label=CI)](https://github.com/cognisoc/mullama/actions)
[![Documentation](https://img.shields.io/badge/docs-cognisoc.com-blue)](https://docs.cognisoc.com/mullama/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Website](https://mullama.cognisoc.com)** · **[Docs](https://docs.cognisoc.com/mullama/)** · **[GitHub](https://github.com/cognisoc/mullama)**

## What is Mullama?

Mullama is a local LLM runtime built on `llama.cpp` that runs GGUF models — Llama 3.2, Qwen 2.5, DeepSeek R1, Mistral, Phi 3, Gemma 2, LLaVA, and any other GGUF file from Hugging Face — directly inside your application. It exposes the same CLI as Ollama (`mullama run`, `pull`, `serve`, `chat`) plus native bindings for **Rust, Python, Node.js, Go, PHP, and C/C++**, so you can either run it as a server with OpenAI- and Anthropic-compatible HTTP APIs, or embed the inference engine in-process with zero HTTP overhead.

If you want to run a local LLM in your stack without an Ollama daemon, without spawning a Python subprocess from your Node app, and without writing your own FFI to `llama.cpp` — Mullama is the single dependency that covers all of it.

## Install

```bash
# One-liner (Linux/macOS)
curl -fsSL https://mullama.cognisoc.com/install.sh | sh

# Windows (PowerShell)
iwr -useb https://mullama.cognisoc.com/install.ps1 | iex

# Or via your language's package manager
cargo add mullama                       # Rust
pip install mullama                     # Python
npm install mullama                     # Node.js / TypeScript
go get github.com/cognisoc/mullama      # Go
composer require mullama/mullama        # PHP
docker pull ghcr.io/cognisoc/mullama    # Docker
```

## Quick start (CLI)

```bash
# Run a model — daemon auto-starts in the background
mullama run llama3.2:1b "What is the capital of France?"

# Interactive chat
mullama chat

# Start an OpenAI-compatible HTTP server on port 11434
mullama serve --model llama3.2:1b
```

Coming from Ollama? Your commands work unchanged — `run`, `pull`, `serve`, `list`, `ps`, `create`, `show`, `rm`, `cp`. Your Modelfiles work unchanged too.

## Quick start (library)

Embed inference directly — no daemon, no HTTP, no separate process.

**Rust:**

```rust
use mullama::{Model, Context, ContextParams};

let model = Model::load("llama3.2-1b.gguf")?;
let mut ctx = Context::new(&model, ContextParams::default())?;
let response = ctx.generate("Hello, AI!", 256)?;
println!("{response}");
```

**Python:**

```python
from mullama import Model, Context
model = Model.load("llama3.2-1b.gguf", n_gpu_layers=32)
ctx = Context(model, n_ctx=4096)
print(ctx.generate("Hello, AI!"))
```

**Node.js:**

```javascript
const { Model, Context } = require("mullama");
const model = Model.load("llama3.2-1b.gguf", { nGpuLayers: 32 });
const ctx = new Context(model, { nCtx: 4096 });
console.log(ctx.generate("Hello, AI!"));
```

**Go:**

```go
import "github.com/cognisoc/mullama"
model, _ := mullama.LoadModel("llama3.2-1b.gguf", &mullama.ModelParams{NGpuLayers: 32})
ctx, _ := mullama.NewContext(model, &mullama.ContextParams{NCtx: 4096})
response, _ := ctx.Generate("Hello, AI!", 256, nil)
```

**PHP:**

```php
use Mullama\Model;
use Mullama\Context;
$model = Model::load("llama3.2-1b.gguf", ["nGpuLayers" => 32]);
$ctx = new Context($model, ["nCtx" => 4096]);
echo $ctx->generate("Hello, AI!");
```

See the binding-specific READMEs for full API docs: [Python](./bindings/python/README.md) · [Node.js](./bindings/node/README.md) · [Go](./bindings/go/README.md) · [PHP](./bindings/php/README.md) · [C/C++ (FFI)](./bindings/ffi/README.md).

## Supported models

Mullama runs any GGUF model from Hugging Face. The following are pre-configured aliases — type the short name and Mullama resolves it to the right HF repository and quantization:

- **Llama** — `llama3.2:1b`, `llama3.2:3b`, `llama3.1:8b`, `llama3.1:70b`, `llama3:8b`
- **Qwen 2.5** — `qwen2.5:0.5b` through `qwen2.5:72b`, plus `qwen2.5-coder:7b` / `:14b` / `:32b`
- **DeepSeek R1** (reasoning) — `deepseek-r1:1.5b`, `:7b`, `:14b`, `:32b`; plus `deepseek-coder:7b`, `:33b`
- **Mistral** — `mistral:7b`, `mixtral:8x7b`, `codestral:22b`
- **Phi** (Microsoft) — `phi3:mini`, `phi3:medium`, `phi3.5:mini`
- **Gemma** (Google) — `gemma2:2b`, `:9b`, `:27b`
- **Vision (multimodal)** — `llava:7b`, `:13b`, `llava-phi3`, `moondream:2b`
- **Embeddings** — `nomic-embed`, `bge:small`, `bge:large`
- **Code completion** — `starcoder2:3b`, `:7b`, `:15b`

Anything else in GGUF format works via `hf:owner/repo` (e.g. `hf:bartowski/Llama-3.2-1B-Instruct-GGUF`) or by giving a direct path to a `.gguf` file. The full registry lives in [`configs/models.toml`](./configs/models.toml).

## Why Mullama?

| | |
|:--|:--|
| **Native bindings for 6 languages** | Rust, Python, Node.js, Go, PHP, C/C++ — call models directly, no HTTP round-trips, no Python subprocess from your Node app |
| **Drop-in Ollama replacement** | Same CLI verbs, same Modelfile format, same model registry, same port (11434) |
| **OpenAI + Anthropic API compatible** | Use your existing OpenAI SDK, LangChain, LlamaIndex, or Anthropic SDK against `localhost:11434` |
| **Embed in any app** | Run inference in-process — no separate daemon, no orchestration |
| **7 GPU backends** | CUDA, Metal (Apple Silicon), ROCm, Vulkan, OpenCL, SYCL (Intel Arc), RPC |
| **Multimodal** | Text, image (LLaVA, Moondream), and real-time audio with voice activity detection |
| **Built-in Web UI and TUI** | Chat interface, model management, and API playground |

## How Mullama compares

| | Mullama | Ollama | llama.cpp | llama-cpp-python | vLLM |
|---|:-:|:-:|:-:|:-:|:-:|
| GGUF models | ✓ | ✓ | ✓ | ✓ | ✗ (uses safetensors) |
| Native language bindings | **6** | HTTP only | C/C++ only | Python only | Python only |
| Embed in-process (no daemon) | ✓ | ✗ | ✓ | ✓ | ✗ |
| OpenAI-compatible API | ✓ | ✓ | partial | ✗ | ✓ |
| Anthropic-compatible API | ✓ | ✗ | ✗ | ✗ | ✗ |
| Ollama Modelfile compatible | ✓ | ✓ | ✗ | ✗ | ✗ |
| Built-in Web UI / TUI | ✓ | ✗ (third-party) | ✗ | ✗ | ✗ |
| GPU backends | 7 | 4 | 7 | 4 | CUDA |
| Multimodal (vision + audio) | ✓ | partial | ✓ | partial | ✗ |

[Full comparison](https://docs.cognisoc.com/mullama/comparison/vs-ollama/) · [Migration guide from Ollama](https://docs.cognisoc.com/mullama/comparison/migration-from-ollama/)

## What you can build

- **Chatbots and assistants** — streaming responses, multi-turn context, custom system prompts.
- **RAG pipelines** — built-in embeddings (BGE, Nomic), ColBERT-style late interaction, and grammar-constrained JSON output.
- **Voice assistants** — real-time audio capture with voice activity detection, speech-to-text, streaming LLM responses.
- **Production API servers** — OpenAI-compatible endpoints with SSE streaming, per-model resource limits, LRU model eviction, Prometheus metrics.
- **Edge deployments** — embed a quantized model directly in your app; no network dependency, no daemon.
- **Batch processing** — parallel inference across documents with work-stealing scheduling (`parallel` feature).
- **Local AI in Electron / Tauri / mobile** — link the static C ABI library directly into your desktop app.

## GPU acceleration

Set the environment variable for your hardware before building, or use the pre-built CUDA Docker image (`ghcr.io/cognisoc/mullama:<version>-cuda`):

| Backend | Env var | Hardware |
|---|---|---|
| CUDA | `LLAMA_CUDA=1` | NVIDIA GPUs |
| Metal | `LLAMA_METAL=1` | Apple Silicon |
| ROCm | `LLAMA_HIPBLAS=1` | AMD Radeon |
| Vulkan | `LLAMA_VULKAN=1` | AMD / NVIDIA / Intel (cross-platform) |
| OpenCL | `LLAMA_CLBLAST=1` | Intel / AMD legacy |
| SYCL | `LLAMA_SYCL=1` | Intel Arc, Data Center GPUs |
| RPC | `LLAMA_RPC=1` | Distributed inference across machines |

## FAQ

### Is Mullama a drop-in replacement for Ollama?

Yes for the CLI, Modelfile format, model registry, and HTTP port (11434). The OpenAI-compatible API matches Ollama's. The Anthropic-compatible API is a Mullama addition that Ollama doesn't have. Existing Ollama Modelfiles work unchanged; existing client code pointed at `localhost:11434` works unchanged.

### How is Mullama different from `llama.cpp`?

Mullama uses `llama.cpp` as its inference engine — it's the same numerical kernel. What Mullama adds: a safe Rust API on top of the C++ library, six language bindings, an Ollama-compatible daemon with OpenAI/Anthropic HTTP APIs, model registry resolution, Modelfile parser, embedded Web UI, multimodal pipeline, streaming audio with VAD, and production features (per-model resource limits, LRU eviction, persistent stats via sled, Prometheus metrics).

### Does Mullama require a GPU?

No. CPU inference works out of the box. GPU acceleration is optional and gated behind build-time environment variables — see the GPU acceleration table above. Apple Silicon Macs use Metal automatically.

### Can I use the OpenAI Python SDK / LangChain / LlamaIndex with Mullama?

Yes. Point any OpenAI-compatible client at `http://localhost:11434/v1`. LangChain's `ChatOpenAI`, LlamaIndex's `OpenAILike`, and the official `openai` SDK all work without modification.

### Does Mullama support function calling / tool use?

Yes for OpenAI-style `tools` in the chat completion API, and for grammar-constrained JSON output via the underlying `llama.cpp` grammar feature. See the [tool calling guide](https://docs.cognisoc.com/mullama/api/tool-calling/).

### What's the license?

MIT. Pure permissive — use it in commercial products without restriction.

### Where do I report bugs or request features?

[GitHub Issues](https://github.com/cognisoc/mullama/issues) for bugs and feature requests. [Discussions](https://github.com/cognisoc/mullama/discussions) for usage questions.

## Documentation

Full documentation: **[docs.cognisoc.com/mullama](https://docs.cognisoc.com/mullama/)**.

Topics: installation, library usage per language, daemon configuration, API reference (OpenAI, Anthropic, Ollama-compatible), Modelfile format, GPU configuration, multimodal pipeline, streaming audio, embeddings, RAG patterns, deployment recipes (Docker, Kubernetes, systemd, launchd).

## Contributing

```bash
git clone --recurse-submodules https://github.com/cognisoc/mullama.git
cd mullama
cargo test --all-features
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).

---

## Part of the Cognisoc stack

**[Cognisoc](https://www.cognisoc.com)** builds open-source LLM inference for every language and every device — *LLM inference, everywhere.* This project is one of six:

| Project | Language | What it does |
|---|---|---|
| mullama **(this project)** | Python · Node · Go · PHP · Rust · C | Local LLM runtime & server, drop-in Ollama alternative |
| [unillm](https://github.com/cognisoc/unillm) | Rust | Modular inference runtime, 47 architectures |
| [llamafu](https://github.com/cognisoc/llamafu) | Dart / Flutter | On-device inference for mobile apps |
| [llmdot](https://github.com/cognisoc/llmdot) | C# / .NET | Local GGUF inference for the .NET ecosystem |
| [cllm](https://github.com/cognisoc/cllm) | C | Bare-metal unikernel — boots straight into inference |
| [zigllm](https://github.com/cognisoc/zigllm) | Zig | Learn LLMs by building one, from tensors to text |

🌐 [cognisoc.com](https://www.cognisoc.com) · 📚 [docs.cognisoc.com](https://docs.cognisoc.com) · 🐙 [github.com/cognisoc](https://github.com/cognisoc)
