# Mullama Language Bindings

This directory contains the language bindings for the Mullama LLM library. All bindings share a common C ABI layer (`ffi/`) so the same `llama.cpp`-backed engine is reachable from Rust, Python, Node.js, Go, PHP, and C/C++.

## Supported languages

| Language | Directory | Package | Status |
|---|---|---|---|
| **Rust** (core) | `../src/` | [`mullama` on crates.io](https://crates.io/crates/mullama) | Stable |
| **C / C++** | [`ffi/`](./ffi) | [`mullama-ffi` on crates.io](https://crates.io/crates/mullama-ffi) + `mullama.h` | Stable |
| **Python** | [`python/`](./python) | [`mullama` on PyPI](https://pypi.org/project/mullama/) | Stable |
| **Node.js / TypeScript** | [`node/`](./node) | [`mullama` on npm](https://www.npmjs.com/package/mullama) | Stable |
| **Go** | [`go/`](./go) | [`github.com/cognisoc/mullama` on pkg.go.dev](https://pkg.go.dev/github.com/cognisoc/mullama) | Stable |
| **PHP** | [`php/`](./php) | [`mullama/mullama` on Packagist](https://packagist.org/packages/mullama/mullama) | Beta |

## Pick the right binding

```
Building in …             Use
────────────────────────────────────────────────
Rust app                  mullama (crates.io)        — direct, no FFI hop
Python app / Jupyter      mullama (PyPI)             — PyO3, in-process
Node.js / TS / Electron   mullama (npm)              — napi-rs, in-process
Bun / Deno                mullama (npm)              — works via Node-API shim
Go service / CLI          github.com/cognisoc/mullama — CGO + FFI library
PHP / Laravel / Symfony   mullama/mullama (Packagist) — PHP FFI extension
C / C++ / new binding     mullama-ffi (crates.io) + mullama.h
Mobile (Android/iOS)      mullama-ffi static library — bundled with JNI / Swift
Browser (WASM)            not supported yet (roadmap)
```

If you only need an **HTTP API**, you don't need a binding at all — run `mullama serve` and point any OpenAI- or Anthropic-compatible client at `http://localhost:11434`.

## Architecture

All bindings share the same C ABI surface:

```
┌─────────────────────────────────────┐
│         mullama (Rust core)         │
└────────────────┬────────────────────┘
                 │
┌────────────────┴────────────────────┐
│         mullama-ffi (C ABI)         │
│  Handle management, error codes,    │
│  streaming callbacks, cancellation  │
└────────────────┬────────────────────┘
                 │
   ┌─────────┬───┴───┬─────────┬────────┐
   │         │       │         │        │
┌──▼────┐ ┌──▼───┐ ┌─▼───┐ ┌───▼──┐ ┌───▼──┐
│napi-rs│ │ PyO3 │ │PHP  │ │ cgo  │ │  C/  │
│Node.js│ │Python│ │FFI  │ │ Go   │ │  C++ │
└───────┘ └──────┘ └─────┘ └──────┘ └──────┘
```

Memory-safe handles, thread-local error messages, callback-based streaming, and roughly 50 FFI functions covering the full Mullama surface — see [`ffi/README.md`](./ffi/README.md) for details.

## Quick-start parity

Every binding exposes the same concepts (model, context, sampler, embeddings) under language-idiomatic names:

| Concept | Rust | Python | Node.js | Go | PHP | C |
|---|---|---|---|---|---|---|
| Load model | `Model::load()` | `Model.load()` | `Model.load()` | `LoadModel()` | `Model::load()` | `mullama_model_load()` |
| Create context | `Context::new()` | `Context()` | `new Context()` | `NewContext()` | `new Context()` | `mullama_context_new()` |
| Generate | `ctx.generate()` | `ctx.generate()` | `ctx.generate()` | `ctx.Generate()` | `$ctx->generate()` | `mullama_generate()` |
| Stream | `ctx.generate_stream()` | `ctx.generate_stream()` | `ctx.generateStream()` | `ctx.GenerateStream()` | `$ctx->generateStream()` | `mullama_generate_streaming()` |
| Tokenize | `model.tokenize()` | `model.tokenize()` | `model.tokenize()` | `model.Tokenize()` | `$model->tokenize()` | `mullama_tokenize()` |
| Embed | `EmbeddingGenerator::new()` | `EmbeddingGenerator()` | `new EmbeddingGenerator()` | `NewEmbeddingGenerator()` | `new EmbeddingGenerator()` | `mullama_embedding_generator_new()` |

Sampler presets (`greedy`, `precise`, `default`, `creative`) are available in every binding under the same names.

## Platform support

Pre-built artifacts ship for every release:

| Platform | CPU | CUDA | Metal | Vulkan | ROCm |
|---|:-:|:-:|:-:|:-:|:-:|
| Linux x86_64 | ✓ | ✓ | – | ✓ (build) | ✓ (build) |
| Linux aarch64 | ✓ | – | – | ✓ (build) | – |
| macOS x86_64 | ✓ | – | – | – | – |
| macOS aarch64 | ✓ | – | ✓ | – | – |
| Windows x86_64 | ✓ | ✓ (build) | – | ✓ (build) | – |

"build" = supported via `cargo build` with the relevant env var, not shipped pre-built.

## Building from source

Prerequisites:
- Rust toolchain (1.75+)
- System dependencies (audio + image + ffmpeg dev packages — see the [top-level README](../README.md#contributing))
- Language tooling:
  - Python — `pip install maturin`
  - Node — Node 18+, npm
  - Go — Go 1.21+
  - PHP — PHP 7.4+ with `ffi` extension

```bash
# Build the FFI library (used by every non-Rust binding)
cargo build --release -p mullama-ffi

# Build the Python wheel
cd bindings/python && maturin build --release

# Build the Node.js native module
cd bindings/node && npm install && npm run build

# Build + check the Go binding
cd bindings/go && go build ./...

# Run the PHP test suite
cd bindings/php && composer install && composer test
```

## CI / release pipeline

Two GitHub Actions workflows cover everything:

- **`.github/workflows/ci.yml`** — runs on every push to `main` and every PR. Tests the core crate, runs clippy + fmt, smoke-builds each binding against the FFI library, builds the daemon with the embedded Web UI, and smoke-builds the CPU Docker image.
- **`.github/workflows/release.yml`** — triggered by a `v*` tag (or `workflow_dispatch` with `dry_run=true`). Builds the CLI binary, FFI library, Python wheels, and Node native modules for every supported target; pushes multi-arch Docker images to GHCR (CPU + CUDA); publishes to crates.io / PyPI / npm via OIDC trusted publishing; and creates the GitHub Release with cargo-dist / Homebrew-friendly artifact names.

See [the release docs](https://docs.cognisoc.com/mullama/contributing/releasing/) for the tagging convention and the one-time trusted-publisher setup.

## Contributing a new binding

1. Create a new directory under `bindings/<lang>/`.
2. Build against the FFI library (`bindings/ffi/`) — don't bind to `llama.cpp` directly.
3. Match the established naming conventions (see the table above).
4. Write tests that exercise model load, generate, streaming, embeddings, and tokenize.
5. Add an SEO/AI-search-optimised README — follow the template used by `bindings/python/README.md` (title, "What is X?", comparison, install, quick start, features, supported models, GPU table, API, FAQ, cross-links).
6. Wire the build into `.github/workflows/ci.yml` (smoke build) and `.github/workflows/release.yml` (matrix build + publish).

## License

MIT OR Apache-2.0

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
