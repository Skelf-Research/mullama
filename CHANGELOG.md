# Changelog

## [0.3.0] - 2026-04-02

### Added

#### Memory Management & Multi-Model Intelligence
- **mimalloc everywhere**: Global allocator now active for library users and all bindings via `use-mimalloc` feature
- **Per-model statistics**: Track requests, tokens, tok/s, memory, and last-used time per model
- **Memory estimation**: `estimate_model_memory()` calculates expected memory before loading
- **Resource limits**: `--max-models`, `--max-memory`, `--eviction-policy`, `--idle-unload` CLI flags
- **LRU eviction**: Automatically unload least-recently-used models when limits are reached
- **sled persistent store**: Per-model stats, metadata cache, prompt cache, and sessions survive daemon restarts
- **Enhanced metrics**: Per-model Prometheus metrics at `/metrics` with tok/s, memory, request counts
- **Rich `mullama ps`**: Table view with per-model memory, requests, tok/s, last used
- **ModelDetailedStats**: Expanded daemon stats in protocol and status endpoints

#### Build System
- **Vulkan backend**: `LLAMA_VULKAN=1` enables AMD/NVIDIA/Intel GPU support via Vulkan
- **SYCL backend**: `LLAMA_SYCL=1` enables Intel Arc GPU support
- **RPC backend**: `LLAMA_RPC=1` enables distributed inference across machines
- **Configurable CPU features**: `LLAMA_AVX=0`, `LLAMA_AVX512=1`, `LLAMA_PORTABLE=1`, etc.
- **Updated CUDA architectures**: Default includes Hopper (H100), configurable via `LLAMA_CUDA_ARCHS`
- **BLAS controls**: `LLAMA_BLAS=1`, `LLAMA_BLAS_VENDOR`, `LLAMA_NO_ACCELERATE=1`

#### Full Model Control
- **Expanded CLI flags**: `--flash-attn`, `--cache-type-k`, `--cache-type-v`, `--mlock`, `--no-mmap`, `--batch-size`, `--rope-freq-base`, `--rope-freq-scale`, `--split-mode`, `--defrag-thold`
- **Expanded IPC protocol**: All model loading params available via `Request::LoadModel`
- **Expanded HTTP API**: `LoadModelRequest` accepts all params
- **DaemonConfig defaults**: Server-level defaults for all model params

#### Developer Experience
- **Hardware presets**: `HardwarePreset` enum with 7 presets (CpuLowMemory through MaxPerformance)
- **Auto-detection**: `HardwarePreset::detect()` picks optimal preset for current hardware
- **Preset API in all bindings**: Python, Node.js, Go, PHP all expose presets
- **Expanded FFI**: Context params include KV cache types, RoPE, defrag threshold
- **PHP ContextParams/ModelParams/HardwarePreset**: Full typed PHP 8.1+ classes

#### Documentation
- **Hardware Configuration Guide**: Recipes for CPU, NVIDIA, AMD, Intel, Apple Silicon, multi-GPU
- **Performance Tuning Guide**: Flash attention, KV cache, mmap/mlock, threading, quantization
- **Runnable examples**: `simple.rs` and `multi_model.rs` with real load->generate->print flows

### Changed
- Version bumped to 0.3.0
- `mimalloc` global allocator moved from binary to library (when `use-mimalloc` feature is active)
- `DaemonStats` expanded with memory and per-model detail fields

### Fixed
- CUDA architecture list now includes Hopper (sm_90, sm_90a) for H100 support

## [0.2.0] - 2026-04-01

### Production Readiness Release

This release focuses on hardening Mullama for production use as a drop-in Ollama replacement.

### Changed
- **Version Sync**: All crates and packages synchronized to 0.2.0
- **Version Derivation**: FFI, Python, Node.js, Go, and PHP versions now derived from `CARGO_PKG_VERSION` or native FFI calls to prevent drift
- **Unwrap Elimination**: Replaced ~30 production `unwrap()` calls in daemon code with proper error handling
  - Multimodal context access in `server.rs` now returns proper errors
  - Regex patterns in `ollama_template.rs` moved to `LazyLock` statics
  - JSON serialization in `anthropic.rs` uses `unwrap_or_else` fallbacks
  - HTTP response builders use explicit `expect` with static messages
- **CI**: Clippy now runs with `--features full` to catch more warnings
- **Cargo.lock**: Added to `include` list for binary crate reproducibility

### Added
- **Tracing/Logging**: Integrated `tracing` crate with `tower-http::TraceLayer`
  - Use `MULLAMA_LOG=debug` or `RUST_LOG=mullama=debug` to control log levels
  - All daemon `eprintln!` calls replaced with structured `tracing` macros
- **Public API Exports**: Enabled previously commented-out advanced modules:
  - `LoRAAdapter`, `LoRAManager` — LoRA adapter management
  - `Grammar`, `GrammarRule` — Grammar-constrained generation
  - `ControlVector`, `ControlVectorManager` — Steering vector support
  - `SpeculativeDecoder`, `SpeculativeConfig` — Speculative decoding
  - `QuantizationEngine`, `QuantizationParams` — Model quantization
  - `GpuManager`, `GpuDevice`, `AllocationStrategy` — GPU management
- **FFI Layer**: New metadata iteration functions:
  - `mullama_model_metadata_count()`
  - `mullama_model_metadata_key()`
  - `mullama_model_metadata_value()`
  - `mullama_apply_chat_template()` convenience alias
- **Python Bindings**: `generate_stream()` now returns a `TokenIterator` with `__iter__`/`__next__` protocol instead of a flat list
- **Node.js Bindings**: Added `generate_stream_full()` returning `{ pieces, text }` object
- **Go Bindings**: Added `ApplyChatTemplate()`, `Metadata()`, `GenerateStreamTokens()`, and version from native FFI
- **PHP Bindings**: Added `Model::chatTemplate()`, `Model::metadata()`, improved `Context::generateStream()`, version from FFI
- **Ollama API**: Full Ollama-compatible REST API endpoints:
  - `POST /api/generate` — Text generation
  - `POST /api/chat` — Chat completion
  - `GET /api/tags` — List local models
  - `POST /api/show` — Show model info
  - `POST /api/copy` — Copy model
  - `DELETE /api/delete` — Delete model
  - `POST /api/embeddings` — Generate embeddings
  - `GET /api/ps` — List running models
  - `GET /api/version` — Version info
- **Docker Support**: `Dockerfile`, `Dockerfile.cuda`, `docker-compose.yml`, `.dockerignore`
- **TLS Support**: `--tls-cert` and `--tls-key` CLI flags, `tls` feature flag with `axum-server`
- **Version Bump Script**: `scripts/bump-version.sh` for atomic version updates

### Fixed
- `registry.rs`: Replaced `unwrap()` after `strip_prefix` with `if let Some`
- `ollama.rs`: Replaced `unwrap()` after `split_once` with `if let Some`
- `hf.rs`: `HfDownloader::default()` no longer panics on initialization failure

## [0.1.1] - Previous Release

Initial public release with core LLM inference capabilities.
