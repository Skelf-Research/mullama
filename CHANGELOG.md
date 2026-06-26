# Changelog

## [0.4.0] - 2026-06-26

### Added

#### Agentic inference — the cache *is* the conversation
- **Cross-turn KV reuse**: a named `session` pins a context-pool slot and keeps its KV cache alive across turns. Each turn matches the new prompt's longest common prefix against the cached tokens, drops only the divergent tail, and decodes just the new suffix — turning per-turn prefill from `O(history)` into `O(delta)`. Numerically identical to a full decode (greedy parity preserved). Measured up to **27–28× less prefill** by turn 11 of a 12-turn agent loop.
- **Sliding-window pruning**: `session_keep_turns=N` trims everything older than the last N user turns before rendering, bounding prompt + pinned KV so long sessions can't overflow `n_ctx` (which otherwise crashes the decode).
- **Durable content-addressed KV store**: a sled-backed CAS (`~/.mullama/kv-cas/`) persists each session's token sequence + per-sequence KV blob, so a daemon restart restores instead of re-prefilling the whole history. Gated by a compatibility digest (model + KV-layout params). Restore keeps the delta-prefill win across restarts; restored output is token-identical.
- **Multi-session scheduling**: sessions pin to pool slots by affinity with durable-safe LRU eviction (an evicted session's KV is persisted, so it restores rather than re-prefilling). Concurrent requests to different sessions decode in parallel across slots.
- **Background hydrator with hydration modes**: pre-warms dormant durable sessions into free slots so their next request is a hot in-memory hit. `--hydration off|idle|active`; default is platform-aware — `active` (parallel-fill, pre-warm during live decodes) on macOS / Apple Silicon where unified memory has the bandwidth headroom, `idle` on x86. `active` mode keeps one slot free as headroom for live traffic.
- **Agent file-access prefetcher**: predicts the files an agent will read next (import-following + directory locality, parsed from conversation content) and warms the OS page cache during idle windows.

#### Constrained decoding
- **Grammar in the streaming path**: structured-output grammars (`response_format`) now apply to streaming responses identically to non-streaming.
- **Tool-call constrained decoding**: `tools` + `tool_choice` (`required` / specific function) synthesize a GBNF grammar that forces a valid JSON tool call with the name constrained to the offered tools. Fixes two latent grammar-engine crashes (sampler ordering and a double-`accept` that aborted the grammar / double-counted penalties).

#### Speculative decoding & quantization
- **Prompt-lookup speculative decoding**: greedy-exact, single-model n-gram speculation — propose next tokens by matching the suffix against history, verify in one batched target pass (`Context::decode_batch_argmax`). Output is token-for-token identical to greedy. Measured **72.6% acceptance / ~6.9 tokens-per-forward-pass / up to 6.8×** on repetitive/structured output (break-even on prose). See `examples/speculative_lookup.rs`.
- **INT4 group quantization + Hadamard rotation kernel**: a self-contained 4-bit group-quantization runtime kernel (6.4× vs f32) with an optional QuaRot-style rotation. See `examples/int4_rotation_demo.rs`. (Standalone kernel — not yet wired into the llama.cpp graph.)

#### Benchmarks & docs
- `docs/AGENTIC_PERFORMANCE.md`: measured per-feature performance map with reproduction commands and honest caveats.
- `bench/run_agentloop.sh`, `bench/concurrent_sessions.py`: reproduction harnesses for the KV-reuse and concurrent-throughput benchmarks.

## [0.3.1] - 2026-05-20

### Fixed
- **Packaging**: `0.3.0` failed to build from crates.io with the `daemon` feature because `configs/models.toml` was not in the published include list. The daemon's model registry no longer embeds that TOML via `include_str!`.

### Changed
- **Model registry lookup**: `ModelRegistry::load_embedded()` is removed. The daemon now loads the registry via `ModelRegistry::load_default()`, which searches:
  1. `$MULLAMA_REGISTRY` (path to a TOML file), then
  2. `<config_dir>/mullama/models.toml` (e.g. `~/.config/mullama/models.toml` on Linux, `~/Library/Application Support/mullama/models.toml` on macOS, `%APPDATA%\mullama\models.toml` on Windows).

  If neither is found, an empty registry is used. Local paths and explicit `hf:` / `ollama:` prefixes still resolve without a registry; only the short-name aliases (e.g. `llama3.2:1b`) require one. The repository's `configs/models.toml` is still shipped as a starter file users can copy into their config directory.

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
