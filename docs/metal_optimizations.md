# Metal / Apple Silicon Optimisation Catalogue

Findings and concrete opportunities surfaced while bringing mullama's M1 numbers
to parity with ollama. Each item lists effort, expected win, the file to touch,
and whether it lands in mullama (Rust) or the cognisoc/llama.cpp fork.

Reference numbers — qwen2.5-0.5b Q4_K_M on M1, after Phase A.2 fixes:
- single-session: decode 1 370 tok/s, prefill 1.3 ms, wall 0.81 s
- 4-session concurrent: 75 tok/s aggregate, 10.2 s wall, 1.0× scaling
- ollama 0.30.10 reference: 60 tok/s baseline, 81 tok/s agg, 9.5 s wall, 1.34× scaling

## Tier 1 — high-leverage, known wins

### T1.1  Single-context, `n_seq_max>1`, batched decode (Phase C)
- **Where:** mullama Rust — `src/daemon/models/pool.rs`,
  `src/daemon/server/generation/*`, `src/context.rs:156`,
  `src/daemon/server/session.rs`
- **Effort:** ~3–5 days (architectural refactor)
- **Why Metal-specific:** Metal exposes a single command queue per device.
  Multiple parallel `llama_decode` calls serialize there (we measured 1 814 s
  for the pool=2 case). The fix is structural: one context, `N` `seq_id`s,
  one `llama_decode` per scheduler tick batching tokens from all active
  sessions. Closes the residual 1.0× → 1.34× scaling gap to ollama.
- **What changes:** `ContextPool` becomes `SeqPool`; per-session `seq_id`
  allocator; continuous-batching scheduler that drains the request queue into a
  single batched decode. `llama_memory_seq_cp/seq_rm/seq_keep` (already in the
  C API) replace `save_state_seq`/`load_state_seq` for cross-turn KV reuse —
  KV stays in Metal residency, no host round-trip.

### T1.2  Skip the host round-trip in session save/restore (unified memory)
- **Where:** mullama Rust + cognisoc fork
  - Rust: `src/context.rs::save_state_seq` / `load_state_seq`
  - Fork: `src/llama-context.cpp::state_seq_get_data` / `state_seq_set_data`
- **Effort:** 1–2 days (mullama uses what fork exposes)
- **Why Metal-specific:** Apple Silicon has unified memory: CPU and GPU
  address the same physical DRAM. `save_state_seq` today bulk-copies the KV
  cache from the Metal buffer into a host `Vec<u8>`. On Mac, that "copy" is a
  pure memcpy through a single memory pool — we should expose a zero-copy
  path that hands out an `MTLBuffer.contents()` view. Then `load_state_seq`
  becomes a pointer swap instead of a 96 MiB memcpy.
- **Side effect:** removes the `state_read_meta: cell_count=N` log spam in
  cross-session swap scenarios (those logs are written by the existing copy
  path).
- **Fallout test:** parity of `save_state_seq → load_state_seq` round-trip;
  ensure the `MTLStorageMode.Shared` mapping doesn't break Metal's cache
  invalidation rules.

### T1.3  Bring `dev-metal`'s per-op source split + parallel compile
- **Where:** cognisoc fork — rebase `5d3eb999b metal : per-op source split +
  parallel compile (#24021)` onto `mullama-parity`
- **Effort:** 1–2 days rebase work
- **Why Metal-specific:** llama.cpp's Metal backend currently compiles the
  entire `default.metallib` as a single library on first context init. The
  patch splits it into 20 small libraries and compiles them in parallel.
  Saves several seconds of cold-start on each first model load — directly
  improves first-request TTFB.
- **Risk:** the upstream patch sits on top of post-`55bb64a` Metal-backend
  refactors; rebasing onto our `mullama-parity` will conflict. Worth doing
  once Phase C lands so we're working on one fewer moving target.

## Tier 2 — measurable wins, modest effort

### T2.1  Lift `ggml_metal_rsets_init` residency keep-alive for server workloads
- **Where:** cognisoc fork — `ggml/src/ggml-metal/ggml-metal.cpp` (the
  `keep_alive = 180 s` daemon log line)
- **Effort:** 1 hour
- **Why Metal-specific:** Metal's residency sets ensure buffers stay
  GPU-resident; once `keep_alive` expires, buffers can be paged out. For a
  long-running server, 180 s is too short — under low traffic the next request
  pays a re-residency cost. Expose `MULLAMA_METAL_KEEP_ALIVE=<seconds>` (env)
  or extend the default to infinity in server mode.

### T2.2  `n_ubatch` sweep on Metal
- **Where:** mullama Rust — `src/daemon/server/config.rs` (add `n_ubatch`
  field), `src/context.rs:155` (default)
- **Effort:** half-day (param plumbing + benchmark sweep)
- **Why Metal-specific:** mullama defaults to `n_ubatch=512`. On Metal, ubatch
  drives kernel-dispatch granularity. The right ubatch trades dispatch
  overhead vs SIMD-group occupancy — on M-series A14 and later the sweet
  spot may be higher than the CPU-optimised default. Bench `256/512/1024/2048`.

### T2.3  `MULLAMA_METAL_QUEUE_DEPTH` — multiple Metal command queues
- **Where:** cognisoc fork — `ggml/src/ggml-metal/ggml-metal.cpp` (where
  `MTLCommandQueue` is created at backend init)
- **Effort:** 2–3 days (needs cross-queue synchronization design)
- **Why Metal-specific:** Today the Metal backend uses a single
  `MTLCommandQueue` per device. Multiple `llama_decode` calls serialize at
  queue submission. With `N` command queues plus `MTLSharedEvent` for KV
  dependencies, two concurrent decodes can overlap encoding while the GPU
  pipelines execution. This is the alternative to Phase C if we want
  *true* parallelism without continuous batching — but Phase C is the
  better path because it also reduces total compute, not just hide latency.

### T2.4  Enable Metal4 tensor API on M5+ / A19+
- **Where:** cognisoc fork — `ggml/src/ggml-metal/ggml-metal.cpp` (the
  `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices`
  log line)
- **Effort:** none for M1 (gate it off); 1–2 days to wire runtime detection
- **Why Metal-specific:** newer Apple GPUs (M5, A19) expose Metal4's tensor
  API — hardware-accelerated low-precision matmul that beats the current
  hand-written MSL kernels by 1.5–2× on prefill. Currently disabled
  unconditionally. Runtime-detect MTLDevice family ≥ Apple10 (or whatever
  the M5 generation maps to) and enable.

## Tier 3 — research-grade, larger investment

### T3.1  MPSGraph for attention block
- **Where:** cognisoc fork — wrap llama.cpp's attention forward pass in an
  `MPSGraph` operator and dispatch via `MPSGraphExecutable`
- **Effort:** 1–2 weeks (proof of concept), more to ship
- **Why Metal-specific:** Apple's `MPSGraph` framework ships highly tuned
  fused attention kernels (the same code path Apple's own ML stack uses).
  llama.cpp's hand-written MSL Flash Attention is competitive but not optimal.
  Worth benching on M1 (modest win expected) and M3 Max (where MPSGraph has
  been observed at 1.4× hand-written MSL on similar workloads).
- **Risk:** MPSGraph's tensor lifecycle differs from ggml's — may need a
  shim layer to avoid extra copies. Best as a Phase D experiment after Phase
  C lands.

### T3.2  Persistent `MTLHeap` for KV pool
- **Where:** cognisoc fork — `ggml-metal.cpp` buffer allocation paths
- **Effort:** 3–5 days
- **Why Metal-specific:** with `n_seq_max=N` (Phase C), KV cache memory
  pressure scales with sessions. Today each layer's K and V tensors are
  separate `MTLBuffer`s — fragmentation reduces residency efficiency.
  Allocating from a single `MTLHeap` of size `N * n_layer * 2 * cell_size`
  lets Metal pack the whole KV cache contiguously, improving cache locality
  and residency-set efficiency.

### T3.3  Lazy command buffer commit (decode pipelining)
- **Where:** cognisoc fork — `ggml-metal.cpp` graph compute path; the
  `ggml_backend_metal_synchronize` callback
- **Effort:** 3–7 days
- **Why Metal-specific:** today llama_decode submits a command buffer, then
  blocks on `waitUntilCompleted` before reading logits. For batched
  multi-seq decode (Phase C), the sampler needs logits *eventually*, but
  the next decode's prefill could start its setup work in parallel.
  Decouple commit-from-wait to overlap CPU-side scheduling with GPU
  execution. Token-level pipelining.

## Quick A/B tests (single-line config flips, run today)

| flag | hypothesis | how to test |
|---|---|---|
| `kv_unified=true` | unified memory has no penalty for shared buffer; may give better cache behavior | flip in `ContextParams::default()`, rerun agent-loop |
| `swa_full=true` (already on) | confirm needed for n_seq_max>1 cases | bench with `false` once Phase C is in |
| `n_batch=2048` (default) vs `1024` / `4096` | bench per-token cost at different batches | add CLI sweep |
| `MULLAMA_OPENMP=1` vs OpenMP off | confirm OpenMP off is still the Metal default win | env-flip rerun |

## Mac-specific defaults already landed (this work)

| change | file | effect |
|---|---|---|
| `n_gpu_layers` default `-1` on macOS, translated to 999 | `src/lib.rs`, `src/model.rs`, `src/daemon/server/config.rs` | all layers on Metal — 65 → 1 370 tok/s decode |
| `DEFAULT_CONTEXT_POOL_SIZE = 1` on macOS | `src/daemon/models/config.rs` | avoid Metal-queue thrash (1 814 s → 10 s for 4 concurrent) |
| Hydrator uses `try_acquire_at` (non-blocking) | `src/daemon/server/hydrator.rs` | no `state_read_meta` thrash during concurrent load |
| `op_offload=true` (matched upstream) | `src/context.rs` | scheduler offloads small host tensor ops; perf wash on 0.5B, free on larger models |
| Fork: Metal `graph_compute` signature fix | `llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp` | unblocks builds on `mullama-parity` |

## What this catalogue did NOT identify as a win

- Going to FP8 / lower-precision KV cache — not Metal-specific; trade-off
  already exposed via `cache_type_k`/`cache_type_v`.
- llama.cpp Vulkan backend on Mac — pointless, Metal is the right path.
- Apple's CoreML — wrong abstraction for GGUF inference; doesn't carry the
  llama.cpp graph structure.
