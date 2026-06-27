# Metal vs Generic — what differs by platform and why

After Phases 0/A/A.2/A.3/B.2/C land, mullama's behavior depends on the build
target. This doc enumerates exactly which decisions vary by platform, where
they live in the code, and the empirical evidence that drove each choice.

## Defaults — one-glance table

| knob | macOS | other | reason |
|---|---|---|---|
| `n_gpu_layers` | `-1` → 999 | `0` | Apple Silicon always has Metal; on x86/Linux GPU is opt-in (CUDA/ROCm builds vary). |
| `DEFAULT_CONTEXT_POOL_SIZE` | `1` | `4` | Metal has one command queue; parallel `llama_decode` thrashes (we measured >100× slowdown). CUDA streams overlap. |
| Hydration mode (`HydrationMode::platform_default`) | `Active` | `Idle` | Unified high-bandwidth memory makes parallel-fill free; DDR doesn't. |
| Phase-C batched scheduler default | **on** (`MULLAMA_BATCHED=1` implicit) | off | Validated on M1 only; CUDA/ROCm scaling story may differ. |
| `MULLAMA_BATCHED_SLOTS` default | `16` | `8` | M1 sweep landed 16 slots at the sweet spot (3.6× scaling); haven't characterized other targets. |

Code reference for each:
- `n_gpu_layers`: `src/lib.rs::default_gpu_layers()`, `src/daemon/server/config.rs:92`
- `DEFAULT_CONTEXT_POOL_SIZE`: `src/daemon/models/config.rs:3`
- `HydrationMode`: `src/daemon/server/config.rs::HydrationMode::platform_default`
- Phase-C on/off: `src/daemon/models/manager.rs::ModelManager::load` (the `batched_enabled` block)

## Generic — works the same everywhere

These either ride on existing llama.cpp portability or were measured to be neutral across platforms:

| change | scope | what it does |
|---|---|---|
| Phase A.2: translate `n_gpu_layers=-1` → `999` | `src/model.rs::load_with_params` | C convention is large positive int; we used `-1` as the "all" sentinel and llama.cpp silently sent everything to CPU. |
| Phase A.3: `op_offload: true` default | `src/context.rs::ContextParams::default()` | Match upstream llama.cpp default; cost-free everywhere (unified memory makes the "copy" free on Mac; on CUDA op_offload was always the better default). |
| Phase B.2: hydrator `try_acquire_at` | `src/daemon/server/hydrator.rs` | Non-blocking slot acquire prevents the hydrator from stealing a slot held by a live request, eliminating `state_read_meta` spam under load. |
| Phase 0: cognisoc fork pin restored | `.gitmodules`, `bench/check_fork.sh` | All llama.cpp work lives on `cognisoc/mullama-parity`. Drift guard prevents accidental upstream-revert. |
| Phase 0: Metal `graph_compute` signature fix | `llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp` (pending fork commit) | The Ollama-0.24.0 alignment patch added a 3rd int arg to the interface but didn't update Metal. Macs couldn't build until this was patched. |
| Phase C: BatchScheduler | `src/daemon/server/batcher/` | Single-context, `n_seq_max>1`, continuous batching. The architectural fix; works on any backend, but matters most on Metal where parallel decode is otherwise serialized. |
| Phase C: per-tick instrumentation | `BatcherStats` in `scheduler.rs` | `MULLAMA_BATCHER_DEBUG=1` emits tick-level decode_ms / sample_ms / overhead_ms. Findings drive future tuning anywhere. |

## Metal-specific — only matters on Apple Silicon

These are wins that don't apply (or even regress) elsewhere.

### 1. Pool size ≤ 1 outside Phase C
On Metal, two parallel `llama_decode` calls serialize at the single
`MTLCommandQueue` *and* fight in the GGML CPU threadpool. We measured the
collapse: `--context-pool-size 2` with 4 concurrent stateless requests went
from a steady-state 10s to a 1 814s tarpit (without Phase C). On CUDA, each
context can have its own stream; pool>1 actually overlaps.

Phase C side-steps this by using one context + multiple `seq_id`s, so the
pool-size lever is now mainly about *memory headroom* (each pool slot is a
full KV cache, ~96 MiB at our default `n_ctx=8192`).

### 2. Hydrator Active mode
The hydrator pre-warms idle durable sessions into free pool slots while live
requests run. On Mac the unified-memory bandwidth absorbs the extra reads;
on DDR x86 it eats into the live decode's bandwidth budget.

### 3. Larger `MULLAMA_BATCHED_SLOTS` default
We see scaling continue improving up to at least 16 slots on M1 (3.58× at 16
sessions). The exact ceiling is GPU-memory-bound, not compute-bound — Apple
Silicon's bigger configurations (M2 Max, M3 Max, M3 Ultra) should profitably
go higher; we haven't measured. CUDA pads its slot count differently because
the per-stream KV pressure interacts with VRAM.

### 4. Flash Attention auto-enables on Metal
Set by llama.cpp itself when `flash_attn_type = AUTO` and the device
supports FA — Apple GPUs do. Worth knowing as a default; if FA ever causes
correctness issues on a Mac model, set `--flash-attn false` to compare.

## Empirical-evidence box: why these defaults

Bench numbers from `/Volumes/Github/mullama/bench/` on Apple M1, qwen2.5-0.5b
Q4_K_M, against ollama 0.30.10:

| build | 1-session wall | 4-conc wall | 4-conc agg tok/s | scaling | latency infl |
|---|---:|---:|---:|---:|---:|
| ollama 0.30.10 | 1.07 s | 9.5 s | 81 | 1.34× | 2.78× |
| mullama pre-fix (upstream, pool=2, hyd active) | 1.38 s | 1 814 s | 0.4 | 0.01× | 575× |
| mullama Phase A (cognisoc, gpu=-1, pool=2) | 1.20 s | timeout >600s/req | n/a | n/a | n/a |
| mullama Phase A.2 (Metal KV) | 0.81 s | 10.2 s | 75 | 1.0× | 3.7× |
| **mullama Phase C (default-on, 16 slots, 4 sessions)** | 0.82 s | **5.13 s** | **150** | **2.02×** | **1.98×** |
| **mullama Phase C (16 sessions)** | 0.82 s | **12.2 s** | **252** | **3.37×** | **4.36×** |

Each row is a deliberately picked single-knob change; the table tells you
which knob bought what.

## What's not segregated (yet)

These either don't have enough data, or are awaiting a future investigation:

- **CUDA pool-size sweep**: we set `DEFAULT_CONTEXT_POOL_SIZE = 4` on Linux
  based on the prior README findings, but never verified Phase C on CUDA.
  Worth a sweep; may flip the default to `1 + batched` like macOS.
- **ROCm/Vulkan**: untested. Behave as "other" → batched off, pool=4,
  hydration=Idle.
- **Per-Apple-chip slot ceiling**: 16 on M1; M2/M3/M4 may scale higher.
  `MULLAMA_BATCHED_SLOTS` overrides for now.

## Knobs every user can flip

User-facing surfaces; not platform-coded.

| flag / env | effect |
|---|---|
| `--gpu-layers N` | override `n_gpu_layers`; `-1` → all on GPU, `0` → CPU |
| `--context-pool-size N` | override pool size; mostly relevant for memory headroom now that Phase C exists |
| `--ubatch-size N` | physical micro-batch size; M1 sweep showed 256 slightly wins for 0.5B model, 512 is default |
| `--n-seq-max N` | override context's sequence max (the substrate Phase C uses) |
| `--hydration {off,idle,active}` | manual override of the platform default |
| `MULLAMA_BATCHED={0,1}` | force Phase-C scheduler off/on regardless of platform default |
| `MULLAMA_BATCHED_SLOTS=N` | scheduler slot count (default 16 on macOS, 8 elsewhere) |
| `MULLAMA_BATCHER_DEBUG=1` | per-tick scheduler telemetry to stderr |
| `MULLAMA_OPENMP=1` | restore legacy OpenMP CPU backend (slower on small models) |
| `MULLAMA_STATIC=1` | static single-binary build (no dynamic backends) |
