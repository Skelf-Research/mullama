# Phase-C Architectural Review

A pass over the current Phase-C-on-dev-metal architecture to find what's
still leaking performance, correctness, or simplicity. Findings grouped by
severity. The dominant remaining cost is `llama_sampler_sample` per slot
(measured at ~10 ms vs 0.6 ms decode). The 7B-at-8-concurrent win
(64.27 s wall, 23.9 tok/s, 1.99× scaling) is bottlenecked here.

## Correctness bugs (must fix)

### C1 — Streaming token loss under back-pressure
`src/daemon/server/batcher/slot.rs::emit_token_chunk` uses `tx.try_send` and
silently discards the chunk if the channel is full (capacity 64). A slow SSE
consumer (or a momentarily-blocked one) loses tokens — the client sees a
truncated/garbled response with no indication anything went wrong. This is
worse than back-pressure because the scheduler keeps generating into a hole.
**Fix**: switch to blocking `send` with a bounded budget (e.g. spawn a
companion task per slot that drains the channel), or mark the slot as
"consumer lagging" and cancel via the existing `cancel` flag.

### C2 — Long-session KV overflow
The legacy `kv_reuse` path had `trim_to_last_n_user_turns` (sliding-window
pruning) via the `session_keep_turns` param. The batched path never calls
it. A long-running agent session accumulates KV until `n_ctx` is exhausted,
at which point `llama_decode` returns "failed to find a memory slot for
batch of size N" (we saw this on the 3B during the bench). **Fix**: at
request-handler level when `session_id.is_some()`, apply
`trim_to_last_n_user_turns` to `messages` before building the prompt — this
also keeps the `slot.cached.tokens` length bounded.

### C3 — Cancellation doesn't reach in-flight prefill chunks
The scheduler checks `cancel` flag for `Generating` slots only. A slot mid-
prefill (e.g. on a multi-thousand-token prompt arriving in 512-token
chunks) keeps prefilling for many ticks before reaching `Generating` state
— cancellation is delayed by the rest of the prefill. **Fix**: also check
the flag in the `Prefilling` arm; if set, finalize_err the slot and skip.

## High-leverage perf (1-2 days each)

### P1 — Parallel per-slot sampling (the big one)
The dominant per-tick cost: `task.sampler.sample(context, batch_pos)` runs
sequentially for every active slot. With 8 slots, that's 8 × 10 ms ≈ 80 ms
of sample-phase per tick. The actual `llama_decode` was 0.6 ms.

Per-slot sampling is *almost* embarrassingly parallel:
- `llama_sampler_sample` reads logits from `ctx` (after the one-time
  post-decode `llama_synchronize`), then applies the per-slot sampler chain.
- Each slot's `SamplerChain` is independent (no shared state).
- The context is only being read after sync.

`unsafe impl Sync for Context` already exists in `src/context.rs`. Rayon is
already a (feature-gated) dep. Sketch:

```rust
// after llama_decode + llama_synchronize once
self.slots
    .par_iter_mut()           // requires unsafe-Sync ctx; already declared
    .zip(logits_pos.par_iter())
    .for_each(|(slot, lpos)| {
        if let Some(pos) = lpos {
            let tok = slot.task.sampler.sample(ctx_ref, *pos as i32);
            slot.task.sampler.accept(tok);
            // (state mutations to slot.state collected into a second pass)
        }
    });
```

Concurrency safety to verify: `llama_sampler_sample` is a small read-only
ctx call after sync; multiple invocations on different samplers should not
race. **Risk**: subtle Metal-driver assumptions about "one thread per
context" — need a careful test pass before flipping the switch.

Expected win: 8-slot tick goes from ~80 ms sample → ~10 ms sample (one
parallel batch) → ~8× faster sampling phase → ~2-4× aggregate-throughput
boost at concurrency. The 7B at 8-conc could go from 64 s to ~25-30 s.

### P2 — Native batched sampler chain (Tier-3 from earlier doc)
dev-metal's `llama_context_params` now has `samplers: *mut llama_sampler_seq_config` and
`n_samplers: size_t`. This is upstream's native batched-sampler API — the
context can sample N seq_ids in one call, on-device.

We already added the field to `sys.rs` (opaque). Wiring it up requires:
- Declaring `llama_sampler_seq_config` struct
- Per-slot config registration at context-build time
- Replacing the per-slot `Sampler::sample` loop with a single call

Bigger lift than P1 but eliminates the bottleneck entirely (no CPU-side
sampling). Probably the right long-term answer.

## Memory / resource (small but free)

### M1 — Vestigial `ContextPool` when batcher is on
`LoadedModel` always allocates a `ContextPool` (size 1 on macOS default)
even when `batcher.is_some()`. The pool context is never used in that case.
That's ~96 MiB (0.5B / 8K context) — ~448 MiB (3B) — ~896 MiB (7B) of KV
buffer allocated and untouched. **Fix**: make pool allocation conditional
(`if batcher_enabled { pool_size = 0; skip pool }`) or move pool inside
the legacy-fallback path entirely.

### M2 — Sampler chain rebuilt per task
`scheduler::assign_to_first_idle` calls `build_chain_with_grammar` for
every assigned task. For typical defaults (no grammar, OpenAI temp=1
top_p=1 etc), the chain construction is identical across requests —
allocating and freeing C-side samplers per request. **Fix**: small LRU
cache keyed by `(sampler_params, grammar_gbnf.is_some())`; reuse the chain
via `llama_sampler_reset`. Maybe 1-2 ms per request saved; meaningful at
high request rate.

### M3 — Hydrator is dead code under batcher
The background `hydrate_idle_sessions` loop wakes every 2 s, looks for
durable sessions to pre-warm into pool slots. With the batcher routing all
work, the pool has nothing to pre-warm — hydrator just locks the unused
pool context briefly and returns. **Fix**: short-circuit when
`loaded.batcher.is_some()`; the batcher does its own restore on-demand.

### M4 — Pre-allocate the per-tick `llama_batch`
`scheduler::tick` calls `llama_batch_init(total_tokens, 0, 1)` and
`llama_batch_free` each tick. For a ~80 tick/s steady state, that's 160
allocations per second. The batch is small (max `n_batch` tokens), but the
allocator churn is unnecessary. **Fix**: allocate once at scheduler init
sized for `n_batch`, reuse across ticks (zero `n_tokens` each tick).

## Scheduling / fairness

### S1 — FIFO queue + no priorities
The mpsc queue is plain FIFO. A burst of fresh requests can push an
agentic-loop's next turn behind newcomers, breaking expectation. **Fix**:
priority queue at `assign_pending`; bump priority for tasks whose
`session_id` already has a hot slot (cache locality + fairness).

### S2 — Slot count is fixed at scheduler construction
`MULLAMA_BATCHED_SLOTS=16` is set once. On macOS unified memory, slots
cost ~100 MB KV each; if a model loads on a 32 GB Mac vs a 16 GB Mac, we
should size differently. **Fix**: derive default from available device
memory at startup.

### S3 — Cross-model serialization not coordinated
With N `LoadedModel`s, each has its own `BatchScheduler` task. Two models
in flight = two parallel `llama_decode` calls into Metal's single command
queue — exactly the pathology we fixed at the single-model level, recurring
at the multi-model level. **Fix**: a single "compute coordinator" task
that serializes Metal-bound decode calls across all model schedulers.
Significant refactor; defer until multi-model usage is real.

### S4 — `assign_pending` uses a coarse heuristic
The "if any_busy && any_idle: try_recv else block on recv" rule briefly
blocks the loop when fully idle, then again drops into try_recv when busy.
**Fix**: `tokio::select!` on `rx.recv() / tokio::time::sleep(1ms)`. Mostly
elegance; perf delta likely tiny.

## Robustness

### R1 — Memory pressure not observed by scheduler
The `MemoryMonitor` exists but the batcher never queries it. If we run
near VRAM/RAM limit, we'll OOM-kill rather than gracefully shedding the
oldest idle slot. **Fix**: scheduler-side LRU eviction of idle slots
under memory pressure.

### R2 — Errors in the scheduler tick fail *all* in-flight slots
`run()` catches a tick error and calls `finalize_err` on every active slot
(see `scheduler.rs::run`). That's coarse — one bad request's
`llama_decode` failure shouldn't kill seven well-behaved ones. **Fix**:
isolate the failing slot by dropping its tokens from the batch and retrying
the tick.

### R3 — No graceful shutdown wait for in-flight requests
SIGTERM kills the daemon; in-flight slots' clients get a closed connection
mid-stream. **Fix**: drain the scheduler — stop accepting new tasks,
let active slots finish (with timeout), then exit.

## Code-shape / maintainability

### X1 — Header still says `ollama-v0.24.0` (fixed in `build.rs`, awaiting rebuild)
Cosmetic but misleading after the dev-metal rebase. Already addressed in
the `build.rs` change.

### X2 — `ReplyMode::Buffered { tx: placeholder_tx }` in `generate_text_streaming_batched`
We create a dummy oneshot just to make the type system happy; `submit_streaming`
replaces it with the real `Streaming` variant. Refactor: split the task
struct into a header (everything but reply) + reply, so each path
constructs the reply directly. Cosmetic.

### X3 — `Slot::cached` couples session_id to slot KV
`SlotCache { session_id, tokens }` says "this slot's seq_id KV corresponds
to these tokens for this session". If a session migrates between slots
(e.g. when slot eviction is added — R1), we'd need to clear the cache.
Document the invariant clearly; consider promoting it to a type-state
encoding.

## Process / bench gaps

### B1 — No per-slot latency histogram
We have per-tick `MULLAMA_BATCHER_DEBUG=1`. We don't track per-slot
end-to-end latency. **Fix**: emit a per-slot finalize event with
`prefill_ns / decode_ns / total_ns / chunks_dropped` so the SLO target
("P95 latency for slot X") is observable.

### B2 — Grammar/structured-output not benched at concurrency
We pass grammar through but never measured its overhead under batched.
Likely fine but unverified.

### B3 — CUDA path entirely unverified
`MULLAMA_BATCHED` is implicit-on macOS only; on Linux+CUDA the legacy pool
path runs. The Phase-C scheduler should work there too (it's
backend-agnostic), but nobody's run the bench.

## Suggested order of attack

If we had a week to spend on this:

1. **Day 1 — Correctness**: C1 (streaming back-pressure), C2 (sliding-window pruning), C3 (cancel during prefill).
2. **Day 2 — Resource cleanups**: M1 (skip pool when batcher on), M3 (hydrator no-op), M4 (pre-alloc batch). Saves ~100-900 MB per loaded model.
3. **Day 3-4 — Parallel sampling (P1)**: implement, test exhaustively for Metal-driver thread-safety quirks, bench. Expected: 2-4× concurrency throughput.
4. **Day 5 — Native batched sampler (P2)**: wire up `llama_sampler_seq_config`. Eliminates the sampling phase entirely.
5. **Beyond**: S1/S2/S3 (scheduling fairness), R1/R3 (robustness), then CUDA bench.

## What this round will land

The high-leverage safe subset:
- C1 streaming back-pressure (correctness)
- C2 sliding-window pruning (correctness, agentic long sessions)
- M1 skip pool when batcher on (memory)
- M4 pre-allocate batch buffer (cleanup)

The bigger swings (P1 parallel sampling, P2 native batched sampler) are
documented and queued — they need a careful afternoon with focused testing,
not auto-mode iteration.
