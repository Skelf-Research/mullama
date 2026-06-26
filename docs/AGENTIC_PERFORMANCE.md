# Agentic Performance: KV Reuse, Speculative Decoding, and Quantization

This document maps the performance advantages of mullama's agent-oriented
features, the benchmarks that measure them, and how to reproduce the numbers.
Unlike [`PERFORMANCE.md`](PERFORMANCE.md) (a config-tuning guide), this covers
*algorithmic* wins that change the asymptotic cost of an agent loop.

All numbers below were measured on **qwen2.5-0.5b (Q4_K_M), CPU, 8 threads**,
with the Ollama-matched backend (`GGML_BACKEND_PATH=.../libggml-cpu-alderlake.so`)
so the token stream is bit-identical to Ollama. Absolute milliseconds are
hardware- and load-dependent; the **ratios and acceptance rates are the stable,
portable metrics**.

---

## TL;DR — where each feature helps

| Feature | What it speeds up | Headline result | Regime |
|---|---|---|---|
| **Cross-turn KV reuse** | Multi-turn agent prefill | **up to 28.5× less prefill** by turn 11 | Any repeated-history loop |
| **Sliding-window pruning** | Long sessions vs `n_ctx` overflow | bounds prefill+KV; prevents OOM crash | Sessions longer than `n_ctx` |
| **Durable KV store** | First turn after a daemon restart | restore keeps the delta-prefill win across restarts | Restart / cold start |
| **Idle hydration + scheduling** | More sessions than pool slots | moves restore cost off the request path | Many concurrent sessions |
| **Concurrent multi-session serving** | Parallel requests across pool slots | **1.48× throughput at 2 sessions** (plateaus — memory-bandwidth bound) | Multi-tenant, ≤2 hot sessions |
| **Prompt-lookup speculation** | Repetitive / structured output | **6.96 tokens/forward-pass, ~6.8× wall-clock** | Code, JSON, repeated text |
| **INT4 group quantization** | Weight memory | **6.4× smaller** than f32 | Always (weight storage) |
| **+ Hadamard rotation** | INT4 accuracy on outliers | error 4.53%→4.35% | Only ~1 outlier per group |

The first four are **the agent-loop story**: an agent re-sends a growing
conversation every turn, so naive prefill is `O(history)` *per turn* —
quadratic over a session. KV reuse makes per-turn prefill `O(delta)`, and the
durable store + scheduling keep that win across restarts and across more
sessions than you have context slots.

---

## 1. Cross-turn KV reuse — the headline

**The cost it removes.** Stock llama.cpp servers `kv_cache_clear()` and
re-decode the *entire* prompt on every request. An agent loop re-sends the full
history each turn, so turn N re-prefills all N turns' tokens — the prefill time
grows linearly with conversation length, and total session cost is quadratic.

**What mullama does.** A session is pinned to a context-pool slot; the KV cache
*is* the conversation. Each turn we match the new prompt's longest common prefix
against the cached tokens, drop only the divergent tail (`kv_cache_seq_rm`), and
decode only the new suffix. The reused path is numerically identical to a full
decode (same tokens at the same positions), so greedy output is unchanged.

### Benchmark: 12-turn `long-repo` trace

Prefill time per turn, **with reuse vs stateless baseline**, same daemon:

| turn | prompt tokens | reuse prefill | baseline prefill | **speedup** |
|-----:|-----:|-----:|-----:|-----:|
| 1 | 20 | 74.7 ms | 54.8 ms | 0.7× |
| 2 | 110 | 85.9 ms | 394.8 ms | 4.6× |
| 4 | 288 | 86.1 ms | 793.1 ms | 9.2× |
| 6 | 466 | 103.5 ms | 1273.6 ms | 12.3× |
| 8 | 640 | 126.4 ms | 1794.3 ms | 14.2× |
| 10 | 821 | 133.0 ms | 2186.2 ms | 16.4× |
| 11 | 905 | 106.4 ms | 3034.4 ms | **28.5×** |
| 12 | 994 | 147.3 ms | 3504.6 ms | 23.8× |

Reuse holds prefill **flat (~80–150 ms)** regardless of history length, while
the baseline climbs to 3.5 s. The speedup grows with the conversation — exactly
the `O(history) → O(delta)` collapse. Turn 1 is a wash (no prior KV to reuse);
the win compounds from turn 2 on.

> Note: the bench's printed "reuse ratio turn1/turnN" line compares within a
> single run and is **not** the reuse-vs-baseline speedup. Use the cross-run
> table above (computed from the two report JSONs).

**Reproduce:**
```bash
# starts daemon, runs agent-loop with reuse and with --no-kv-reuse
bash bench/run_agentloop.sh        # see "Reproduction" section
# or manually:
mullama-bench --mullama-url http://127.0.0.1:8080 --models qwen2.5-0.5b \
  --mode agent-loop --trace-file bench/trace.jsonl --agent-max-tokens 64 \
  --temperature 0 --report reuse.json
mullama-bench ... --no-kv-reuse --report noreuse.json
```

---

## 2. Sliding-window pruning — surviving long sessions

Reuse wants a stable prefix; a bounded context wants to drop it. For sessions
that exceed `n_ctx`, `session_keep_turns=N` trims everything older than the last
N user turns before rendering the prompt, bounding both prompt and pinned KV.

**Measured:** an unbounded 12-turn trace eventually overflows `n_ctx` and
llama.cpp returns *"decode: failed to find a memory slot for batch of size 1"*
(a 500 error). With `--keep-turns 4`, every turn completes, prompt bounded at
~265 tokens. This is a **crash-vs-completes** result, not a speedup — pruning is
what makes very long agent sessions possible at all.

---

## 3. Durable KV store — keeping the win across restarts

In-memory reuse dies with the process. The durable content-addressed store
(sled, `~/.mullama/kv-cas/`) persists each session's token sequence + the
per-sequence KV state blob, so a fresh daemon restores instead of re-prefilling
the whole history.

**Measured** (3-turn conversation, restart between warm-up and turn 3):

| path | turn-3 prefill |
|---|---|
| stateless (full decode) | 1334 ms |
| in-memory reuse | 556 ms |
| **durable restore (after restart)** | **600 ms** |

Restore (600 ms) ≈ in-memory (556 ms) ≪ stateless (1334 ms): the delta-prefill
win **survives the restart**, and restored output is token-identical to both.
Blobs are small (per-sequence, not full-context: ~0.4–1.8 MB for these turns).

---

## 4. Idle hydration + multi-session scheduling

With more active sessions than context-pool slots, naive round-robin thrashes —
each turn clobbers another session's KV. mullama pins sessions to slots by
**affinity** with **durable-safe LRU eviction**: an evicted session's KV is
persisted, so it just restores on its next request rather than re-prefilling.

A background **idle hydrator** pre-warms durable-but-not-live sessions into free
slots during idle windows (no active requests), moving the restore cost *off*
the request path — so the next turn is a hot in-memory hit. Verified: after a
restart the hydrator pre-warms a session (logged `idle-hydrated session into
slot`) and the subsequent request is served live, output identical to stateless.

This is a **latency-shaping** win (cost moved to idle time), not a throughput
number; correctness is guaranteed by a reserve/commit/abort protocol that only
marks a session live after its KV is genuinely populated.

### Concurrent multi-session throughput — what the pool actually buys

The context pool has N independently-lockable slots, so concurrent requests to
different sessions *can* decode in parallel. We measured whether they actually
do, by firing S sessions at the daemon at once (pool size 4, 4 threads/session,
24-core box) and comparing aggregate tokens/sec to a single session alone:

| concurrent sessions | aggregate tok/s | **throughput scaling** | median latency inflation |
|---:|---:|---:|---:|
| 1 | 53.7 | 1.00× | 1.0× |
| 2 | 78.1 | **1.48×** | 1.35× |
| 4 | 79.3 | 1.43× | 2.42× |

**Verdict: parallel serving is real but plateaus at ~1.5×.** Scaling rises from
1.0× to 1.48× at two sessions — proof the pool serves concurrently and is *not*
single-flight (a lock would pin it at 1.0×). But it saturates there: a 3rd/4th
session adds latency without adding throughput, because decoding a small model
is **memory-bandwidth bound** — each token streams the full weight set, and two
concurrent decodes already saturate the memory subsystem. This is a hardware
ceiling, not a software lock. (Confirmed: with pool size 1, scaling is 1.1× —
sessions correctly serialize on the single slot.)

**Implication for "parallel idle-fill".** The original idea of running one
session's prefill *during* another's decode would only help if there were spare
memory bandwidth to fill — and there isn't past ~2 concurrent decodes on this
class of machine. So overlapping work inside the active window is **not worth
building** here; the idle hydrator (which runs only when fully idle) already
captures the available headroom. On a GPU or a bandwidth-rich server the ceiling
would be higher and the calculus could change.

**Reproduce:**
```bash
# sweep concurrency at fixed pool size
python3 bench/concurrent_sessions.py --url http://127.0.0.1:8110 \
  --model qwen2.5-0.5b --sessions 2 --turns 3 --max-tokens 48
```

---

## 5. Prompt-lookup speculative decoding

**Greedy-exact** speculation with no draft model: propose the next K tokens by
matching the current suffix n-gram against the token history and copying what
followed, then verify all K in **one** batched target forward pass
(`Context::decode_batch_argmax`). Accept the longest prefix matching the
target's own greedy argmax. Every emitted token equals the target's greedy
choice, so output is **token-for-token identical** to plain greedy — speculation
is pure latency, never a quality trade-off.

### Benchmark: 160 new tokens, two regimes

| prompt regime | acceptance | tokens/forward-pass | wall-clock |
|---|---:|---:|---:|
| **Repetitive** (JSON-array continuation) | **72.6%** | **6.96** | **~6.8× faster** |
| Low-repetition prose | 37.8% | 1.25 | 0.45× (slower) |

Both regimes are **token-identical** to the greedy baseline (parity ✓). The win
scales with repetitiveness: on structured/code/JSON output (the agent workload
this targets) the n-gram drafter hits often and amortizes ~7 tokens per forward
pass. On free-form prose it rarely hits, and the per-round verification overhead
makes it slower — an honest break-even-to-loss. **Use it for structured output,
not creative writing.**

`tokens/forward-pass` and `acceptance` are the portable metrics; wall-clock
varies with machine load (we observed 2.66×–6.8× on the same repetitive prompt
under different load).

**Reproduce:**
```bash
GGML_BACKEND_PATH=.../libggml-cpu-alderlake.so \
  cargo run --release --example speculative_lookup -- <model.gguf> 160
# prose regime:
SPEC_PROMPT="Explain relativity..." cargo run --release --example speculative_lookup -- <model.gguf> 80
```

---

## 6. INT4 quantization + Hadamard rotation

A self-contained 4-bit group-quantization kernel (symmetric, 32-weight groups,
one f32 scale per group) with an optional QuaRot-style Hadamard rotation folded
into the weights. The matvec runs directly from 4-bit storage.

### Compression

| format | bytes/weight | ratio |
|---|---:|---:|
| f32 | 4.0 | 1× |
| **INT4 group** | **0.62** | **6.4×** |

(0.5 byte for the nibble + 0.125 byte for the amortized per-group f32 scale.)

### Rotation: an honest, narrow win

Rotation is **not** a universal improvement for weight-only group INT4. Group
quantization already *isolates* sparse outliers into a few bad groups; rotation
spreads them across every group, which only helps once (nearly) every group
already has an outlier. Relative matvec error vs exact f32, by outlier density
(64×256 matrix, 8 groups/row):

| outliers/group | plain INT4 | + rotation | winner |
|---:|---:|---:|---|
| 0 | 4.66% | 42.91% | plain |
| **1** | 4.53% | **4.35%** | **rotation** |
| 2 | 3.86% | 6.49% | plain |
| 4 | 2.68% | 9.05% | plain |
| 8 | 2.86% | 22.09% | plain |

Rotation wins **only** at ~1 outlier per group; on smooth weights it concentrates
energy and worsens error sharply. This matches the QuaRot literature: weight-only
rotation has a narrow regime, and the larger wins come from rotating *activations*
(not yet wired here). The kernel is verified by unit tests but is **not yet
integrated into the llama.cpp graph** (that needs a GGML custom op).

**Reproduce:**
```bash
cargo run --release --example int4_rotation_demo -- 64 256   # density sweep
```

---

## Reproduction

All benchmarks use `bench/trace.jsonl` (3 agent traces: `repo-qna` 5 turns,
`bug-hunt` 5 turns, `long-repo` 12 turns) and the qwen2.5-0.5b blob.

| Benchmark | Command |
|---|---|
| KV reuse vs baseline | `mullama-bench --mode agent-loop` with/without `--no-kv-reuse` |
| Pruning (overflow) | `mullama-bench --mode agent-loop --keep-turns 4` |
| Durable restore | restart daemon between turns, compare prefill |
| Concurrent throughput | `python3 bench/concurrent_sessions.py --sessions N` |
| Speculative decoding | `cargo run --release --example speculative_lookup` |
| INT4 + rotation | `cargo run --release --example int4_rotation_demo` |

For exact Ollama-matched numerics set
`GGML_BACKEND_PATH=/usr/local/lib/ollama/libggml-cpu-alderlake.so`.

### Caveats / honesty notes

- **Absolute ms are not portable.** Ratios, acceptance rates, and
  tokens/forward-pass are. The same speculative prompt measured 2.66× and 6.8×
  on the same box under different load.
- **Speculation can lose** on low-repetition text (0.45×). It's a structured-
  output optimization.
- **Rotation can lose** outside the ~1-outlier-per-group regime; plain INT4 is
  often better.
- **Concurrent serving plateaus at ~1.5×** (memory-bandwidth bound on CPU), so
  in-active-window prefill overlap ("parallel idle-fill") is not worth building
  on this hardware class — the idle hydrator already captures the headroom.
- **INT4 + rotation is a standalone kernel**, not yet in the inference graph.
- Greedy parity vs Ollama is exact only with the matched backend; long
  completions diverge mid-stream on a stock backend (a known build-numerics
  gap, independent of these features).
