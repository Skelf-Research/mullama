# Phase C — Single-context multi-seq batched decode

## Why

The Phase A.2 + B.2 result on M1 is excellent at per-request perf (mullama
beats ollama on single-session: 0.81s vs 1.07s wall), but the concurrency
scaling is 1.0× because we serialize requests at a single context's write
lock. ollama scales to 1.34× by interleaving N sessions' tokens into one
`llama_decode` call. Phase C delivers that.

## What llama.cpp's server does (the reference)

`llama.cpp/tools/server/server.cpp` runs **one context** with
`cparams.n_seq_max = N`. It allocates a `slot_t` per concurrent client:

```cpp
struct slot_t {
    int      id_seq;            // unique seq_id within the context
    state_t  state;             // IDLE / PROCESSING / WAITING_TOKEN / ...
    task_t * task;              // current task (prompt + sampling params)
    int      n_prompt_processed;
    int      n_predict;
    sampler  smpl;
};
```

The main loop on each tick:

1. Walk `slots` and find any that need prompt prefill or are ready for next
   token sampling.
2. Build a single `llama_batch` containing tokens from multiple slots:

   ```cpp
   for (slot : slots) {
       if (slot.state == PROCESSING) {
           // prefill remaining prompt tokens
           common_batch_add(batch, slot.prompt_tokens[i], pos, {slot.id_seq}, true);
       } else if (slot.state == GENERATING) {
           // append the previously-sampled token for one decode step
           common_batch_add(batch, slot.last_sampled, slot.n_past++, {slot.id_seq}, true);
       }
   }
   ```

3. `llama_decode(ctx, batch)` — **one** GPU dispatch interleaving N slots'
   work.
4. After decode, for each slot whose token is at the end of the batch, call
   the sampler and emit the token to its client.

The KV cache stores all `N` slots' KV simultaneously; `seq_id` indexes which
cells belong to which slot.

## Why this lets Metal scale

A single `llama_decode` is one GPU dispatch (one Metal command buffer
submission). The GPU pipelines tokens for all slots inside the kernel
launches. No queue contention. Effective scaling proportional to slot count,
bounded by KV bandwidth, not command-queue serialization.

## Mapping to mullama's architecture

### What to keep
- `LoadedModel` + `Model` + `Context` types are fine
- the OpenAI/Ollama HTTP handlers are fine
- sessions / `KvReuse` model is fine (sessions get a stable `seq_id` instead
  of a slot index)

### What to replace
- `ContextPool` (N contexts × `n_seq_max=1`) → `SeqPool` (1 context ×
  `n_seq_max=N`)
- `generate_text` (synchronous, holds context lock for full decode) →
  enqueue task on the scheduler, await response on a channel
- `acquire_context_at(slot)` → `seq_pool.alloc_seq() -> SeqHandle`

### New module: `src/daemon/server/batcher/`

```
batcher/
  mod.rs           # public API: BatchScheduler, BatchTask
  scheduler.rs     # main loop: pick slots, build batch, decode, dispatch sampling
  slot.rs          # Slot state machine
  task.rs          # BatchTask: prompt + sampler + response channel
```

### Slot state machine

```rust
enum SlotState {
    Idle,                              // no task; available for assignment
    Prefilling { remaining: Vec<TokenId>, pos: i32 },
    Generating { n_past: i32, n_predict: u32, last_token: TokenId },
    Finalising,                        // EOG or max_tokens hit; flushing response
}

struct Slot {
    id_seq: i32,
    state: SlotState,
    task: Option<BatchTask>,
    sampler: SamplerChain,
}
```

### Per-request flow (replaces `Daemon::generate_text`)

```rust
async fn generate_text(&self, ...) -> Result<...> {
    let (tx, mut rx) = mpsc::channel(16);
    let task = BatchTask {
        prompt_tokens,
        max_tokens, sampler_params, stop_sequences, grammar_gbnf,
        kv_reuse,                       // optional pinned-seq id
        response_tx: tx,
    };
    self.batcher.submit(task).await?;   // queues in scheduler's input
    let mut text = String::new();
    let mut completion_tokens = 0;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            BatchOut::Token(tok_str) => { text.push_str(&tok_str); completion_tokens += 1; }
            BatchOut::Done { timings } => return Ok((text, prompt_tokens, completion_tokens, timings, ...)),
            BatchOut::Err(e) => return Err(e),
        }
    }
}
```

### Scheduler main loop

Runs in a dedicated tokio task per `LoadedModel`. Owns the single `Context`
exclusively (no RwLock — there's only one owner). Drains the input queue,
maintains the slot table, and runs decode steps:

```rust
loop {
    // Try to assign any idle slot to a new task
    while let Some(task) = self.try_recv_task() {
        let slot = self.find_idle_slot()?;
        let id_seq = slot.id_seq;
        if let Some(reuse) = &task.kv_reuse {
            // keep [0, l) of the cached prefix in the slot's seq_id
            self.ctx.kv_cache_seq_rm(id_seq, l as i32, -1);
        } else {
            self.ctx.kv_cache_seq_rm(id_seq, 0, -1);    // clear this seq only
        }
        slot.state = Prefilling { remaining: task.prompt_tokens[l..].to_vec(), pos: l as i32 };
        slot.task = Some(task);
        slot.sampler = build_sampler(...);
    }

    // Build one batch with one token per slot (or many for prefill catch-up)
    let mut batch = LlamaBatch::new();
    for slot in &mut self.slots {
        match &mut slot.state {
            Prefilling { remaining, pos } => {
                let chunk_size = (n_batch - batch.n_tokens()).min(remaining.len()).min(MAX_PREFILL_CHUNK);
                for k in 0..chunk_size {
                    let last = k == chunk_size - 1 && chunk_size == remaining.len();
                    batch.add(remaining[k], *pos + k as i32, slot.id_seq, /*logits=*/ last);
                }
                if chunk_size == remaining.len() {
                    let last_pos = *pos + chunk_size as i32 - 1;
                    slot.state = Generating { n_past: last_pos + 1, n_predict: slot.task.unwrap().max_tokens, last_token: 0 };
                } else {
                    *pos += chunk_size as i32;
                    remaining.drain(..chunk_size);
                }
            }
            Generating { n_past, last_token, .. } => {
                batch.add(*last_token, *n_past, slot.id_seq, /*logits=*/ true);
                *n_past += 1;
            }
            _ => {}
        }
    }
    if batch.is_empty() { tokio::task::yield_now().await; continue; }

    self.ctx.decode(&batch)?;            // ONE GPU dispatch for all slots

    // Sample per slot from its logits row, dispatch token to its client
    for slot in &mut self.slots {
        if let Generating { last_token, n_predict, .. } = &mut slot.state {
            let logits_idx = batch.logits_idx_for(slot.id_seq);
            let tok = slot.sampler.sample_and_accept(logits_idx);
            *last_token = tok;
            let s = self.model.token_to_piece(tok)?;
            slot.task.as_ref().unwrap().response_tx.send(BatchOut::Token(s)).await?;
            if is_eog(tok) || *n_predict == 0 { slot.finalise(); }
        }
    }
}
```

## Implementation order (estimated effort)

| step | effort | deliverable |
|---|---|---|
| 1. Wire `n_seq_max` cfg → CLI → manager | 1 hr | done (this work) |
| 2. `Context::decode_batch_seq(tokens_by_seq)` API | 4 hr | parameterise existing `decode` |
| 3. `LlamaBatch` builder Rust wrapper | 4 hr | wrap `llama_batch_init/add/free` |
| 4. `Slot` state machine + free-list of `seq_id`s | 4 hr | unit-testable |
| 5. `BatchScheduler` main loop (single-model) | 1.5 day | the heart |
| 6. Wire `generate_text` → submit task → await channel | 1 day | replaces lock-based flow |
| 7. Streaming endpoints (token-by-token) wiring | 1 day | per-slot `response_tx` |
| 8. Multi-model support (scheduler per LoadedModel) | 0.5 day | done if each model owns a scheduler |
| 9. Bench + tune (MAX_PREFILL_CHUNK, slot count) | 1 day | confirm 1.34× scaling |

Total: ~6–8 working days for production-quality.

## Risks

- **Cross-slot sampling determinism**: greedy parity may shift if two slots
  produce logits in different batch positions across runs. Mitigation: tie
  the sampler order to `seq_id`, not batch order.
- **Streaming back-pressure**: slow clients can stall a slot, blocking the
  batcher. Mitigation: per-slot bounded mpsc channel; if full, finalise the
  slot with a timeout error.
- **KV memory growth**: `n_seq_max=8` means 8 sessions' KV in the cache.
  Sliding-window pruning (already in mullama) must be aware of per-seq
  positions.
- **Hydrator**: the durable session restore path becomes simpler — there's no
  context pool to thrash, just `llama_state_seq_set_data(seq_id, blob)` into
  the chosen slot. The current try_acquire fix becomes a no-op (good).

## Tracer-bullet milestone

A *minimal viable* Phase C that proves scaling, before any of the streaming
work: one model, no streaming, fixed slot count `= num_cpus`, prompt-only
prefill (no `kv_reuse`). Single bench: `bench/concurrent_sessions.py
--no-session --sessions 4`. If scaling hits ≥ 1.2× and wall time matches
ollama's ~9.5s, the design is validated and the rest is plumbing.

## Why this is a Metal-grade fix, not just a generic concurrency one

Mullama's per-request perf (Phase A.2: 1 370 tok/s decode) already beats
ollama on M1. The reason concurrency tanks on Metal specifically is the
single command queue (CUDA has independent streams). One unified
`llama_decode` per tick is the only way to use Metal at its true parallelism
ceiling — and now it does so while *keeping* mullama's superior per-token
decode rate, since the batched decode kernels are the same hot path.

Once Phase C lands, expected M1 numbers:
- 4 concurrent sessions: ~8 s wall (beats ollama's 9.5 s)
- aggregate throughput: ~110 tok/s (beats ollama's 81)
- scaling vs 1-session: ~1.5× (beats ollama's 1.34×)
