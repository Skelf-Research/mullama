//! Continuous-batched decode loop.
//!
//! Owns the single [`Context`] exclusively; no `RwLock` because there is
//! exactly one owner — the spawned scheduler task. All concurrency comes
//! from interleaving `N` `seq_id`s in one `llama_decode` per tick.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::memory_monitor::{MemoryMonitor, MemoryPressure};
use crate::sys;
use crate::token::TokenId;
use crate::{Context, Model, MullamaError};

use super::slot::{Slot, SlotState};
use super::{BatchTask, ReplyMode};

/// Length of the longest shared prefix of two token slices.
fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Best-effort error reply on a fresh task that never got into a slot.
/// Buffered tasks get the error on their oneshot; streaming tasks see the
/// sender drop, which the consumer sees as an empty stream — matches the
/// streaming-contract "channel close = stream end".
fn reply_err(reply: ReplyMode, err: MullamaError) {
    match reply {
        ReplyMode::Buffered { tx } => {
            let _ = tx.send(Err(err));
        }
        ReplyMode::Streaming { tx, .. } => {
            drop(tx);
            let _ = err;
        }
    }
}

/// Maximum prompt tokens we'll prefill into the batch in one tick, per slot.
/// Caps the per-tick latency contribution from any one slot's prompt — keeps
/// other slots' decode latency bounded even if a giant prompt arrives.
const MAX_PREFILL_CHUNK_PER_SLOT: usize = 512;

/// Per-tick telemetry, emitted at `MULLAMA_BATCHER_DEBUG=1`. Counters are
/// reset each tick; histograms (e.g. queue depth) aggregate at the
/// `BatcherStats` level.
#[derive(Default, Debug)]
struct TickStats {
    /// Tokens in the batch from prefill chunks.
    prefill_tokens: usize,
    /// Tokens in the batch from generate (one per active Generating slot).
    decode_tokens: usize,
    /// Active slots this tick (Prefilling + Generating).
    active_slots: usize,
    /// Wall time for `llama_decode`.
    decode_ns: u64,
    /// Wall time for the per-slot sampler+token-to-string work after decode.
    sample_ns: u64,
    /// Wall time for the rest of the tick (batch alloc + per-slot setup +
    /// finalization).
    overhead_ns: u64,
}

/// Cross-tick aggregates so we can print summary numbers without an external
/// metrics system. Reset whenever an explicit `summary` log is emitted (so
/// the per-bench-run rollup is clean).
#[derive(Default)]
struct BatcherStats {
    enabled: bool,
    ticks: u64,
    decode_ns_total: u64,
    sample_ns_total: u64,
    overhead_ns_total: u64,
    prefill_tokens_total: u64,
    decode_tokens_total: u64,
    // Histogram of how many slots were active per tick (index = count).
    active_slots_hist: Vec<u64>,
}

impl BatcherStats {
    fn from_env(n_slots: usize) -> Self {
        Self {
            enabled: std::env::var("MULLAMA_BATCHER_DEBUG").ok().as_deref() == Some("1"),
            active_slots_hist: vec![0; n_slots + 1],
            ..Default::default()
        }
    }

    fn record(&mut self, t: &TickStats) {
        if !self.enabled {
            return;
        }
        self.ticks += 1;
        self.decode_ns_total += t.decode_ns;
        self.sample_ns_total += t.sample_ns;
        self.overhead_ns_total += t.overhead_ns;
        self.prefill_tokens_total += t.prefill_tokens as u64;
        self.decode_tokens_total += t.decode_tokens as u64;
        if let Some(slot_count_bucket) = self.active_slots_hist.get_mut(t.active_slots) {
            *slot_count_bucket += 1;
        }
        // Per-tick lines — cheap; one per llama_decode.
        tracing::info!(
            target: "batcher",
            active = t.active_slots,
            prefill = t.prefill_tokens,
            decode = t.decode_tokens,
            decode_ms = t.decode_ns as f64 / 1e6,
            sample_ms = t.sample_ns as f64 / 1e6,
            overhead_ms = t.overhead_ns as f64 / 1e6,
            "batcher tick"
        );
    }

    /// Emit a rollup line. Called on shutdown (sched.run exits) and on
    /// demand if a long-lived daemon wants periodic summaries (not wired
    /// yet — first user is shutdown).
    fn summary(&self) {
        if !self.enabled || self.ticks == 0 {
            return;
        }
        let total_ns = self.decode_ns_total + self.sample_ns_total + self.overhead_ns_total;
        let active_avg: f64 = self
            .active_slots_hist
            .iter()
            .enumerate()
            .map(|(k, c)| (k as f64) * (*c as f64))
            .sum::<f64>()
            / (self.ticks as f64);
        tracing::info!(
            target: "batcher",
            ticks = self.ticks,
            decode_ms_total = self.decode_ns_total as f64 / 1e6,
            sample_ms_total = self.sample_ns_total as f64 / 1e6,
            overhead_ms_total = self.overhead_ns_total as f64 / 1e6,
            decode_pct = (self.decode_ns_total as f64) / (total_ns as f64) * 100.0,
            avg_active_slots = active_avg,
            prefill_tokens = self.prefill_tokens_total,
            decode_tokens = self.decode_tokens_total,
            hist = ?self.active_slots_hist,
            "batcher summary"
        );
    }
}

pub struct BatchScheduler {
    model: Arc<Model>,
    context: Context,
    slots: Vec<Slot>,
    rx: mpsc::Receiver<BatchTask>,
    stats: BatcherStats,
    /// Pre-allocated `llama_batch` sized for the context's `n_batch`. Reused
    /// across ticks via `n_tokens = 0` reset instead of init/free per tick.
    /// At ~80 ticks/s steady state, removes ~160 allocator round-trips per
    /// second. Freed in `Drop`.
    batch_buf: sys::llama_batch,
    batch_capacity: usize,
    /// Shared shutdown flag. When set, the scheduler stops accepting new
    /// tasks from the channel and drains all active slots before exiting
    /// the run loop.
    shutdown: Arc<AtomicBool>,
    /// Optional system memory monitor. When present and pressure rises to
    /// Warning or above, idle slots are LRU-evicted to free memory.
    memory_monitor: Option<Arc<MemoryMonitor>>,
}

impl Drop for BatchScheduler {
    fn drop(&mut self) {
        unsafe {
            sys::llama_batch_free(self.batch_buf.clone());
        }
    }
}

// SAFETY: `BatchScheduler` is moved into a single `tokio::spawn` task and
// only ever touched from that one task. The raw pointers inside
// `batch_buf` (which `llama_batch` carries) point into the buffer that
// llama.cpp allocated for *us*; we are the sole writer. `Context` already
// carries its own `unsafe impl Send` for the same reason.
unsafe impl Send for BatchScheduler {}

impl BatchScheduler {
    pub fn new(
        model: Arc<Model>,
        context: Context,
        n_slots: u32,
        rx: mpsc::Receiver<BatchTask>,
        shutdown: Arc<AtomicBool>,
        memory_monitor: Option<Arc<MemoryMonitor>>,
    ) -> Self {
        let slots = (0..n_slots as i32).map(Slot::new).collect();
        let stats = BatcherStats::from_env(n_slots as usize);
        let batch_capacity = context.n_batch() as usize;
        let batch_buf = unsafe { sys::llama_batch_init(batch_capacity as i32, 0, 1) };
        Self {
            model,
            context,
            slots,
            rx,
            stats,
            batch_buf,
            batch_capacity,
            shutdown,
            memory_monitor,
        }
    }

    pub async fn run(&mut self) {
        loop {
            if self.assign_pending().await.is_none() {
                self.stats.summary();
                return;
            }
            if let Err(e) = self.tick() {
                tracing::error!(error = %e, "batch scheduler tick failed");
                for slot in &mut self.slots {
                    if !slot.is_idle() {
                        slot.finalize_err(MullamaError::GenerationError(format!(
                            "batched decode failed: {}",
                            e
                        )));
                    }
                }
            }
        }
    }

    async fn assign_pending(&mut self) -> Option<()> {
        let any_busy = self.slots.iter().any(|s| !s.is_idle());
        let any_idle = self.slots.iter().any(|s| s.is_idle());
        let shutting_down = self.shutdown.load(Ordering::Relaxed);

        if any_idle && !any_busy {
            if shutting_down {
                return None;
            }
            let Some(task) = self.rx.recv().await else { return None; };
            self.assign_to_first_idle(task);
        }
        // Drain pending tasks into a buffer, up to the number of idle
        // slots. Then sort: tasks whose session_id already matches a
        // cached slot (session-hot) come first. This prevents a burst of
        // cold-start requests from starving an agentic-loop's next turn
        // which already has warm KV cache waiting.
        let idle_count = self.slots.iter().filter(|s| s.is_idle()).count();
        let mut pending: Vec<BatchTask> = Vec::with_capacity(idle_count);
        while pending.len() < idle_count {
            match self.rx.try_recv() {
                Ok(task) => {
                    if shutting_down {
                        reply_err(
                            task.reply,
                            MullamaError::OperationFailed("daemon shutting down".into()),
                        );
                        continue;
                    }
                    pending.push(task);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if !self.slots.iter().any(|s| !s.is_idle()) {
                        return None;
                    }
                    break;
                }
            }
        }
        // Session-hot-first sort: tasks whose session_id matches a
        // currently-idle slot's cached session get priority.
        if !pending.is_empty() {
            pending.sort_by(|a, b| {
                let a_hot = a
                    .session_id
                    .as_deref()
                    .is_some_and(|sid| self.slots.iter().any(|s| s.is_idle() && s.cached.as_ref().is_some_and(|c| c.session_id.as_str() == sid)));
                let b_hot = b
                    .session_id
                    .as_deref()
                    .is_some_and(|sid| self.slots.iter().any(|s| s.is_idle() && s.cached.as_ref().is_some_and(|c| c.session_id.as_str() == sid)));
                b_hot.cmp(&a_hot)
            });
            for task in pending {
                self.assign_to_first_idle(task);
            }
        }

        // Memory-pressure-aware LRU eviction: when system/GPU memory is
        // under pressure (≥Warning), shed the idle slot that finished
        // longest ago. This frees its KV-cache allocation without
        // disturbing active slots. Under Emergency pressure, evict two.
        if let Some(ref monitor) = self.memory_monitor {
            let pressure = monitor.pressure();
            if pressure >= MemoryPressure::Warning {
                let evict_count = if pressure >= MemoryPressure::Critical { 2 } else { 1 };
                for _ in 0..evict_count {
                    let victim = self
                        .slots
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.is_idle() && s.cached.is_some())
                        .min_by_key(|(_, s)| s.last_finalized);
                    if let Some((idx, _)) = victim {
                        self.evict_slot(idx);
                    }
                }
            }
        }

        if shutting_down && self.slots.iter().all(|s| s.is_idle()) {
            return None;
        }
        Some(())
    }

    /// Clear a slot's KV cache and cached metadata so its memory can be
    /// reused by the OS or other allocations. The slot stays Idle — it
    /// just loses its cached session affinity.
    fn evict_slot(&mut self, idx: usize) {
        let slot = &mut self.slots[idx];
        let id_seq = slot.id_seq;
        self.context.kv_cache_seq_rm(id_seq, 0, -1);
        slot.cached = None;
        tracing::warn!(
            slot = idx,
            "LRU-evicted slot due to memory pressure"
        );
    }

    fn assign_to_first_idle(&mut self, task: BatchTask) -> bool {
        // Pre-flight overflow guard: under `kv_unified=true` (the mullama
        // default), all seq_ids share a single n_ctx-cell KV pool. A
        // request whose prompt + max_tokens couldn't possibly fit on its
        // own should fail fast with a clear error rather than crash
        // `llama_decode` deep in the tick (which produces the cryptic
        // "decode: failed to find a memory slot for batch of size N").
        let n_ctx = self.context.n_ctx() as usize;
        let prompt_len = task.prompt_tokens.len();
        if prompt_len.saturating_add(task.max_tokens as usize) > n_ctx {
            reply_err(
                task.reply,
                MullamaError::OperationFailed(format!(
                    "request exceeds context window: prompt {} + max_tokens {} > n_ctx {}; \
                     set `session_keep_turns` to enable sliding-window pruning or raise `-c <N>`",
                    prompt_len, task.max_tokens, n_ctx
                )),
            );
            return true; // task handled (errored); keep scheduler running
        }
        // Slot selection: prefer a slot whose cached session_id matches
        // this task's (cross-turn KV reuse); fall back to any idle slot.
        let chosen_idx = if let Some(sid) = task.session_id.as_deref() {
            self.slots
                .iter()
                .position(|s| {
                    s.is_idle()
                        && s.cached.as_ref().map(|c| c.session_id.as_str()) == Some(sid)
                })
                .or_else(|| self.slots.iter().position(|s| s.is_idle()))
        } else {
            self.slots.iter().position(|s| s.is_idle())
        };
        let slot_idx = match chosen_idx {
            Some(i) => i,
            None => {
                reply_err(
                    task.reply,
                    MullamaError::OperationFailed("no idle slot — try again".into()),
                );
                return false;
            }
        };
        let id_seq = self.slots[slot_idx].id_seq;

        // Durable session restore: if the task carried a saved KV blob
        // (cold session start after a daemon restart), load it into this
        // slot's seq_id BEFORE prefix-matching. The restore's `tokens` are
        // what the blob corresponds to — slot.cached becomes that, so the
        // subsequent common_prefix_len matches against the right history.
        // A failed load (incompatible build/state-version) falls through
        // to a fresh prefill; never incorrect.
        if let Some(restore) = task.restore.as_ref() {
            // Drop whatever was in this seq_id, then load the blob.
            self.context.kv_cache_seq_rm(id_seq, 0, -1);
            let ok = self
                .context
                .load_state_seq(id_seq, &restore.blob)
                .is_ok();
            if ok {
                // Mark the slot as holding these tokens so the prefix-reuse
                // path below sees the restored prefix and only prefills the
                // new delta.
                if let Some(sid) = task.session_id.as_ref() {
                    self.slots[slot_idx].cached =
                        Some(super::slot::SlotCache {
                            session_id: sid.clone(),
                            tokens: restore.tokens.clone(),
                        });
                }
            } else {
                // Wipe again so we re-prefill cleanly; clear the cache so
                // the common_prefix_len below returns 0.
                self.context.kv_cache_seq_rm(id_seq, 0, -1);
                self.slots[slot_idx].cached = None;
            }
        }

        // Compute the common prefix length between this task's prompt and
        // the slot's cached tokens (only if same session). That gives us
        // the prefill_start_pos — we drop the divergent KV tail and
        // prefill only the new suffix. Matches the legacy `kv_reuse` path
        // semantics but batched.
        let prefill_start_pos: i32 = match (&task.session_id, &self.slots[slot_idx].cached) {
            (Some(sid), Some(cache)) if cache.session_id == *sid => {
                let prompt_len = task.prompt_tokens.len();
                let mut l = common_prefix_len(&cache.tokens, &task.prompt_tokens);
                // Edge case: the new prompt is a *prefix or full match* of
                // the cached tokens (l == prompt_len). We need at least one
                // token to prefill so the model produces fresh logits; back
                // off by one so the last prompt token is re-prefilled.
                if l == prompt_len && prompt_len > 0 {
                    l = prompt_len - 1;
                }
                // Drop the divergent tail from THIS seq_id's KV
                // (positions >= l). Other seq_ids untouched. Note the
                // semantics of `llama_memory_seq_rm(seq, p0, p1)` — it
                // removes cells in the half-open range `[p0, p1)`, with
                // `p1 = -1` meaning "to the end".
                self.context.kv_cache_seq_rm(id_seq, l as i32, -1);
                l as i32
            }
            _ => {
                // Fresh slot OR different session: clear and start over.
                self.context.kv_cache_seq_rm(id_seq, 0, -1);
                0
            }
        };

        let sampler = match task.sampler_params.build_chain_with_grammar(
            self.model.clone(),
            task.grammar_gbnf.as_deref(),
        ) {
            Ok(s) => s,
            Err(e) => {
                reply_err(task.reply, e);
                return true;
            }
        };
        let max_stop_len = task
            .stop_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        let slot = &mut self.slots[slot_idx];
        if let Err(e) = slot.assign(task, sampler, max_stop_len, prefill_start_pos) {
            slot.finalize_err(e);
        }
        true
    }

    fn tick(&mut self) -> Result<(), MullamaError> {
        let tick_start = Instant::now();
        let mut tstats = TickStats::default();
        // Cancellation pass: any slot whose cancel flag fired finalizes
        // early so the caller doesn't wait for max_tokens (Generating)
        // or the rest of a long prompt (Prefilling). We do this BEFORE
        // building the batch so cancelled slots don't consume a token
        // slot. Previously only Generating slots were checked — a
        // multi-chunk prefill could remain un-cancellable for many ticks.
        let mut cancel_finalize: Vec<usize> = Vec::new();
        for (si, slot) in self.slots.iter().enumerate() {
            let cancelled = slot
                .task
                .as_ref()
                .and_then(|t| t.cancel.as_ref())
                .map(|f| f.load(Ordering::Relaxed))
                .unwrap_or(false);
            if !cancelled {
                continue;
            }
            match slot.state {
                SlotState::Generating { .. } | SlotState::Prefilling { .. } => {
                    cancel_finalize.push(si);
                }
                SlotState::Idle => {}
            }
        }
        for si in cancel_finalize {
            self.slots[si].finalize_ok();
        }

        // Total batch capacity = the context's `n_batch`. We must keep
        // sum(per-slot tokens) ≤ n_batch or llama_decode will assert.
        // Generating slots always contribute exactly 1; the prefill chunks
        // get whatever room is left, divided up so no single slot starves
        // the others.
        let n_batch_cap = self.context.n_batch() as usize;
        let n_generating = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Generating { .. }))
            .count();
        let n_prefilling = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Prefilling { .. }))
            .count();
        tstats.active_slots = n_generating + n_prefilling;
        if n_generating + n_prefilling == 0 {
            return Ok(());
        }
        // Reserve one slot of room per Generating slot; split the rest evenly
        // across Prefilling slots, capped per slot.
        let prefill_room = n_batch_cap.saturating_sub(n_generating);
        let per_slot_chunk_cap = if n_prefilling == 0 {
            0
        } else {
            (prefill_room / n_prefilling).min(MAX_PREFILL_CHUNK_PER_SLOT)
        };

        // Now compute the real total.
        let mut total_tokens = 0usize;
        for slot in &self.slots {
            match &slot.state {
                SlotState::Prefilling { remaining, .. } => {
                    total_tokens += remaining.len().min(per_slot_chunk_cap);
                }
                SlotState::Generating { .. } => total_tokens += 1,
                SlotState::Idle => {}
            }
        }
        if total_tokens == 0 {
            return Ok(());
        }
        if total_tokens > self.batch_capacity {
            // Shouldn't happen — the per-slot cap above keeps the sum ≤
            // n_batch — but defense in depth. Fail loudly so a future
            // change to the slot-chunk math surfaces here, not as a crash
            // 200 lines deep in Metal.
            return Err(MullamaError::GenerationError(format!(
                "scheduler bug: tick wants {} tokens but batch capacity is {}",
                total_tokens, self.batch_capacity
            )));
        }

        // Reuse the pre-allocated batch buffer (M4): reset its token count
        // and re-fill its slots in place. The buffer is sized to n_batch at
        // scheduler init and freed in Drop, so per-tick alloc churn is zero.
        let mut batch = self.batch_buf.clone();
        unsafe {
            batch.n_tokens = 0;
        }
        // For each slot index, the batch position whose logits we'll sample
        // after this decode. None = the slot isn't producing a sampled token
        // this tick (e.g. it's still mid-prefill in a multi-chunk prompt).
        let mut logits_pos: Vec<Option<usize>> = vec![None; self.slots.len()];
        // Slot indices that finished prefill in this tick — they need their
        // state flipped from Prefilling → Generating after the decode.
        let mut started_decodes: Vec<(usize, i32)> = Vec::new(); // (slot_idx, final_pos)
        let mut batch_idx: usize = 0;

        for (si, slot) in self.slots.iter_mut().enumerate() {
            match &mut slot.state {
                SlotState::Prefilling { remaining, pos } => {
                    let chunk = remaining.len().min(per_slot_chunk_cap);
                    if chunk == 0 {
                        continue;
                    }
                    let last_prompt_token = chunk == remaining.len();
                    unsafe {
                        for k in 0..chunk {
                            *batch.token.add(batch_idx) = remaining[k] as sys::llama_token;
                            *batch.pos.add(batch_idx) = *pos + k as i32;
                            *batch.n_seq_id.add(batch_idx) = 1;
                            *(*batch.seq_id.add(batch_idx)).add(0) = slot.id_seq;
                            // Only the last prompt token needs logits.
                            *batch.logits.add(batch_idx) =
                                if k == chunk - 1 && last_prompt_token { 1 } else { 0 };
                            batch_idx += 1;
                        }
                    }
                    if last_prompt_token {
                        let final_pos = *pos + chunk as i32 - 1;
                        logits_pos[si] = Some(batch_idx - 1);
                        started_decodes.push((si, final_pos));
                    } else {
                        let drained: Vec<TokenId> = remaining.drain(chunk..).collect();
                        *pos += chunk as i32;
                        *remaining = drained;
                    }
                }
                SlotState::Generating {
                    n_past, next_token, ..
                } => {
                    unsafe {
                        *batch.token.add(batch_idx) = *next_token as sys::llama_token;
                        *batch.pos.add(batch_idx) = *n_past;
                        *batch.n_seq_id.add(batch_idx) = 1;
                        *(*batch.seq_id.add(batch_idx)).add(0) = slot.id_seq;
                        *batch.logits.add(batch_idx) = 1;
                    }
                    // Increment AFTER writing: `n_past` is "the position we'll
                    // write the next token at". The sample we'll get from this
                    // tick's logits is the prediction for the *next* position,
                    // so it's correct to bump now and not when sampling.
                    *n_past += 1;
                    logits_pos[si] = Some(batch_idx);
                    batch_idx += 1;
                }
                SlotState::Idle => {}
            }
        }
        unsafe {
            batch.n_tokens = batch_idx as i32;
        }

        // Account: batch composition + setup overhead so far.
        let setup_done = Instant::now();
        let overhead_pre = setup_done.duration_since(tick_start).as_nanos() as u64;
        for (i, slot) in self.slots.iter().enumerate() {
            if logits_pos[i].is_some() {
                match &slot.state {
                    SlotState::Prefilling { .. } => {
                        // Counted via started_decodes below — Prefilling slot
                        // that's flipping to Generating contributes its whole
                        // chunk to prefill_tokens.
                    }
                    SlotState::Generating { .. } => tstats.decode_tokens += 1,
                    _ => {}
                }
            }
        }
        // Prefill tokens = batch tokens that aren't generate-step tokens.
        tstats.prefill_tokens = batch_idx.saturating_sub(tstats.decode_tokens);

        // ONE GPU dispatch for all slots.
        let decode_start = Instant::now();
        let rc = unsafe { sys::llama_decode(self.context.ctx_ptr, batch.clone()) };
        tstats.decode_ns = decode_start.elapsed().as_nanos() as u64;
        if rc != 0 {
            // No per-tick batch_free here: the buffer is pre-allocated and
            // freed in `BatchScheduler::Drop`.
            return Err(MullamaError::GenerationError(format!(
                "llama_decode failed (rc = {})",
                rc
            )));
        }
        let sample_start = Instant::now();

        // Flip just-finished-prefill slots into Generating before sampling.
        for &(si, final_pos) in &started_decodes {
            let max_tokens = self.slots[si]
                .task
                .as_ref()
                .map(|t| t.max_tokens)
                .unwrap_or(0);
            if let Some(task) = self.slots[si].task.as_mut() {
                task.prefill_ns = Some(task.prefill_start.elapsed().as_nanos() as u64);
                task.decode_start = Some(Instant::now());
            }
            self.slots[si].state = SlotState::Generating {
                n_past: final_pos + 1,
                next_token: 0,
                generated: Vec::with_capacity(max_tokens as usize),
                text_so_far: String::new(),
            };
        }

        // Sample + accept per slot. Borrows are split: `context` (one field
        // of self) and `slot.task.sampler` (different field) can coexist.
        let mut to_finalize: Vec<usize> = Vec::new();
        for si in 0..self.slots.len() {
            let Some(batch_pos) = logits_pos[si] else { continue };
            // Split borrow: context and slots[si] are sibling fields of Self.
            let context = &mut self.context;
            let slot = &mut self.slots[si];
            let Some(task) = slot.task.as_mut() else { continue };
            let token = task.sampler.sample(context, batch_pos as i32);
            task.sampler.accept(token);

            let max_tokens = task.max_tokens;
            let max_stop_len = task.max_stop_len;
            // Avoid per-tick `Vec<String>::clone` (heap + string clone × N).
            // We need to borrow `task.stop_sequences` while also mutating
            // `slot.state` — they're sibling fields of `slot`, so Rust
            // allows independent borrows when accessed through field
            // projections.
            let stop_seqs: &Vec<String> = &task.stop_sequences;
            // Re-borrow so the long-lived `task` doesn't keep `slot` aliased.
            let state = &mut slot.state;
            let mut finished = false;

            // Match the legacy `generate_tokens` (common.rs) semantics:
            // - EOG terminates immediately, with the token *counted* but its
            //   text *not* appended (special=false on token_to_str hides it
            //   anyway). This matches ollama's `eval_count` convention.
            // - special=false on token_to_str so special tokens (chat-template
            //   markers, EOG) don't leak into the response text.
            if let SlotState::Generating { generated, .. } = state {
                generated.push(token);
                if generated.len() as u32 >= max_tokens {
                    finished = true;
                }
            }
            if self.model.token_is_eog(token) {
                finished = true;
            }
            if !finished {
                let new_piece = self
                    .model
                    .token_to_str(token, 0, false)
                    .unwrap_or_default();
                if let SlotState::Generating {
                    next_token,
                    text_so_far,
                    ..
                } = state
                {
                    *next_token = token;
                    text_so_far.push_str(&new_piece);
                    if max_stop_len > 0 && stop_seqs.iter().any(|s| text_so_far.contains(s)) {
                        finished = true;
                    }
                }
                // Stream the per-token chunk if this slot is in streaming
                // mode; cheap no-op for buffered. Done *after* state mutation
                // so a slow consumer can't stall the scheduler.
                self.slots[si].emit_token_chunk(new_piece, token);
            }
            if finished {
                to_finalize.push(si);
            }
        }
        for si in to_finalize {
            self.slots[si].finalize_ok();
        }
        // No per-tick `llama_batch_free` here: `self.batch_buf` is reused
        // across ticks and freed once in `BatchScheduler::Drop`. The `batch`
        // local is a `Clone` of `batch_buf` — same underlying allocations,
        // not an independent buffer.

        tstats.sample_ns = sample_start.elapsed().as_nanos() as u64;
        let tick_total_ns = tick_start.elapsed().as_nanos() as u64;
        tstats.overhead_ns = tick_total_ns
            .saturating_sub(tstats.decode_ns)
            .saturating_sub(tstats.sample_ns);
        // overhead_pre is the pre-decode setup, already included above via
        // `tick_total - decode - sample`. Kept as a local debug aid.
        let _ = overhead_pre;
        self.stats.record(&tstats);
        Ok(())
    }
}
