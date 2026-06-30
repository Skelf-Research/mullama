//! Per-`seq_id` slot state for the [`super::BatchScheduler`].
//!
//! A `Slot` owns one of the context's sequence ids. The scheduler walks the
//! slot table on every tick, builds a [`llama_batch`] that asks the model for
//! the next-token logits for each `Generating` slot (and prompt-prefill
//! tokens for each `Prefilling` slot), and dispatches the results back.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::daemon::protocol::StreamChunk;
use crate::sampling::SamplerChain;
use crate::token::TokenId;
use crate::MullamaError;

use super::{BatchOutcome, BatchTask, ReplyMode};

pub(super) enum SlotState {
    /// No task; available for assignment.
    Idle,
    /// Prompt is being streamed into the KV cache in `n_batch`-sized chunks.
    /// `remaining` holds the unsubmitted tail; `pos` is the next `llama_pos`
    /// to write.
    Prefilling { remaining: Vec<TokenId>, pos: i32 },
    /// All prompt tokens are in the KV; the slot now generates one token per
    /// scheduler tick. `next_token` is what to submit on the next tick (the
    /// last sampled one, or the final prompt token for the very first
    /// generate step). `n_past` is the position to write it at.
    Generating {
        n_past: i32,
        next_token: TokenId,
        generated: Vec<TokenId>,
        text_so_far: String,
    },
}

pub(super) struct Slot {
    pub(super) id_seq: i32,
    pub(super) state: SlotState,
    pub(super) task: Option<TaskCtx>,
    /// What this slot's seq_id KV currently contains, post-finalize. Used
    /// to detect cross-turn cache hits: if a new task arrives with the
    /// same `session_id`, the scheduler can keep the common prefix in the
    /// KV and prefill only the divergent tail. `None` = slot is fresh /
    /// just-cleared / belonged to no session.
    pub(super) cached: Option<SlotCache>,
    /// Wall-clock timestamp of the last `finalize` call. Used for
    /// memory-pressure-aware LRU eviction: the slot that finished longest
    /// ago is the best candidate to evict when memory is tight.
    pub(super) last_finalized: Instant,
}

pub(super) struct SlotCache {
    pub(super) session_id: String,
    pub(super) tokens: Vec<TokenId>,
}

/// Bundles the per-task plumbing — sampler, stop sequences, max tokens, and
/// the reply channel — so the slot can finalize without reaching back into
/// the task queue.
pub(super) enum ReplySink {
    /// Holds the oneshot until finalize; sends [`BatchOutcome`] then.
    Buffered(Option<oneshot::Sender<Result<BatchOutcome, MullamaError>>>),
    /// Holds the mpsc + request_id/index so per-token chunks can be sent
    /// from the scheduler's per-tick path.
    Streaming {
        tx: mpsc::Sender<StreamChunk>,
        request_id: Arc<str>,
        index: u32,
    },
}

pub(super) struct TaskCtx {
    pub(super) reply: ReplySink,
    pub(super) sampler: SamplerChain,
    pub(super) max_tokens: u32,
    pub(super) prompt_tokens: u32,
    /// Full prompt tokens (for stashing as the slot's cache on finalize,
    /// so the next turn of this session can reuse the prefix). Kept here
    /// rather than rebuilt from `Prefilling::remaining` because the latter
    /// shrinks as we consume.
    pub(super) prompt_tokens_vec: Vec<TokenId>,
    pub(super) session_id: Option<String>,
    pub(super) stop_sequences: Vec<String>,
    pub(super) max_stop_len: usize,
    pub(super) prefill_start: Instant,
    pub(super) prefill_ns: Option<u64>,
    pub(super) decode_start: Option<Instant>,
    /// Cooperative cancellation. The scheduler polls this each tick on
    /// every `Generating` slot; if set, the slot finalizes early. `None`
    /// for tasks the caller doesn't track (no cancel needed).
    pub(super) cancel: Option<Arc<AtomicBool>>,
}

impl Slot {
    pub(super) fn new(id_seq: i32) -> Self {
        Self {
            id_seq,
            state: SlotState::Idle,
            task: None,
            cached: None,
            last_finalized: Instant::now(),
        }
    }

    pub(super) fn is_idle(&self) -> bool {
        matches!(self.state, SlotState::Idle)
    }

    /// Assign a freshly-pulled task to this slot. Sets `Prefilling`.
    pub(super) fn assign(
        &mut self,
        task: BatchTask,
        sampler: SamplerChain,
        max_stop_len: usize,
        prefill_start_pos: i32,
    ) -> Result<(), MullamaError> {
        let prompt_len = task.prompt_tokens.len();
        let reply = match task.reply {
            ReplyMode::Buffered { tx } => ReplySink::Buffered(Some(tx)),
            ReplyMode::Streaming {
                tx,
                request_id,
                index,
            } => ReplySink::Streaming {
                tx,
                request_id,
                index,
            },
        };
        if prompt_len == 0 {
            // Refuse zero-prompt; safer to fail fast than feed an empty batch.
            if let ReplySink::Buffered(Some(tx)) = reply {
                let _ = tx.send(Err(MullamaError::GenerationError(
                    "prompt is empty".into(),
                )));
            }
            return Ok(());
        }
        // Scheduler guarantees `prefill_start_pos < prompt_len` so we
        // always have at least one token to prefill — that token's logits
        // become the seed for sampling. The clamp is defense-in-depth.
        let prefill_start_pos = prefill_start_pos
            .max(0)
            .min(prompt_len.saturating_sub(1) as i32);
        let remaining: Vec<TokenId> = task.prompt_tokens[prefill_start_pos as usize..].to_vec();
        let prompt_tokens_vec = task.prompt_tokens;
        self.task = Some(TaskCtx {
            reply,
            sampler,
            max_tokens: task.max_tokens,
            prompt_tokens: prompt_len as u32,
            prompt_tokens_vec,
            session_id: task.session_id,
            stop_sequences: task.stop_sequences,
            max_stop_len,
            prefill_start: Instant::now(),
            prefill_ns: None,
            decode_start: None,
            cancel: task.cancel,
        });
        self.state = SlotState::Prefilling {
            remaining,
            pos: prefill_start_pos,
        };
        Ok(())
    }

    /// Emit one sampled-token chunk on the streaming channel. No-op for
    /// buffered tasks. Called from the scheduler per-tick after sampling.
    pub(super) fn emit_token_chunk(&self, delta: String, token_id: TokenId) {
        let Some(task) = self.task.as_ref() else { return };
        let ReplySink::Streaming {
            tx,
            request_id,
            index,
        } = &task.reply
        else {
            return;
        };
        // If the SSE consumer disconnected (HTTP client gone), the channel
        // is closed. Set the cancel flag so the scheduler finalizes this
        // slot on the next tick instead of running the full generation
        // into a black hole. This is mullama's HTTP-cancel implementation
        // for the batched path: no separate cancel endpoint needed.
        if tx.is_closed() {
            if let Some(flag) = task.cancel.as_ref() {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        }
        if delta.is_empty() {
            return;
        }
        let chunk = StreamChunk {
            request_id: request_id.clone(),
            index: *index,
            delta,
            token_id,
            thinking: None,
            tool_calls: None,
        };
        // Try non-blocking first — common case is the consumer is keeping
        // up and try_send fits in one cheap atomic.
        match tx.try_send(chunk) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                // The consumer (SSE encoder) is lagging. Previously we
                // dropped the chunk silently — that loses tokens and
                // corrupts the user-visible response without warning.
                // Now we cancel the slot instead: the slow consumer can't
                // keep up with the scheduler, so terminating here is
                // better than silently truncating the assistant message.
                // The client sees an early-EOF SSE stream — observable.
                tracing::warn!(
                    request_id = %request_id,
                    "streaming consumer back-pressured; cancelling slot to avoid silent token loss"
                );
                if let Some(flag) = task.cancel.as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                // Receiver dropped — same as the is_closed() path above.
                if let Some(flag) = task.cancel.as_ref() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// Mark the slot as finished and ship the response. Resets to Idle so the
    /// scheduler can claim it for the next task on the next tick.
    pub(super) fn finalize_ok(&mut self) {
        let Some(task) = self.task.take() else { return };
        let (text, completion_tokens, generated_tokens) =
            match std::mem::replace(&mut self.state, SlotState::Idle) {
                SlotState::Generating {
                    text_so_far,
                    generated,
                    ..
                } => {
                    let n = generated.len() as u32;
                    (text_so_far, n, generated)
                }
                _ => (String::new(), 0, Vec::new()),
            };
        // Stash the cache so next turn of this session can prefix-reuse.
        //
        // What's actually in KV: the prompt + every Generating-tick token
        // EXCEPT the very last one. The last sampled token sits in `generated`
        // (so the API-visible response text contains it) but its KV was
        // never committed because we finalized the slot before the next tick
        // that would have written it. If we put it in `cached.tokens` here,
        // the next turn would compute `common_prefix_len` against a length
        // that includes that uncommitted token and skip a position in KV —
        // producing the `inconsistent sequence positions: Y = X + 2` error
        // we observed under multi-turn pressure.
        //
        // Truncating to `generated_tokens[..len-1]` keeps `cached.tokens` in
        // perfect lock-step with what `seq_id`'s KV actually holds.
        if let Some(sid) = &task.session_id {
            let mut tokens = task.prompt_tokens_vec.clone();
            let committed_gen_len = generated_tokens.len().saturating_sub(1);
            tokens.extend_from_slice(&generated_tokens[..committed_gen_len]);
            self.cached = Some(SlotCache {
                session_id: sid.clone(),
                tokens,
            });
        } else {
            // No session affinity — slot is up for any caller; clear the
            // cache so the next assign doesn't think it has reuse.
            self.cached = None;
        }
        let timings = crate::daemon::protocol::Timings {
            prompt_eval_ns: task.prefill_ns.unwrap_or(0),
            eval_ns: task
                .decode_start
                .map(|s| s.elapsed().as_nanos() as u64)
                .unwrap_or(0),
            prompt_tokens: task.prompt_tokens,
            completion_tokens,
        };
        match task.reply {
            ReplySink::Buffered(Some(tx)) => {
                let _ = tx.send(Ok(BatchOutcome {
                    text,
                    prompt_tokens: task.prompt_tokens,
                    completion_tokens,
                    timings,
                }));
            }
            ReplySink::Streaming { tx, .. } => {
                drop(tx);
            }
            ReplySink::Buffered(None) => {} // already replied (shouldn't happen)
        }
        self.last_finalized = Instant::now();
    }

    pub(super) fn finalize_err(&mut self, err: MullamaError) {
        if let Some(task) = self.task.take() {
            match task.reply {
                ReplySink::Buffered(Some(tx)) => {
                    let _ = tx.send(Err(err));
                }
                ReplySink::Streaming { tx, .. } => {
                    drop(tx);
                    // Best-effort: error surfaces as the channel closing
                    // early; consumer sees an empty stream.
                    let _ = err;
                }
                ReplySink::Buffered(None) => {}
            }
        }
        self.state = SlotState::Idle;
        self.last_finalized = Instant::now();
    }
}
