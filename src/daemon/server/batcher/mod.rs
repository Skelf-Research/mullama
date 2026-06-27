//! Continuous-batched decode scheduler (Phase C tracer-bullet).
//!
//! One [`Context`] with `n_seq_max=N`, served by a single async task that
//! drains incoming generation requests into one [`llama_decode`] per tick —
//! the model llama.cpp's `tools/server/server.cpp` uses, and the only way to
//! get real concurrent scaling on Metal (one command queue, no parallel
//! `llama_decode`s).
//!
//! Today this is opt-in (`MULLAMA_BATCHED=1` env or `batched: true` in
//! [`crate::daemon::DaemonConfig`]) so the existing pool-per-context path
//! stays the default while the new path is benchmarked. Non-streaming only;
//! KV reuse / sessions / streaming get wired once the tracer bullet shows
//! the expected concurrency win.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use crate::daemon::protocol::{StreamChunk, Timings};
use crate::token::TokenId;
use crate::{Context, Model, MullamaError, SamplerParams};

mod scheduler;
mod slot;

pub use scheduler::BatchScheduler;

/// A unit of work submitted to the scheduler. Carries everything the
/// scheduler needs to run a prompt to completion and ship the response back
/// over either a oneshot (buffered) or an mpsc (streaming) channel.
pub struct BatchTask {
    pub prompt_tokens: Vec<TokenId>,
    pub max_tokens: u32,
    pub sampler_params: SamplerParams,
    pub stop_sequences: Vec<String>,
    pub grammar_gbnf: Option<String>,
    pub reply: ReplyMode,
    /// Optional session affinity. When set, the scheduler tries to route
    /// this task to a slot whose `seq_id`'s KV already holds the previous
    /// turn's prefix for the same session — then drops the divergent tail
    /// and prefills only the new suffix. Hits the same algorithmic O(delta)
    /// behavior as the legacy `kv_reuse` path, but interleaved with other
    /// slots in a single `llama_decode` per tick.
    pub session_id: Option<String>,
    /// Durable KV-restore payload. Set when this is the first turn of a
    /// session after a cold start: the scheduler `load_state_seq`s `blob`
    /// into the chosen slot's `seq_id` before prefix-matching against
    /// `tokens`. Lets the batched path serve cross-restart agent loops
    /// the same way the legacy [`crate::daemon::server::generation::KvReuse`]
    /// path does. `None` for in-memory sessions and stateless requests.
    pub restore: Option<BatchRestore>,
    /// Cooperative cancellation flag. Checked at every scheduler tick;
    /// when set, the slot is finalized early (the partial response is
    /// shipped via the configured `reply`). Mirrors the legacy
    /// `StreamingSetup::cancel_flag` so clients that disconnect mid-stream
    /// release the slot promptly instead of running to `max_tokens`.
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Saved KV state for a session, restored into the slot's `seq_id` before
/// the prefix-match runs. `tokens` is the cached prompt+generated token
/// sequence that `blob` corresponds to (used by `common_prefix_len`).
pub struct BatchRestore {
    pub blob: Vec<u8>,
    pub tokens: Vec<TokenId>,
}

/// How the scheduler returns generated tokens to the caller.
///
/// `Buffered`: collect the whole response, ship a single
/// [`BatchOutcome`] when the slot finalizes.
/// `Streaming`: emit one [`StreamChunk`] per sampled token as it lands,
/// then close the channel — matches the legacy `generate_text_streaming`
/// shape so the SSE wrapper can be reused unchanged.
pub enum ReplyMode {
    Buffered {
        tx: oneshot::Sender<Result<BatchOutcome, MullamaError>>,
    },
    Streaming {
        tx: mpsc::Sender<StreamChunk>,
        request_id: Arc<str>,
        index: u32,
    },
}

pub struct BatchOutcome {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub timings: Timings,
}

/// Handle to a running scheduler. Cheap to clone (just an mpsc sender + arc
/// to the shared model). Cloning does *not* spawn a new scheduler; all
/// clones funnel into the same `llama_decode` loop.
#[derive(Clone)]
pub struct BatcherHandle {
    tx: mpsc::Sender<BatchTask>,
    /// Kept so callers can still reach the model for tokenization and
    /// chat-template work without contending for the scheduler's lock.
    pub model: Arc<Model>,
}

impl BatcherHandle {
    /// Buffered submit: scheduler builds the full response and returns it
    /// via the oneshot. Errors propagate from either side.
    pub async fn submit(&self, task: BatchTask) -> Result<BatchOutcome, MullamaError> {
        let (tx, rx) = oneshot::channel();
        let task = BatchTask {
            reply: ReplyMode::Buffered { tx },
            ..task
        };
        if self.tx.send(task).await.is_err() {
            return Err(MullamaError::OperationFailed(
                "batch scheduler is shut down".into(),
            ));
        }
        rx.await.map_err(|_| {
            MullamaError::OperationFailed("batch scheduler dropped the reply channel".into())
        })?
    }

    /// Streaming submit: scheduler emits one [`StreamChunk`] per sampled
    /// token over the returned receiver. The channel is closed when the
    /// slot terminates (EOG / max_tokens / stop). Errors land as the
    /// channel closing early — caller should treat an empty stream as a
    /// scheduler-side failure.
    pub async fn submit_streaming(
        &self,
        task: BatchTask,
        request_id: Arc<str>,
        index: u32,
    ) -> Result<mpsc::Receiver<StreamChunk>, MullamaError> {
        // Buffer a few chunks per slot so a transient slow consumer doesn't
        // block the scheduler tick — but small enough that a stuck consumer
        // surfaces as back-pressure quickly. 64 ≈ ~0.8s of generation at
        // 80 tok/s, room to absorb a typical SSE flush delay.
        let (tx, rx) = mpsc::channel(64);
        let task = BatchTask {
            reply: ReplyMode::Streaming {
                tx,
                request_id,
                index,
            },
            ..task
        };
        if self.tx.send(task).await.is_err() {
            return Err(MullamaError::OperationFailed(
                "batch scheduler is shut down".into(),
            ));
        }
        Ok(rx)
    }
}

/// Spawn a new scheduler driving `context` and return its handle. The
/// scheduler keeps running until `handle` (and all clones) are dropped.
///
/// `n_slots` is the number of `seq_id`s the scheduler will multiplex; it
/// must be ≤ the context's configured `n_seq_max`. Typical: equal to it.
pub fn spawn(model: Arc<Model>, context: Context, n_slots: u32) -> BatcherHandle {
    // Bounded queue: back-pressure protects the scheduler from memory blow-up
    // under load. 256 is a generous bound for typical request rates and small
    // enough that an unhealthy scheduler can't queue a working set of
    // half-GB prompts.
    let (tx, rx) = mpsc::channel::<BatchTask>(256);
    let handle = BatcherHandle {
        tx,
        model: model.clone(),
    };
    tokio::spawn(async move {
        let mut sched = BatchScheduler::new(model, context, n_slots, rx);
        sched.run().await;
    });
    handle
}
