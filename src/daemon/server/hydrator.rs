//! Background KV hydration: pre-warm durable sessions into free slots.
//!
//! Agent workloads are bursty: a session fires several turns, then goes idle
//! while the user thinks (or another session runs). During that window the
//! daemon often has free context slots. This module uses them to restore — from
//! the durable [`KvStore`] — sessions whose KV isn't currently live in memory,
//! so their *next* request is a hot in-memory hit instead of a cold restore +
//! delta prefill.
//!
//! ## Idle vs active (parallel) fill
//!
//! [`HydrationMode`] controls *when* this runs:
//! - **Idle**: only when the daemon has no active requests. Safe everywhere; it
//!   never competes with a live decode for memory bandwidth.
//! - **Active**: whenever a free slot exists, *including while other slots are
//!   decoding* — "parallel fill". This overlaps a waiting session's prefill
//!   with another session's in-flight decode. It costs memory bandwidth, so it
//!   only pays off on high-bandwidth hardware (Apple Silicon's unified memory +
//!   Metal GPU); it's the macOS default. To protect incoming live traffic, in
//!   Active mode we leave one slot free as headroom when the pool has >1 slot.
//! - **Off**: never pre-warm.
//!
//! Regardless of mode, hydration is correctness-safe:
//! - only takes a *free* slot (never evicts a live session),
//! - only commits the session as live after `load_state_seq` succeeds (a failed
//!   restore releases the reservation — never a stale "hot hit" that would
//!   decode a suffix against a missing prefix KV),
//! - skips a session if a request served it in the meantime.
//!
//! A failed or racy hydration just wastes some bandwidth; it can never corrupt
//! output, because a session is only marked live once its KV is genuinely
//! populated in its pinned slot.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::config::HydrationMode;
use super::Daemon;

/// How often the background hydrator wakes to look for pre-warm work.
const HYDRATE_INTERVAL: Duration = Duration::from_secs(2);

impl Daemon {
    /// Pre-warm durable-but-not-live sessions into free slots, honoring the
    /// configured [`HydrationMode`]. Returns the number of sessions hydrated.
    /// Safe to call directly (used by tests and the background loop).
    pub async fn hydrate_idle_sessions(&self) -> usize {
        let mode = self.config.resources.hydration_mode;
        if mode == HydrationMode::Off {
            return 0;
        }
        // Idle mode: only run when the daemon is fully quiescent.
        if mode == HydrationMode::Idle && self.active_requests.load(Ordering::SeqCst) > 0 {
            return 0;
        }

        let candidates = self.sessions.idle_durable_sessions();
        let mut hydrated = 0;
        for (id, alias, digest) in candidates {
            // Idle mode: bail the moment real work arrives mid-loop. Active mode
            // keeps going — it's allowed to overlap live decodes — but still
            // respects the per-slot reservation below.
            if mode == HydrationMode::Idle && self.active_requests.load(Ordering::SeqCst) > 0 {
                break;
            }
            let loaded = match self.models.get(Some(&alias)).await {
                Ok(l) => l,
                Err(_) => continue, // model not loaded for this session
            };
            // Phase-C: skip hydration when the model has a batched scheduler
            // attached. The batcher does its own restore on-demand in
            // `assign_to_first_idle` (the request handler builds the kv_reuse
            // via SessionStore::get and the scheduler calls load_state_seq),
            // so pre-warming the unused legacy pool slot just burns memory
            // bandwidth for nothing.
            if loaded.batcher.read().await.is_some() {
                continue;
            }
            // Stale blob (config changed) — leave it; the digest gate in `get`
            // would refuse it too. Don't waste a slot on incompatible state.
            if loaded.kv_compat != digest {
                continue;
            }
            // Active mode: keep one slot free as headroom for incoming live
            // requests, so parallel fill never starves new traffic of a slot.
            let pool = loaded.pool_size();
            let reservable = if mode == HydrationMode::Active && pool > 1 {
                pool - 1
            } else {
                pool
            };
            let Some((slot, tokens, blob)) =
                self.sessions
                    .reserve_hydrate(&id, reservable, &loaded.kv_compat)
            else {
                continue; // already live, no free slot within headroom, or blob gone
            };
            // Non-blocking acquire: if the slot is held by a live request, skip
            // this candidate and retry on the next hydrate tick. Without this,
            // an Active-mode hydrator queues behind a long decode (holding its
            // own reservation the whole time), then load_state_seqs into the
            // slot just as the live decode releases it — guaranteeing the
            // `state_read_meta` thrash we observed under concurrent load.
            let Some(mut ctx) = loaded.try_acquire_context_at(slot) else {
                self.sessions.abort_hydrate(slot, &id);
                continue;
            };
            let ok = tokio::task::block_in_place(|| {
                ctx.kv_cache_clear();
                ctx.load_state_seq(0, &blob).is_ok()
            });
            drop(ctx);
            if ok {
                if self.sessions.commit_hydrate(&id, &loaded.alias, slot, tokens) {
                    hydrated += 1;
                    tracing::info!(session = %id, slot, mode = ?mode, "hydrated session into slot");
                }
            } else {
                self.sessions.abort_hydrate(slot, &id);
            }
        }
        hydrated
    }

    /// Predict the files each active agent session is about to read next and
    /// surface them. Returns the predicted candidate paths (existing on disk),
    /// per session, so a caller can pre-read them into page cache or warm
    /// derived state. This is the file-access prefetch policy applied: the
    /// observer (fed from conversation content at request time) supplies the
    /// touched-file history; [`super::prefetch::predict_fs`] ranks the next
    /// reads via import-following and directory locality.
    ///
    /// Pre-reading the predicted files warms the OS page cache so the agent's
    /// next `read` (and the prompt that embeds it) hits warm pages instead of
    /// cold disk — a safe, correctness-neutral win that never touches KV or
    /// output. Disk reads barely touch memory bandwidth, so in `Active` mode
    /// this runs even alongside live decodes; in `Idle` mode it waits for
    /// quiescence; in `Off` mode it's skipped.
    pub async fn prefetch_predicted_files(&self, per_session_limit: usize) -> Vec<(String, Vec<std::path::PathBuf>)> {
        let mode = self.config.resources.hydration_mode;
        if mode == HydrationMode::Off {
            return Vec::new();
        }
        if mode == HydrationMode::Idle && self.active_requests.load(Ordering::SeqCst) > 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for session in self.prefetch.sessions() {
            let touched = self.prefetch.history(&session);
            if touched.is_empty() {
                continue;
            }
            let preds = tokio::task::block_in_place(|| {
                super::prefetch::predict_fs(&touched, per_session_limit)
            });
            if preds.is_empty() {
                continue;
            }
            let paths: Vec<std::path::PathBuf> = preds.iter().map(|c| c.path.clone()).collect();
            // Warm the OS page cache by touching each predicted file's bytes.
            tokio::task::block_in_place(|| {
                for p in &paths {
                    let _ = std::fs::read(p);
                }
            });
            for c in &preds {
                tracing::debug!(
                    session = %session,
                    path = %c.path.display(),
                    score = c.score,
                    reason = ?c.reason,
                    "prefetch: warmed predicted next-read"
                );
            }
            out.push((session, paths));
        }
        out
    }

    /// Run the background hydrator forever, waking every [`HYDRATE_INTERVAL`].
    /// Stops when the daemon is shutting down. Spawn this on a tokio task at
    /// serve time; it's a no-op when the durable store is disabled, when the
    /// [`HydrationMode`] is `Off`, or (in `Idle` mode) while the daemon is busy.
    pub async fn run_idle_hydrator(self: std::sync::Arc<Self>) {
        // Nothing to do for the lifetime of the process if hydration is off.
        if self.config.resources.hydration_mode == HydrationMode::Off {
            return;
        }
        while !self.shutdown.load(Ordering::SeqCst) {
            // Sleep first, then check — avoids a burst of work right at startup
            // competing with model loads.
            tokio::time::sleep(HYDRATE_INTERVAL).await;
            if self.shutdown.load(Ordering::SeqCst) {
                break;
            }
            let _ = self.hydrate_idle_sessions().await;
            // Warm the page cache for each agent's predicted next file reads.
            let _ = self.prefetch_predicted_files(8).await;
        }
    }
}