//! Idle-window KV hydration: pre-warm durable sessions into free slots.
//!
//! Agent workloads are bursty: a session fires several turns, then goes idle
//! while the user thinks (or another session runs). During that idle window
//! the daemon has free context slots and idle CPU. This module uses that idle
//! time to restore — from the durable [`KvStore`] — sessions whose KV isn't
//! currently live in memory, so their *next* request is a hot in-memory hit
//! instead of a cold restore + delta prefill.
//!
//! Hydration is purely opportunistic:
//! - only runs when the daemon has no active requests (`active_requests == 0`),
//! - only takes a *free* slot (never evicts a live session),
//! - only commits the session as live after `load_state_seq` succeeds (a failed
//!   restore releases the reservation — never a stale "hot hit" that would
//!   decode a suffix against a missing prefix KV),
//! - skips a session if a request served it in the meantime.
//!
//! A failed or racy hydration just wastes some CPU; it can never corrupt
//! output, because a session is only marked live once its KV is genuinely
//! populated in its pinned slot.

use std::time::Duration;

use super::Daemon;

/// How often the background hydrator wakes to look for idle pre-warm work.
const HYDRATE_INTERVAL: Duration = Duration::from_secs(2);

impl Daemon {
    /// Restore every durable-but-not-live session into a free slot, opportunistically.
    /// Returns the number of sessions hydrated. Safe to call directly (used by tests and the
    /// background loop). Only does work when the daemon is idle; otherwise returns 0.
    pub async fn hydrate_idle_sessions(&self) -> usize {
        // Don't compete with real work.
        if self.active_requests.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            return 0;
        }
        let candidates = self.sessions.idle_durable_sessions();
        let mut hydrated = 0;
        for (id, alias, digest) in candidates {
            // Re-check idleness per session: a burst may have arrived mid-loop.
            if self.active_requests.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                break;
            }
            let loaded = match self.models.get(Some(&alias)).await {
                Ok(l) => l,
                Err(_) => continue, // model not loaded for this session
            };
            // Stale blob (config changed) — leave it; the digest gate in `get`
            // would refuse it too. Don't waste a slot on incompatible state.
            if loaded.kv_compat != digest {
                continue;
            }
            let Some((slot, tokens, blob)) =
                self.sessions
                    .reserve_hydrate(&id, loaded.pool_size(), &loaded.kv_compat)
            else {
                continue; // already live, no free slot, or blob gone
            };
            let mut ctx = loaded.acquire_context_at(slot).await;
            let ok = tokio::task::block_in_place(|| {
                ctx.kv_cache_clear();
                ctx.load_state_seq(0, &blob).is_ok()
            });
            drop(ctx);
            if ok {
                if self.sessions.commit_hydrate(&id, &loaded.alias, slot, tokens) {
                    hydrated += 1;
                    tracing::info!(session = %id, slot, "idle-hydrated session into slot");
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
    /// Only runs when idle. Pre-reading the predicted files warms the OS page
    /// cache so the agent's next `read` (and the prompt that embeds it) hits
    /// warm pages instead of cold disk — a safe, correctness-neutral win that
    /// never touches KV or output.
    pub async fn prefetch_predicted_files(&self, per_session_limit: usize) -> Vec<(String, Vec<std::path::PathBuf>)> {
        if self.active_requests.load(std::sync::atomic::Ordering::SeqCst) > 0 {
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

    /// Run the idle hydrator forever, waking every [`HYDRATE_INTERVAL`]. Stops
    /// when the daemon is shutting down. Spawn this on a tokio task at serve
    /// time; it's a no-op whenever the durable store is disabled or the daemon
    /// is busy.
    pub async fn run_idle_hydrator(self: std::sync::Arc<Self>) {
        while !self
            .shutdown
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            // Sleep first, then check — avoids a burst of work right at startup
            // competing with model loads.
            tokio::time::sleep(HYDRATE_INTERVAL).await;
            if self
                .shutdown
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                break;
            }
            let _ = self.hydrate_idle_sessions().await;
            // Warm the page cache for each agent's predicted next file reads.
            let _ = self.prefetch_predicted_files(8).await;
        }
    }
}