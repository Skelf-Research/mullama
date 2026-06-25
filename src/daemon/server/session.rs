//! Cross-turn KV reuse: the cache *is* the conversation.
//!
//! Stock behaviour is to `kv_cache_clear()` and re-decode the full prompt on
//! every request. For an agent loop that re-sends the growing history each
//! turn, that re-prefills the whole history every turn — an O(history) cost on
//! a workload that only adds a small delta per turn.
//!
//! This module keeps the KV cache alive across turns for a named session. A
//! session is pinned to one [`ContextPool`] slot, and we track the token-id
//! sequence currently held in that slot's KV. On the next turn we tokenize the
//! new prompt, find the longest common prefix with the cached tokens, drop the
//! divergent tail from the KV (`kv_cache_seq_rm`), and decode only the new
//! suffix — `llama_batch_get_one` auto-continues positions from
//! `seq_pos_max+1`, so no explicit-position batch is needed.
//!
//! Crucially the reused path is *numerically identical* to clear-and-full-decode
//! (same tokens at the same positions), so greedy parity is preserved. The
//! agent-loop bench verifies this: the same trace run with and without a
//! session must produce identical output, while turn-2+ prefill collapses.
//!
//! ## Durable restore
//!
//! With an optional [`KvStore`], the cached token sequence *and* the seq-state
//! blob are persisted after each turn. A fresh daemon (the in-memory map is
//! gone) restores on first lookup: the blob hydrates the pinned slot's KV via
//! `Context::load_state_seq`, after which the reuse path runs unchanged. The
//! restore is gated on a compatibility digest (model + KV-layout-affecting
//! context params) so a config change refuses to reuse a stale blob and falls
//! back to a full decode instead.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use super::kvstore::KvStore;
use crate::token::TokenId;

/// Per-session cross-turn KV reuse state. `cached_tokens` is the token-id
/// sequence currently held in the pinned `slot`'s KV cache (prompt prefix +
/// generated tokens from prior turns).
#[derive(Debug)]
pub struct SessionState {
    pub model_alias: String,
    pub cached_tokens: Vec<TokenId>,
    pub slot: usize,
}

/// Result of a session lookup. `restore` carries the seq-state blob to
/// hydrate the pinned slot's KV with, and is `Some` *only* on the first turn
/// after a durable restore (an in-memory hit already has the KV in the slot,
/// so nothing to hydrate).
pub(crate) struct SessionLookup {
    pub slot: usize,
    pub cached_tokens: Vec<TokenId>,
    pub restore: Option<Vec<u8>>,
}

/// A concurrency-safe map of session id -> [`SessionState`]. The KV tensors
/// themselves live in the [`ContextPool`] slots; this just tracks which slot and
/// the cached token sequence so the next turn can match its prefix. With a
/// [`KvStore`] attached, sessions also persist to disk for restart-tolerance.
pub struct SessionStore {
    by_id: DashMap<String, SessionState>,
    /// Round-robin slot allocator for new sessions, bounded by the model's
    /// context pool size at lookup time.
    next_slot: AtomicUsize,
    /// Optional durable content-addressed KV store. `None` disables
    /// restart-tolerance (in-memory reuse still works within a process).
    kv: Option<Arc<KvStore>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            by_id: DashMap::new(),
            next_slot: AtomicUsize::new(0),
            kv: None,
        }
    }

    /// Attach a durable KV store. After this, sessions persist their seq-state
    /// after each turn and a fresh daemon can restore them on first lookup.
    pub(crate) fn with_kv(mut self, kv: Arc<KvStore>) -> Self {
        self.kv = Some(kv);
        self
    }

    /// Get or create the session for `id`, pinned to a pool slot of `model`.
    /// If the session exists but the model changed, it is reset (different
    /// model => stale KV). `compat_digest` gates durable restore against the
    /// current model + context params. Returns the slot, a clone of the cached
    /// tokens, and an optional seq-state blob to hydrate the slot's KV with
    /// (only present on a fresh durable restore). The caller writes the updated
    /// tokens back via [`Self::put`].
    pub(crate) fn get(
        &self,
        id: &str,
        model_alias: &str,
        pool_size: usize,
        compat_digest: &str,
    ) -> SessionLookup {
        if let Some(entry) = self.by_id.get(id) {
            if entry.model_alias == model_alias {
                return SessionLookup {
                    slot: entry.slot,
                    cached_tokens: entry.cached_tokens.clone(),
                    restore: None,
                };
            }
            // Model changed: drop the stale entry and fall through to recreate.
        }
        self.by_id.remove(id);

        // Fresh in-memory entry. If a durable blob exists and its compat digest
        // matches the current model+config, restore it: hydrate the slot's KV
        // with the blob and seed cached_tokens from the persisted sequence so
        // the reuse path only prefills the new delta.
        let (cached_tokens, restore) = match self.kv.as_ref() {
            Some(kv) => match kv.get(id, compat_digest) {
                Some(r) => (r.tokens, Some(r.state)),
                None => (Vec::new(), None),
            },
            None => (Vec::new(), None),
        };

        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) % pool_size.max(1);
        let entry = SessionState {
            model_alias: model_alias.to_string(),
            cached_tokens: cached_tokens.clone(),
            slot,
        };
        self.by_id.insert(id.to_string(), entry);

        SessionLookup {
            slot,
            cached_tokens,
            restore,
        }
    }

    /// Write back the updated cached-token sequence after a turn. If `seq_state`
    /// is present and a durable store is attached, persist it (best-effort).
    pub(crate) fn put(
        &self,
        id: &str,
        model_alias: &str,
        compat_digest: &str,
        cached_tokens: Vec<TokenId>,
        seq_state: Option<Vec<u8>>,
    ) {
        if let Some(mut entry) = self.by_id.get_mut(id) {
            entry.cached_tokens = cached_tokens.clone();
        }
        if let (Some(kv), Some(state)) = (self.kv.as_ref(), seq_state) {
            kv.put(id, model_alias, compat_digest, &cached_tokens, &state);
        }
    }

    /// Number of active in-memory sessions (for observability / tests).
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Length of the longest common prefix of two token slices. Used to decide how
/// much of the cached KV is still valid for the new prompt.
pub fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i;
        }
    }
    n
}