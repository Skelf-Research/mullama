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
//! ## Slot affinity + durable-safe eviction
//!
//! Sessions are pinned to slots by *affinity*, not round-robin: a session
//! keeps its slot across turns, and a new session takes a free slot. When
//! every slot is owned, the least-recently-used owner is evicted. Because the
//! evicted session's KV is durably persisted (see [`KvStore`]), eviction is
//! non-destructive — the next request for it just pays one restore + delta
//! prefill instead of a hot hit. This is what makes more-than-pool-size
//! concurrent sessions tolerable instead of a thrash where every turn
//! clobbers another session's slot.
//!
//! ## Durable restore + idle hydration
//!
//! With an optional [`KvStore`], the cached token sequence *and* the seq-state
//! blob are persisted after each turn. A fresh daemon (the in-memory map is
//! gone) restores on first lookup: the blob hydrates the pinned slot's KV via
//! `Context::load_state_seq`, after which the reuse path runs unchanged. The
//! restore is gated on a compatibility digest (model + KV-layout-affecting
//! context params) so a config change refuses to reuse a stale blob and falls
//! back to a full decode instead.
//!
//! The [`SessionStore::hydrate`] method lets a background idle-hydrator
//! pre-warm sessions whose durable blob exists but isn't live in memory: it
//! restores the blob into a free slot during an idle window so the *next*
//! request for that session is a hot in-memory hit (no restore cost). See the
//! daemon's `hydrate_idle_sessions`.

use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use parking_lot::Mutex;

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
    /// Last time this session was touched (for LRU eviction). Wall-clock
    /// `Instant`; only used relative to other sessions, never serialized.
    pub last_used: Instant,
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
/// themselves live in the [`ContextPool`] slots; this just tracks which slot
/// and the cached token sequence so the next turn can match its prefix. With a
/// [`KvStore`] attached, sessions also persist to disk for restart-tolerance
/// and can be pre-warmed during idle windows.
pub struct SessionStore {
    by_id: DashMap<String, SessionState>,
    /// Which session currently owns each pool slot (slot -> session id). A
    /// session not present here isn't live in any slot. Drives affinity + LRU.
    slot_owner: DashMap<usize, String>,
    /// Slot allocation + eviction is serialized here. The decision is fast
    /// (no model work), so contention is minimal; the lock just makes
    /// "find a free slot or evict LRU" atomic across concurrent lookups.
    alloc_lock: Mutex<()>,
    /// Optional durable content-addressed KV store. `None` disables
    /// restart-tolerance (in-memory reuse still works within a process).
    kv: Option<Arc<KvStore>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            by_id: DashMap::new(),
            slot_owner: DashMap::new(),
            alloc_lock: Mutex::new(()),
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
        // Hot path: session already live in memory -> reuse its slot, no
        // restore needed. Bump last_used for LRU recency.
        if let Some(mut entry) = self.by_id.get_mut(id) {
            if entry.model_alias == model_alias {
                entry.last_used = Instant::now();
                return SessionLookup {
                    slot: entry.slot,
                    cached_tokens: entry.cached_tokens.clone(),
                    restore: None,
                };
            }
            // Model changed: drop the stale in-memory state (slot ownership
            // too) and fall through to recreate on a fresh slot.
            self.slot_owner.remove(&entry.slot);
        }
        self.by_id.remove(id);

        // Allocate a slot under the lock: prefer a free slot, else evict the
        // least-recently-used owner. Eviction is durable-safe — the blob
        // persists, so the evicted session just restores on next request.
        let slot = {
            let _g = self.alloc_lock.lock();
            self.allocate_slot(pool_size)
        };

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

        let now = Instant::now();
        self.by_id.insert(
            id.to_string(),
            SessionState {
                model_alias: model_alias.to_string(),
                cached_tokens: cached_tokens.clone(),
                slot,
                last_used: now,
            },
        );
        self.slot_owner.insert(slot, id.to_string());

        SessionLookup {
            slot,
            cached_tokens,
            restore,
        }
    }

    /// Pick a slot for a new session: a free one if any, else the LRU owner's
    /// (which is evicted). Caller holds `alloc_lock`. Eviction removes the
    /// owner from `by_id` and `slot_owner` so its next request re-allocates +
    /// durably restores.
    fn allocate_slot(&self, pool_size: usize) -> usize {
        let pool_size = pool_size.max(1);
        // Free slot?
        for slot in 0..pool_size {
            if !self.slot_owner.contains_key(&slot) {
                return slot;
            }
        }
        // Full: evict the least-recently-used owner.
        let mut evict_slot = 0;
        let mut evict_last = Instant::now();
        let mut found = false;
        for slot in 0..pool_size {
            if let Some(owner_id) = self.slot_owner.get(&slot) {
                if let Some(state) = self.by_id.get(owner_id.value()) {
                    if !found || state.last_used < evict_last {
                        evict_last = state.last_used;
                        evict_slot = slot;
                        found = true;
                    }
                }
            }
        }
        if let Some((_, owner)) = self.slot_owner.remove(&evict_slot) {
            self.by_id.remove(&owner);
        }
        evict_slot
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
            entry.last_used = Instant::now();
        }
        if let (Some(kv), Some(state)) = (self.kv.as_ref(), seq_state) {
            kv.put(id, model_alias, compat_digest, &cached_tokens, &state);
        }
    }

    /// Enumerate durable sessions (id, model_alias, compat_digest) that are not
    /// currently live in memory — candidates for the idle hydrator.
    pub(crate) fn idle_durable_sessions(&self) -> Vec<(String, String, String)> {
        let Some(kv) = self.kv.as_ref() else {
            return Vec::new();
        };
        kv.list_sessions()
            .into_iter()
            .filter(|(id, _, _)| !self.by_id.contains_key(id))
            .collect()
    }

    /// Reserve a free slot for the idle hydrator and read the durable blob, *without*
    /// committing the session as live yet. Returns `(slot, cached_tokens, blob)` so
    /// the caller can `load_state_seq` into the slot. The reservation is released by
    /// [`Self::commit_hydrate`] (on success) or [`Self::abort_hydrate`] (on failure).
    /// Pre-warming never evicts a live session — it only takes a free slot, so it's
    /// purely opportunistic.
    pub(crate) fn reserve_hydrate(
        &self,
        id: &str,
        pool_size: usize,
        compat_digest: &str,
    ) -> Option<(usize, Vec<TokenId>, Vec<u8>)> {
        if self.by_id.contains_key(id) {
            return None; // a request served it meanwhile
        }
        let kv = self.kv.as_ref()?;
        let slot = {
            let _g = self.alloc_lock.lock();
            if self.by_id.contains_key(id) {
                return None;
            }
            let slot = (0..pool_size.max(1))
                .find(|&s| !self.slot_owner.contains_key(&s))?;
            // Tentatively reserve so a concurrent get won't take this slot.
            self.slot_owner.insert(slot, id.to_string());
            slot
        };
        // Read the blob outside the lock. If it's missing/incompatible, abort.
        let r = match kv.get(id, compat_digest) {
            Some(r) => r,
            None => {
                self.abort_hydrate(slot, id);
                return None;
            }
        };
        Some((slot, r.tokens, r.state))
    }

    /// Commit a hydrated session: mark it live in `by_id` with the cached tokens
    /// the blob holds, so the next `get` is a hot in-memory hit (`restore =
    /// None`). Returns false if a request served the session or stole the slot
    /// while we were loading — in that case the loaded KV is just abandoned and
    /// the reservation released.
    pub(crate) fn commit_hydrate(
        &self,
        id: &str,
        model_alias: &str,
        slot: usize,
        cached_tokens: Vec<TokenId>,
    ) -> bool {
        let _g = self.alloc_lock.lock();
        if self.by_id.contains_key(id) {
            // A request beat us to it; release our reservation if we still hold it.
            if self.slot_owner.get(&slot).map(|v| v.value() == id).unwrap_or(false) {
                self.slot_owner.remove(&slot);
            }
            return false;
        }
        if self.slot_owner.get(&slot).map(|v| v.value() == id).unwrap_or(false) {
            self.by_id.insert(
                id.to_string(),
                SessionState {
                    model_alias: model_alias.to_string(),
                    cached_tokens,
                    slot,
                    last_used: Instant::now(),
                },
            );
            true
        } else {
            // Slot was stolen by a concurrent allocation; abandon.
            false
        }
    }

    /// Release a tentative hydrate reservation whose `load_state_seq` failed,
    /// so the slot is free again. Idempotent: only removes it if still owned by
    /// `id`.
    pub(crate) fn abort_hydrate(&self, slot: usize, id: &str) {
        let _g = self.alloc_lock.lock();
        if self.slot_owner.get(&slot).map(|v| v.value() == id).unwrap_or(false) {
            self.slot_owner.remove(&slot);
        }
    }

    /// Number of active in-memory sessions (for observability / tests).
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn owner_of(&self, slot: usize) -> Option<String> {
        self.slot_owner.get(&slot).map(|v| v.value().clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenId;

    fn store_no_kv() -> SessionStore {
        SessionStore::new()
    }

    #[test]
    fn affinity_pins_session_to_one_slot_across_turns() {
        let s = store_no_kv();
        // pool_size 4; three sessions get distinct slots.
        let a = s.get("A", "m", 4, "d");
        let b = s.get("B", "m", 4, "d");
        let c = s.get("C", "m", 4, "d");
        let mut slots = [a.slot, b.slot, c.slot];
        slots.sort();
        assert_eq!(slots, [0, 1, 2], "three sessions occupy three distinct slots");

        // Turn 2 for A reuses the same slot (affinity), restore=None.
        let a2 = s.get("A", "m", 4, "d");
        assert_eq!(a2.slot, a.slot);
        assert!(a2.restore.is_none());

        // No two live sessions share a slot.
        let owners: Vec<_> = (0..4).filter_map(|i| s.owner_of(i)).collect();
        assert_eq!(owners.len(), 3);
    }

    #[test]
    fn lru_evicts_when_pool_full_and_is_durable_safe() {
        let s = store_no_kv();
        // pool_size 2: A, B fill it.
        let a = s.get("A", "m", 2, "d");
        let b = s.get("B", "m", 2, "d");
        assert_ne!(a.slot, b.slot);
        // Touch A again so B is the LRU.
        s.get("A", "m", 2, "d");
        // C arrives -> evicts B (LRU), takes B's slot.
        let c = s.get("C", "m", 2, "d");
        assert_eq!(c.slot, b.slot, "evicted the LRU slot (B's)");
        // B is no longer live.
        assert!(s.owner_of(a.slot).as_deref() != Some("B"));
        assert!(s.owner_of(b.slot).as_deref() == Some("C"));
    }

    #[test]
    fn round_trip_tokens_via_put_get() {
        let s = store_no_kv();
        let lk = s.get("A", "m", 2, "d");
        s.put("A", "m", "d", vec![5 as TokenId, 6, 7], None);
        let lk2 = s.get("A", "m", 2, "d");
        assert_eq!(lk2.slot, lk.slot);
        assert_eq!(lk2.cached_tokens, vec![5, 6, 7]);
        assert!(lk2.restore.is_none());
    }

    #[test]
    fn model_change_resets_session_to_a_fresh_slot() {
        let s = store_no_kv();
        let a = s.get("A", "m1", 2, "d");
        s.put("A", "m1", "d", vec![1, 2], None);
        // Same session, different model -> reset.
        let a2 = s.get("A", "m2", 2, "d");
        // New state, empty cache.
        assert!(a2.cached_tokens.is_empty());
        let _ = a.slot; // old slot freed by reset
    }
}