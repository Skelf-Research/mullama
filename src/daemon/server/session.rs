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

use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;

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

/// A concurrency-safe map of session id -> [`SessionState`]. The KV tensors
/// themselves live in the [`ContextPool`] slots; this just tracks which slot and
/// the cached token sequence so the next turn can match its prefix.
pub struct SessionStore {
    by_id: DashMap<String, SessionState>,
    /// Round-robin slot allocator for new sessions, bounded by the model's
    /// context pool size at lookup time.
    next_slot: AtomicUsize,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            by_id: DashMap::new(),
            next_slot: AtomicUsize::new(0),
        }
    }

    /// Get or create the session for `id`, pinned to a pool slot of `model`.
    /// If the session exists but the model changed, it is reset (different
    /// model => stale KV). Returns the slot index and a *clone* of the cached
    /// tokens; the caller writes the updated tokens back via [`Self::put`].
    pub fn get(&self, id: &str, model_alias: &str, pool_size: usize) -> (usize, Vec<TokenId>) {
        if let Some(entry) = self.by_id.get(id) {
            if entry.model_alias == model_alias {
                return (entry.slot, entry.cached_tokens.clone());
            }
            // Model changed: drop the stale entry and fall through to recreate.
        }
        self.by_id.remove(id);

        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) % pool_size.max(1);
        self.by_id.insert(
            id.to_string(),
            SessionState {
                model_alias: model_alias.to_string(),
                cached_tokens: Vec::new(),
                slot,
            },
        );
        (slot, Vec::new())
    }

    /// Write back the updated cached-token sequence after a turn.
    pub fn put(&self, id: &str, cached_tokens: Vec<TokenId>) {
        if let Some(mut entry) = self.by_id.get_mut(id) {
            entry.cached_tokens = cached_tokens;
        }
    }

    /// Number of active sessions (for observability / tests).
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