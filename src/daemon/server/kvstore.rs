//! Durable, content-addressed KV-cache store for cross-session reuse.
//!
//! [`crate::daemon::server::session`] keeps a session's KV alive *in memory*
//! across turns. That wins within a single daemon process, but a restart
//! wipes the pinned slot and the next turn re-prefills the whole history.
//! This module persists the KV so a restart can restore it instead.
//!
//! The store is a sled-backed content-addressable store (CAS):
//! - **blobs** tree: `sha256(token sequence) -> llama_state_seq` bytes. Keyed by
//!   the token sequence, so two sessions that reached the same token history
//!   share one blob (dedup). The blob is the opaque `Context::save_state_seq`
//!   payload — positions and KV tensors included.
//! - **manifest** tree: `session_id -> {model_alias, compat_digest, token_hash,
//!   token_count}`. Maps a session to its last persisted blob.
//!
//! ## Determinism gate
//!
//! A saved KV is only valid for the *same* model weights and the KV-layout-
//! affecting context params (RoPE base/scale, cache dtype, n_ctx, defrag
//! threshold). Restoring under a different config would corrupt the cache.
//! So [`KvStore::compat_digest`] folds those into a digest stored in the
//! manifest; [`KvStore::get`] refuses to restore when the digest no longer
//! matches (returns `None`, and the caller falls back to a full decode).
//! llama.cpp's own state-version header is the backstop: a blob from an
//! incompatible build fails `load_state_seq`, which we treat as "no restore".
//!
//! Everything here is best-effort: if sled can't open, the store disables
//! itself (gets return `None`, puts are no-ops) and the daemon runs exactly as
//! before — just without durable KV.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::context::ContextParams;
use crate::token::TokenId;

/// Don't persist blobs larger than this; a pathological session shouldn't
/// balloon the store. Persistence is an optimization, never a correctness
/// requirement, so dropping a too-big blob is fine.
const MAX_BLOB_BYTES: usize = 64 * 1024 * 1024;

/// Per-session manifest entry pointing at a content-addressed blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    model_alias: String,
    compat_digest: String,
    token_hash: String,
    token_count: u32,
    /// The cached token sequence itself, so a fresh daemon (no in-memory
    /// session state) can still do prefix matching after restoring the blob.
    tokens: Vec<TokenId>,
}

/// A restored session's KV: the cached token sequence (so the reuse path can
/// match its prefix) and the opaque seq-state bytes to hydrate a context with.
pub(crate) struct RestoredKv {
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) state: Vec<u8>,
}

/// Durable content-addressed KV-cache store. Wrap in `Option` / `Arc` at the
/// call site: `None` means durability is disabled (sled unavailable).
pub(crate) struct KvStore {
    db: sled::Db,
    blobs: sled::Tree,
    manifest: sled::Tree,
}

impl KvStore {
    /// Open (or create) the store under `~/.mullama/kv-cas/`.
    pub(crate) fn open_default() -> Result<Self, sled::Error> {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mullama")
            .join("kv-cas");
        Self::open(&path)
    }

    pub(crate) fn open(path: &PathBuf) -> Result<Self, sled::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let db = sled::open(path)?;
        let blobs = db.open_tree("blobs")?;
        let manifest = db.open_tree("manifest")?;
        Ok(Self { db, blobs, manifest })
    }

    /// Compatibility digest over the params that change the cached K/V bytes
    /// or their layout. Two contexts with the same digest can safely share a
    /// blob; different digests cannot.
    pub(crate) fn compat_digest(
        model_path: &str,
        ctx: &ContextParams,
    ) -> String {
        let mut h = Sha256::new();
        h.update(b"mullama-kv-v1|");
        h.update(model_path.as_bytes());
        h.update(b"|n_ctx=");
        h.update(ctx.n_ctx.to_le_bytes());
        h.update(b"|n_batch=");
        h.update(ctx.n_batch.to_le_bytes());
        h.update(b"|rope_base=");
        h.update(ctx.rope_freq_base.to_le_bytes());
        h.update(b"|rope_scale=");
        h.update(ctx.rope_freq_scale.to_le_bytes());
        h.update(b"|yarn_ext=");
        h.update(ctx.yarn_ext_factor.to_le_bytes());
        h.update(b"|yarn_attn=");
        h.update(ctx.yarn_attn_factor.to_le_bytes());
        h.update(b"|yarn_orig=");
        h.update(ctx.yarn_orig_ctx.to_le_bytes());
        h.update(b"|type_k=");
        h.update(format!("{:?}", ctx.type_k).as_bytes());
        h.update(b"|type_v=");
        h.update(format!("{:?}", ctx.type_v).as_bytes());
        h.update(b"|defrag=");
        h.update(ctx.defrag_thold.to_le_bytes());
        hex(&h.finalize())
    }

    /// Persist a session's KV. `seq_state` is `Context::save_state_seq(0)`.
    /// Returns silently if the blob is too large or persistence is disabled.
    pub(crate) fn put(
        &self,
        session_id: &str,
        model_alias: &str,
        compat_digest: &str,
        tokens: &[TokenId],
        seq_state: &[u8],
    ) {
        if seq_state.len() > MAX_BLOB_BYTES || tokens.is_empty() {
            return;
        }
        let token_hash = content_hash(tokens);
        // Content-addressed: write the blob only if absent (dedup).
        if self.blobs.get(token_hash.as_bytes()).ok().flatten().is_none() {
            if self.blobs.insert(token_hash.as_bytes(), seq_state).is_err() {
                return;
            }
        }
        let entry = Manifest {
            model_alias: model_alias.to_string(),
            compat_digest: compat_digest.to_string(),
            token_hash: token_hash.clone(),
            token_count: tokens.len() as u32,
            tokens: tokens.to_vec(),
        };
        let Ok(bytes) = serde_json::to_vec(&entry) else {
            return;
        };
        let _ = self.manifest.insert(session_id.as_bytes(), bytes);
        let _ = self.db.flush();
    }

    /// Look up a session's persisted KV. Returns `None` if absent, if the
    /// compatibility digest no longer matches, or if the blob can't be read —
    /// in every such case the caller must fall back to a full decode.
    pub(crate) fn get(
        &self,
        session_id: &str,
        compat_digest: &str,
    ) -> Option<RestoredKv> {
        let bytes = self.manifest.get(session_id.as_bytes()).ok()??;
        let entry: Manifest = serde_json::from_slice(&bytes).ok()?;
        if entry.compat_digest != compat_digest {
            // Config or model changed since the blob was written: refuse to
            // restore (the cache would be invalid).
            return None;
        }
        let blob = self.blobs.get(entry.token_hash.as_bytes()).ok()??;
        Some(RestoredKv {
            tokens: entry.tokens,
            state: blob.to_vec(),
        })
    }

    /// Drop a session's manifest entry (blob kept — content-addressed, may be
    /// shared). Used on session reset.
    #[allow(dead_code)]
    pub(crate) fn remove(&self, session_id: &str) {
        let _ = self.manifest.remove(session_id.as_bytes());
        let _ = self.db.flush();
    }

    /// Enumerate every persisted session: `(session_id, model_alias,
    /// compat_digest)`. The idle hydrator walks this list to pre-warm sessions
    /// that aren't currently live in memory.
    pub(crate) fn list_sessions(&self) -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        for entry in self.manifest.iter() {
            let Ok((key, bytes)) = entry else { continue };
            let Ok(m): Result<Manifest, _> = serde_json::from_slice(&bytes) else {
                continue;
            };
            out.push((
                String::from_utf8_lossy(&key).to_string(),
                m.model_alias,
                m.compat_digest,
            ));
        }
        out
    }

    /// Number of tracked sessions (observability / tests).
    #[cfg(test)]
    pub(crate) fn session_count(&self) -> usize {
        self.manifest.len()
    }
}

/// SHA-256 content hash of a token sequence, hex-encoded for use as a sled key.
fn content_hash(tokens: &[TokenId]) -> String {
    let mut h = Sha256::new();
    for t in tokens {
        h.update(i32::from(*t).to_le_bytes());
    }
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::KvCacheType;

    fn dummy_ctx() -> ContextParams {
        ContextParams::default()
    }

    #[test]
    fn compat_digest_is_stable_for_same_params() {
        let a = KvStore::compat_digest("/m/x.gguf", &dummy_ctx());
        let b = KvStore::compat_digest("/m/x.gguf", &dummy_ctx());
        assert_eq!(a, b);
    }

    #[test]
    fn compat_digest_changes_with_model_path() {
        let a = KvStore::compat_digest("/m/x.gguf", &dummy_ctx());
        let b = KvStore::compat_digest("/m/y.gguf", &dummy_ctx());
        assert_ne!(a, b);
    }

    #[test]
    fn compat_digest_changes_with_rope_scale() {
        let mut ctx = dummy_ctx();
        let a = KvStore::compat_digest("/m/x.gguf", &ctx);
        ctx.rope_freq_scale = 2.0;
        let b = KvStore::compat_digest("/m/x.gguf", &ctx);
        assert_ne!(a, b);
    }

    #[test]
    fn compat_digest_changes_with_cache_dtype() {
        let mut ctx = dummy_ctx();
        let a = KvStore::compat_digest("/m/x.gguf", &ctx);
        ctx.type_k = KvCacheType::Q8_0;
        let b = KvStore::compat_digest("/m/x.gguf", &ctx);
        assert_ne!(a, b);
    }

    #[test]
    fn put_get_roundtrip_respects_compat_digest() {
        let dir = tempfile::tempdir().unwrap();
        let store = KvStore::open(&dir.path().join("kv")).unwrap();
        let ctx = dummy_ctx();
        let digest = KvStore::compat_digest("/m/x.gguf", &ctx);
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4];
        let state = vec![9u8; 16];
        store.put("sess-1", "x", &digest, &tokens, &state);
        assert_eq!(store.session_count(), 1);

        // Matching digest -> restored.
        let r = store.get("sess-1", &digest).expect("should restore");
        assert_eq!(r.state, state);

        // Different digest -> refused (None).
        let other = KvStore::compat_digest("/m/y.gguf", &ctx);
        assert!(store.get("sess-1", &other).is_none());

        // Unknown session -> None.
        assert!(store.get("sess-2", &digest).is_none());
    }

    #[test]
    fn put_dedups_identical_token_sequences() {
        let dir = tempfile::tempdir().unwrap();
        let store = KvStore::open(&dir.path().join("kv")).unwrap();
        let digest = KvStore::compat_digest("/m/x.gguf", &dummy_ctx());
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        store.put("a", "x", &digest, &tokens, &[7u8; 4]);
        store.put("b", "x", &digest, &tokens, &[7u8; 4]); // same seq -> same blob
        assert_eq!(store.session_count(), 2);
        // Both resolve to the same blob bytes.
        let ra = store.get("a", &digest).unwrap();
        let rb = store.get("b", &digest).unwrap();
        assert_eq!(ra.state, rb.state);
    }

    #[test]
    fn oversized_blob_is_not_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let store = KvStore::open(&dir.path().join("kv")).unwrap();
        let digest = KvStore::compat_digest("/m/x.gguf", &dummy_ctx());
        let huge = vec![0u8; MAX_BLOB_BYTES + 1];
        store.put("big", "x", &digest, &[1, 2], &huge);
        assert_eq!(store.session_count(), 0);
    }
}