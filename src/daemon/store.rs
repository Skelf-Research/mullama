//! Persistent storage for daemon state using sled embedded database.
//!
//! Stores per-model stats, model metadata cache, prompt token cache,
//! conversation sessions, and daemon configuration across restarts.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Persistent store backed by sled embedded database
pub struct DaemonStore {
    db: sled::Db,
    model_stats: sled::Tree,
    model_meta: sled::Tree,
    prompt_cache: sled::Tree,
    sessions: sled::Tree,
    config: sled::Tree,
}

/// Persisted per-model statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedModelStats {
    pub requests_total: u64,
    pub tokens_generated: u64,
    pub tokens_prompt: u64,
    pub avg_tokens_per_sec_x100: u64,
    pub load_count: u64,
    pub last_used: u64,
    pub total_load_time_ms: u64,
}

/// Cached model metadata (from GGUF headers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModelMeta {
    pub n_params: u64,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub file_size: u64,
    pub quant_type: Option<String>,
    pub mtime: u64,
}

/// A stored conversation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSession {
    pub id: String,
    pub model: String,
    pub messages: Vec<StoredMessage>,
    pub created_at: u64,
    pub updated_at: u64,
}

/// A stored chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
}

impl DaemonStore {
    /// Open or create the store at the default location (~/.mullama/db/)
    pub fn open_default() -> Result<Self, sled::Error> {
        let db_path = Self::default_path();
        Self::open(&db_path)
    }

    /// Open or create the store at a specific path
    pub fn open(path: &PathBuf) -> Result<Self, sled::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let db = sled::open(path)?;
        let model_stats = db.open_tree("model_stats")?;
        let model_meta = db.open_tree("model_meta")?;
        let prompt_cache = db.open_tree("prompt_cache")?;
        let sessions = db.open_tree("sessions")?;
        let config = db.open_tree("config")?;

        Ok(Self {
            db,
            model_stats,
            model_meta,
            prompt_cache,
            sessions,
            config,
        })
    }

    /// Get the default database path
    pub fn default_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mullama")
            .join("db")
    }

    // ==================== Model Stats ====================

    /// Get persisted stats for a model alias
    pub fn get_model_stats(&self, alias: &str) -> Option<PersistedModelStats> {
        self.model_stats
            .get(alias.as_bytes())
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// Update persisted stats for a model alias (merge with existing)
    pub fn update_model_stats(
        &self,
        alias: &str,
        requests_delta: u64,
        tokens_generated_delta: u64,
        tokens_prompt_delta: u64,
        avg_tps_x100: u64,
    ) {
        let mut stats = self.get_model_stats(alias).unwrap_or_default();
        stats.requests_total += requests_delta;
        stats.tokens_generated += tokens_generated_delta;
        stats.tokens_prompt += tokens_prompt_delta;
        if avg_tps_x100 > 0 {
            // Running average
            if stats.avg_tokens_per_sec_x100 == 0 {
                stats.avg_tokens_per_sec_x100 = avg_tps_x100;
            } else {
                stats.avg_tokens_per_sec_x100 =
                    (stats.avg_tokens_per_sec_x100 * 3 + avg_tps_x100) / 4;
            }
        }
        stats.last_used = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Ok(bytes) = serde_json::to_vec(&stats) {
            let _ = self.model_stats.insert(alias.as_bytes(), bytes);
        }
    }

    /// Record a model load event
    pub fn record_model_load(&self, alias: &str, load_time_ms: u64) {
        let mut stats = self.get_model_stats(alias).unwrap_or_default();
        stats.load_count += 1;
        stats.total_load_time_ms += load_time_ms;
        stats.last_used = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Ok(bytes) = serde_json::to_vec(&stats) {
            let _ = self.model_stats.insert(alias.as_bytes(), bytes);
        }
    }

    /// Get all persisted model stats
    pub fn all_model_stats(&self) -> Vec<(String, PersistedModelStats)> {
        self.model_stats
            .iter()
            .filter_map(|item| {
                let (key, value) = item.ok()?;
                let alias = String::from_utf8(key.to_vec()).ok()?;
                let stats: PersistedModelStats = serde_json::from_slice(&value).ok()?;
                Some((alias, stats))
            })
            .collect()
    }

    // ==================== Model Metadata Cache ====================

    /// Get cached metadata for a model file
    pub fn get_model_meta(&self, path: &str, mtime: u64) -> Option<CachedModelMeta> {
        let key = format!("{}:{}", path, mtime);
        self.model_meta
            .get(key.as_bytes())
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// Cache metadata for a model file
    pub fn set_model_meta(&self, path: &str, meta: &CachedModelMeta) {
        let key = format!("{}:{}", path, meta.mtime);
        if let Ok(bytes) = serde_json::to_vec(meta) {
            let _ = self.model_meta.insert(key.as_bytes(), bytes);
        }
    }

    // ==================== Prompt Cache ====================

    /// Get cached token IDs for a prompt hash
    pub fn get_prompt_tokens(&self, hash: &[u8]) -> Option<Vec<i32>> {
        self.prompt_cache
            .get(hash)
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// Cache token IDs for a prompt hash
    pub fn set_prompt_tokens(&self, hash: &[u8], tokens: &[i32]) {
        if let Ok(bytes) = serde_json::to_vec(tokens) {
            let _ = self.prompt_cache.insert(hash, bytes);
        }
    }

    // ==================== Sessions ====================

    /// Save a conversation session
    pub fn save_session(&self, session: &StoredSession) {
        if let Ok(bytes) = serde_json::to_vec(session) {
            let _ = self.sessions.insert(session.id.as_bytes(), bytes);
        }
    }

    /// Get a conversation session by ID
    pub fn get_session(&self, id: &str) -> Option<StoredSession> {
        self.sessions
            .get(id.as_bytes())
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// List all sessions (most recent first)
    pub fn list_sessions(&self) -> Vec<StoredSession> {
        let mut sessions: Vec<StoredSession> = self
            .sessions
            .iter()
            .filter_map(|item| {
                let (_key, value) = item.ok()?;
                serde_json::from_slice(&value).ok()
            })
            .collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    /// Delete a session
    pub fn delete_session(&self, id: &str) {
        let _ = self.sessions.remove(id.as_bytes());
    }

    // ==================== Config ====================

    /// Get a config value
    pub fn get_config(&self, key: &str) -> Option<String> {
        self.config
            .get(key.as_bytes())
            .ok()
            .flatten()
            .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok())
    }

    /// Set a config value
    pub fn set_config(&self, key: &str, value: &str) {
        let _ = self.config.insert(key.as_bytes(), value.as_bytes());
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) {
        let _ = self.db.flush();
    }
}

impl Drop for DaemonStore {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_model_stats() {
        let dir = tempfile::tempdir().unwrap();
        let store = DaemonStore::open(&dir.path().join("test.db")).unwrap();

        assert!(store.get_model_stats("test").is_none());

        store.update_model_stats("test", 10, 500, 100, 4500);
        let stats = store.get_model_stats("test").unwrap();
        assert_eq!(stats.requests_total, 10);
        assert_eq!(stats.tokens_generated, 500);
        assert_eq!(stats.tokens_prompt, 100);

        // Merge additional stats
        store.update_model_stats("test", 5, 200, 50, 5000);
        let stats = store.get_model_stats("test").unwrap();
        assert_eq!(stats.requests_total, 15);
        assert_eq!(stats.tokens_generated, 700);
    }

    #[test]
    fn test_store_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let store = DaemonStore::open(&dir.path().join("test.db")).unwrap();

        let session = StoredSession {
            id: "sess_1".to_string(),
            model: "llama3".to_string(),
            messages: vec![
                StoredMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                },
                StoredMessage {
                    role: "assistant".to_string(),
                    content: "Hi there!".to_string(),
                },
            ],
            created_at: 1000,
            updated_at: 2000,
        };
        store.save_session(&session);

        let loaded = store.get_session("sess_1").unwrap();
        assert_eq!(loaded.model, "llama3");
        assert_eq!(loaded.messages.len(), 2);
    }

    #[test]
    fn test_store_config() {
        let dir = tempfile::tempdir().unwrap();
        let store = DaemonStore::open(&dir.path().join("test.db")).unwrap();

        store.set_config("last_gpu_layers", "33");
        assert_eq!(store.get_config("last_gpu_layers").unwrap(), "33");
    }
}
