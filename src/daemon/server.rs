//! Daemon server implementation
//!
//! Core daemon that manages models and handles requests from IPC and HTTP.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;

mod batcher;
mod builder;
mod config;
mod dispatch;
mod generation;
mod handlers;
mod hydrator;
mod kvstore;
mod prefetch;
mod prompt;
mod session;

pub(crate) use batcher::{spawn as spawn_batcher, BatchRestore, BatchTask, BatcherHandle, ReplyMode};
pub use builder::DaemonBuilder;
pub(crate) use kvstore::KvStore;
pub(crate) use prefetch::PrefetchObserver;
pub(crate) use session::SessionStore;
pub use config::{
    DaemonConfig, EvictionPolicy, HttpConfig, HydrationMode, ModelDefaultsConfig, ResourceConfig,
};
pub(crate) use prompt::resolve_chat_stop_sequences;

use super::models::ModelManager;
use super::protocol::*;
use super::store::{DaemonStore, StorageBackend};
use crate::memory_monitor::{MemoryMonitor, MemoryPressure, RecoveryManager};

/// The daemon server
pub struct Daemon {
    pub config: DaemonConfig,
    pub models: Arc<ModelManager>,
    pub start_time: Instant,
    pub shutdown: Arc<AtomicBool>,
    pub active_requests: Arc<AtomicU32>,
    pub total_requests: AtomicU64,
    /// Cancellation flags for streaming requests (request_id -> cancel flag)
    pub cancellations: Arc<DashMap<String, Arc<AtomicBool>>>,
    /// Cross-turn KV reuse sessions (session id -> cached tokens + pinned pool
    /// slot). Lets an agent loop prefill only the new delta each turn instead
    /// of re-decoding the whole history.
    pub sessions: Arc<SessionStore>,
    /// Per-session agent file-access tracker. Records source paths mentioned in
    /// conversation turns so the idle hydrator can predict (and pre-warm) the
    /// files an agent is about to read next. See [`prefetch`].
    pub(crate) prefetch: Arc<PrefetchObserver>,
    /// Memory monitor for tracking system and GPU memory
    pub memory_monitor: Option<Arc<MemoryMonitor>>,
    /// Recovery manager for handling OOM situations
    pub recovery_manager: RecoveryManager,
    /// Persistent store for daemon state (pluggable via [`StorageBackend`] trait)
    pub store: Arc<dyn StorageBackend>,
    /// Model providers for resolving model specs to local paths
    pub providers: Vec<Box<dyn super::provider::ModelProvider>>,
}

impl Daemon {
    /// Create a new daemon
    pub fn new(config: DaemonConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));

        let memory_monitor = if config.resources.enable_memory_monitoring {
            let monitor = MemoryMonitor::new(config.resources.memory_config.clone());
            monitor.start();
            Some(monitor)
        } else {
            None
        };

        let recovery_manager = if let Some(ref monitor) = memory_monitor {
            RecoveryManager::new().with_monitor(Arc::clone(monitor))
        } else {
            RecoveryManager::new()
        };

        let store: Arc<dyn StorageBackend> = match DaemonStore::open_default() {
            Ok(s) => Arc::new(s),
            Err(e) => {
                eprintln!(
                    "Warning: Failed to open persistent store: {}. Using in-memory fallback.",
                    e
                );
                let tmp = tempfile::tempdir().expect("Failed to create temp dir");
                Arc::new(
                    DaemonStore::open(&tmp.path().join("mullama.db"))
                        .expect("Failed to create temp store"),
                )
            }
        };

        let mut providers: Vec<Box<dyn super::provider::ModelProvider>> = Vec::new();
        if let Ok(ollama) = super::ollama::OllamaClient::new() {
            providers.push(Box::new(ollama));
        }
        if let Ok(hf) = crate::hf::HfDownloader::new() {
            providers.push(Box::new(hf));
        }

        let models = Arc::new(ModelManager::new());
        models.set_shutdown(shutdown.clone());
        if let Some(ref monitor) = memory_monitor {
            models.set_memory_monitor(Arc::clone(monitor));
        }

        Self {
            config,
            models,
            start_time: Instant::now(),
            shutdown,
            active_requests: Arc::new(AtomicU32::new(0)),
            total_requests: AtomicU64::new(0),
            cancellations: Arc::new(DashMap::new()),
            sessions: match kvstore::KvStore::open_default() {
                Ok(kv) => Arc::new(SessionStore::new().with_kv(Arc::new(kv))),
                Err(e) => {
                    eprintln!(
                        "Warning: durable KV store disabled (sled open failed: {}). \
                         Cross-turn reuse still works in-process; restarts will re-prefill.",
                        e
                    );
                    Arc::new(SessionStore::new())
                }
            },
            prefetch: Arc::new(PrefetchObserver::new()),
            memory_monitor,
            recovery_manager,
            store,
            providers,
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_max_tokens(&self, max_tokens: u32) -> Result<(), Response> {
        if max_tokens == 0 {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "max_tokens must be greater than 0",
            ));
        }

        if max_tokens > self.config.resources.max_tokens_per_request {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                format!(
                    "max_tokens {} exceeds server limit {}",
                    max_tokens, self.config.resources.max_tokens_per_request
                ),
            ));
        }

        Ok(())
    }

    fn register_cancellation(&self, request_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.cancellations
            .insert(request_id.to_string(), Arc::clone(&flag));
        flag
    }

    pub fn cancel_request(&self, request_id: &str) -> bool {
        if let Some(flag) = self.cancellations.get(request_id) {
            flag.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Resolve a model spec to a local path using the provider chain.
    ///
    /// Iterates through registered providers (Ollama, HuggingFace) and returns
    /// the first successful resolution. Falls through on provider errors.
    pub async fn resolve_model_spec(
        &self,
        spec: &str,
    ) -> Result<super::provider::ResolvedModelPath, crate::MullamaError> {
        for provider in &self.providers {
            if provider.supports(spec) {
                match provider.resolve(spec).await {
                    Ok(resolved) => return Ok(resolved),
                    Err(e) => {
                        tracing::warn!(
                            "Provider '{}' failed for '{}': {}",
                            provider.name(),
                            spec,
                            e
                        );
                        continue;
                    }
                }
            }
        }
        Err(crate::MullamaError::OperationFailed(format!(
            "No provider can resolve: {}",
            spec
        )))
    }

    /// Get current memory pressure level
    pub fn memory_pressure(&self) -> MemoryPressure {
        self.memory_monitor
            .as_ref()
            .map(|m| m.pressure())
            .unwrap_or(MemoryPressure::Normal)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Option<crate::memory_monitor::MemoryStats> {
        self.memory_monitor.as_ref().map(|m| m.stats())
    }

    /// Check if memory recovery is needed
    pub fn needs_memory_recovery(&self) -> bool {
        self.recovery_manager.needs_recovery()
    }

    /// Log memory warning if pressure is elevated
    #[allow(dead_code)]
    fn log_memory_pressure(&self) {
        if let Some(monitor) = &self.memory_monitor {
            let pressure = monitor.pressure();
            let stats = monitor.stats();

            match pressure {
                MemoryPressure::Warning => {
                    tracing::warn!(
                        gpu_usage = stats.gpu_usage() * 100.0,
                        system_usage = stats.system_usage() * 100.0,
                        "Memory pressure elevated"
                    );
                }
                MemoryPressure::Critical => {
                    tracing::error!(
                        gpu_usage = stats.gpu_usage() * 100.0,
                        system_usage = stats.system_usage() * 100.0,
                        "Memory pressure CRITICAL"
                    );
                }
                MemoryPressure::Emergency => {
                    tracing::error!(
                        gpu_usage = stats.gpu_usage() * 100.0,
                        system_usage = stats.system_usage() * 100.0,
                        "Memory EMERGENCY - recovery needed"
                    );
                }
                MemoryPressure::Normal => {}
            }
        }
    }

    /// Check if shutdown was requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_daemon() -> Daemon {
        let config = DaemonConfig {
            resources: ResourceConfig {
                enable_memory_monitoring: false,
                ..ResourceConfig::default()
            },
            ..DaemonConfig::default()
        };
        Daemon::new(config)
    }

    #[test]
    fn merge_stop_sequences_deduplicates_and_filters_empty() {
        let merged = super::prompt::merge_stop_sequences(
            vec!["</s>".to_string(), "".to_string()],
            vec!["<|eot_id|>".to_string(), "</s>".to_string()],
        );
        assert_eq!(merged, vec!["</s>", "<|eot_id|>"]);
    }

    #[test]
    fn find_stop_in_recent_window_detects_cross_token_boundary() {
        let generated = "hello<|eot_id|>";
        let previous_len = "hello<|eo".len();
        let stop_sequences = vec!["<|eot_id|>".to_string()];
        let pos =
            super::prompt::find_stop_in_recent_window(generated, previous_len, &stop_sequences, 10);
        assert_eq!(pos, Some("hello".len()));
    }

    #[test]
    fn apply_default_system_prompt_only_when_missing() {
        let daemon = test_daemon();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string().into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let with_system =
            daemon.apply_default_system_prompt(messages.clone(), Some("You are helpful."));
        assert_eq!(with_system.len(), 2);
        assert_eq!(with_system[0].role, "system");

        let with_existing = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "existing".to_string().into(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            messages[0].clone(),
        ];
        let unchanged = daemon.apply_default_system_prompt(with_existing.clone(), Some("ignored"));
        assert_eq!(unchanged.len(), with_existing.len());
        assert_eq!(unchanged[0].content.text(), "existing");
    }
}
