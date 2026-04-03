//! Daemon server implementation
//!
//! Core daemon that manages models and handles requests from IPC and HTTP.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;

mod builder;
mod config;
mod generation;
mod handlers;
mod prompt;

pub use builder::DaemonBuilder;
pub use config::{DaemonConfig, EvictionPolicy};

use super::models::ModelManager;
use super::protocol::*;
use super::store::DaemonStore;
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
    /// Memory monitor for tracking system and GPU memory
    pub memory_monitor: Option<Arc<MemoryMonitor>>,
    /// Recovery manager for handling OOM situations
    pub recovery_manager: RecoveryManager,
    /// Persistent store for daemon state
    pub store: Arc<DaemonStore>,
}

impl Daemon {
    /// Create a new daemon
    pub fn new(config: DaemonConfig) -> Self {
        let memory_monitor = if config.enable_memory_monitoring {
            let monitor = MemoryMonitor::new(config.memory_config.clone());
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

        let store = match DaemonStore::open_default() {
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

        Self {
            config,
            models: Arc::new(ModelManager::new()),
            start_time: Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            active_requests: Arc::new(AtomicU32::new(0)),
            total_requests: AtomicU64::new(0),
            cancellations: Arc::new(DashMap::new()),
            memory_monitor,
            recovery_manager,
            store,
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

        if max_tokens > self.config.max_tokens_per_request {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                format!(
                    "max_tokens {} exceeds server limit {}",
                    max_tokens, self.config.max_tokens_per_request
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

    /// Handle a request
    pub async fn handle_request(&self, request: Request) -> Response {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match request {
            Request::Ping => Response::Pong {
                uptime_secs: self.start_time.elapsed().as_secs(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Request::Status => self.handle_status().await,
            Request::ListModels => self.handle_list_models().await,

            Request::LoadModel {
                alias,
                path,
                gpu_layers,
                context_size,
                use_mmap,
                use_mlock,
                flash_attn,
                cache_type_k,
                cache_type_v,
                rope_freq_base,
                rope_freq_scale,
                n_batch,
                defrag_thold,
                split_mode,
            } => {
                self.handle_load_model(
                    alias,
                    path,
                    gpu_layers,
                    context_size,
                    use_mmap,
                    use_mlock,
                    flash_attn,
                    cache_type_k,
                    cache_type_v,
                    rope_freq_base,
                    rope_freq_scale,
                    n_batch,
                    defrag_thold,
                    split_mode,
                )
                .await
            }

            Request::UnloadModel { alias } => self.handle_unload_model(&alias).await,
            Request::SetDefaultModel { alias } => self.handle_set_default(&alias).await,

            Request::ChatCompletion {
                model,
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                stream,
                stop,
                response_format,
                tools: _,
                tool_choice: _,
                thinking: _,
            } => {
                self.handle_chat_completion(
                    model,
                    messages,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    presence_penalty,
                    stream,
                    stop,
                    response_format,
                )
                .await
            }

            Request::Completion {
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                stream,
                stop,
            } => {
                self.handle_completion(
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    presence_penalty,
                    stream,
                    stop,
                )
                .await
            }

            Request::Embeddings { model, input } => self.handle_embeddings(model, input).await,

            Request::Tokenize { model, text } => self.handle_tokenize(model, &text).await,

            Request::Cancel { request_id } => {
                if self.cancel_request(&request_id) {
                    Response::Cancelled { request_id }
                } else {
                    Response::error(
                        ErrorCode::InvalidRequest,
                        format!("No active request found with id '{}'", request_id),
                    )
                }
            }

            Request::Shutdown => {
                self.shutdown.store(true, Ordering::SeqCst);
                Response::ShuttingDown
            }
        }
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
            enable_memory_monitoring: false,
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
        let pos = super::prompt::find_stop_in_recent_window(
            generated,
            previous_len,
            &stop_sequences,
            10,
        );
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
