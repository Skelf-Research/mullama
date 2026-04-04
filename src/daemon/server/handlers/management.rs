use std::sync::atomic::Ordering;

use super::super::{prompt::infer_ollama_model_config, Daemon};
use crate::daemon::models::ModelLoadConfig;
use crate::daemon::protocol::{
    DaemonStats, DaemonStatus, ErrorCode, ModelDetailedStats, ModelLoadParams, ModelStatus,
    Response,
};

impl Daemon {
    pub(crate) async fn handle_status(&self) -> Response {
        let default_model = self.models.default_alias();

        let memory_used_mb = self
            .memory_monitor
            .as_ref()
            .map(|m| {
                let stats = m.stats();
                let used = if stats.gpu_total > 0 {
                    stats.gpu_used
                } else {
                    stats.system_used
                };
                used / (1024 * 1024)
            })
            .unwrap_or(0);

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let model_details: Vec<ModelDetailedStats> = self
            .store
            .all_model_stats()
            .into_iter()
            .map(|(alias, ps)| {
                let avg_tps = if ps.avg_tokens_per_sec_x100 > 0 {
                    ps.avg_tokens_per_sec_x100 as f32 / 100.0
                } else {
                    0.0
                };
                let last_used_secs_ago = now_secs.saturating_sub(ps.last_used);
                let avg_load_time_ms = if ps.load_count > 0 {
                    ps.total_load_time_ms / ps.load_count
                } else {
                    0
                };
                ModelDetailedStats {
                    alias,
                    requests_total: ps.requests_total,
                    tokens_generated: ps.tokens_generated,
                    tokens_prompt: ps.tokens_prompt,
                    avg_tokens_per_sec: avg_tps,
                    memory_bytes: 0,
                    active_requests: 0,
                    last_used_secs_ago,
                    load_time_ms: avg_load_time_ms,
                    pool_size: 0,
                }
            })
            .collect();

        Response::Status(DaemonStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            models_loaded: self.models.count(),
            default_model,
            http_endpoint: self
                .config
                .http
                .port
                .map(|p| format!("http://{}:{}", self.config.http.addr, p)),
            ipc_endpoint: self.config.ipc_addr.clone(),
            stats: DaemonStats {
                requests_total: self.total_requests.load(Ordering::Relaxed),
                tokens_generated: self.models.total_tokens(),
                active_requests: self.active_requests.load(Ordering::Relaxed),
                memory_used_mb,
                gpu_available: crate::supports_gpu_offload(),
                memory_total_mb: 0,
                memory_available_mb: 0,
                memory_pressure: String::new(),
                model_details,
            },
        })
    }

    pub(crate) async fn handle_list_models(&self) -> Response {
        let models = self.models.list();
        Response::Models(
            models
                .into_iter()
                .map(|(alias, info, is_default, active)| ModelStatus {
                    alias,
                    info,
                    is_default,
                    active_requests: active,
                })
                .collect(),
        )
    }

    pub(crate) async fn handle_load_model(&self, params: ModelLoadParams) -> Response {
        let load_start = std::time::Instant::now();
        let md = &self.config.model_defaults;
        let mut resolved_context_size = if params.context_size == 0 {
            md.context_size
        } else {
            params.context_size
        };

        let mut config = ModelLoadConfig::new(&params.alias, &params.path)
            .gpu_layers(if params.gpu_layers == 0 {
                md.gpu_layers
            } else {
                params.gpu_layers
            })
            .context_size(resolved_context_size)
            .context_pool_size(md.context_pool_size)
            .threads(md.threads_per_model);

        if let Some(mmap) = params.use_mmap.or(md.use_mmap) {
            config = config.use_mmap(mmap);
        }
        if params.use_mlock || md.use_mlock {
            config = config.use_mlock(true);
        }
        if params.flash_attn || md.flash_attn {
            config = config.flash_attn(true);
        }
        if let Some(ref k) = params.cache_type_k.as_ref().or(md.cache_type_k.as_ref()) {
            config = config.cache_type_k(k.as_str());
        }
        if let Some(ref v) = params.cache_type_v.as_ref().or(md.cache_type_v.as_ref()) {
            config = config.cache_type_v(v.as_str());
        }
        if let Some(base) = params.rope_freq_base.or(md.rope_freq_base) {
            config = config.rope_freq_base(base);
        }
        if let Some(scale) = params.rope_freq_scale.or(md.rope_freq_scale) {
            config = config.rope_freq_scale(scale);
        }
        if let Some(batch) = params.n_batch.or(md.n_batch) {
            config = config.n_batch(batch);
        }
        if let Some(thold) = params.defrag_thold.or(md.defrag_thold) {
            config = config.defrag_thold(thold);
        }
        if let Some(ref mode) = params.split_mode.as_ref().or(md.split_mode.as_ref()) {
            config = config.split_mode(mode.as_str());
        }

        if let Some(ollama_config) = infer_ollama_model_config(&params.path) {
            if params.context_size == 0 {
                if let Some(ctx) = ollama_config.context_size {
                    resolved_context_size = ctx;
                }
                config = config.context_size(resolved_context_size);
            }
            config = config.with_config(ollama_config);
        }

        match self.models.load(config).await {
            Ok(info) => {
                let elapsed = load_start.elapsed();
                self.store.record_model_load(&params.alias, elapsed.as_millis() as u64);
                Response::ModelLoaded {
                    alias: params.alias,
                    info,
                }
            }
            Err(e) => Response::error(ErrorCode::ModelLoadFailed, e.to_string()),
        }
    }

    pub(crate) async fn handle_unload_model(&self, alias: &str) -> Response {
        match self.models.unload(alias).await {
            Ok(()) => Response::ModelUnloaded {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    pub(crate) async fn handle_set_default(&self, alias: &str) -> Response {
        match self.models.set_default(alias).await {
            Ok(()) => Response::DefaultModelSet {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }
}
