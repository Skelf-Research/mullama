use std::sync::atomic::Ordering;

use super::super::{prompt::infer_ollama_model_config, Daemon};
use crate::daemon::models::ModelLoadConfig;
use crate::daemon::protocol::{DaemonStats, DaemonStatus, ErrorCode, ModelStatus, Response};

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

        Response::Status(DaemonStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            models_loaded: self.models.count(),
            default_model,
            http_endpoint: self
                .config
                .http_port
                .map(|p| format!("http://{}:{}", self.config.http_addr, p)),
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
                model_details: Vec::new(),
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn handle_load_model(
        &self,
        alias: String,
        path: String,
        gpu_layers: i32,
        context_size: u32,
        use_mmap: Option<bool>,
        use_mlock: bool,
        flash_attn: bool,
        cache_type_k: Option<String>,
        cache_type_v: Option<String>,
        rope_freq_base: Option<f32>,
        rope_freq_scale: Option<f32>,
        n_batch: Option<u32>,
        defrag_thold: Option<f32>,
        split_mode: Option<String>,
    ) -> Response {
        let mut resolved_context_size = if context_size == 0 {
            self.config.default_context_size
        } else {
            context_size
        };

        let mut config = ModelLoadConfig::new(&alias, &path)
            .gpu_layers(if gpu_layers == 0 {
                self.config.default_gpu_layers
            } else {
                gpu_layers
            })
            .context_size(resolved_context_size)
            .context_pool_size(self.config.default_context_pool_size)
            .threads(self.config.threads_per_model);

        if let Some(mmap) = use_mmap.or(self.config.default_use_mmap) {
            config = config.use_mmap(mmap);
        }
        if use_mlock || self.config.default_use_mlock {
            config = config.use_mlock(true);
        }
        if flash_attn || self.config.default_flash_attn {
            config = config.flash_attn(true);
        }
        if let Some(ref k) = cache_type_k.as_ref().or(self.config.default_cache_type_k.as_ref()) {
            config = config.cache_type_k(k.as_str());
        }
        if let Some(ref v) = cache_type_v.as_ref().or(self.config.default_cache_type_v.as_ref()) {
            config = config.cache_type_v(v.as_str());
        }
        if let Some(base) = rope_freq_base.or(self.config.default_rope_freq_base) {
            config = config.rope_freq_base(base);
        }
        if let Some(scale) = rope_freq_scale.or(self.config.default_rope_freq_scale) {
            config = config.rope_freq_scale(scale);
        }
        if let Some(batch) = n_batch.or(self.config.default_n_batch) {
            config = config.n_batch(batch);
        }
        if let Some(thold) = defrag_thold.or(self.config.default_defrag_thold) {
            config = config.defrag_thold(thold);
        }
        if let Some(ref mode) = split_mode.as_ref().or(self.config.default_split_mode.as_ref()) {
            config = config.split_mode(mode.as_str());
        }

        if let Some(ollama_config) = infer_ollama_model_config(&path) {
            if context_size == 0 {
                if let Some(ctx) = ollama_config.context_size {
                    resolved_context_size = ctx;
                }
                config = config.context_size(resolved_context_size);
            }
            config = config.with_config(ollama_config);
        }

        match self.models.load(config).await {
            Ok(info) => Response::ModelLoaded { alias, info },
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
