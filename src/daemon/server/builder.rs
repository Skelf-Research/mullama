use crate::memory_monitor::MemoryConfig;

use super::super::models::ModelLoadConfig;
use super::{Daemon, DaemonConfig};

/// Builder for daemon configuration
pub struct DaemonBuilder {
    config: DaemonConfig,
    initial_models: Vec<ModelLoadConfig>,
}

impl DaemonBuilder {
    pub fn new() -> Self {
        Self {
            config: DaemonConfig::default(),
            initial_models: Vec::new(),
        }
    }

    pub fn ipc_socket(mut self, addr: impl Into<String>) -> Self {
        self.config.ipc_addr = addr.into();
        self
    }

    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = Some(port);
        self
    }

    pub fn disable_http(mut self) -> Self {
        self.config.http_port = None;
        self
    }

    pub fn http_addr(mut self, addr: impl Into<String>) -> Self {
        self.config.http_addr = addr.into();
        self
    }

    pub fn http_api_key(mut self, api_key: Option<String>) -> Self {
        self.config.http_api_key = api_key;
        self
    }

    pub fn enforce_http_api_key(mut self, enforce: bool) -> Self {
        self.config.enforce_http_api_key = enforce;
        self
    }

    pub fn max_tokens_per_request(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens_per_request = max_tokens;
        self
    }

    pub fn max_request_body_bytes(mut self, bytes: usize) -> Self {
        self.config.max_request_body_bytes = bytes;
        self
    }

    pub fn max_concurrent_http_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_http_requests = max;
        self
    }

    pub fn max_requests_per_second(mut self, max: u64) -> Self {
        self.config.max_requests_per_second = max;
        self
    }

    pub fn default_context_size(mut self, size: u32) -> Self {
        self.config.default_context_size = size;
        self
    }

    pub fn default_gpu_layers(mut self, layers: i32) -> Self {
        self.config.default_gpu_layers = layers;
        self
    }

    pub fn default_context_pool_size(mut self, size: usize) -> Self {
        self.config.default_context_pool_size = size.max(1);
        self
    }

    pub fn threads_per_model(mut self, threads: i32) -> Self {
        self.config.threads_per_model = threads;
        self
    }

    /// Configure memory monitoring
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory_config = config;
        self
    }

    /// Enable or disable memory monitoring
    pub fn enable_memory_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_memory_monitoring = enable;
        self
    }

    /// Add a model to load on startup (format: "alias:path" or just "path")
    pub fn model(mut self, spec: impl Into<String>) -> Self {
        let spec = spec.into();
        let (alias, path) = if let Some(pos) = spec.find(':') {
            (spec[..pos].to_string(), spec[pos + 1..].to_string())
        } else {
            let path = std::path::Path::new(&spec);
            let alias = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".to_string());
            (alias, spec)
        };

        self.initial_models.push(
            ModelLoadConfig::new(alias, path)
                .gpu_layers(self.config.default_gpu_layers)
                .context_size(self.config.default_context_size)
                .context_pool_size(self.config.default_context_pool_size)
                .threads(self.config.threads_per_model),
        );
        self
    }

    pub fn build(self) -> (Daemon, Vec<ModelLoadConfig>) {
        (Daemon::new(self.config), self.initial_models)
    }
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self::new()
    }
}
