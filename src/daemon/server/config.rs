use crate::memory_monitor::MemoryConfig;

use super::super::{models::DEFAULT_CONTEXT_POOL_SIZE, DEFAULT_HTTP_PORT, DEFAULT_SOCKET};

/// Policy for handling model eviction when resource limits are reached
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict least-recently-used model when at limit
    Lru,
    /// Never auto-evict, return error at limit
    Manual,
    /// No limits enforced
    None,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::Lru
    }
}

/// Daemon server configuration
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// IPC socket address
    pub ipc_addr: String,
    /// HTTP port (None to disable)
    pub http_port: Option<u16>,
    /// HTTP bind address
    pub http_addr: String,
    /// Default context size for new models
    pub default_context_size: u32,
    /// Default GPU layers
    pub default_gpu_layers: i32,
    /// Number of contexts in each model's context pool
    pub default_context_pool_size: usize,
    /// Number of threads per model
    pub threads_per_model: i32,
    /// HTTP API key (Bearer token / x-api-key)
    pub http_api_key: Option<String>,
    /// Enforce API key authentication for HTTP endpoints
    pub enforce_http_api_key: bool,
    /// Hard cap on max_tokens for generation requests
    pub max_tokens_per_request: u32,
    /// Maximum accepted HTTP request body size in bytes
    pub max_request_body_bytes: usize,
    /// Maximum concurrent in-flight HTTP requests
    pub max_concurrent_http_requests: usize,
    /// Maximum requests per second for HTTP endpoints
    pub max_requests_per_second: u64,
    /// Memory monitoring configuration
    pub memory_config: MemoryConfig,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    /// TLS certificate file path (enables HTTPS when set)
    pub tls_cert_path: Option<String>,
    /// TLS private key file path
    pub tls_key_path: Option<String>,
    /// Default flash attention setting for new models
    pub default_flash_attn: bool,
    /// Default use_mmap setting for new models
    pub default_use_mmap: Option<bool>,
    /// Default use_mlock setting for new models
    pub default_use_mlock: bool,
    /// Default KV cache type for keys
    pub default_cache_type_k: Option<String>,
    /// Default KV cache type for values
    pub default_cache_type_v: Option<String>,
    /// Default batch size for prompt processing
    pub default_n_batch: Option<u32>,
    /// Default RoPE frequency base
    pub default_rope_freq_base: Option<f32>,
    /// Default RoPE frequency scale
    pub default_rope_freq_scale: Option<f32>,
    /// Default KV cache defragmentation threshold
    pub default_defrag_thold: Option<f32>,
    /// Default tensor split mode
    pub default_split_mode: Option<String>,
    /// Maximum number of concurrently loaded models
    pub max_loaded_models: Option<usize>,
    /// Maximum total memory for all loaded models (bytes)
    pub max_memory_bytes: Option<u64>,
    /// Model eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Auto-unload models idle for this many seconds
    pub idle_unload_secs: Option<u64>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            ipc_addr: DEFAULT_SOCKET.to_string(),
            http_port: Some(DEFAULT_HTTP_PORT),
            http_addr: "127.0.0.1".to_string(),
            default_context_size: 4096,
            default_gpu_layers: 0,
            default_context_pool_size: DEFAULT_CONTEXT_POOL_SIZE,
            threads_per_model: (num_cpus::get() / 2).max(1) as i32,
            http_api_key: None,
            enforce_http_api_key: false,
            max_tokens_per_request: 4096,
            max_request_body_bytes: 2 * 1024 * 1024,
            max_concurrent_http_requests: 64,
            max_requests_per_second: 200,
            memory_config: MemoryConfig::default(),
            enable_memory_monitoring: true,
            tls_cert_path: None,
            tls_key_path: None,
            default_flash_attn: false,
            default_use_mmap: None,
            default_use_mlock: false,
            default_cache_type_k: None,
            default_cache_type_v: None,
            default_n_batch: None,
            default_rope_freq_base: None,
            default_rope_freq_scale: None,
            default_defrag_thold: None,
            default_split_mode: None,
            max_loaded_models: None,
            max_memory_bytes: None,
            eviction_policy: EvictionPolicy::default(),
            idle_unload_secs: None,
        }
    }
}
