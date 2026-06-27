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

/// HTTP server configuration
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// HTTP port (None to disable)
    pub port: Option<u16>,
    /// HTTP bind address
    pub addr: String,
    /// API key (Bearer token / x-api-key)
    pub api_key: Option<String>,
    /// Enforce API key authentication
    pub enforce_api_key: bool,
    /// Maximum accepted request body size in bytes
    pub max_request_body_bytes: usize,
    /// Maximum concurrent in-flight requests
    pub max_concurrent_requests: usize,
    /// Maximum requests per second
    pub max_requests_per_second: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            port: Some(DEFAULT_HTTP_PORT),
            addr: "127.0.0.1".to_string(),
            api_key: None,
            enforce_api_key: false,
            max_request_body_bytes: 2 * 1024 * 1024,
            max_concurrent_requests: 64,
            max_requests_per_second: 200,
        }
    }
}

/// Default settings applied when loading new models
#[derive(Debug, Clone)]
pub struct ModelDefaultsConfig {
    /// Context size
    pub context_size: u32,
    /// GPU layers
    pub gpu_layers: i32,
    /// Number of contexts in each model's context pool
    pub context_pool_size: usize,
    /// Number of threads per model
    pub threads_per_model: i32,
    /// Flash attention
    pub flash_attn: bool,
    /// Memory-mapped model loading
    pub use_mmap: Option<bool>,
    /// Lock model in memory
    pub use_mlock: bool,
    /// KV cache type for keys
    pub cache_type_k: Option<String>,
    /// KV cache type for values
    pub cache_type_v: Option<String>,
    /// Batch size for prompt processing
    pub n_batch: Option<u32>,
    /// Physical micro-batch size (kernel dispatch granularity). On Metal this
    /// trades dispatch overhead vs SIMD-group occupancy; tunable per workload.
    pub n_ubatch: Option<u32>,
    /// Max concurrent sequences per context. Phase-C scaffolding: today the
    /// daemon serves one request per context, so a higher value mostly buys
    /// you cheap `kv_cache_seq_cp` for branching agentic patterns. Once the
    /// continuous-batched scheduler lands this becomes the concurrency knob.
    pub n_seq_max: Option<u32>,
    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,
    /// RoPE frequency scale
    pub rope_freq_scale: Option<f32>,
    /// KV cache defragmentation threshold
    pub defrag_thold: Option<f32>,
    /// Tensor split mode
    pub split_mode: Option<String>,
}

impl Default for ModelDefaultsConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            gpu_layers: crate::default_gpu_layers(),
            context_pool_size: DEFAULT_CONTEXT_POOL_SIZE,
            threads_per_model: (num_cpus::get() / 2).max(1) as i32,
            flash_attn: false,
            use_mmap: None,
            use_mlock: false,
            cache_type_k: None,
            cache_type_v: None,
            n_batch: None,
            n_ubatch: None,
            n_seq_max: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            defrag_thold: None,
            split_mode: None,
        }
    }
}

/// Resource limits and memory management configuration
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// Hard cap on max_tokens for generation requests
    pub max_tokens_per_request: u32,
    /// Memory monitoring configuration
    pub memory_config: MemoryConfig,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    /// Maximum number of concurrently loaded models
    pub max_loaded_models: Option<usize>,
    /// Maximum total memory for all loaded models (bytes)
    pub max_memory_bytes: Option<u64>,
    /// Model eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Auto-unload models idle for this many seconds
    pub idle_unload_secs: Option<u64>,
    /// When the background session hydrator pre-warms durable sessions into
    /// free context-pool slots. See [`HydrationMode`].
    pub hydration_mode: HydrationMode,
}

/// Controls when the background hydrator pre-warms durable-but-not-live
/// sessions into free context-pool slots.
///
/// Pre-warming reads a session's persisted KV (`load_state_seq`) into a free
/// slot so its *next* request is a hot in-memory hit instead of a cold restore.
/// The question is whether to do it only when the daemon is fully idle, or also
/// while other slots are actively decoding ("active-window" / parallel fill).
///
/// Parallel fill competes for memory bandwidth with the in-flight decode. On a
/// bandwidth-constrained box (typical x86 desktop, ~50 GB/s DDR) that slows the
/// active request, so the default there is [`HydrationMode::Idle`]. On Apple
/// Silicon — unified memory at ~100–400 GB/s plus a Metal GPU for the matmuls —
/// there is ample headroom, so the macOS default is [`HydrationMode::Active`]:
/// a waiting session's prefill overlaps another session's decode for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrationMode {
    /// Never pre-warm; sessions restore lazily on their next request.
    Off,
    /// Pre-warm only when the daemon has no active requests (safe everywhere).
    Idle,
    /// Pre-warm whenever a free slot exists, even during active decodes
    /// (parallel fill). Best on high-bandwidth hardware (Apple Silicon).
    Active,
}

impl HydrationMode {
    /// Platform-aware default: `Active` on macOS / Apple Silicon (unified
    /// high-bandwidth memory makes parallel fill free), `Idle` elsewhere.
    pub fn platform_default() -> Self {
        if cfg!(target_os = "macos") {
            HydrationMode::Active
        } else {
            HydrationMode::Idle
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_request: 4096,
            memory_config: MemoryConfig::default(),
            enable_memory_monitoring: true,
            max_loaded_models: None,
            max_memory_bytes: None,
            eviction_policy: EvictionPolicy::default(),
            idle_unload_secs: None,
            hydration_mode: HydrationMode::platform_default(),
        }
    }
}

/// Daemon server configuration
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// IPC socket address
    pub ipc_addr: String,
    /// HTTP server settings
    pub http: HttpConfig,
    /// Default model loading settings
    pub model_defaults: ModelDefaultsConfig,
    /// Resource limits and memory management
    pub resources: ResourceConfig,
    /// TLS certificate file path (enables HTTPS when set)
    pub tls_cert_path: Option<String>,
    /// TLS private key file path
    pub tls_key_path: Option<String>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            ipc_addr: DEFAULT_SOCKET.to_string(),
            http: HttpConfig::default(),
            model_defaults: ModelDefaultsConfig::default(),
            resources: ResourceConfig::default(),
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_default_is_active_on_macos_idle_elsewhere() {
        let m = HydrationMode::platform_default();
        if cfg!(target_os = "macos") {
            assert_eq!(m, HydrationMode::Active);
        } else {
            assert_eq!(m, HydrationMode::Idle);
        }
    }

    #[test]
    fn resource_config_default_uses_platform_hydration() {
        let r = ResourceConfig::default();
        assert_eq!(r.hydration_mode, HydrationMode::platform_default());
    }

    #[test]
    fn hydration_modes_are_distinct() {
        assert_ne!(HydrationMode::Off, HydrationMode::Idle);
        assert_ne!(HydrationMode::Idle, HydrationMode::Active);
        assert_ne!(HydrationMode::Off, HydrationMode::Active);
    }
}
