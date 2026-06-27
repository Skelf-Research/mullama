/// Default number of contexts in the pool per model.
///
/// On macOS we default to **1**. Apple's Metal driver has a single command
/// queue per GPU device, and llama.cpp's GGML CPU threadpool over-subscribes
/// the cores when N contexts call `llama_decode` concurrently (`N * n_threads`
/// workers fighting for ~num_cpu cores). The combined effect is pathological:
/// in the M1 bench, `--context-pool-size 2` with 4 in-flight requests took
/// 811 s vs 13.6 s with `--context-pool-size 1` (clean serialization at the
/// slot's write lock). Until single-context multi-seq batched decode lands
/// (one `llama_decode` interleaving `n_seq_max>1` sessions, matching ollama's
/// design), pool=1 is the only stable concurrency story on Metal.
///
/// Elsewhere we keep the historical default of 4. On CUDA/ROCm each context
/// gets its own CUDA stream and the GPU schedules them in parallel, so the
/// queue-serialization pathology doesn't apply.
#[cfg(target_os = "macos")]
pub const DEFAULT_CONTEXT_POOL_SIZE: usize = 1;
#[cfg(not(target_os = "macos"))]
pub const DEFAULT_CONTEXT_POOL_SIZE: usize = 4;

pub(super) fn detect_quantization_from_path(path: &str) -> Option<String> {
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())?
        .to_ascii_uppercase();

    let known = [
        "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S",
        "Q6_K", "Q8_0", "F16", "F32",
    ];

    for q in known {
        if filename.contains(q) {
            return Some(q.to_string());
        }
    }

    None
}

/// Runtime configuration for a loaded model (from Ollama, Modelfile, or defaults)
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub context_size: Option<u32>,
}

/// Configuration for loading a model
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub alias: String,
    pub path: String,
    pub gpu_layers: i32,
    pub context_size: u32,
    pub threads: i32,
    /// Number of contexts to keep in the per-model pool.
    pub context_pool_size: usize,
    /// Path to multimodal projector file (mmproj) for vision/audio models
    pub mmproj_path: Option<String>,
    /// Runtime configuration from Ollama registry or Modelfile
    pub model_config: Option<ModelConfig>,
    /// Use memory-mapped file for model weights
    pub use_mmap: Option<bool>,
    /// Lock model weights in memory
    pub use_mlock: bool,
    /// Enable flash attention
    pub flash_attn: bool,
    /// KV cache type for keys (default: f16)
    pub cache_type_k: Option<String>,
    /// KV cache type for values (default: f16)
    pub cache_type_v: Option<String>,
    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,
    /// RoPE frequency scale
    pub rope_freq_scale: Option<f32>,
    /// Batch size for prompt processing
    pub n_batch: Option<u32>,
    /// Physical micro-batch size on Metal/CUDA. Controls kernel-dispatch
    /// granularity; the right value is hardware-specific.
    pub n_ubatch: Option<u32>,
    /// Max concurrent sequences per context (Phase-C prerequisite).
    pub n_seq_max: Option<u32>,
    /// KV cache defragmentation threshold
    pub defrag_thold: Option<f32>,
    /// Tensor split mode for multi-GPU
    pub split_mode: Option<String>,
}

impl ModelLoadConfig {
    pub fn new(alias: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            path: path.into(),
            gpu_layers: 0,
            context_size: 4096,
            threads: num_cpus::get() as i32,
            context_pool_size: DEFAULT_CONTEXT_POOL_SIZE,
            mmproj_path: None,
            model_config: None,
            use_mmap: None,
            use_mlock: false,
            flash_attn: false,
            cache_type_k: None,
            cache_type_v: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            n_batch: None,
            n_ubatch: None,
            n_seq_max: None,
            defrag_thold: None,
            split_mode: None,
        }
    }

    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    pub fn context_size(mut self, size: u32) -> Self {
        self.context_size = size;
        self
    }

    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    pub fn context_pool_size(mut self, size: usize) -> Self {
        self.context_pool_size = size.max(1);
        self
    }

    pub fn mmproj(mut self, path: impl Into<String>) -> Self {
        self.mmproj_path = Some(path.into());
        self
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    pub fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = Some(use_mmap);
        self
    }

    pub fn use_mlock(mut self, mlock: bool) -> Self {
        self.use_mlock = mlock;
        self
    }

    pub fn flash_attn(mut self, enabled: bool) -> Self {
        self.flash_attn = enabled;
        self
    }

    pub fn cache_type_k(mut self, cache_type: impl Into<String>) -> Self {
        self.cache_type_k = Some(cache_type.into());
        self
    }

    pub fn cache_type_v(mut self, cache_type: impl Into<String>) -> Self {
        self.cache_type_v = Some(cache_type.into());
        self
    }

    pub fn rope_freq_base(mut self, base: f32) -> Self {
        self.rope_freq_base = Some(base);
        self
    }

    pub fn rope_freq_scale(mut self, scale: f32) -> Self {
        self.rope_freq_scale = Some(scale);
        self
    }

    pub fn n_batch(mut self, batch: u32) -> Self {
        self.n_batch = Some(batch);
        self
    }

    pub fn defrag_thold(mut self, thold: f32) -> Self {
        self.defrag_thold = Some(thold);
        self
    }

    pub fn split_mode(mut self, mode: impl Into<String>) -> Self {
        self.split_mode = Some(mode.into());
        self
    }
}
