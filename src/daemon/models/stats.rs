use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Per-model statistics tracking
pub struct ModelStats {
    pub requests_total: AtomicU64,
    pub tokens_generated: AtomicU64,
    pub tokens_prompt: AtomicU64,
    pub avg_tokens_per_sec: AtomicU64,
    pub last_used: AtomicU64,
    pub estimated_memory_bytes: AtomicU64,
    pub load_time_ms: AtomicU64,
}

impl ModelStats {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            tokens_prompt: AtomicU64::new(0),
            avg_tokens_per_sec: AtomicU64::new(0),
            last_used: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            estimated_memory_bytes: AtomicU64::new(0),
            load_time_ms: AtomicU64::new(0),
        }
    }

    pub fn record_request(&self, prompt_tokens: u32, completion_tokens: u32, duration_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.tokens_generated
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.tokens_prompt
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.last_used.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );

        if duration_ms > 0 && completion_tokens > 0 {
            let tps_x100 = (completion_tokens as u64 * 100_000) / duration_ms;
            let prev = self.avg_tokens_per_sec.load(Ordering::Relaxed);
            if prev == 0 {
                self.avg_tokens_per_sec.store(tps_x100, Ordering::Relaxed);
            } else {
                let new_avg = (prev * 3 + tps_x100) / 4;
                self.avg_tokens_per_sec.store(new_avg, Ordering::Relaxed);
            }
        }
    }

    pub fn touch(&self) {
        self.last_used.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );
    }
}

impl Default for ModelStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory estimation result for a model
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub model_bytes: u64,
    pub kv_cache_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
}

impl MemoryEstimate {
    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }
}

/// Estimate model memory requirements from file size and parameters
pub fn estimate_model_memory(
    file_size: u64,
    context_size: u32,
    gpu_layers: i32,
    n_layers: u32,
) -> MemoryEstimate {
    let model_bytes = file_size;
    let kv_bytes_per_token = (n_layers as u64) * 256;
    let kv_cache_bytes = kv_bytes_per_token * (context_size as u64);
    let overhead_bytes = model_bytes / 5;

    let _gpu_layers = gpu_layers;

    let total_bytes = model_bytes + kv_cache_bytes + overhead_bytes;

    MemoryEstimate {
        model_bytes,
        kv_cache_bytes,
        overhead_bytes,
        total_bytes,
    }
}
