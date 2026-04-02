//! Hardware presets for common deployment configurations.
//!
//! Presets provide sensible defaults for model and context parameters
//! based on the target hardware. Users can start with a preset and
//! override individual settings as needed.

use crate::{ContextParams, KvCacheType, ModelParams};

/// Hardware preset for common deployment configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwarePreset {
    /// CPU-only, low memory (4GB RAM)
    CpuLowMemory,
    /// CPU-only, standard (8-16GB RAM)
    CpuStandard,
    /// GPU with low VRAM (4GB)
    GpuLowVram,
    /// GPU with medium VRAM (8GB)
    GpuMediumVram,
    /// GPU with high VRAM (16GB+)
    GpuHighVram,
    /// Apple Silicon (M-series unified memory)
    AppleSilicon,
    /// Maximum performance (all resources)
    MaxPerformance,
}

impl HardwarePreset {
    /// Get recommended model parameters for this preset
    pub fn model_params(&self) -> ModelParams {
        let mut params = ModelParams::default();
        match self {
            Self::CpuLowMemory => {
                params.n_gpu_layers = 0;
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::CpuStandard => {
                params.n_gpu_layers = 0;
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::GpuLowVram => {
                params.n_gpu_layers = 20;
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::GpuMediumVram => {
                params.n_gpu_layers = 33;
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::GpuHighVram => {
                params.n_gpu_layers = -1; // All layers
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::AppleSilicon => {
                params.n_gpu_layers = -1; // All layers via Metal
                params.use_mmap = true;
                params.use_mlock = false;
            }
            Self::MaxPerformance => {
                params.n_gpu_layers = -1;
                params.use_mmap = true;
                params.use_mlock = true;
            }
        }
        params
    }

    /// Get recommended context parameters for this preset
    pub fn context_params(&self) -> ContextParams {
        let mut params = ContextParams::default();
        match self {
            Self::CpuLowMemory => {
                params.n_ctx = 2048;
                params.n_batch = 256;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
                params.type_k = KvCacheType::Q4_0;
                params.type_v = KvCacheType::Q4_0;
            }
            Self::CpuStandard => {
                params.n_ctx = 4096;
                params.n_batch = 512;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
                params.type_k = KvCacheType::F16;
                params.type_v = KvCacheType::F16;
            }
            Self::GpuLowVram => {
                params.n_ctx = 4096;
                params.n_batch = 512;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                params.type_k = KvCacheType::Q8_0;
                params.type_v = KvCacheType::Q8_0;
            }
            Self::GpuMediumVram => {
                params.n_ctx = 8192;
                params.n_batch = 512;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                params.type_k = KvCacheType::Q8_0;
                params.type_v = KvCacheType::Q8_0;
            }
            Self::GpuHighVram => {
                params.n_ctx = 16384;
                params.n_batch = 1024;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                params.type_k = KvCacheType::F16;
                params.type_v = KvCacheType::F16;
            }
            Self::AppleSilicon => {
                params.n_ctx = 8192;
                params.n_batch = 512;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                params.type_k = KvCacheType::Q8_0;
                params.type_v = KvCacheType::Q8_0;
            }
            Self::MaxPerformance => {
                params.n_ctx = 32768;
                params.n_batch = 2048;
                params.flash_attn_type =
                    crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                params.type_k = KvCacheType::F16;
                params.type_v = KvCacheType::F16;
            }
        }
        params
    }

    /// Get recommended quantization format name for this preset
    pub fn recommended_quant(&self) -> &'static str {
        match self {
            Self::CpuLowMemory => "Q4_K_S",
            Self::CpuStandard => "Q4_K_M",
            Self::GpuLowVram => "Q4_K_M",
            Self::GpuMediumVram => "Q5_K_M",
            Self::GpuHighVram => "Q6_K",
            Self::AppleSilicon => "Q5_K_M",
            Self::MaxPerformance => "Q8_0",
        }
    }

    /// Get a human-readable name for this preset
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuLowMemory => "CPU Low Memory (4GB RAM)",
            Self::CpuStandard => "CPU Standard (8-16GB RAM)",
            Self::GpuLowVram => "GPU Low VRAM (4GB)",
            Self::GpuMediumVram => "GPU Medium VRAM (8GB)",
            Self::GpuHighVram => "GPU High VRAM (16GB+)",
            Self::AppleSilicon => "Apple Silicon (M-series)",
            Self::MaxPerformance => "Maximum Performance",
        }
    }

    /// Get a short description of this preset
    pub fn description(&self) -> &'static str {
        match self {
            Self::CpuLowMemory => "Minimal memory usage, quantized KV cache, small context",
            Self::CpuStandard => "Balanced CPU performance with standard context",
            Self::GpuLowVram => "Partial GPU offload with quantized KV cache",
            Self::GpuMediumVram => "Full GPU offload with flash attention, 8K context",
            Self::GpuHighVram => "Full GPU offload, F16 KV cache, large context",
            Self::AppleSilicon => "Optimized for Apple unified memory with Metal",
            Self::MaxPerformance => "Maximum quality and context, all resources",
        }
    }

    /// List all available presets
    pub fn all() -> &'static [HardwarePreset] {
        &[
            Self::CpuLowMemory,
            Self::CpuStandard,
            Self::GpuLowVram,
            Self::GpuMediumVram,
            Self::GpuHighVram,
            Self::AppleSilicon,
            Self::MaxPerformance,
        ]
    }

    /// Get preset index (for FFI)
    pub fn index(&self) -> usize {
        match self {
            Self::CpuLowMemory => 0,
            Self::CpuStandard => 1,
            Self::GpuLowVram => 2,
            Self::GpuMediumVram => 3,
            Self::GpuHighVram => 4,
            Self::AppleSilicon => 5,
            Self::MaxPerformance => 6,
        }
    }

    /// Get preset from index (for FFI)
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::CpuLowMemory),
            1 => Some(Self::CpuStandard),
            2 => Some(Self::GpuLowVram),
            3 => Some(Self::GpuMediumVram),
            4 => Some(Self::GpuHighVram),
            5 => Some(Self::AppleSilicon),
            6 => Some(Self::MaxPerformance),
            _ => None,
        }
    }

    /// Detect the best preset for the current hardware
    pub fn detect() -> Self {
        if crate::supports_gpu_offload() {
            #[cfg(target_os = "macos")]
            {
                return Self::AppleSilicon;
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Self::GpuMediumVram;
            }
        }
        Self::CpuStandard
    }

    /// Parse from string name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "cpu-low" | "cpu_low_memory" | "cpulowmemory" => Some(Self::CpuLowMemory),
            "cpu" | "cpu-standard" | "cpu_standard" | "cpustandard" => Some(Self::CpuStandard),
            "gpu-low" | "gpu_low_vram" | "gpulowvram" => Some(Self::GpuLowVram),
            "gpu" | "gpu-medium" | "gpu_medium_vram" | "gpumediumvram" => {
                Some(Self::GpuMediumVram)
            }
            "gpu-high" | "gpu_high_vram" | "gpuhighvram" => Some(Self::GpuHighVram),
            "apple" | "apple-silicon" | "apple_silicon" | "applesilicon" | "metal" => {
                Some(Self::AppleSilicon)
            }
            "max" | "max-performance" | "max_performance" | "maxperformance" => {
                Some(Self::MaxPerformance)
            }
            "auto" | "detect" => Some(Self::detect()),
            _ => None,
        }
    }

    /// Check if this preset enables flash attention
    pub fn flash_attn(&self) -> bool {
        match self {
            Self::CpuLowMemory | Self::CpuStandard => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_params() {
        for preset in HardwarePreset::all() {
            let mp = preset.model_params();
            let cp = preset.context_params();
            assert!(cp.n_ctx > 0, "Preset {:?} has zero context", preset);
            assert!(cp.n_batch > 0, "Preset {:?} has zero batch", preset);
            let _ = mp; // just verify it doesn't panic
        }
    }

    #[test]
    fn test_preset_roundtrip() {
        for preset in HardwarePreset::all() {
            let idx = preset.index();
            assert_eq!(HardwarePreset::from_index(idx), Some(*preset));
        }
    }

    #[test]
    fn test_preset_from_name() {
        assert_eq!(
            HardwarePreset::from_name("cpu"),
            Some(HardwarePreset::CpuStandard)
        );
        assert_eq!(
            HardwarePreset::from_name("gpu"),
            Some(HardwarePreset::GpuMediumVram)
        );
        assert_eq!(
            HardwarePreset::from_name("apple-silicon"),
            Some(HardwarePreset::AppleSilicon)
        );
        assert_eq!(
            HardwarePreset::from_name("max"),
            Some(HardwarePreset::MaxPerformance)
        );
        assert_eq!(HardwarePreset::from_name("invalid"), None);
    }

    #[test]
    fn test_detect() {
        // Just verify it doesn't panic
        let _ = HardwarePreset::detect();
    }
}
