<?php

declare(strict_types=1);

namespace Mullama;

/**
 * Hardware preset for common deployment configurations.
 *
 * Presets provide sensible defaults for model and context parameters
 * based on the target hardware.
 */
enum HardwarePreset: int
{
    case CpuLowMemory = 0;
    case CpuStandard = 1;
    case GpuLowVram = 2;
    case GpuMediumVram = 3;
    case GpuHighVram = 4;
    case AppleSilicon = 5;
    case MaxPerformance = 6;

    /**
     * Get the human-readable name of this preset.
     */
    public function presetName(): string
    {
        return match ($this) {
            self::CpuLowMemory => 'CPU Low Memory (4GB RAM)',
            self::CpuStandard => 'CPU Standard (8-16GB RAM)',
            self::GpuLowVram => 'GPU Low VRAM (4GB)',
            self::GpuMediumVram => 'GPU Medium VRAM (8GB)',
            self::GpuHighVram => 'GPU High VRAM (16GB+)',
            self::AppleSilicon => 'Apple Silicon (M-series)',
            self::MaxPerformance => 'Maximum Performance',
        };
    }

    /**
     * Get the short description of this preset.
     */
    public function description(): string
    {
        return match ($this) {
            self::CpuLowMemory => 'Minimal memory usage, quantized KV cache, small context',
            self::CpuStandard => 'Balanced CPU performance with standard context',
            self::GpuLowVram => 'Partial GPU offload with quantized KV cache',
            self::GpuMediumVram => 'Full GPU offload with flash attention, 8K context',
            self::GpuHighVram => 'Full GPU offload, F16 KV cache, large context',
            self::AppleSilicon => 'Optimized for Apple unified memory with Metal',
            self::MaxPerformance => 'Maximum quality and context, all resources',
        };
    }

    /**
     * Get the recommended number of GPU layers (-1 = all).
     */
    public function gpuLayers(): int
    {
        return match ($this) {
            self::CpuLowMemory, self::CpuStandard => 0,
            self::GpuLowVram => 20,
            self::GpuMediumVram => 33,
            self::GpuHighVram, self::AppleSilicon, self::MaxPerformance => -1,
        };
    }

    /**
     * Get the recommended context size.
     */
    public function contextSize(): int
    {
        return match ($this) {
            self::CpuLowMemory => 2048,
            self::CpuStandard, self::GpuLowVram => 4096,
            self::GpuMediumVram, self::AppleSilicon => 8192,
            self::GpuHighVram => 16384,
            self::MaxPerformance => 32768,
        };
    }

    /**
     * Get the recommended quantization format.
     */
    public function recommendedQuant(): string
    {
        return match ($this) {
            self::CpuLowMemory => 'Q4_K_S',
            self::CpuStandard, self::GpuLowVram => 'Q4_K_M',
            self::GpuMediumVram, self::AppleSilicon => 'Q5_K_M',
            self::GpuHighVram => 'Q6_K',
            self::MaxPerformance => 'Q8_0',
        };
    }

    /**
     * Check if this preset enables flash attention.
     */
    public function flashAttn(): bool
    {
        return match ($this) {
            self::CpuLowMemory, self::CpuStandard => false,
            default => true,
        };
    }
}
