<?php

declare(strict_types=1);

namespace Mullama;

/**
 * Model loading parameters.
 *
 * Provides structured configuration for loading models,
 * with support for creating from hardware presets.
 */
class ModelParams
{
    public function __construct(
        public readonly int $gpuLayers = 0,
        public readonly bool $useMmap = true,
        public readonly bool $useMlock = false,
        public readonly ?string $splitMode = null,
    ) {
    }

    /**
     * Create from a hardware preset.
     */
    public static function fromPreset(HardwarePreset $preset): self
    {
        return new self(
            gpuLayers: $preset->gpuLayers(),
        );
    }

    /**
     * Convert to array for FFI/API calls.
     *
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'gpu_layers' => $this->gpuLayers,
            'use_mmap' => $this->useMmap,
            'use_mlock' => $this->useMlock,
            'split_mode' => $this->splitMode,
        ];
    }
}
