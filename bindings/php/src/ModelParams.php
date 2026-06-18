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
        public readonly int $nGpuLayers = 0,
        public readonly bool $useMmap = true,
        public readonly bool $useMlock = false,
        public readonly bool $vocabOnly = false,
        public readonly bool $checkTensors = false,
        public readonly int $splitMode = 0,
    ) {
    }

    /**
     * Create from a hardware preset.
     */
    public static function fromPreset(HardwarePreset $preset): self
    {
        return new self(
            nGpuLayers: $preset->gpuLayers(),
        );
    }

    /**
     * Convert to array for Model::load().
     *
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'nGpuLayers' => $this->nGpuLayers,
            'useMmap' => $this->useMmap,
            'useMlock' => $this->useMlock,
            'vocabOnly' => $this->vocabOnly,
            'checkTensors' => $this->checkTensors,
            'splitMode' => $this->splitMode,
        ];
    }
}
