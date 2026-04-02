<?php

declare(strict_types=1);

namespace Mullama;

/**
 * Context parameters for model inference.
 *
 * Provides structured configuration for creating inference contexts,
 * with support for creating from hardware presets.
 */
class ContextParams
{
    public function __construct(
        public readonly int $contextSize = 4096,
        public readonly int $batchSize = 512,
        public readonly int $threads = 0,
        public readonly bool $flashAttn = false,
        public readonly string $cacheTypeK = 'f16',
        public readonly string $cacheTypeV = 'f16',
        public readonly float $ropeFreqBase = 0.0,
        public readonly float $ropeFreqScale = 0.0,
        public readonly float $defragThold = -1.0,
    ) {
    }

    /**
     * Create from a hardware preset.
     */
    public static function fromPreset(HardwarePreset $preset): self
    {
        return new self(
            contextSize: $preset->contextSize(),
            flashAttn: $preset->flashAttn(),
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
            'context_size' => $this->contextSize,
            'batch_size' => $this->batchSize,
            'threads' => $this->threads,
            'flash_attn' => $this->flashAttn,
            'cache_type_k' => $this->cacheTypeK,
            'cache_type_v' => $this->cacheTypeV,
            'rope_freq_base' => $this->ropeFreqBase,
            'rope_freq_scale' => $this->ropeFreqScale,
            'defrag_thold' => $this->defragThold,
        ];
    }
}
