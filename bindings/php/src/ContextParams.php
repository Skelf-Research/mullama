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
        public readonly int $nCtx = 0,
        public readonly int $nBatch = 2048,
        public readonly int $nUbatch = 512,
        public readonly int $nThreads = 0,
        public readonly bool $embeddings = false,
        public readonly bool $offloadKqv = true,
        public readonly int $flashAttn = 0,
    ) {
    }

    /**
     * Create from a hardware preset.
     */
    public static function fromPreset(HardwarePreset $preset): self
    {
        return new self(
            nCtx: $preset->contextSize(),
            flashAttn: $preset->flashAttn() ? 2 : 0,
        );
    }

    /**
     * Convert to array for Context constructor.
     *
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'nCtx' => $this->nCtx,
            'nBatch' => $this->nBatch,
            'nUbatch' => $this->nUbatch,
            'nThreads' => $this->nThreads,
            'embeddings' => $this->embeddings,
            'offloadKqv' => $this->offloadKqv,
            'flashAttn' => $this->flashAttn,
        ];
    }
}
