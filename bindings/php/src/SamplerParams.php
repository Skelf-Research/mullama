<?php

declare(strict_types=1);

namespace Mullama;

/**
 * Sampler parameters for text generation.
 */
final class SamplerParams
{
    public float $temperature = 0.8;
    public int $topK = 40;
    public float $topP = 0.95;
    public float $minP = 0.05;
    public float $typicalP = 1.0;
    public float $penaltyRepeat = 1.1;
    public float $penaltyFreq = 0.0;
    public float $penaltyPresent = 0.0;
    public int $penaltyLastN = 64;
    public int $seed = 0;

    public function __construct(array $params = [])
    {
        foreach ($params as $key => $value) {
            if (property_exists($this, $key)) {
                $this->$key = $value;
            }
        }
    }

    /**
     * Create greedy (deterministic) sampler parameters.
     */
    public static function greedy(): self
    {
        return new self([
            'temperature' => 0.0,
            'topK' => 1,
            'topP' => 1.0,
            'minP' => 0.0,
            'typicalP' => 1.0,
            'penaltyRepeat' => 1.0,
            'penaltyFreq' => 0.0,
            'penaltyPresent' => 0.0,
            'penaltyLastN' => 0,
            'seed' => 0,
        ]);
    }

    /**
     * Create creative (high randomness) sampler parameters.
     */
    public static function creative(): self
    {
        return new self([
            'temperature' => 1.2,
            'topK' => 100,
            'topP' => 0.95,
            'minP' => 0.02,
            'typicalP' => 1.0,
            'penaltyRepeat' => 1.15,
            'penaltyFreq' => 0.1,
            'penaltyPresent' => 0.1,
            'penaltyLastN' => 128,
            'seed' => 0,
        ]);
    }

    /**
     * Create precise (low randomness) sampler parameters.
     */
    public static function precise(): self
    {
        return new self([
            'temperature' => 0.3,
            'topK' => 20,
            'topP' => 0.8,
            'minP' => 0.1,
            'typicalP' => 1.0,
            'penaltyRepeat' => 1.05,
            'penaltyFreq' => 0.0,
            'penaltyPresent' => 0.0,
            'penaltyLastN' => 32,
            'seed' => 0,
        ]);
    }
}
