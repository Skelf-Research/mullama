<?php

declare(strict_types=1);

namespace Mullama;

use FFI;
use FFI\CData;
use RuntimeException;

/**
 * Context class for model inference.
 */
final class Context
{
    private ?CData $ptr;
    private Model $model;

    /**
     * Create a new context from a model.
     *
     * @param Model $model The model to use
     * @param array $params Optional parameters: nCtx, nBatch, nThreads, embeddings
     */
    public function __construct(Model $model, array $params = [])
    {
        $ffi = Mullama::ffi();

        $cParams = $ffi->new('MullamaMullamaContextParams');
        $cParams->n_ctx = $params['nCtx'] ?? 0;
        $cParams->n_batch = $params['nBatch'] ?? 2048;
        $cParams->n_ubatch = $params['nBatch'] ?? 512;
        $cParams->n_seq_max = 1;
        $cParams->n_threads = $params['nThreads'] ?? 0;
        $cParams->n_threads_batch = $params['nThreads'] ?? 0;
        $cParams->embeddings = $params['embeddings'] ?? false;
        $cParams->offload_kqv = true;
        $cParams->flash_attn = 0;

        $this->ptr = $ffi->mullama_context_new($model->getPtr(), FFI::addr($cParams));

        if ($this->ptr === null) {
            throw new RuntimeException(Mullama::getLastError());
        }

        $this->model = $model;
    }

    public function __destruct()
    {
        $this->free();
    }

    /**
     * Free the context resources.
     */
    public function free(): void
    {
        if ($this->ptr !== null) {
            Mullama::ffi()->mullama_context_free($this->ptr);
            $this->ptr = null;
        }
    }

    /**
     * Generate text from a prompt.
     *
     * @param string $prompt Text prompt
     * @param int $maxTokens Maximum tokens to generate
     * @param SamplerParams|null $params Sampler parameters
     * @return string Generated text
     */
    public function generate(string $prompt, int $maxTokens = 100, ?SamplerParams $params = null): string
    {
        $tokens = $this->model->tokenize($prompt, true, false);
        return $this->generateFromTokens($tokens, $maxTokens, $params);
    }

    /**
     * Generate text from token IDs.
     *
     * @param int[] $tokens Token IDs
     * @param int $maxTokens Maximum tokens to generate
     * @param SamplerParams|null $params Sampler parameters
     * @return string Generated text
     */
    public function generateFromTokens(array $tokens, int $maxTokens = 100, ?SamplerParams $params = null): string
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Context has been freed');
        }

        $params = $params ?? new SamplerParams();

        $ffi = Mullama::ffi();

        $n = count($tokens);
        $cTokens = FFI::new("int[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $cTokens[$i] = $tokens[$i];
        }

        $cParams = $ffi->new('MullamaMullamaSamplerParams');
        $cParams->temperature = $params->temperature;
        $cParams->top_k = $params->topK;
        $cParams->top_p = $params->topP;
        $cParams->min_p = $params->minP;
        $cParams->typical_p = $params->typicalP;
        $cParams->penalty_repeat = $params->penaltyRepeat;
        $cParams->penalty_freq = $params->penaltyFreq;
        $cParams->penalty_present = $params->penaltyPresent;
        $cParams->penalty_last_n = $params->penaltyLastN;
        $cParams->penalize_nl = false;
        $cParams->ignore_eos = false;
        $cParams->seed = $params->seed;

        $maxOutput = $maxTokens * 32;
        $buf = FFI::new("char[{$maxOutput}]");

        $result = $ffi->mullama_generate(
            $this->ptr,
            FFI::addr($cTokens[0]),
            $n,
            $maxTokens,
            FFI::addr($cParams),
            FFI::addr($buf[0]),
            $maxOutput
        );

        if ($result < 0) {
            throw new RuntimeException(Mullama::getLastError());
        }

        return FFI::string($buf);
    }

    /**
     * Generate text with streaming (returns array of token strings).
     *
     * @param string $prompt Text prompt
     * @param int $maxTokens Maximum tokens to generate
     * @param SamplerParams|null $params Sampler parameters
     * @return string[] Array of generated token strings
     */
    public function generateStream(string $prompt, int $maxTokens = 100, ?SamplerParams $params = null): array
    {
        // Simplified implementation - returns full text as single array element
        $text = $this->generate($prompt, $maxTokens, $params);
        return [$text];
    }

    /**
     * Clear the KV cache.
     */
    public function clearCache(): void
    {
        if ($this->ptr !== null) {
            Mullama::ffi()->mullama_context_kv_cache_clear($this->ptr);
        }
    }

    /**
     * Get the context size.
     */
    public function nCtx(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_context_n_ctx($this->ptr);
    }

    /**
     * Get the batch size.
     */
    public function nBatch(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_context_n_batch($this->ptr);
    }
}
