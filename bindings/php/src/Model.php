<?php

declare(strict_types=1);

namespace Mullama;

use FFI;
use FFI\CData;
use RuntimeException;

/**
 * Model class for loading and managing LLM models.
 */
final class Model
{
    private ?CData $ptr;

    private function __construct(CData $ptr)
    {
        $this->ptr = $ptr;
    }

    public function __destruct()
    {
        $this->free();
    }

    /**
     * Load a model from a GGUF file.
     *
     * @param string $path Path to the GGUF model file
     * @param array $params Optional parameters: nGpuLayers, useMmap, useMlock, vocabOnly
     * @return Model
     * @throws RuntimeException if loading fails
     */
    public static function load(string $path, array $params = []): self
    {
        $ffi = Mullama::ffi();

        $cParams = $ffi->mullama_model_default_params();
        if (isset($params['nGpuLayers'])) {
            $cParams->n_gpu_layers = $params['nGpuLayers'];
        }
        if (isset($params['useMmap'])) {
            $cParams->use_mmap = $params['useMmap'];
        }
        if (isset($params['useMlock'])) {
            $cParams->use_mlock = $params['useMlock'];
        }
        if (isset($params['vocabOnly'])) {
            $cParams->vocab_only = $params['vocabOnly'];
        }
        if (isset($params['checkTensors'])) {
            $cParams->check_tensors = $params['checkTensors'];
        }
        if (isset($params['splitMode'])) {
            $cParams->split_mode = $params['splitMode'];
        }

        $ptr = $ffi->mullama_model_load($path, FFI::addr($cParams));

        if ($ptr === null) {
            throw new RuntimeException(Mullama::getLastError());
        }

        return new self($ptr);
    }

    /**
     * Free the model resources.
     */
    public function free(): void
    {
        if ($this->ptr !== null) {
            Mullama::ffi()->mullama_model_free($this->ptr);
            $this->ptr = null;
        }
    }

    /**
     * Get the internal pointer (for use by other classes).
     */
    public function getPtr(): ?CData
    {
        return $this->ptr;
    }

    /**
     * Tokenize text into token IDs.
     *
     * @param string $text Text to tokenize
     * @param bool $addBos Add beginning-of-sequence token
     * @param bool $special Parse special tokens
     * @return int[]
     */
    public function tokenize(string $text, bool $addBos = true, bool $special = false): array
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Model has been freed');
        }

        $ffi = Mullama::ffi();
        $maxTokens = strlen($text) * 2 + 10;
        $tokens = FFI::new("int[{$maxTokens}]");

        $n = $ffi->mullama_tokenize(
            $this->ptr,
            $text,
            FFI::addr($tokens[0]),
            $maxTokens,
            $addBos,
            $special
        );

        if ($n < 0) {
            throw new RuntimeException(Mullama::getLastError());
        }

        $result = [];
        for ($i = 0; $i < $n; $i++) {
            $result[] = $tokens[$i];
        }

        return $result;
    }

    /**
     * Detokenize token IDs back to text.
     *
     * @param int[] $tokens Token IDs
     * @return string
     */
    public function detokenize(array $tokens): string
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Model has been freed');
        }

        if (empty($tokens)) {
            return '';
        }

        $ffi = Mullama::ffi();
        $n = count($tokens);
        $cTokens = FFI::new("int[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $cTokens[$i] = $tokens[$i];
        }

        $maxLen = $n * 32;
        $buf = FFI::new("char[{$maxLen}]");

        $result = $ffi->mullama_detokenize(
            $this->ptr,
            FFI::addr($cTokens[0]),
            $n,
            FFI::addr($buf[0]),
            $maxLen
        );

        if ($result < 0) {
            throw new RuntimeException(Mullama::getLastError());
        }

        return FFI::string($buf);
    }

    /**
     * Get the model's training context size.
     */
    public function nCtxTrain(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_model_n_ctx_train($this->ptr);
    }

    /**
     * Get the embedding dimension.
     */
    public function nEmbd(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_model_n_embd($this->ptr);
    }

    /**
     * Get the vocabulary size.
     */
    public function nVocab(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_model_n_vocab($this->ptr);
    }

    /**
     * Get the number of layers.
     */
    public function nLayer(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_model_n_layer($this->ptr);
    }

    /**
     * Get the number of attention heads.
     */
    public function nHead(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_model_n_head($this->ptr);
    }

    /**
     * Get the BOS token ID.
     */
    public function tokenBos(): int
    {
        if ($this->ptr === null) {
            return -1;
        }
        return (int) Mullama::ffi()->mullama_model_token_bos($this->ptr);
    }

    /**
     * Get the EOS token ID.
     */
    public function tokenEos(): int
    {
        if ($this->ptr === null) {
            return -1;
        }
        return (int) Mullama::ffi()->mullama_model_token_eos($this->ptr);
    }

    /**
     * Get the model size in bytes.
     */
    public function size(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        $result = Mullama::ffi()->mullama_model_size($this->ptr);
        if (is_object($result) && $result instanceof \FFI\CData) {
            return (int) FFI::cast('int64_t', $result)->cdata;
        }
        return (int) $result;
    }

    /**
     * Get the number of parameters.
     */
    public function nParams(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        $result = Mullama::ffi()->mullama_model_n_params($this->ptr);
        if (is_object($result) && $result instanceof \FFI\CData) {
            return (int) FFI::cast('int64_t', $result)->cdata;
        }
        return (int) $result;
    }

    /**
     * Get the model description.
     */
    public function description(): string
    {
        if ($this->ptr === null) {
            return '';
        }
        $buf = FFI::new('char[256]');
        Mullama::ffi()->mullama_model_desc($this->ptr, FFI::addr($buf[0]), 256);
        return FFI::string($buf);
    }

    /**
     * Check if a token is end-of-generation.
     */
    public function tokenIsEog(int $token): bool
    {
        if ($this->ptr === null) {
            return false;
        }
        return (bool) Mullama::ffi()->mullama_model_token_is_eog($this->ptr, $token);
    }

    /**
     * Apply a chat template to format messages.
     *
     * @param array $messages Array of ['role' => string, 'content' => string]
     * @param bool $addGenerationPrompt Whether to add generation prompt
     * @return string Formatted prompt
     */
    public function chatTemplate(array $messages, bool $addGenerationPrompt = true): string
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Model has been freed');
        }

        $ffi = Mullama::ffi();
        $n = count($messages);
        $cMessages = $ffi->new("MullamaChatMessage[{$n}]");

        for ($i = 0; $i < $n; $i++) {
            $cMessages[$i]->role = $messages[$i]['role'];
            $cMessages[$i]->content = $messages[$i]['content'];
        }

        $maxOutput = 8192;
        $buf = FFI::new("char[{$maxOutput}]");

        $result = $ffi->mullama_apply_chat_template(
            $this->ptr,
            FFI::addr($cMessages[0]),
            $n,
            $addGenerationPrompt,
            FFI::addr($buf[0]),
            $maxOutput
        );

        if ($result < 0) {
            throw new RuntimeException(Mullama::getLastError());
        }

        return FFI::string($buf);
    }

    /**
     * Get all model metadata as an associative array.
     *
     * @return array<string, string>
     */
    public function metadata(): array
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Model has been freed');
        }

        $ffi = Mullama::ffi();
        $count = (int) $ffi->mullama_model_metadata_count($this->ptr);
        $result = [];

        $keyBuf = FFI::new('char[1024]');
        $valBuf = FFI::new('char[4096]');

        for ($i = 0; $i < $count; $i++) {
            $kn = $ffi->mullama_model_metadata_key($this->ptr, $i, FFI::addr($keyBuf[0]), 1024);
            if ($kn < 0) {
                continue;
            }
            $vn = $ffi->mullama_model_metadata_value($this->ptr, $i, FFI::addr($valBuf[0]), 4096);
            if ($vn < 0) {
                continue;
            }
            $result[FFI::string($keyBuf)] = FFI::string($valBuf);
        }

        return $result;
    }
}
