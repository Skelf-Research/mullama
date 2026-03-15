<?php

declare(strict_types=1);

namespace Mullama;

use FFI;
use FFI\CData;
use RuntimeException;

/**
 * Embedding generator for creating text embeddings.
 */
final class EmbeddingGenerator
{
    private ?CData $ptr;
    private Model $model;

    /**
     * Create a new embedding generator.
     *
     * @param Model $model The model to use
     * @param int $nCtx Context size (0 = default)
     * @param bool $normalize Normalize embeddings
     */
    public function __construct(Model $model, int $nCtx = 512, bool $normalize = true)
    {
        $ffi = Mullama::ffi();

        $config = $ffi->mullama_embedding_default_config();
        $config->n_ctx = $nCtx;
        $config->normalize = $normalize;

        $this->ptr = $ffi->mullama_embedding_generator_new($model->getPtr(), FFI::addr($config));

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
     * Free the embedding generator resources.
     */
    public function free(): void
    {
        if ($this->ptr !== null) {
            Mullama::ffi()->mullama_embedding_generator_free($this->ptr);
            $this->ptr = null;
        }
    }

    /**
     * Generate embeddings for text.
     *
     * @param string $text Text to embed
     * @return float[] Embedding vector
     */
    public function embed(string $text): array
    {
        if ($this->ptr === null) {
            throw new RuntimeException('Embedding generator has been freed');
        }

        $ffi = Mullama::ffi();
        $nEmbd = $this->nEmbd();
        $embeddings = FFI::new("float[{$nEmbd}]");

        $n = $ffi->mullama_embed_text($this->ptr, $text, FFI::addr($embeddings[0]), $nEmbd);

        if ($n < 0) {
            throw new RuntimeException(Mullama::getLastError());
        }

        $result = [];
        for ($i = 0; $i < $n; $i++) {
            $result[] = $embeddings[$i];
        }

        return $result;
    }

    /**
     * Generate embeddings for multiple texts.
     *
     * @param string[] $texts Texts to embed
     * @return float[][] Array of embedding vectors
     */
    public function embedBatch(array $texts): array
    {
        $results = [];
        foreach ($texts as $text) {
            $results[] = $this->embed($text);
        }
        return $results;
    }

    /**
     * Get the embedding dimension.
     */
    public function nEmbd(): int
    {
        if ($this->ptr === null) {
            return 0;
        }
        return (int) Mullama::ffi()->mullama_embedding_generator_n_embd($this->ptr);
    }

    /**
     * Compute cosine similarity between two vectors.
     *
     * @param float[] $a First vector
     * @param float[] $b Second vector
     * @return float Cosine similarity
     */
    public static function cosineSimilarity(array $a, array $b): float
    {
        if (count($a) !== count($b)) {
            throw new RuntimeException('Vectors must have the same length');
        }

        $dot = 0.0;
        $normA = 0.0;
        $normB = 0.0;

        foreach ($a as $i => $val) {
            $dot += $val * $b[$i];
            $normA += $val * $val;
            $normB += $b[$i] * $b[$i];
        }

        $norm = sqrt($normA) * sqrt($normB);
        if ($norm == 0) {
            return 0.0;
        }

        return $dot / $norm;
    }
}
