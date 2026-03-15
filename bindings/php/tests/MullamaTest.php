<?php

declare(strict_types=1);

namespace Mullama\Tests;

use Mullama\Mullama;
use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;
use Mullama\EmbeddingGenerator;
use PHPUnit\Framework\TestCase;

/**
 * Tests for the Mullama PHP bindings.
 *
 * Note: Some tests require a model file. Set MULLAMA_TEST_MODEL environment variable.
 */
final class MullamaTest extends TestCase
{
    private static ?string $testModelPath = null;
    private static bool $modelAvailable = false;

    public static function setUpBeforeClass(): void
    {
        self::$testModelPath = getenv('MULLAMA_TEST_MODEL') ?: null;
        self::$modelAvailable = self::$testModelPath && file_exists(self::$testModelPath);
    }

    // === Version and System Info Tests ===

    public function testVersion(): void
    {
        $version = Mullama::version();
        $this->assertIsString($version);
        $this->assertEquals('0.1.0', $version);
    }

    public function testSystemInfo(): void
    {
        $this->markTestSkipped('FFI initialization required');
        // $info = Mullama::systemInfo();
        // $this->assertIsString($info);
        // $this->assertNotEmpty($info);
    }

    public function testSupportsGpuOffload(): void
    {
        $this->markTestSkipped('FFI initialization required');
        // $result = Mullama::supportsGpuOffload();
        // $this->assertIsBool($result);
    }

    public function testMaxDevices(): void
    {
        $this->markTestSkipped('FFI initialization required');
        // $devices = Mullama::maxDevices();
        // $this->assertIsInt($devices);
        // $this->assertGreaterThanOrEqual(1, $devices);
    }

    // === Sampler Params Tests ===

    public function testDefaultSamplerParams(): void
    {
        $params = new SamplerParams();
        $this->assertEquals(0.8, $params->temperature, '', 0.01);
        $this->assertEquals(40, $params->topK);
        $this->assertEquals(0.95, $params->topP, '', 0.01);
    }

    public function testCustomSamplerParams(): void
    {
        $params = new SamplerParams([
            'temperature' => 0.5,
            'topK' => 20,
            'topP' => 0.8,
        ]);
        $this->assertEquals(0.5, $params->temperature, '', 0.01);
        $this->assertEquals(20, $params->topK);
        $this->assertEquals(0.8, $params->topP, '', 0.01);
    }

    public function testGreedySamplerParams(): void
    {
        $params = SamplerParams::greedy();
        $this->assertEquals(0.0, $params->temperature, '', 0.01);
        $this->assertEquals(1, $params->topK);
    }

    public function testCreativeSamplerParams(): void
    {
        $params = SamplerParams::creative();
        $this->assertGreaterThan(1.0, $params->temperature);
        $this->assertGreaterThan(40, $params->topK);
    }

    public function testPreciseSamplerParams(): void
    {
        $params = SamplerParams::precise();
        $this->assertLessThan(0.5, $params->temperature);
        $this->assertLessThan(40, $params->topK);
    }

    // === Cosine Similarity Tests ===

    public function testCosineSimilarityIdentical(): void
    {
        $a = [1.0, 0.0, 0.0];
        $b = [1.0, 0.0, 0.0];
        $sim = EmbeddingGenerator::cosineSimilarity($a, $b);
        $this->assertEquals(1.0, $sim, '', 0.001);
    }

    public function testCosineSimilarityOrthogonal(): void
    {
        $a = [1.0, 0.0, 0.0];
        $b = [0.0, 1.0, 0.0];
        $sim = EmbeddingGenerator::cosineSimilarity($a, $b);
        $this->assertEquals(0.0, $sim, '', 0.001);
    }

    public function testCosineSimilarityOpposite(): void
    {
        $a = [1.0, 0.0, 0.0];
        $b = [-1.0, 0.0, 0.0];
        $sim = EmbeddingGenerator::cosineSimilarity($a, $b);
        $this->assertEquals(-1.0, $sim, '', 0.001);
    }

    public function testCosineSimilarityRandom(): void
    {
        $a = array_map(fn() => mt_rand(-100, 100) / 100, range(0, 127));
        $b = array_map(fn() => mt_rand(-100, 100) / 100, range(0, 127));
        $sim = EmbeddingGenerator::cosineSimilarity($a, $b);
        $this->assertGreaterThanOrEqual(-1.0, $sim);
        $this->assertLessThanOrEqual(1.0, $sim);
    }

    public function testCosineSimilarityMismatchedLength(): void
    {
        $this->expectException(\RuntimeException::class);
        $a = [1.0, 0.0];
        $b = [1.0, 0.0, 0.0];
        EmbeddingGenerator::cosineSimilarity($a, $b);
    }

    // === Model Loading Tests (require FFI and model) ===

    public function testLoadNonexistentModel(): void
    {
        $this->markTestSkipped('FFI initialization required');
        // $this->expectException(\RuntimeException::class);
        // Model::load('/nonexistent/path/model.gguf');
    }

    public function testLoadModel(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $this->assertGreaterThan(0, $model->nVocab());
        $this->assertGreaterThan(0, $model->nEmbd());
        $model->free();
    }

    public function testModelProperties(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);

        $this->assertGreaterThan(0, $model->nCtxTrain());
        $this->assertGreaterThan(0, $model->nEmbd());
        $this->assertGreaterThan(0, $model->nVocab());
        $this->assertGreaterThan(0, $model->nLayer());
        $this->assertGreaterThan(0, $model->nHead());
        $this->assertGreaterThan(0, $model->size());
        $this->assertGreaterThan(0, $model->nParams());
        $this->assertGreaterThanOrEqual(0, $model->tokenBos());
        $this->assertGreaterThanOrEqual(0, $model->tokenEos());

        $model->free();
    }

    public function testTokenization(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);

        $text = 'Hello, world!';
        $tokens = $model->tokenize($text);

        $this->assertIsArray($tokens);
        $this->assertNotEmpty($tokens);

        $decoded = $model->detokenize($tokens);
        $this->assertIsString($decoded);

        $model->free();
    }

    // === Context Tests (require FFI and model) ===

    public function testCreateContext(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $ctx = new Context($model);

        $this->assertGreaterThan(0, $ctx->nCtx());
        $this->assertGreaterThan(0, $ctx->nBatch());

        $ctx->free();
        $model->free();
    }

    public function testContextWithParams(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $ctx = new Context($model, ['nCtx' => 512, 'nBatch' => 256]);

        $this->assertEquals(512, $ctx->nCtx());
        $this->assertEquals(256, $ctx->nBatch());

        $ctx->free();
        $model->free();
    }

    public function testGenerateText(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $ctx = new Context($model, ['nCtx' => 256]);

        $params = SamplerParams::greedy();
        $text = $ctx->generate('Hello', 10, $params);

        $this->assertIsString($text);
        $this->assertNotEmpty($text);

        $ctx->free();
        $model->free();
    }

    public function testClearCache(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $ctx = new Context($model);

        // Should not throw
        $ctx->clearCache();

        $ctx->free();
        $model->free();
    }

    // === Embedding Generator Tests (require FFI and model) ===

    public function testCreateEmbeddingGenerator(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $gen = new EmbeddingGenerator($model);

        $this->assertGreaterThan(0, $gen->nEmbd());

        $gen->free();
        $model->free();
    }

    public function testEmbedText(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $gen = new EmbeddingGenerator($model);

        $embedding = $gen->embed('Hello, world!');

        $this->assertIsArray($embedding);
        $this->assertCount($gen->nEmbd(), $embedding);

        $gen->free();
        $model->free();
    }

    public function testEmbedBatch(): void
    {
        if (!self::$modelAvailable) {
            $this->markTestSkipped('No test model available');
        }

        $model = Model::load(self::$testModelPath);
        $gen = new EmbeddingGenerator($model);

        $texts = ['Hello', 'World', 'Test'];
        $embeddings = $gen->embedBatch($texts);

        $this->assertCount(3, $embeddings);
        foreach ($embeddings as $emb) {
            $this->assertIsArray($emb);
            $this->assertCount($gen->nEmbd(), $emb);
        }

        $gen->free();
        $model->free();
    }
}
