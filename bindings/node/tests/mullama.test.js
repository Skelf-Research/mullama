/**
 * Tests for mullama Node.js bindings.
 *
 * These tests cover the core functionality of the mullama library.
 * Note: Some tests require a model file to be present at the specified path.
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');

// Path to test model (set via environment variable)
const TEST_MODEL_PATH = process.env.MULLAMA_TEST_MODEL || '';
const MODEL_AVAILABLE = TEST_MODEL_PATH && require('fs').existsSync(TEST_MODEL_PATH);

// Try to load mullama
let mullama = null;
let MULLAMA_AVAILABLE = false;

try {
  mullama = require('../index.js');
  MULLAMA_AVAILABLE = true;
} catch (e) {
  console.log('mullama not available:', e.message);
}

describe('Module Import', { skip: !MULLAMA_AVAILABLE }, () => {
  test('module exports JsModel', () => {
    assert.ok(mullama.JsModel);
  });

  test('module exports JsContext', () => {
    assert.ok(mullama.JsContext);
  });

  test('module exports JsEmbeddingGenerator', () => {
    assert.ok(mullama.JsEmbeddingGenerator);
  });

  test('module exports utility functions', () => {
    assert.ok(typeof mullama.backendInit === 'function');
    assert.ok(typeof mullama.backendFree === 'function');
    assert.ok(typeof mullama.supportsGpuOffload === 'function');
    assert.ok(typeof mullama.systemInfo === 'function');
    assert.ok(typeof mullama.maxDevices === 'function');
    assert.ok(typeof mullama.cosineSimilarity === 'function');
    assert.ok(typeof mullama.version === 'function');
  });

  test('version returns string', () => {
    const ver = mullama.version();
    assert.ok(typeof ver === 'string');
    assert.ok(ver.length > 0);
  });

  test('systemInfo returns string', () => {
    const info = mullama.systemInfo();
    assert.ok(typeof info === 'string');
    assert.ok(info.length > 0);
  });

  test('supportsGpuOffload returns boolean', () => {
    const result = mullama.supportsGpuOffload();
    assert.ok(typeof result === 'boolean');
  });

  test('maxDevices returns positive number', () => {
    const devices = mullama.maxDevices();
    assert.ok(typeof devices === 'number');
    assert.ok(devices >= 1);
  });
});

describe('Sampler Params', { skip: !MULLAMA_AVAILABLE }, () => {
  test('samplerParamsGreedy returns deterministic params', () => {
    const params = mullama.samplerParamsGreedy();
    assert.strictEqual(params.temperature, 0.0);
    assert.strictEqual(params.topK, 1);
  });

  test('samplerParamsCreative returns high randomness params', () => {
    const params = mullama.samplerParamsCreative();
    assert.ok(params.temperature > 1.0);
    assert.ok(params.topK > 40);
  });

  test('samplerParamsPrecise returns low randomness params', () => {
    const params = mullama.samplerParamsPrecise();
    assert.ok(params.temperature < 0.5);
    assert.ok(params.topK < 40);
  });
});

describe('Cosine Similarity', { skip: !MULLAMA_AVAILABLE }, () => {
  test('identical vectors have similarity 1', () => {
    const a = [1.0, 0.0, 0.0];
    const b = [1.0, 0.0, 0.0];
    const sim = mullama.cosineSimilarity(a, b);
    assert.ok(Math.abs(sim - 1.0) < 0.001);
  });

  test('orthogonal vectors have similarity 0', () => {
    const a = [1.0, 0.0, 0.0];
    const b = [0.0, 1.0, 0.0];
    const sim = mullama.cosineSimilarity(a, b);
    assert.ok(Math.abs(sim) < 0.001);
  });

  test('opposite vectors have similarity -1', () => {
    const a = [1.0, 0.0, 0.0];
    const b = [-1.0, 0.0, 0.0];
    const sim = mullama.cosineSimilarity(a, b);
    assert.ok(Math.abs(sim + 1.0) < 0.001);
  });

  test('random vectors have similarity in [-1, 1]', () => {
    const a = Array.from({ length: 128 }, () => Math.random() - 0.5);
    const b = Array.from({ length: 128 }, () => Math.random() - 0.5);
    const sim = mullama.cosineSimilarity(a, b);
    assert.ok(sim >= -1.0 && sim <= 1.0);
  });

  test('throws on mismatched lengths', () => {
    const a = [1.0, 0.0];
    const b = [1.0, 0.0, 0.0];
    assert.throws(() => {
      mullama.cosineSimilarity(a, b);
    });
  });
});

describe('Model Loading', { skip: !MULLAMA_AVAILABLE }, () => {
  test('loading nonexistent model throws', () => {
    assert.throws(() => {
      mullama.JsModel.load('/nonexistent/path/model.gguf');
    });
  });
});

describe('Model Loading (with model)', { skip: !MULLAMA_AVAILABLE || !MODEL_AVAILABLE }, () => {
  test('load model successfully', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    assert.ok(model);
    assert.ok(model.nVocab > 0);
    assert.ok(model.nEmbd > 0);
  });

  test('model properties are accessible', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);

    assert.ok(model.nCtxTrain > 0);
    assert.ok(model.nEmbd > 0);
    assert.ok(model.nVocab > 0);
    assert.ok(model.nLayer > 0);
    assert.ok(model.nHead > 0);
    assert.ok(model.size > 0);
    assert.ok(model.nParams > 0);
    assert.ok(model.tokenBos >= 0);
    assert.ok(model.tokenEos >= 0);
  });

  test('tokenize returns tokens', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const tokens = model.tokenize('Hello, world!');

    assert.ok(Array.isArray(tokens));
    assert.ok(tokens.length > 0);
    assert.ok(tokens.every((t) => typeof t === 'number'));
  });

  test('tokenize and detokenize roundtrip', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const text = 'Hello, world!';
    const tokens = model.tokenize(text);
    const decoded = model.detokenize(tokens);

    assert.ok(typeof decoded === 'string');
  });

  test('model metadata returns object', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const metadata = model.metadata();

    assert.ok(typeof metadata === 'object');
  });
});

describe('Context', { skip: !MULLAMA_AVAILABLE || !MODEL_AVAILABLE }, () => {
  test('create context successfully', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model);

    assert.ok(ctx);
    assert.ok(ctx.nCtx > 0);
    assert.ok(ctx.nBatch > 0);
  });

  test('create context with params', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model, { nCtx: 512, nBatch: 256 });

    assert.strictEqual(ctx.nCtx, 512);
    assert.strictEqual(ctx.nBatch, 256);
  });

  test('generate text', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model, { nCtx: 256 });
    const params = mullama.samplerParamsGreedy();

    const text = ctx.generate('Hello', 10, params);

    assert.ok(typeof text === 'string');
    assert.ok(text.length > 0);
  });

  test('generate from tokens', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model, { nCtx: 256 });

    const tokens = model.tokenize('Hello');
    const params = mullama.samplerParamsGreedy();
    const text = ctx.generateFromTokens(tokens, 10, params);

    assert.ok(typeof text === 'string');
  });

  test('generate stream returns array', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model, { nCtx: 256 });
    const params = mullama.samplerParamsGreedy();

    const pieces = ctx.generateStream('Hello', 10, params);

    assert.ok(Array.isArray(pieces));
    assert.ok(pieces.every((p) => typeof p === 'string'));
  });

  test('clear cache does not throw', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const ctx = new mullama.JsContext(model, { nCtx: 256 });

    assert.doesNotThrow(() => {
      ctx.clearCache();
    });
  });
});

describe('Embedding Generator', { skip: !MULLAMA_AVAILABLE || !MODEL_AVAILABLE }, () => {
  test('create generator successfully', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const gen = new mullama.JsEmbeddingGenerator(model);

    assert.ok(gen);
    assert.ok(gen.nEmbd > 0);
  });

  test('embed text returns array', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const gen = new mullama.JsEmbeddingGenerator(model);

    const embedding = gen.embed('Hello, world!');

    assert.ok(Array.isArray(embedding));
    assert.strictEqual(embedding.length, gen.nEmbd);
    assert.ok(embedding.every((v) => typeof v === 'number'));
  });

  test('embed batch returns array of arrays', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const gen = new mullama.JsEmbeddingGenerator(model);

    const texts = ['Hello', 'World', 'Test'];
    const embeddings = gen.embedBatch(texts);

    assert.ok(Array.isArray(embeddings));
    assert.strictEqual(embeddings.length, 3);
    for (const emb of embeddings) {
      assert.ok(Array.isArray(emb));
      assert.strictEqual(emb.length, gen.nEmbd);
    }
  });

  test('normalized embeddings have unit norm', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const gen = new mullama.JsEmbeddingGenerator(model, 512, true);

    const embedding = gen.embed('Hello');
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));

    assert.ok(Math.abs(norm - 1.0) < 0.01);
  });

  test('same text produces similar embeddings', () => {
    const model = mullama.JsModel.load(TEST_MODEL_PATH);
    const gen = new mullama.JsEmbeddingGenerator(model);

    const emb1 = gen.embed('Hello, world!');
    const emb2 = gen.embed('Hello, world!');

    const sim = mullama.cosineSimilarity(emb1, emb2);
    assert.ok(sim > 0.99);
  });
});
