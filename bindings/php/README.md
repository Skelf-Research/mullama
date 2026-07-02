# Mullama for PHP — local LLM inference with GGUF models via PHP FFI

[![Packagist Version](https://img.shields.io/packagist/v/mullama/mullama)](https://packagist.org/packages/mullama/mullama)
[![PHP Version](https://img.shields.io/packagist/php-v/mullama/mullama)](https://packagist.org/packages/mullama/mullama)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/cognisoc/mullama/blob/main/LICENSE)

## What is Mullama for PHP?

`mullama/mullama` is a PHP package that runs local LLMs from GGUF files — Llama 3.2, Qwen 2.5, DeepSeek R1, Mistral, Phi 3, Gemma 2, and LLaVA — directly inside your PHP process via the FFI extension. It loads the Mullama FFI library (a Rust core wrapping `llama.cpp`), giving you native llama.cpp throughput without spawning the Ollama daemon and without shelling out to a Python or Node sidecar.

Use it from a Laravel or Symfony app to add chat, RAG, or local AI features without depending on an external LLM service.

## Why use Mullama from PHP?

| | `mullama/mullama` | HTTP to `ollama serve` from PHP | Shelling out to `llama.cpp` |
|---|:-:|:-:|:-:|
| In-process inference (no daemon) | ✓ | ✗ | ✓ |
| Streaming generation | ✓ | ✓ (via SSE) | requires pipe parsing |
| Built-in embeddings | ✓ | ✓ (via HTTP) | requires extra binary |
| Idiomatic PHP API (classes + exceptions) | ✓ | partial | ✗ |
| Works with Laravel / Symfony | ✓ | ✓ | brittle |
| Vision (LLaVA / Moondream) | ✓ | ✓ | ✓ |

## Requirements

- PHP **7.4+** with the `ffi` extension enabled (`extension=ffi` in `php.ini`)
- Pre-built `libmullama_ffi` shared library for your platform (attached to every GitHub release as `mullama-ffi-<version>-<target>.tar.gz`)

## Install

```bash
composer require mullama/mullama
```

Then point the runtime at the FFI library:

```bash
export MULLAMA_LIB_PATH=/usr/local/lib/libmullama_ffi.so
export MULLAMA_HEADER_PATH=/usr/local/include/mullama.h
```

Or programmatically in PHP:

```php
putenv('MULLAMA_LIB_PATH=/path/to/libmullama_ffi.so');
putenv('MULLAMA_HEADER_PATH=/path/to/mullama.h');
```

| Platform | Library file |
|---|---|
| Linux | `libmullama_ffi.so` |
| macOS | `libmullama_ffi.dylib` |
| Windows | `mullama_ffi.dll` |

## Quick start

```php
<?php

use Mullama\Mullama;
use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;

Mullama::initialize();

$model = Model::load('./llama3.2-1b.gguf', ['nGpuLayers' => 32]);
$ctx = new Context($model, ['nCtx' => 2048]);

$params = new SamplerParams(['temperature' => 0.8, 'topP' => 0.95]);
echo $ctx->generate('What is the capital of France?', 100, $params);

$ctx->free();
$model->free();
Mullama::shutdown();
```

## Streaming

```php
foreach ($ctx->generateStream('Once upon a time', 100, $params) as $chunk) {
    echo $chunk;
}
```

## Embeddings (RAG-ready)

```php
use Mullama\EmbeddingGenerator;

$gen = new EmbeddingGenerator($model, 512, true);

$a = $gen->embed('How do I cook pasta?');
$b = $gen->embed("What's the best way to boil noodles?");
echo EmbeddingGenerator::cosineSimilarity($a, $b);  // ≈ 0.9

$vectors = $gen->embedBatch(['doc 1', 'doc 2', 'doc 3']);

$gen->free();
```

## Sampler presets

```php
$greedy   = SamplerParams::greedy();    // deterministic
$precise  = SamplerParams::precise();   // low randomness, factual
$creative = SamplerParams::creative();  // high randomness

echo $ctx->generate('Tell me a fact about France.', 100, $greedy);
```

| Preset | Temperature | Top-K | Top-P | Use case |
|---|---|---|---|---|
| `greedy()` | 0.0 | 1 | 1.0 | Deterministic, factual responses |
| `precise()` | 0.3 | 20 | 0.8 | Focused, consistent output |
| default | 0.8 | 40 | 0.95 | Balanced |
| `creative()` | 1.2 | 100 | 0.95 | Creative writing |

## Laravel / Symfony

Wire the model as a singleton in your service container so it loads once per worker:

```php
// Laravel: AppServiceProvider::register()
$this->app->singleton(\Mullama\Model::class, function () {
    \Mullama\Mullama::initialize();
    return \Mullama\Model::load(storage_path('models/llama3.2-1b.gguf'), [
        'nGpuLayers' => env('MULLAMA_GPU_LAYERS', 32),
    ]);
});
```

A `Context` is cheap to create per request; the heavy lifting is in `Model::load()`, which you do once.

See the [Laravel recipe](https://docs.cognisoc.com/mullama/recipes/laravel/) for a full chat-endpoint walkthrough.

## Supported models

Any GGUF file works. Common picks:

- **Llama 3.2** — 1B, 3B Instruct
- **Llama 3.1** — 8B, 70B
- **Qwen 2.5** — 0.5B – 72B, Coder variants
- **DeepSeek R1** — distilled 1.5B / 7B / 14B / 32B (reasoning)
- **Mistral / Mixtral / Codestral**
- **Phi 3 / 3.5** — mini, medium
- **Gemma 2** — 2B, 9B, 27B
- **Vision** — LLaVA 1.5, Moondream
- **Embeddings** — `nomic-embed`, `bge:small`, `bge:large`

## API reference

### `Mullama` (static utilities)

```php
Mullama::initialize();
Mullama::shutdown();
Mullama::version();           // e.g. "0.3.1"
Mullama::systemInfo();
Mullama::supportsGpuOffload();
Mullama::maxDevices();
```

### `Model`

```php
$model = Model::load('./model.gguf', [
    'nGpuLayers' => 32,
    'useMmap'    => true,
    'useMlock'   => false,
    'vocabOnly'  => false,
]);

$model->nCtxTrain();   $model->nEmbd();    $model->nVocab();
$model->nLayer();      $model->nHead();
$model->size();        $model->nParams();  $model->description();
$model->tokenBos();    $model->tokenEos();
$model->tokenIsEog($token);
$tokens = $model->tokenize('Hello, world!', true, false);
$text   = $model->detokenize($tokens);

$model->free();
```

### `Context`

```php
$ctx = new Context($model, [
    'nCtx'       => 2048,
    'nBatch'     => 512,
    'nThreads'   => 4,
    'embeddings' => false,
]);

$text   = $ctx->generate('Hello', 100, $params);
$text   = $ctx->generateFromTokens($tokens, 100, $params);
$chunks = $ctx->generateStream('Hello', 100, $params);  // iterable
$ctx->clearCache();
$ctx->free();
```

### `SamplerParams`

```php
$params = new SamplerParams([
    'temperature'    => 0.8,
    'topK'           => 40,
    'topP'           => 0.95,
    'minP'           => 0.05,
    'typicalP'       => 1.0,
    'penaltyRepeat'  => 1.1,
    'penaltyFreq'    => 0.0,
    'penaltyPresent' => 0.0,
    'penaltyLastN'   => 64,
    'seed'           => 0,
]);

SamplerParams::greedy();
SamplerParams::creative();
SamplerParams::precise();
```

### `EmbeddingGenerator`

```php
$gen = new EmbeddingGenerator($model, 512, true);
$vector  = $gen->embed('Hello, world!');
$vectors = $gen->embedBatch(['a', 'b', 'c']);
$dim     = $gen->nEmbd();
EmbeddingGenerator::cosineSimilarity($v1, $v2);
$gen->free();
```

## Error handling

All methods throw `RuntimeException` on failure:

```php
try {
    $model = Model::load('./nonexistent.gguf');
} catch (\RuntimeException $e) {
    error_log("Mullama: " . $e->getMessage());
}
```

## FAQ

### Is Mullama for PHP production ready?

The PHP binding is the youngest of the official bindings — Rust/Python/Node/Go are 0.3.x stable. The PHP layer is functional but expect more breaking changes through 0.3 minor versions. The underlying FFI library is the same one driving the others.

### Why PHP FFI instead of a PHP extension?

FFI lets us ship one shared library that every binding consumes; no per-language native extension to compile. Trade-off: slightly more startup overhead than a native ext (FFI has to declare the C ABI on first load), but well-amortised for any LLM call.

### Does it work with Laravel Octane / Swoole / RoadRunner?

Yes. With long-lived workers, load `Mullama::initialize()` and `Model::load(...)` once at worker boot. `Model` is safe to share across requests on the same worker; create a fresh `Context` per request.

### Does it work with PHP-FPM?

Yes, but every FPM worker pays the model-load cost on cold start. For a 1B model that's ~200ms; for 70B it's measured in seconds. Octane / Swoole / RoadRunner avoid this by keeping the worker warm.

### Does Mullama work with NVIDIA GPUs from PHP?

Yes — point `MULLAMA_LIB_PATH` at the CUDA variant of the FFI library (`mullama-ffi-<version>-x86_64-unknown-linux-gnu-cuda.tar.gz` from a GitHub release).

### Where do I report bugs?

[GitHub Issues](https://github.com/cognisoc/mullama/issues). For usage questions, [Discussions](https://github.com/cognisoc/mullama/discussions).

## Other language bindings

- [Rust core (`mullama` on crates.io)](https://crates.io/crates/mullama)
- [Python (`mullama` on PyPI)](https://pypi.org/project/mullama/) · [README](../python/README.md)
- [Node.js / TypeScript (`mullama` on npm)](https://www.npmjs.com/package/mullama) · [README](../node/README.md)
- [Go (`github.com/cognisoc/mullama` on pkg.go.dev)](https://pkg.go.dev/github.com/cognisoc/mullama) · [README](../go/README.md)
- [C/C++ (FFI library)](../ffi/README.md)

## Documentation

Full PHP docs at **[docs.cognisoc.com/mullama/bindings/php](https://docs.cognisoc.com/mullama/bindings/php/)**.

## Building the FFI library

```bash
cd bindings/ffi
cargo build --release
# Library at target/release/libmullama_ffi.{so,dylib} or target/release/mullama_ffi.dll
```

## Testing

```bash
composer test
MULLAMA_TEST_MODEL=./model.gguf composer test
```

## License

MIT

---

## Part of the Cognisoc stack

**[Cognisoc](https://www.cognisoc.com)** builds open-source LLM inference for every language and every device — *LLM inference, everywhere.* This project is one of six:

| Project | Language | What it does |
|---|---|---|
| mullama **(this project)** | Python · Node · Go · PHP · Rust · C | Local LLM runtime & server, drop-in Ollama alternative |
| [unillm](https://github.com/cognisoc/unillm) | Rust | Modular inference runtime, 47 architectures |
| [llamafu](https://github.com/cognisoc/llamafu) | Dart / Flutter | On-device inference for mobile apps |
| [llmdot](https://github.com/cognisoc/llmdot) | C# / .NET | Local GGUF inference for the .NET ecosystem |
| [cllm](https://github.com/cognisoc/cllm) | C | Bare-metal unikernel — boots straight into inference |
| [zigllm](https://github.com/cognisoc/zigllm) | Zig | Learn LLMs by building one, from tensors to text |

🌐 [cognisoc.com](https://www.cognisoc.com) · 📚 [docs.cognisoc.com](https://docs.cognisoc.com) · 🐙 [github.com/cognisoc](https://github.com/cognisoc)
