---
title: PHP Bindings
description: PHP bindings for Mullama LLM inference using PHP FFI for direct access to the native shared library, with Laravel and Symfony integration examples.
---

# PHP Bindings

PHP bindings for the Mullama LLM library, using PHP's built-in [FFI extension](https://www.php.net/manual/en/book.ffi.php) for direct access to the native shared library. Provides model loading, text generation, embeddings, and sampler presets with framework integrations for Laravel and Symfony.

## Installation

### Via Composer

```bash
composer require mullama/mullama
```

### Prerequisites

The PHP bindings require:

1. **PHP >= 8.1** with the FFI extension enabled
2. The pre-built `libmullama_ffi` shared library
3. The `mullama.h` header file

#### Building the Shared Library

```bash
# From the mullama source directory
cargo build --release -p mullama-ffi

# Output locations:
# Linux:  target/release/libmullama_ffi.so
# macOS:  target/release/libmullama_ffi.dylib
# Windows: target/release/mullama_ffi.dll
```

#### Enabling PHP FFI

Ensure the FFI extension is enabled in your `php.ini`:

```ini
extension=ffi

; For development (allows FFI::cdef)
ffi.enable=true

; For production (preloaded only)
; ffi.enable=preload
```

#### Library Placement

Place the shared library and header file where the PHP bindings can find them. The bindings search in this order:

1. Path set via `Mullama::setLibraryPath()`
2. `bindings/ffi/include/mullama.h` (relative to package)
3. `/usr/local/include/mullama.h` and `/usr/local/lib/libmullama_ffi.so`
4. `/usr/include/mullama.h` and `/usr/lib/libmullama_ffi.so`

```bash
# System-wide installation (Linux)
sudo cp target/release/libmullama_ffi.so /usr/local/lib/
sudo cp bindings/ffi/include/mullama.h /usr/local/include/
sudo ldconfig
```

---

## Quick Start

```php
<?php

require_once 'vendor/autoload.php';

use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;

// Load a model
$model = Model::load('./model.gguf', ['nGpuLayers' => 32]);

// Create a context
$ctx = new Context($model, ['nCtx' => 2048]);

// Generate text
$text = $ctx->generate("Once upon a time", 100, SamplerParams::greedy());
echo $text . "\n";
```

---

## API Reference

### Mullama (Main Class)

The `Mullama` class manages the FFI library loading and backend lifecycle.

#### `Mullama::initialize()`

Initialize the backend. Called automatically on first use.

```php
public static function initialize(): void
```

#### `Mullama::shutdown()`

Free backend resources. Call before process exit.

```php
public static function shutdown(): void
```

#### `Mullama::setLibraryPath(path)`

Set a custom path to the `libmullama_ffi` library before initialization.

```php
public static function setLibraryPath(string $path): void
```

```php
// Must be called before any other Mullama operations
Mullama\Mullama::setLibraryPath('/opt/mullama/lib/libmullama_ffi.so');
```

#### `Mullama::supportsGpuOffload()`

Check if GPU offloading is available.

```php
public static function supportsGpuOffload(): bool
```

#### `Mullama::systemInfo()`

Get system information about the backend.

```php
public static function systemInfo(): string
```

#### `Mullama::maxDevices()`

Get the maximum number of compute devices.

```php
public static function maxDevices(): int
```

#### `Mullama::version()`

Get the library version.

```php
public static function version(): string
```

#### `Mullama::getLastError()`

Get the last error message from the FFI layer.

```php
public static function getLastError(): string
```

---

### Model

The `Model` class handles model loading, tokenization, and model information.

#### `Model::load(path, params)`

Load a model from a GGUF file.

```php
public static function load(string $path, array $params = []): self
```

**Parameters (via `$params` array):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nGpuLayers` | `int` | `0` | Layers to offload to GPU (0 = CPU, -1 = all) |
| `useMmap` | `bool` | `true` | Use memory mapping |
| `useMlock` | `bool` | `false` | Lock model in memory |
| `vocabOnly` | `bool` | `false` | Only load vocabulary |

**Throws:** `RuntimeException` if loading fails

```php
// CPU only
$model = Model::load('./model.gguf');

// GPU accelerated
$model = Model::load('./model.gguf', ['nGpuLayers' => -1]);

// Vocabulary only
$model = Model::load('./model.gguf', ['vocabOnly' => true]);
```

---

#### `$model->free()`

Release model resources. Called automatically by the destructor.

```php
public function free(): void
```

---

#### `$model->tokenize(text, addBos, special)`

Convert text to token IDs.

```php
public function tokenize(string $text, bool $addBos = true, bool $special = false): array
```

**Returns:** `int[]` array of token IDs

```php
$tokens = $model->tokenize('Hello, world!');
print_r($tokens); // [1, 10994, 29892, 3186, 29991]
```

---

#### `$model->detokenize(tokens)`

Convert token IDs back to text.

```php
public function detokenize(array $tokens): string
```

```php
$text = $model->detokenize([1, 10994, 29892, 3186, 29991]);
echo $text; // "Hello, world!"
```

---

#### Model Properties

```php
public function nCtxTrain(): int    // Training context size
public function nEmbd(): int        // Embedding dimension
public function nVocab(): int       // Vocabulary size
public function nLayer(): int       // Number of layers
public function nHead(): int        // Number of attention heads
public function tokenBos(): int     // BOS token ID
public function tokenEos(): int     // EOS token ID
public function size(): int         // Model size in bytes
public function nParams(): int      // Number of parameters
public function description(): string  // Model description
public function tokenIsEog(int $token): bool  // Check if token is EOG
```

```php
$model = Model::load('./model.gguf');

echo "Description: " . $model->description() . "\n";
echo "Parameters: " . number_format($model->nParams()) . "\n";
echo "Layers: " . $model->nLayer() . "\n";
echo "Embedding dim: " . $model->nEmbd() . "\n";
echo "Size: " . round($model->size() / 1e9, 2) . " GB\n";
```

---

### Context

The `Context` class provides text generation capabilities.

#### `new Context(model, params)`

Create a new inference context.

```php
public function __construct(Model $model, array $params = [])
```

**Parameters (via `$params` array):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nCtx` | `int` | `0` | Context size (0 = model default) |
| `nBatch` | `int` | `2048` | Batch size |
| `nThreads` | `int` | `0` | Thread count (0 = auto) |
| `embeddings` | `bool` | `false` | Enable embeddings mode |

**Throws:** `RuntimeException` if creation fails

```php
$ctx = new Context($model, [
    'nCtx' => 4096,
    'nBatch' => 512,
]);
```

---

#### `$ctx->generate(prompt, maxTokens, params)`

Generate text from a prompt.

```php
public function generate(string $prompt, int $maxTokens = 100, ?SamplerParams $params = null): string
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `$prompt` | `string` | (required) | Text prompt |
| `$maxTokens` | `int` | `100` | Maximum tokens to generate |
| `$params` | `?SamplerParams` | `null` | Sampling parameters (null = defaults) |

```php
$text = $ctx->generate("Hello, AI!", 100);
echo $text;

// With custom params
$text = $ctx->generate("Write a poem:", 200, new SamplerParams([
    'temperature' => 0.9,
    'topP' => 0.95,
]));
```

---

#### `$ctx->generateFromTokens(tokens, maxTokens, params)`

Generate text from pre-tokenized input.

```php
public function generateFromTokens(array $tokens, int $maxTokens = 100, ?SamplerParams $params = null): string
```

```php
$tokens = $model->tokenize("Hello!");
$text = $ctx->generateFromTokens($tokens, 100);
```

---

#### `$ctx->generateStream(prompt, maxTokens, params)`

Generate text and return token pieces as an array.

```php
public function generateStream(string $prompt, int $maxTokens = 100, ?SamplerParams $params = null): array
```

**Returns:** `string[]` array of generated text segments

```php
$pieces = $ctx->generateStream("Once upon a time", 100);
foreach ($pieces as $piece) {
    echo $piece;
    flush();
}
```

!!! note
    The current PHP implementation returns the full result as a single-element array. True token-by-token streaming requires the C-level callback mechanism which is not directly exposed through PHP FFI.

---

#### `$ctx->clearCache()`

Clear the KV cache.

```php
public function clearCache(): void
```

---

#### Context Properties

```php
public function nCtx(): int    // Context size
public function nBatch(): int  // Batch size
```

---

### SamplerParams

The `SamplerParams` class configures text generation sampling.

#### Constructor

```php
public function __construct(array $params = [])
```

**Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `$temperature` | `float` | `0.8` | Randomness (0.0 = deterministic) |
| `$topK` | `int` | `40` | Top-k sampling (0 = disabled) |
| `$topP` | `float` | `0.95` | Nucleus sampling (1.0 = disabled) |
| `$minP` | `float` | `0.05` | Min-p sampling (0.0 = disabled) |
| `$typicalP` | `float` | `1.0` | Typical sampling (1.0 = disabled) |
| `$penaltyRepeat` | `float` | `1.1` | Repeat penalty (1.0 = disabled) |
| `$penaltyFreq` | `float` | `0.0` | Frequency penalty |
| `$penaltyPresent` | `float` | `0.0` | Presence penalty |
| `$penaltyLastN` | `int` | `64` | Token window for penalties |
| `$seed` | `int` | `0` | Random seed (0 = random) |

```php
// Default parameters
$params = new SamplerParams();

// Custom parameters
$params = new SamplerParams([
    'temperature' => 0.7,
    'topK' => 50,
    'topP' => 0.9,
]);

// Direct property access
$params->temperature = 0.5;
echo $params->topK; // 50
```

---

#### Preset Methods

```php
// Deterministic generation
$params = SamplerParams::greedy();
// temperature=0.0, topK=1

// Creative generation
$params = SamplerParams::creative();
// temperature=1.2, topK=100

// Focused generation
$params = SamplerParams::precise();
// temperature=0.3, topK=20
```

---

### EmbeddingGenerator

The `EmbeddingGenerator` class creates text embeddings.

#### Constructor

```php
public function __construct(Model $model, int $nCtx = 512, bool $normalize = true)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `$model` | `Model` | (required) | Model for embeddings |
| `$nCtx` | `int` | `512` | Context size |
| `$normalize` | `bool` | `true` | Normalize to unit length |

```php
$gen = new EmbeddingGenerator($model);
// or
$gen = new EmbeddingGenerator($model, 1024, true);
```

---

#### `$gen->embed(text)`

Generate an embedding vector for text.

```php
public function embed(string $text): array
```

**Returns:** `float[]` embedding vector

```php
$embedding = $gen->embed("Hello, world!");
echo "Dimensions: " . count($embedding) . "\n";
```

---

#### `$gen->embedBatch(texts)`

Generate embeddings for multiple texts.

```php
public function embedBatch(array $texts): array
```

**Returns:** `float[][]` array of embedding vectors

```php
$texts = ['Hello', 'World', 'Test'];
$embeddings = $gen->embedBatch($texts);
echo "Count: " . count($embeddings) . "\n";
```

---

#### `$gen->nEmbd()`

Get the embedding dimension.

```php
public function nEmbd(): int
```

---

#### `EmbeddingGenerator::cosineSimilarity(a, b)`

Compute cosine similarity between two vectors.

```php
public static function cosineSimilarity(array $a, array $b): float
```

**Throws:** `RuntimeException` if vectors have different lengths.

```php
$sim = EmbeddingGenerator::cosineSimilarity($emb1, $emb2);
echo "Similarity: " . round($sim, 4) . "\n";
```

---

## Examples

### Basic Text Generation

```php
<?php

require_once 'vendor/autoload.php';

use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;

$model = Model::load('./model.gguf', ['nGpuLayers' => -1]);
$ctx = new Context($model, ['nCtx' => 2048]);

$response = $ctx->generate(
    "Explain PHP in one paragraph:",
    150,
    new SamplerParams(['temperature' => 0.7, 'topP' => 0.9])
);

echo $response . "\n";
```

### Embeddings and Similarity

```php
<?php

require_once 'vendor/autoload.php';

use Mullama\Model;
use Mullama\EmbeddingGenerator;

$model = Model::load('./embedding-model.gguf');
$gen = new EmbeddingGenerator($model);

// Index documents
$documents = [
    'PHP is a server-side scripting language',
    'JavaScript runs in web browsers',
    'Python is popular for machine learning',
    'The weather is sunny today',
];

$docEmbeddings = $gen->embedBatch($documents);

// Query
$queryEmb = $gen->embed("What language is used for AI?");

// Rank by similarity
$results = [];
foreach ($docEmbeddings as $i => $docEmb) {
    $score = EmbeddingGenerator::cosineSimilarity($queryEmb, $docEmb);
    $results[] = ['text' => $documents[$i], 'score' => $score];
}

usort($results, fn($a, $b) => $b['score'] <=> $a['score']);

echo "Search results:\n";
foreach ($results as $result) {
    printf("  [%.4f] %s\n", $result['score'], $result['text']);
}
```

### Tokenization

```php
<?php

use Mullama\Model;

$model = Model::load('./model.gguf');

// Tokenize
$tokens = $model->tokenize("Hello, world!");
echo "Tokens: " . implode(', ', $tokens) . "\n";
echo "Token count: " . count($tokens) . "\n";

// Detokenize
$text = $model->detokenize($tokens);
echo "Text: {$text}\n";

// Model info
echo "Vocab size: " . $model->nVocab() . "\n";
echo "BOS token: " . $model->tokenBos() . "\n";
echo "EOS token: " . $model->tokenEos() . "\n";
```

### Laravel Service Provider

```php
<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use Mullama\Model;
use Mullama\Context;

class MullamaServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->app->singleton(Model::class, function () {
            return Model::load(
                config('mullama.model_path'),
                ['nGpuLayers' => config('mullama.gpu_layers', 0)]
            );
        });

        $this->app->bind(Context::class, function ($app) {
            return new Context($app->make(Model::class), [
                'nCtx' => config('mullama.context_size', 2048),
            ]);
        });
    }
}
```

Usage in a controller:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Mullama\Context;
use Mullama\SamplerParams;

class GenerateController extends Controller
{
    public function __construct(
        private Context $context,
    ) {}

    public function generate(Request $request)
    {
        $validated = $request->validate([
            'prompt' => 'required|string|max:4096',
            'max_tokens' => 'integer|min:1|max:2000',
            'temperature' => 'numeric|min:0|max:2',
        ]);

        $params = new SamplerParams([
            'temperature' => $validated['temperature'] ?? 0.8,
        ]);

        $text = $this->context->generate(
            $validated['prompt'],
            $validated['max_tokens'] ?? 100,
            $params
        );

        return response()->json(['text' => $text]);
    }
}
```

### Symfony Bundle Integration

```php
<?php
// config/services.yaml equivalent in PHP

namespace App;

use Mullama\Model;
use Mullama\Context;
use Mullama\EmbeddingGenerator;

// Service definitions
class MullamaFactory
{
    private ?Model $model = null;

    public function __construct(
        private string $modelPath,
        private int $gpuLayers = 0,
        private int $contextSize = 2048,
    ) {}

    public function getModel(): Model
    {
        if ($this->model === null) {
            $this->model = Model::load($this->modelPath, [
                'nGpuLayers' => $this->gpuLayers,
            ]);
        }
        return $this->model;
    }

    public function createContext(): Context
    {
        return new Context($this->getModel(), [
            'nCtx' => $this->contextSize,
        ]);
    }

    public function createEmbeddingGenerator(): EmbeddingGenerator
    {
        return new EmbeddingGenerator($this->getModel());
    }
}
```

---

## Error Handling

PHP bindings throw `RuntimeException` on errors:

```php
use Mullama\Model;
use Mullama\Context;
use RuntimeException;

try {
    $model = Model::load('./nonexistent.gguf');
} catch (RuntimeException $e) {
    echo "Load failed: " . $e->getMessage() . "\n";
}

try {
    $model = Model::load('./model.gguf');
    $ctx = new Context($model, ['nCtx' => 2048]);
    $text = $ctx->generate("Hello", 100);
} catch (RuntimeException $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
```

Errors from the FFI layer are automatically retrieved and included in the exception message.

---

## Configuration

### php.ini Settings

```ini
; Required: enable the FFI extension
extension=ffi

; FFI access level:
; "true" - allow FFI::cdef() (development)
; "preload" - only allow preloaded FFI (production)
; "false" - disable FFI
ffi.enable=true

; Preload the mullama FFI definitions (production)
; ffi.preload=/path/to/mullama_preload.php
```

### Custom Library Path

```php
<?php

use Mullama\Mullama;
use Mullama\Model;

// Set before any other Mullama operations
Mullama::setLibraryPath('/opt/mullama/lib/libmullama_ffi.so');

// Now load models as usual
$model = Model::load('./model.gguf');
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| PHP | >= 8.1 |
| FFI extension | enabled |
| `libmullama_ffi` | shared library |
| `mullama.h` | header file |
| Composer | >= 2.0 (for installation) |

---

## Performance Tips

1. **Reuse models** -- model loading is expensive. Use a singleton pattern or dependency injection to share a single model instance.

2. **GPU offloading** -- set `nGpuLayers` to `-1` to use all GPU layers when available.

3. **Context lifecycle** -- create contexts as needed and let them be garbage collected, or call `free()` explicitly in long-running processes.

4. **Batch embeddings** -- use `embedBatch()` rather than calling `embed()` in a loop.

5. **FFI preloading** -- in production, use `ffi.enable=preload` and preload the FFI definitions for better security and performance.

6. **Memory limits** -- large models may exceed PHP's default memory limit. Increase with `ini_set('memory_limit', '4G')` or in `php.ini`.
