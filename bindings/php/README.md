# Mullama PHP Bindings

PHP bindings for the Mullama LLM library using PHP FFI (Foreign Function Interface).

## Requirements

- PHP 7.4 or later
- PHP FFI extension enabled
- Pre-built `libmullama_ffi` shared library

## Installation

```bash
composer require mullama/mullama
```

Or add to your `composer.json`:

```json
{
    "require": {
        "mullama/mullama": "^0.1"
    }
}
```

## Quick Start

```php
<?php

use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;
use Mullama\EmbeddingGenerator;

// Initialize the backend (call once at startup)
Mullama\Mullama::backendInit();

// Load a model
$model = Model::load('./model.gguf', [
    'nGpuLayers' => 32,  // GPU acceleration
]);

// Create a context
$ctx = new Context($model, [
    'nCtx' => 2048,
    'nBatch' => 512,
]);

// Generate text
$params = new SamplerParams([
    'temperature' => 0.8,
    'topK' => 40,
    'topP' => 0.95,
]);

$response = $ctx->generate("Once upon a time", 100, $params);
echo $response;

// Clean up
$ctx->free();
$model->free();
Mullama\Mullama::backendFree();
```

## API Reference

### Mullama (Static Utilities)

```php
// Initialize/shutdown backend
Mullama::backendInit();
Mullama::backendFree();

// Get library version
$version = Mullama::version();  // "0.1.0"

// System information
$info = Mullama::systemInfo();

// GPU support
$hasGpu = Mullama::supportsGpuOffload();
$maxDevices = Mullama::maxDevices();
```

### Model

```php
// Load a model
$model = Model::load('./model.gguf', [
    'nGpuLayers' => 32,     // Layers to offload to GPU
    'useMmap' => true,      // Memory-map the model
    'useMlock' => false,    // Lock model in RAM
    'vocabOnly' => false,   // Load only vocabulary
]);

// Model properties
$model->nCtxTrain();   // Training context size
$model->nEmbd();       // Embedding dimension
$model->nVocab();      // Vocabulary size
$model->nLayer();      // Number of layers
$model->nHead();       // Number of attention heads
$model->size();        // Model size in bytes
$model->nParams();     // Number of parameters
$model->description(); // Model description

// Special tokens
$model->tokenBos();    // Beginning-of-sequence token
$model->tokenEos();    // End-of-sequence token
$model->tokenIsEog($token);  // Check if token is end-of-generation

// Tokenization
$tokens = $model->tokenize("Hello, world!", true, false);
$text = $model->detokenize($tokens);

// Free resources
$model->free();
```

### Context

```php
// Create a context
$ctx = new Context($model, [
    'nCtx' => 2048,       // Context size
    'nBatch' => 512,      // Batch size
    'nThreads' => 4,      // CPU threads
    'embeddings' => false, // Enable embeddings
]);

// Context properties
$ctx->nCtx();    // Context size
$ctx->nBatch();  // Batch size

// Generate text
$text = $ctx->generate("Hello", 100, $params);

// Generate from tokens
$tokens = $model->tokenize("Hello", true, false);
$text = $ctx->generateFromTokens($tokens, 100, $params);

// Streaming generation (simplified)
$chunks = $ctx->generateStream("Hello", 100, $params);
foreach ($chunks as $chunk) {
    echo $chunk;
}

// Clear KV cache
$ctx->clearCache();

// Free resources
$ctx->free();
```

### SamplerParams

```php
// Default parameters
$params = new SamplerParams();

// Custom parameters
$params = new SamplerParams([
    'temperature' => 0.8,
    'topK' => 40,
    'topP' => 0.95,
    'minP' => 0.05,
    'typicalP' => 1.0,
    'penaltyRepeat' => 1.1,
    'penaltyFreq' => 0.0,
    'penaltyPresent' => 0.0,
    'penaltyLastN' => 64,
    'seed' => 0,
]);

// Preset configurations
$greedy = SamplerParams::greedy();    // Deterministic output
$creative = SamplerParams::creative(); // High randomness
$precise = SamplerParams::precise();   // Low randomness
```

### EmbeddingGenerator

```php
// Create embedding generator
$gen = new EmbeddingGenerator($model, 512, true);  // nCtx, normalize

// Generate embeddings
$embedding = $gen->embed("Hello, world!");

// Batch embedding
$embeddings = $gen->embedBatch(["Hello", "World", "Test"]);

// Get embedding dimension
$dim = $gen->nEmbd();

// Compute cosine similarity (static method)
$similarity = EmbeddingGenerator::cosineSimilarity($vec1, $vec2);

// Free resources
$gen->free();
```

## Sampler Presets

| Preset | Temperature | Top-K | Top-P | Use Case |
|--------|-------------|-------|-------|----------|
| `greedy()` | 0.0 | 1 | 1.0 | Deterministic, factual responses |
| `precise()` | 0.3 | 20 | 0.8 | Focused, consistent output |
| `default` | 0.8 | 40 | 0.95 | Balanced creativity |
| `creative()` | 1.2 | 100 | 0.95 | Creative writing, brainstorming |

## Error Handling

All methods throw `RuntimeException` on errors:

```php
try {
    $model = Model::load('./nonexistent.gguf');
} catch (RuntimeException $e) {
    echo "Error: " . $e->getMessage();
}
```

## Configuration

### Library Path

Set the `MULLAMA_LIB_PATH` environment variable to specify the library location:

```bash
export MULLAMA_LIB_PATH=/path/to/libmullama_ffi.so
```

Or specify programmatically:

```php
putenv('MULLAMA_LIB_PATH=/path/to/libmullama_ffi.so');
```

### Header Path

Set the `MULLAMA_HEADER_PATH` environment variable for the C header:

```bash
export MULLAMA_HEADER_PATH=/path/to/mullama.h
```

## Testing

```bash
# Run tests
composer test

# With a model file
MULLAMA_TEST_MODEL=./model.gguf composer test
```

## Building the FFI Library

```bash
# From the repository root
cd bindings/ffi
cargo build --release

# The library will be at target/release/libmullama_ffi.so (Linux)
# or target/release/libmullama_ffi.dylib (macOS)
# or target/release/mullama_ffi.dll (Windows)
```

## Platform Support

| Platform | Library Name |
|----------|-------------|
| Linux | `libmullama_ffi.so` |
| macOS | `libmullama_ffi.dylib` |
| Windows | `mullama_ffi.dll` |

## License

MIT License - see LICENSE file for details.
