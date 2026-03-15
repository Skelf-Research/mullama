# Mullama Language Bindings

This directory contains language bindings for the Mullama LLM library, providing consistent APIs across multiple programming languages.

## Supported Languages

| Language | Directory | Package | Status |
|----------|-----------|---------|--------|
| **C/C++** | [`ffi/`](./ffi) | Header + shared library | Stable |
| **Python** | [`python/`](./python) | `mullama` on PyPI | Stable |
| **Node.js** | [`node/`](./node) | `mullama` on npm | Stable |
| **Go** | [`go/`](./go) | `github.com/neul-labs/mullama-go` | Stable |
| **PHP** | [`php/`](./php) | `mullama/mullama` on Packagist | Stable |

## Architecture

All bindings share a common C ABI layer (`ffi/`) that provides:

- Memory-safe handle management using Arc reference counting
- Thread-local error messages
- Callback-based streaming
- ~50 FFI functions covering the full Mullama API

```
┌─────────────────────────────────────┐
│         mullama (Rust core)          │
└────────────────┬────────────────────┘
                 │
┌────────────────┴────────────────────┐
│         mullama-ffi (C ABI)          │
│  Handle management, error codes      │
└────────────────┬────────────────────┘
                 │
   ┌─────────┬───┴───┬─────────┬─────────┐
   │         │       │         │         │
┌──▼───┐ ┌───▼──┐ ┌──▼───┐ ┌───▼──┐ ┌───▼──┐
│napi-rs│ │ PyO3 │ │PHP FFI│ │ cgo  │ │  C   │
│Node.js│ │Python│ │ PHP   │ │ Go   │ │ C++  │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

## Quick Start

### Python

```python
from mullama import Model, Context, SamplerParams

model = Model.load("./model.gguf", n_gpu_layers=32)
ctx = Context(model, n_ctx=2048)

for token in ctx.generate_stream("Once upon a time", max_tokens=100):
    print(token, end="", flush=True)
```

### Node.js

```javascript
import { JsModel, JsContext, samplerParamsGreedy } from 'mullama';

const model = await JsModel.load('./model.gguf', { nGpuLayers: 32 });
const ctx = new JsContext(model, { nCtx: 2048 });

const text = ctx.generate("Once upon a time", 100, samplerParamsGreedy());
console.log(text);
```

### Go

```go
import "github.com/neul-labs/mullama-go"

model, _ := mullama.LoadModel("./model.gguf", &mullama.ModelParams{NGpuLayers: 32})
ctx, _ := mullama.NewContext(model, &mullama.ContextParams{NCtx: 2048})

for token := range ctx.GenerateStream("Once upon a time", 100, nil) {
    fmt.Print(token)
}
```

### PHP

```php
use Mullama\Model;
use Mullama\Context;
use Mullama\SamplerParams;

$model = Model::load('./model.gguf', ['nGpuLayers' => 32]);
$ctx = new Context($model, ['nCtx' => 2048]);

$text = $ctx->generate("Once upon a time", 100, SamplerParams::greedy());
echo $text;
```

### C/C++

```c
#include "mullama.h"

MullamaMullamaModelParams params = {.n_gpu_layers = 32};
MullamaMullamaModel* model = mullama_model_load("./model.gguf", &params);

MullamaMullamaContextParams ctx_params = {.n_ctx = 2048};
MullamaMullamaContext* ctx = mullama_context_new(model, &ctx_params);

// ... generate text ...

mullama_context_free(ctx);
mullama_model_free(model);
```

## Consistent API

All bindings follow consistent naming conventions:

| Concept | Python | Node.js | Go | PHP | C |
|---------|--------|---------|-----|-----|---|
| Load model | `Model.load()` | `JsModel.load()` | `LoadModel()` | `Model::load()` | `mullama_model_load()` |
| Create context | `Context()` | `JsContext()` | `NewContext()` | `new Context()` | `mullama_context_new()` |
| Generate | `ctx.generate()` | `ctx.generate()` | `ctx.Generate()` | `$ctx->generate()` | `mullama_generate()` |
| Tokenize | `model.tokenize()` | `model.tokenize()` | `model.Tokenize()` | `$model->tokenize()` | `mullama_tokenize()` |

### Sampler Presets

All bindings provide the same sampler presets:

| Preset | Temperature | Top-K | Use Case |
|--------|-------------|-------|----------|
| `greedy()` | 0.0 | 1 | Deterministic output |
| `precise()` | 0.3 | 20 | Focused, factual |
| `default` | 0.8 | 40 | Balanced |
| `creative()` | 1.2 | 100 | Creative writing |

## Platform Support

Pre-built binaries are available for:

| Platform | CPU | CUDA | Metal |
|----------|-----|------|-------|
| Linux x64 | ✓ | ✓ | - |
| Linux ARM64 | ✓ | - | - |
| macOS x64 | ✓ | - | - |
| macOS ARM64 | ✓ | - | ✓ |
| Windows x64 | ✓ | ✓ | - |

## Building from Source

### Prerequisites

1. Rust toolchain (1.75+)
2. System dependencies (see main README)
3. Language-specific tools:
   - Python: maturin (`pip install maturin`)
   - Node.js: Node 16+, npm
   - Go: Go 1.21+
   - PHP: PHP 7.4+ with FFI extension

### Build Commands

```bash
# Build FFI library
cargo build --release -p mullama-ffi

# Build Python wheel
cd bindings/python && maturin build --release

# Build Node.js native module
cd bindings/node && npm install && npm run build

# Build Go (compilation check)
cd bindings/go && go build ./...

# Run PHP tests
cd bindings/php && composer install && composer test
```

## CI/CD

The bindings are built and tested via GitHub Actions:

- **`bindings.yml`**: CI workflow for PRs and pushes
  - Tests FFI layer
  - Builds Python wheels for all platforms
  - Builds Node.js native modules for all platforms
  - Tests Go bindings
  - Tests PHP bindings

- **`release-bindings.yml`**: Release workflow
  - Triggered manually with version input
  - Publishes Python to PyPI
  - Publishes Node.js to npm
  - Creates GitHub release with pre-built libraries

## Contributing

When adding a new binding:

1. Create a new directory under `bindings/`
2. Use the FFI layer (`bindings/ffi/`) for native interop
3. Follow the established API patterns for consistency
4. Add comprehensive tests
5. Create a README with installation and usage instructions
6. Update the CI workflows

## License

MIT OR Apache-2.0
