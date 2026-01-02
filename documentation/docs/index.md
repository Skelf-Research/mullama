# Mullama

**A comprehensive Rust wrapper for llama.cpp with multimodal support**

Mullama provides safe, idiomatic Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling high-performance LLM inference with support for text, vision, and audio models.

## Features

- **Complete llama.cpp Integration** - 200+ FFI bindings covering the full API
- **Multimodal Support** - Vision-language models (LLaVA, Qwen-VL) and audio models
- **Safe Rust API** - Memory-safe wrappers with RAII resource management
- **Async/Streaming** - First-class support for streaming generation and async operations
- **GPU Acceleration** - CUDA, Metal, ROCm, and OpenCL support
- **Production Ready** - Used in real-world applications

## Quick Example

```rust
use mullama::{Model, Context, ContextParams, SamplingParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load a GGUF model
    let model = Arc::new(Model::load("model.gguf")?);

    // Create inference context
    let ctx_params = ContextParams::default();
    let mut context = Context::new(model.clone(), ctx_params)?;

    // Generate text
    let output = context.generate("Hello, world!", 100)?;
    println!("{}", output);

    Ok(())
}
```

## Multimodal Example

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load VLM model
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams::default())?;

    // Create multimodal context
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    // Load and process image
    let image = mtmd.bitmap_from_file("photo.jpg")?;
    let chunks = mtmd.tokenize("Describe this image: <__media__>", &[&image])?;

    // Evaluate and generate
    let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;
    // ... continue with text generation

    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1"
```

For multimodal support:

```toml
[dependencies]
mullama = { version = "0.1", features = ["multimodal"] }
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `multimodal` | Vision and audio model support |
| `async` | Tokio-based async operations |
| `streaming` | Real-time token streaming |
| `streaming-audio` | Live audio capture and processing |
| `web` | Axum web framework integration |
| `websockets` | WebSocket support for real-time APIs |
| `full` | Enable all features |

## System Requirements

### Linux (Ubuntu/Debian)

```bash
# Audio support
sudo apt install libasound2-dev libpulse-dev

# Image support
sudo apt install libpng-dev libjpeg-dev

# GPU acceleration (optional)
# CUDA: Install NVIDIA drivers and CUDA toolkit
# ROCm: Install AMD ROCm
```

### macOS

```bash
# Metal support is automatic on Apple Silicon
# For Intel Macs, OpenCL is used
```

### Windows

```powershell
# Visual Studio Build Tools required
# CUDA support available with NVIDIA drivers
```

## Next Steps

- [Getting Started](getting-started.md) - Detailed setup guide
- [User Guide](guide/models.md) - Learn the core concepts
- [API Reference](api/model.md) - Complete API documentation
- [Examples](examples/basic.md) - Working code examples
