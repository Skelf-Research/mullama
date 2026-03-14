# Getting Started

This guide will help you set up Mullama and run your first LLM inference.

## Prerequisites

### Rust Toolchain

Mullama requires Rust 1.70 or later:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### System Dependencies

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Build essentials
    sudo apt install build-essential cmake git

    # Audio support (optional)
    sudo apt install libasound2-dev libpulse-dev

    # Image support (optional)
    sudo apt install libpng-dev libjpeg-dev
    ```

=== "macOS"

    ```bash
    # Xcode command line tools
    xcode-select --install

    # CMake
    brew install cmake
    ```

=== "Windows"

    ```powershell
    # Install Visual Studio Build Tools
    # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

    # Install CMake
    winget install Kitware.CMake
    ```

## Installation

Add Mullama to your project:

```bash
cargo add mullama
```

Or manually edit `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1"
```

### Feature Selection

Enable only what you need:

```toml
# Minimal - text generation only
mullama = "0.1"

# With multimodal support
mullama = { version = "0.1", features = ["multimodal"] }

# With async and streaming
mullama = { version = "0.1", features = ["async", "streaming"] }

# Everything
mullama = { version = "0.1", features = ["full"] }
```

## Downloading Models

Mullama uses GGUF format models. You can download them from Hugging Face:

### Text Models

```bash
# Small model for testing (1.1B parameters)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Medium model (7B parameters)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### Vision-Language Models

```bash
# NanoLLaVA (small VLM for testing)
wget https://huggingface.co/qnguyen3/nanoLLaVA/resolve/main/nanollava-text-model-f16.gguf
wget https://huggingface.co/qnguyen3/nanoLLaVA/resolve/main/nanollava-mmproj-f16.gguf

# LLaVA 1.5 7B
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf
```

## Your First Program

Create a new project:

```bash
cargo new my-llm-app
cd my-llm-app
cargo add mullama
```

Edit `src/main.rs`:

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load the model
    println!("Loading model...");
    let model = Arc::new(Model::load("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")?);

    println!("Model loaded!");
    println!("  Vocabulary size: {}", model.vocab_size());
    println!("  Context length: {}", model.n_ctx_train());

    // Create inference context
    let params = ContextParams {
        n_ctx: 2048,      // Context window size
        n_batch: 512,     // Batch size for prompt processing
        n_threads: 4,     // CPU threads
        ..Default::default()
    };

    let mut context = Context::new(model.clone(), params)?;

    // Generate text
    println!("\nGenerating response...\n");

    let prompt = "Write a haiku about programming:";
    let response = context.generate(prompt, 100)?;

    println!("Prompt: {}", prompt);
    println!("Response: {}", response);

    Ok(())
}
```

Run it:

```bash
cargo run --release
```

!!! tip "Always use release mode"
    LLM inference is computationally intensive. Always build with `--release` for 10-50x better performance.

## GPU Acceleration

### NVIDIA CUDA

```bash
# Set environment variable before building
export LLAMA_CUDA=1
cargo build --release
```

### Apple Metal

```bash
# Automatic on Apple Silicon
# For explicit control:
export LLAMA_METAL=1
cargo build --release
```

### AMD ROCm

```bash
export LLAMA_HIPBLAS=1
cargo build --release
```

## Streaming Generation

For real-time output:

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams::default())?;

    // Stream tokens as they're generated
    context.generate_streaming("Tell me a story:", 200, |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        true  // Continue generation
    })?;

    Ok(())
}
```

## Next Steps

Now that you have Mullama running:

- [Loading Models](guide/models.md) - Advanced model configuration
- [Text Generation](guide/generation.md) - Sampling parameters and techniques
- [Multimodal](guide/multimodal.md) - Vision and audio support
- [Examples](examples/basic.md) - More code examples

## Troubleshooting

### Model Loading Fails

```
Error: ModelLoadError("Failed to load model")
```

- Verify the file path is correct
- Check the file is a valid GGUF format
- Ensure you have enough RAM (model size + ~2GB overhead)

### Out of Memory

```
Error: MemoryError("Failed to allocate")
```

- Use a smaller quantized model (Q4_K_M instead of F16)
- Reduce context size: `n_ctx: 1024`
- Enable GPU offloading if available

### Slow Generation

- Build with `--release` flag
- Enable GPU acceleration
- Increase batch size for longer prompts
- Use quantized models (Q4_K_M offers good speed/quality balance)
