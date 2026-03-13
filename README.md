# Mullama

A safe Rust wrapper for llama.cpp with built-in compilation.

## Features

- Builds llama.cpp as part of the Rust build process
- Safe Rust API with automatic memory management
- Session management for saving/restoring model states
- Support for embeddings, tokenization, and generation
- Cross-platform compatibility

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1"
```

Basic usage:

```rust
use mullama::{Model, ContextParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a model
    let model = Model::load("path/to/model.gguf")?;
    
    // Create a context
    let mut ctx = model.create_context(ContextParams::default())?;
    
    // Generate text
    let tokens = model.tokenize("Hello, world!")?;
    let result = ctx.generate(&tokens, 100)?;
    
    println!("Generated: {}", result);
    Ok(())
}
```

## Building

The crate automatically builds llama.cpp during `cargo build`. Make sure you have the required dependencies:

- C++ compiler (GCC/Clang/MSVC)
- CMake 3.12 or higher
- For GPU support: CUDA toolkit (for NVIDIA) or ROCm (for AMD)

## License

This project is licensed under the MIT License - see the LICENSE file for details.