# Getting Started with Mullama

Welcome to Mullama, the most comprehensive Rust bindings for llama.cpp! This guide will help you get up and running quickly.

## Installation

Add Mullama to your `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1.0"
```

## Prerequisites

### System Requirements
- **Rust**: 1.70 or later
- **CMake**: 3.12 or later
- **C++ Compiler**: GCC 8+, Clang 7+, or MSVC 2019+
- **Git**: For submodule support

### Optional GPU Support
- **NVIDIA CUDA**: 11.0+ for GPU acceleration
- **Apple Metal**: macOS 10.15+ for Metal acceleration
- **AMD ROCm**: 5.0+ for ROCm acceleration

## Your First Mullama Application

Let's create a simple text generation application:

### 1. Create a New Project

```bash
cargo new my_llama_app
cd my_llama_app
```

### 2. Add Dependencies

Edit `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

### 3. Basic Text Generation

Replace `src/main.rs`:

```rust
use mullama::{Model, Context, ContextParams, SamplerParams, MullamaError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the backend (required)
    mullama::init();

    println!("Loading model...");

    // Load your model (download a GGUF model first)
    let model = Model::from_file("path/to/your/model.gguf")?;

    println!("Model loaded successfully!");
    println!("Vocabulary size: {}", model.vocab_size());
    println!("Context size: {}", model.n_ctx_train());

    // Create a context with default parameters
    let ctx_params = ContextParams::default()
        .with_n_ctx(2048)     // Context window size
        .with_n_batch(512);   // Batch size for processing

    let mut context = Context::new(&model, ctx_params)?;

    // Set up sampling parameters
    let sampler_params = SamplerParams::default()
        .with_temperature(0.7)   // Controls randomness
        .with_top_k(40)          // Consider top 40 tokens
        .with_top_p(0.9);        // Nucleus sampling

    let sampler = sampler_params.build_chain(&model);

    // Generate text
    let prompt = "The future of artificial intelligence is";
    println!("Prompt: {}", prompt);
    print!("Generated: ");

    // Tokenize the prompt
    let tokens = model.tokenize(prompt, true)?;

    // Evaluate the prompt tokens
    for token in tokens {
        context.eval_token(token)?;
    }

    // Generate tokens one by one
    for i in 0..100 {
        let next_token = sampler.sample(&context)?;

        // Check for end of generation
        if model.is_eog_token(next_token) {
            println!("\n[End of generation]");
            break;
        }

        // Convert token to text and print
        let text = model.token_to_str(next_token)?;
        print!("{}", text);

        // Evaluate the new token
        context.eval_token(next_token)?;

        // Flush output for real-time display
        use std::io::{self, Write};
        io::stdout().flush()?;
    }

    println!("\n\nGeneration complete!");
    Ok(())
}
```

### 4. Download a Model

You'll need a GGUF format model. Popular options:

```bash
# Example: Download a small model for testing
wget https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.gguf

# Or use any other GGUF model from Hugging Face
```

### 5. Run Your Application

```bash
cargo run
```

## Understanding the Code

Let's break down what each part does:

### Backend Initialization
```rust
mullama::init();
```
This initializes the llama.cpp backend. Must be called before using any other functions.

### Model Loading
```rust
let model = Model::from_file("path/to/your/model.gguf")?;
```
Loads a GGUF model file. The model contains the neural network weights and metadata.

### Context Creation
```rust
let ctx_params = ContextParams::default()
    .with_n_ctx(2048)
    .with_n_batch(512);
let mut context = Context::new(&model, ctx_params)?;
```
Creates a context for inference. The context maintains the conversation state and KV cache.

### Sampling Configuration
```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9);
let sampler = sampler_params.build_chain(&model);
```
Configures how the model selects the next token. Multiple sampling strategies can be chained.

### Text Generation Loop
```rust
for i in 0..100 {
    let next_token = sampler.sample(&context)?;
    // ... process token ...
    context.eval_token(next_token)?;
}
```
The generation loop: sample a token, convert to text, and feed back to the context.

## Error Handling

Mullama uses comprehensive error types:

```rust
use mullama::MullamaError;

match model.tokenize(prompt, true) {
    Ok(tokens) => println!("Tokenized: {:?}", tokens),
    Err(MullamaError::ModelError(msg)) => eprintln!("Model error: {}", msg),
    Err(MullamaError::TokenizationError(msg)) => eprintln!("Tokenization failed: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Configuration Options

### Context Parameters
```rust
let ctx_params = ContextParams::default()
    .with_n_ctx(4096)           // Context window size
    .with_n_batch(512)          // Batch size
    .with_n_threads(8)          // CPU threads
    .with_rope_freq_base(10000.0) // RoPE frequency base
    .with_rope_freq_scale(1.0); // RoPE frequency scale
```

### Model Parameters
```rust
let model_params = ModelParams::default()
    .with_n_gpu_layers(35)      // GPU layers
    .with_split_mode(SplitMode::Layer) // GPU split mode
    .with_vocab_only(false)     // Load vocab only
    .with_use_mmap(true);       // Use memory mapping
```

### Sampling Parameters
```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.8)      // Randomness
    .with_top_k(50)            // Top-K filtering
    .with_top_p(0.95)          // Nucleus sampling
    .with_min_p(0.05)          // Minimum probability
    .with_typical_p(1.0)       // Typical sampling
    .with_repeat_penalty(1.1)   // Repetition penalty
    .with_repeat_last_n(64);   // Penalty lookback
```

## Next Steps

Now that you have a basic application running, explore these topics:

1. **[Model Loading](model-loading.md)** - Learn about different model formats and parameters
2. **[Text Generation](text-generation.md)** - Advanced generation techniques
3. **[Advanced Sampling](sampling.md)** - Fine-tune generation quality
4. **[GPU Acceleration](gpu.md)** - Speed up inference with GPU support
5. **[Examples](../examples/)** - Ready-to-use example applications

## Common Issues

### Build Errors
If you encounter build errors:

```bash
# Make sure submodules are initialized
git submodule update --init --recursive

# Clean and rebuild
cargo clean
cargo build
```

### Model Loading Errors
- Ensure the model file exists and is readable
- Check that the model is in GGUF format
- Verify you have enough RAM for the model

### Performance Issues
- Try reducing `n_ctx` if you're running out of memory
- Increase `n_batch` for better throughput
- Use GPU acceleration for large models

## Getting Help

- 📚 [Documentation](https://docs.rs/mullama)
- 💬 [GitHub Discussions](https://github.com/username/mullama/discussions)
- 🐛 [Report Issues](https://github.com/username/mullama/issues)

Happy coding with Mullama! 🦙