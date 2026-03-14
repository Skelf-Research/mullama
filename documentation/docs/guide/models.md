# Loading Models

Mullama supports loading GGUF format models with flexible configuration options.

## Basic Model Loading

```rust
use mullama::Model;
use std::sync::Arc;

let model = Arc::new(Model::load("path/to/model.gguf")?);
```

!!! note "Arc for Shared Ownership"
    Models are typically wrapped in `Arc` to allow sharing between contexts and threads.

## Model Builder

For advanced configuration, use the builder pattern:

```rust
use mullama::ModelBuilder;

let model = ModelBuilder::new("model.gguf")
    .with_n_gpu_layers(35)      // Offload layers to GPU
    .with_use_mmap(true)        // Memory-map the model file
    .with_use_mlock(false)      // Don't lock in RAM
    .with_vocab_only(false)     // Load full model, not just vocab
    .build()?;
```

## GPU Layer Offloading

Offload transformer layers to GPU for faster inference:

```rust
let model = ModelBuilder::new("model.gguf")
    .with_n_gpu_layers(35)  // Offload 35 layers
    .build()?;
```

| Value | Behavior |
|-------|----------|
| `0` | CPU only |
| `1-N` | Offload N layers to GPU |
| `-1` or large number | Offload all layers |

!!! tip "Finding the Right Balance"
    Start with all layers on GPU. If you run out of VRAM, reduce until it fits.
    Monitor with `nvidia-smi` (CUDA) or Activity Monitor (Metal).

## Memory Mapping

Memory mapping allows the OS to load model weights on-demand:

```rust
let model = ModelBuilder::new("model.gguf")
    .with_use_mmap(true)   // Enable mmap (default)
    .with_use_mlock(true)  // Lock pages in RAM (requires privileges)
    .build()?;
```

**Benefits of mmap:**

- Faster initial load time
- Shared memory between processes
- OS manages memory paging

**When to disable mmap:**

- Network file systems (NFS)
- Encrypted filesystems with poor random access
- When you need predictable latency

## Model Information

Query model properties after loading:

```rust
let model = Model::load("model.gguf")?;

// Vocabulary
println!("Vocab size: {}", model.vocab_size());
println!("BOS token: {:?}", model.bos_token());
println!("EOS token: {:?}", model.eos_token());

// Architecture
println!("Context length: {}", model.n_ctx_train());
println!("Embedding size: {}", model.n_embd());
println!("Layer count: {}", model.n_layer());

// Model metadata
println!("Description: {:?}", model.description());
```

## Tokenization

Convert between text and tokens:

```rust
// Text to tokens
let tokens = model.tokenize("Hello, world!", true, false)?;
// Args: text, add_bos, parse_special

println!("Tokens: {:?}", tokens);

// Token to text
let text = model.token_to_str(tokens[0], 0, false)?;
// Args: token, lstrip, special

println!("First token: {}", text);
```

### Special Tokens

```rust
// Check token types
if model.token_is_eog(token) {
    println!("End of generation");
}

// Get special token IDs
let bos = model.bos_token();  // Beginning of sequence
let eos = model.eos_token();  // End of sequence
```

## LoRA Adapters

Load fine-tuned LoRA adapters:

```rust
use mullama::{Model, LoraAdapter};

let model = Model::load("base-model.gguf")?;

// Load LoRA adapter
let lora = LoraAdapter::load("adapter.gguf")?;
model.apply_lora(&lora, 1.0)?;  // scale = 1.0
```

## Quantization Levels

GGUF models come in various quantization levels:

| Quantization | Bits | Quality | Speed | Size |
|--------------|------|---------|-------|------|
| F16 | 16 | Best | Slow | Large |
| Q8_0 | 8 | Excellent | Medium | Medium |
| Q6_K | 6 | Very Good | Fast | Small |
| Q5_K_M | 5 | Good | Fast | Smaller |
| Q4_K_M | 4 | Good | Fastest | Smallest |
| Q3_K_M | 3 | Acceptable | Fastest | Tiny |
| Q2_K | 2 | Poor | Fastest | Tiny |

!!! recommendation
    **Q4_K_M** offers the best balance of quality, speed, and size for most use cases.

## Thread Safety

Models are thread-safe for read operations:

```rust
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::load("model.gguf")?);

let handles: Vec<_> = (0..4).map(|i| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        // Each thread can create its own context
        let ctx = Context::new(model, ContextParams::default())?;
        // ... use context
        Ok::<_, mullama::MullamaError>(())
    })
}).collect();

for handle in handles {
    handle.join().unwrap()?;
}
```

## Error Handling

```rust
use mullama::{Model, MullamaError};

match Model::load("model.gguf") {
    Ok(model) => println!("Loaded successfully"),
    Err(MullamaError::ModelLoadError(msg)) => {
        eprintln!("Failed to load model: {}", msg);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Best Practices

1. **Use Arc for sharing** - Wrap models in `Arc` for multi-threaded use
2. **Match quantization to hardware** - Use Q4_K_M for consumer GPUs, Q8_0 for high-end
3. **Enable GPU offloading** - Significant speedup when VRAM allows
4. **Use mmap for large models** - Faster loading, better memory efficiency
5. **Cache loaded models** - Loading is expensive, reuse when possible
