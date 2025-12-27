# Migration Guide

This guide helps developers migrate to Mullama from other Rust llama.cpp bindings or direct C++ usage.

## Mullama Version Upgrades

### Upgrading to v0.1.x (llama.cpp b7542)

#### Breaking Change: `flash_attn` → `flash_attn_type`

The `flash_attn: bool` field in `ContextParams` has been replaced with `flash_attn_type: llama_flash_attn_type` enum.

**Before (v0.0.x):**
```rust
let params = ContextParams {
    flash_attn: true,
    ..Default::default()
};
```

**After (v0.1.x):**
```rust
use mullama::sys::llama_flash_attn_type;

let params = ContextParams {
    flash_attn_type: llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED,
    ..Default::default()
};

// Or use AUTO for automatic detection (recommended):
let params = ContextParams {
    flash_attn_type: llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
    ..Default::default()
};
```

**Flash Attention Types:**
| Value | Description |
|-------|-------------|
| `LLAMA_FLASH_ATTN_TYPE_AUTO` | Auto-detect best setting (default) |
| `LLAMA_FLASH_ATTN_TYPE_DISABLED` | Disable flash attention |
| `LLAMA_FLASH_ATTN_TYPE_ENABLED` | Enable flash attention |

**Note:** The `ContextConfig` struct (used for file-based configuration) still uses `flash_attn: bool` for simplicity. It's automatically converted to the enum type when creating a context.

#### New Features in v0.1.x

- **Text Generation**: `context.generate()`, `generate_with_params()`, `generate_streaming()`
- **Automatic Batch Chunking**: Long prompts are automatically split into batches
- **LoRA Support**: Full adapter loading with `LoRAAdapter::load()`
- **llama.cpp b7542**: Latest model architecture support

---

## From Other Rust Bindings

### From `llama-rs`

Mullama provides a more comprehensive and memory-safe API:

```rust
// llama-rs
use llama_rs::models::Llama;

let model = Llama::load_from_file("model.gguf", params)?;
let result = model.predict("Hello".to_string(), params)?;

// Mullama (more features, better safety)
use mullama::{Model, Context, ContextParams, SamplerParams};

let model = Model::from_file("model.gguf")?;
let ctx_params = ContextParams::default().with_n_ctx(2048);
let mut context = Context::new(&model, ctx_params)?;

let sampler_params = SamplerParams::default()
    .with_temperature(0.7)
    .with_top_k(40);
let sampler = sampler_params.build_chain(&model);

// Much more control over generation process
```

### From `candle-llama`

```rust
// candle-llama
use candle_llama::LlamaModel;

let model = LlamaModel::load(&device, "model.gguf")?;

// Mullama (no device management needed)
use mullama::{Model, ModelParams};

let params = ModelParams::default()
    .with_n_gpu_layers(35);  // Automatic GPU management
let model = Model::from_file_with_params("model.gguf", params)?;
```

### From `rustformers/llm`

```rust
// rustformers/llm
use llm::models::Llama;

let model = Llama::load("model.gguf", &params, load_progress_callback)?;

// Mullama (simpler, more features)
use mullama::Model;

let model = Model::from_file("model.gguf")?;
// Built-in progress tracking, comprehensive error handling
```

## From Direct llama.cpp C++ Usage

### Basic Model Loading

```cpp
// C++ llama.cpp
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 35;
llama_model* model = llama_load_model_from_file("model.gguf", model_params);

llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;
llama_context* ctx = llama_new_context_with_model(model, ctx_params);
```

```rust
// Mullama (memory safe)
use mullama::{Model, Context, ModelParams, ContextParams};

let model_params = ModelParams::default()
    .with_n_gpu_layers(35);
let model = Model::from_file_with_params("model.gguf", model_params)?;

let ctx_params = ContextParams::default()
    .with_n_ctx(2048);
let context = Context::new(&model, ctx_params)?;

// Automatic memory management - no manual cleanup needed
```

### Tokenization

```cpp
// C++ llama.cpp
std::vector<llama_token> tokens = llama_tokenize(model, text, add_bos, false);

// Manual memory management required
```

```rust
// Mullama
let tokens = model.tokenize(text, add_bos)?;
// Automatic cleanup, error handling included
```

### Generation Loop

```cpp
// C++ llama.cpp
for (int i = 0; i < max_tokens; i++) {
    llama_token next_token = llama_sample_token(ctx, candidates);
    if (llama_token_is_eog(model, next_token)) break;

    llama_eval(ctx, &next_token, 1, tokens.size(), 0);
    tokens.push_back(next_token);

    // Manual string conversion
    std::string text = llama_token_to_piece(ctx, next_token);
    std::cout << text;
}
```

```rust
// Mullama (much cleaner)
for _ in 0..max_tokens {
    let next_token = sampler.sample(&context)?;
    if model.is_eog_token(next_token) { break; }

    let text = model.token_to_str(next_token)?;
    print!("{}", text);

    context.eval_token(next_token)?;
}
// Comprehensive error handling, automatic resource management
```

## Feature Comparison

| Feature | Other Bindings | Mullama |
|---------|---------------|---------|
| **Memory Safety** | ⚠️ Often unsafe | ✅ Zero unsafe in public API |
| **API Coverage** | ❌ Partial (30-60%) | ✅ Complete (100%) |
| **Error Handling** | ❌ Basic | ✅ Comprehensive enum-based |
| **GPU Support** | ⚠️ Limited | ✅ Full multi-GPU support |
| **Sampling** | ❌ Basic | ✅ All 15+ strategies |
| **Documentation** | ❌ Minimal | ✅ Comprehensive guides |
| **Testing** | ❌ Limited | ✅ 450+ test cases |
| **Performance** | ⚠️ Variable | ✅ Optimized with benchmarks |

## Common Migration Patterns

### Error Handling

```rust
// Old pattern (many bindings)
match model.generate(prompt) {
    Ok(result) => println!("{}", result),
    Err(e) => eprintln!("Error: {}", e),
}

// Mullama (comprehensive errors)
use mullama::MullamaError;

match model.tokenize(prompt, true) {
    Ok(tokens) => { /* handle success */ }
    Err(MullamaError::ModelError(msg)) => eprintln!("Model error: {}", msg),
    Err(MullamaError::TokenizationError(msg)) => eprintln!("Tokenization: {}", msg),
    Err(MullamaError::GpuError(msg)) => eprintln!("GPU error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Configuration

```rust
// Simple configuration (typical in other bindings)
let config = Config {
    temperature: 0.7,
    max_tokens: 100,
};

// Mullama (comprehensive configuration)
let model_params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_split_mode(SplitMode::Layer)
    .with_use_mmap(true);

let ctx_params = ContextParams::default()
    .with_n_ctx(4096)
    .with_n_batch(512)
    .with_rope_freq_scale(1.0);

let sampler_params = SamplerParams::default()
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9)
    .with_repeat_penalty(1.1);
```

### Resource Management

```rust
// Manual cleanup (some bindings)
{
    let model = load_model("model.gguf")?;
    let context = create_context(&model)?;

    // Use model and context

    // Manual cleanup required
    drop(context);
    drop(model);
}

// Mullama (automatic RAII)
{
    let model = Model::from_file("model.gguf")?;
    let context = Context::new(&model, params)?;

    // Use model and context

    // Automatic cleanup when going out of scope
}
```

## Performance Migration

### Threading

```rust
// Manual threading (typical pattern)
use std::thread;

let model = Arc::new(load_model("model.gguf")?);
let handles: Vec<_> = (0..4).map(|_| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        // Each thread needs its own context
        let context = create_context(&model)?;
        // Generate text...
    })
}).collect();

// Mullama (thread-safe by design)
use std::sync::Arc;

let model = Arc::new(Model::from_file("model.gguf")?);
let handles: Vec<_> = (0..4).map(|_| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        let context = Context::new(&model, params)?;
        // Thread-safe, automatic resource management
    })
}).collect();
```

### GPU Memory Management

```rust
// Manual GPU management (complex)
fn setup_gpu(layers: i32) -> Result<Model, Error> {
    // Check GPU memory
    // Calculate optimal layer distribution
    // Handle memory allocation failures
    // ...complex error handling...
}

// Mullama (automatic)
let model = Model::from_file_with_params("model.gguf",
    ModelParams::default().with_n_gpu_layers(35)
)?;
// Automatic GPU detection, memory management, error handling
```

## Migration Checklist

### Pre-Migration Assessment

- [ ] **Identify Dependencies**: List current binding and version
- [ ] **Audit Feature Usage**: What features does your code use?
- [ ] **Performance Requirements**: Note current performance characteristics
- [ ] **Error Handling**: Document current error scenarios
- [ ] **Threading Model**: Understand current concurrency patterns

### Migration Steps

1. **Add Mullama Dependency**
   ```toml
   [dependencies]
   mullama = "0.1.0"
   # Remove old binding
   ```

2. **Replace Imports**
   ```rust
   // Old
   use old_binding::*;

   // New
   use mullama::{Model, Context, ContextParams, SamplerParams};
   ```

3. **Update Model Loading**
   ```rust
   // Replace old model loading with Mullama equivalent
   let model = Model::from_file("model.gguf")?;
   ```

4. **Migrate Configuration**
   ```rust
   // Convert old config to Mullama parameter structs
   let params = ModelParams::default()
       .with_n_gpu_layers(old_config.gpu_layers);
   ```

5. **Update Generation Logic**
   ```rust
   // Replace generation loops with Mullama sampling API
   let sampler = SamplerParams::default()
       .with_temperature(old_config.temperature)
       .build_chain(&model);
   ```

6. **Test and Validate**
   - [ ] Run existing tests
   - [ ] Verify performance characteristics
   - [ ] Check memory usage
   - [ ] Validate output quality

### Post-Migration Optimization

1. **Leverage New Features**
   ```rust
   // Take advantage of advanced sampling
   let sampler = SamplerChain::new()
       .add_repetition_penalty(1.1, 64)
       .add_top_k(40)
       .add_top_p(0.9)
       .add_temperature(0.7);
   ```

2. **Optimize Performance**
   ```rust
   // Use batch processing for multiple sequences
   let batch = Batch::new()
       .add_sequence(tokens1, 0)
       .add_sequence(tokens2, 1);

   context.eval_batch(&batch)?;
   ```

3. **Add Comprehensive Error Handling**
   ```rust
   use mullama::MullamaError;

   match operation() {
       Ok(result) => handle_success(result),
       Err(MullamaError::GpuError(msg)) => handle_gpu_error(msg),
       Err(MullamaError::ModelError(msg)) => handle_model_error(msg),
       Err(e) => handle_other_error(e),
   }
   ```

## Common Issues and Solutions

### Build Errors

**Issue**: CMake or compiler errors during build
```bash
error: failed to run custom build command for `mullama`
```

**Solution**: Ensure prerequisites are installed
```bash
# Ubuntu/Debian
sudo apt install cmake build-essential

# macOS
brew install cmake

# Verify
cmake --version
```

### Performance Regression

**Issue**: Slower generation than previous binding

**Solution**: Optimize configuration
```rust
// Ensure GPU layers are set appropriately
let params = ModelParams::default()
    .with_n_gpu_layers(35)  // Adjust for your hardware
    .with_split_mode(SplitMode::Layer);

// Use larger batch sizes for throughput
let ctx_params = ContextParams::default()
    .with_n_batch(1024);  // Increase from default
```

### Memory Usage

**Issue**: Higher memory usage

**Solution**: Tune memory parameters
```rust
let params = ModelParams::default()
    .with_use_mmap(true)    // Use memory mapping
    .with_use_mlock(false); // Don't lock all memory

let ctx_params = ContextParams::default()
    .with_n_ctx(2048);      // Reduce if needed
```

## Getting Help

- **Migration Questions**: [GitHub Discussions](https://github.com/username/mullama/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/username/mullama/issues)
- **Performance Help**: Share your use case in discussions
- **Feature Requests**: Check our [Feature Status](FEATURE_STATUS.md) first

---

Welcome to Mullama! We're excited to have you experience the most comprehensive and safe Rust bindings for llama.cpp. 🦙✨