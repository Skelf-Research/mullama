# GPU Acceleration Guide

Mullama supports GPU acceleration through multiple backends, dramatically improving inference speed for large models. This guide covers setup, configuration, and optimization for different GPU types.

## Supported GPU Backends

- **NVIDIA CUDA** - Best performance for NVIDIA GPUs
- **Apple Metal** - Optimized for Apple Silicon and AMD GPUs on macOS
- **AMD ROCm** - Support for AMD GPUs on Linux
- **Vulkan** - Cross-platform GPU compute (experimental)
- **OpenCL** - Broad compatibility (basic support)

## Quick Setup

### NVIDIA CUDA

Install CUDA Toolkit 11.0 or later:

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
```

Build with CUDA support:

```bash
export LLAMA_CUDA=1
cargo build --release
```

### Apple Metal

On macOS, Metal support is automatically available:

```bash
export LLAMA_METAL=1
cargo build --release
```

### AMD ROCm

Install ROCm 5.0+:

```bash
# Ubuntu/Debian
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev hip-dev
```

Build with ROCm:

```bash
export LLAMA_HIPBLAS=1
cargo build --release
```

## Configuration

### Basic GPU Usage

```rust
use mullama::{Model, ModelParams};

// Load model with GPU acceleration
let params = ModelParams::default()
    .with_n_gpu_layers(35);  // Offload 35 layers to GPU

let model = Model::from_file_with_params("model.gguf", params)?;
```

### Advanced GPU Configuration

```rust
use mullama::{ModelParams, SplitMode};

let params = ModelParams::default()
    .with_n_gpu_layers(35)              // Number of layers on GPU
    .with_split_mode(SplitMode::Layer)  // How to split across GPUs
    .with_main_gpu(0)                   // Primary GPU device
    .with_tensor_split(&[0.7, 0.3]);    // Split ratio across GPUs

let model = Model::from_file_with_params("model.gguf", params)?;
```

## Performance Optimization

### Determining Optimal Layer Count

Find the right balance between GPU memory and performance:

```rust
use mullama::{Model, ModelParams};

fn find_optimal_gpu_layers(model_path: &str) -> Result<i32, Box<dyn std::error::Error>> {
    // Start with all layers on GPU
    let mut best_layers = 0;
    let mut max_layers = 50; // Adjust based on your model

    for layers in (0..=max_layers).step_by(5) {
        let params = ModelParams::default()
            .with_n_gpu_layers(layers);

        match Model::from_file_with_params(model_path, params) {
            Ok(_) => {
                best_layers = layers;
                println!("✅ {} layers: OK", layers);
            }
            Err(e) => {
                println!("❌ {} layers: {}", layers, e);
                break;
            }
        }
    }

    Ok(best_layers)
}
```

### Memory Management

Monitor GPU memory usage:

```rust
use mullama::{Model, Context, ContextParams};

fn create_optimized_context(model: &Model) -> Result<Context, Box<dyn std::error::Error>> {
    // Start with smaller context and increase if memory allows
    let ctx_params = ContextParams::default()
        .with_n_ctx(2048)    // Start conservative
        .with_n_batch(512);  // Batch size affects GPU memory

    match Context::new(model, ctx_params) {
        Ok(ctx) => Ok(ctx),
        Err(_) => {
            // Try with smaller parameters
            let ctx_params = ContextParams::default()
                .with_n_ctx(1024)
                .with_n_batch(256);
            Context::new(model, ctx_params).map_err(Into::into)
        }
    }
}
```

## Multi-GPU Setup

### Load Balancing

```rust
use mullama::{ModelParams, SplitMode};

// Split model across multiple GPUs
let params = ModelParams::default()
    .with_n_gpu_layers(70)              // Total layers for GPU
    .with_split_mode(SplitMode::Layer)  // Split by layers
    .with_tensor_split(&[0.6, 0.4]);    // GPU 0 gets 60%, GPU 1 gets 40%

let model = Model::from_file_with_params("large_model.gguf", params)?;
```

### Custom GPU Selection

```rust
use mullama::{ModelParams, SplitMode};

// Use specific GPUs
let params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_main_gpu(1)                   // Use GPU 1 as primary
    .with_split_mode(SplitMode::Row);   // Alternative split method

let model = Model::from_file_with_params("model.gguf", params)?;
```

## Platform-Specific Optimization

### NVIDIA CUDA

```rust
use mullama::{ModelParams, CudaParams};

let params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_cuda_params(
        CudaParams::default()
            .with_mul_mat_q(true)       // Use quantized matrix multiplication
            .with_flash_attention(true) // Enable flash attention
    );
```

### Apple Metal

```rust
use mullama::{ModelParams, MetalParams};

let params = ModelParams::default()
    .with_n_gpu_layers(-1)  // Use all available GPU memory
    .with_metal_params(
        MetalParams::default()
            .with_unified_memory(true)  // Use unified memory architecture
    );
```

### AMD ROCm

```rust
use mullama::{ModelParams, RocmParams};

let params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_rocm_params(
        RocmParams::default()
            .with_device_id(0)          // Select specific AMD GPU
    );
```

## Benchmarking GPU Performance

```rust
use mullama::{Model, Context, ContextParams, ModelParams};
use std::time::Instant;

fn benchmark_gpu_performance(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = vec![
        ("CPU Only", 0),
        ("GPU 10 layers", 10),
        ("GPU 20 layers", 20),
        ("GPU 35 layers", 35),
    ];

    for (name, gpu_layers) in test_cases {
        println!("Testing: {}", name);

        let params = ModelParams::default()
            .with_n_gpu_layers(gpu_layers);

        let model = Model::from_file_with_params(model_path, params)?;

        let ctx_params = ContextParams::default()
            .with_n_ctx(2048)
            .with_n_batch(512);

        let mut context = Context::new(&model, ctx_params)?;

        // Benchmark tokenization and evaluation
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let tokens = model.tokenize(&prompt, true)?;

        let start = Instant::now();

        for token in tokens.iter().take(100) {
            context.eval_token(*token)?;
        }

        let duration = start.elapsed();
        let tokens_per_sec = 100.0 / duration.as_secs_f64();

        println!("  {} tokens/sec: {:.2}", name, tokens_per_sec);
    }

    Ok(())
}
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```rust
// Reduce GPU layers or context size
let params = ModelParams::default()
    .with_n_gpu_layers(20)  // Reduce from higher number
    .with_split_mode(SplitMode::Layer);

let ctx_params = ContextParams::default()
    .with_n_ctx(1024)       // Reduce context size
    .with_n_batch(256);     // Reduce batch size
```

#### GPU Not Detected

```rust
use mullama::gpu;

fn check_gpu_availability() {
    if gpu::cuda_available() {
        println!("✅ CUDA available");
        println!("   Devices: {}", gpu::cuda_device_count());
    } else {
        println!("❌ CUDA not available");
    }

    if gpu::metal_available() {
        println!("✅ Metal available");
    } else {
        println!("❌ Metal not available");
    }

    if gpu::rocm_available() {
        println!("✅ ROCm available");
        println!("   Devices: {}", gpu::rocm_device_count());
    } else {
        println!("❌ ROCm not available");
    }
}
```

#### Performance Issues

```rust
use mullama::{ModelParams, ContextParams};

// Optimize for throughput
let model_params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_split_mode(SplitMode::Layer)
    .with_use_mmap(false);      // Load into RAM for faster access

let ctx_params = ContextParams::default()
    .with_n_ctx(4096)
    .with_n_batch(1024)         // Larger batch for better GPU utilization
    .with_n_threads(1);         // Fewer CPU threads when using GPU
```

### Memory Optimization

#### Dynamic Layer Adjustment

```rust
use mullama::{Model, ModelParams, MullamaError};

fn load_model_with_auto_gpu(model_path: &str) -> Result<Model, MullamaError> {
    // Try with maximum GPU utilization first
    for layers in (0..=50).rev().step_by(5) {
        let params = ModelParams::default()
            .with_n_gpu_layers(layers);

        match Model::from_file_with_params(model_path, params) {
            Ok(model) => {
                println!("Successfully loaded with {} GPU layers", layers);
                return Ok(model);
            }
            Err(e) if e.to_string().contains("memory") => {
                continue; // Try fewer layers
            }
            Err(e) => return Err(e), // Other error
        }
    }

    // Fallback to CPU only
    Model::from_file(model_path)
}
```

#### Context Size Optimization

```rust
use mullama::{Context, ContextParams, Model};

fn create_context_with_available_memory(model: &Model) -> Result<Context, Box<dyn std::error::Error>> {
    let context_sizes = vec![8192, 4096, 2048, 1024, 512];

    for ctx_size in context_sizes {
        let params = ContextParams::default()
            .with_n_ctx(ctx_size)
            .with_n_batch(std::cmp::min(ctx_size / 4, 1024));

        match Context::new(model, params) {
            Ok(context) => {
                println!("Created context with size: {}", ctx_size);
                return Ok(context);
            }
            Err(_) => continue,
        }
    }

    Err("Unable to create context with any size".into())
}
```

## Best Practices

### Model Loading

1. **Start Conservative**: Begin with fewer GPU layers and increase
2. **Monitor Memory**: Watch GPU memory usage during loading
3. **Use Appropriate Split**: Choose split mode based on your setup
4. **Test Performance**: Benchmark different configurations

### Runtime Optimization

1. **Batch Size**: Larger batches improve GPU utilization
2. **Context Management**: Reuse contexts when possible
3. **Memory Monitoring**: Watch for memory leaks
4. **Thread Configuration**: Fewer CPU threads when using GPU

### Production Deployment

```rust
use mullama::{Model, Context, ModelParams, ContextParams};

pub struct OptimizedInference {
    model: Model,
    context: Context,
}

impl OptimizedInference {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Production-optimized parameters
        let model_params = ModelParams::default()
            .with_n_gpu_layers(35)          // Adjust for your hardware
            .with_split_mode(SplitMode::Layer)
            .with_use_mmap(true)            // Memory mapping for large models
            .with_use_mlock(true);          // Lock in memory

        let model = Model::from_file_with_params(model_path, model_params)?;

        let ctx_params = ContextParams::default()
            .with_n_ctx(4096)               // Production context size
            .with_n_batch(512)              // Optimized batch size
            .with_n_threads(num_cpus::get() as i32 / 2); // Leave CPU for other tasks

        let context = Context::new(&model, ctx_params)?;

        Ok(Self { model, context })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        // Implementation with optimized generation
        // ... generation logic ...
        Ok("Generated text".to_string())
    }
}
```

## Monitoring and Debugging

### GPU Memory Usage

```rust
use mullama::gpu;

fn monitor_gpu_memory() {
    if gpu::cuda_available() {
        for device in 0..gpu::cuda_device_count() {
            let (used, total) = gpu::cuda_memory_info(device);
            println!("GPU {}: {:.1}% used ({:.1} GB / {:.1} GB)",
                device,
                100.0 * used as f64 / total as f64,
                used as f64 / 1e9,
                total as f64 / 1e9
            );
        }
    }
}
```

### Performance Profiling

```rust
use std::time::Instant;
use mullama::{Model, Context};

fn profile_inference(model: &Model, context: &mut Context) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "The future of AI is";
    let tokens = model.tokenize(prompt, true)?;

    // Profile prompt processing
    let start = Instant::now();
    for token in tokens {
        context.eval_token(token)?;
    }
    let prompt_time = start.elapsed();

    println!("Prompt processing: {:.2}ms", prompt_time.as_millis());

    // Profile token generation
    let start = Instant::now();
    for _ in 0..100 {
        // Sample and evaluate tokens
        // ... sampling logic ...
    }
    let generation_time = start.elapsed();

    println!("Token generation: {:.2} tokens/sec",
        100.0 / generation_time.as_secs_f64());

    Ok(())
}
```

GPU acceleration can provide dramatic performance improvements, especially for large models. Start with the basic configuration and gradually optimize based on your specific hardware and use case.