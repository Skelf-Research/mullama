# Memory Management

Mullama provides comprehensive memory management through RAII patterns, configurable KV cache quantization, and monitoring tools for production environments. Understanding memory usage is essential for deploying models effectively.

## RAII in Mullama

All Mullama resources follow the RAII (Resource Acquisition Is Initialization) pattern: resources are automatically cleaned up when they go out of scope. No manual memory management is required.

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    async function processQuery(prompt) {
      // Context is created and used within this function scope
      const model = await Model.load('./model.gguf');
      const context = new Context(model);
      const response = await context.generate(prompt, 200);

      // When context/model go out of scope, they are garbage collected
      // and native resources are freed automatically
      return response;
    }

    // Explicit cleanup if needed
    async function explicitCleanup() {
      const model = await Model.load('./model.gguf');
      const context = new Context(model);

      const response = await context.generate("Hello!", 100);

      // Explicitly free resources before GC
      context.dispose();
      model.dispose();

      return response;
    }
    ```

=== "Python"

    ```python
    from mullama import Model, Context

    def process_query(prompt: str) -> str:
        # Resources freed when references are garbage collected
        model = Model.load("./model.gguf")
        context = Context(model)
        return context.generate(prompt, max_tokens=200)

    # Context manager for explicit cleanup
    def explicit_cleanup(prompt: str) -> str:
        model = Model.load("./model.gguf")
        with Context(model) as context:
            response = context.generate(prompt, max_tokens=200)
        # Context resources freed at end of 'with' block
        return response
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    fn process_query(prompt: &str) -> Result<String, mullama::MullamaError> {
        let model = Arc::new(Model::load("model.gguf")?);
        let mut context = Context::new(model, ContextParams::default())?;
        let response = context.generate(prompt, 200)?;

        // model and context are dropped here, freeing all C++ resources
        // via their Drop implementations
        Ok(response)
    }
    ```

=== "CLI"

    ```bash
    # CLI manages memory automatically
    mullama run llama3.2:1b "Hello!"
    # All resources freed after command completes
    ```

!!! info "Drop Implementations"
    In Rust, `Model`, `Context`, `Batch`, and `SamplerChain` all implement `Drop`, which calls the appropriate llama.cpp free functions. This guarantees no memory leaks even when errors occur.

## Arc for Shared Model Ownership

Models are expensive to load but safe to share. Use reference counting to share a single model across multiple contexts:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    // Model is reference-counted internally
    const model = await Model.load('./model.gguf');

    // Multiple contexts share the same model
    const context1 = new Context(model, { nCtx: 2048 });
    const context2 = new Context(model, { nCtx: 2048 });
    const context3 = new Context(model, { nCtx: 2048 });

    // Model stays alive as long as any reference exists
    // Total memory: 1x model + 3x context KV cache
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    # Python handles reference counting automatically
    model = Model.load("./model.gguf")

    # Multiple contexts share the same model
    context1 = Context(model, ContextParams(n_ctx=2048))
    context2 = Context(model, ContextParams(n_ctx=2048))
    context3 = Context(model, ContextParams(n_ctx=2048))

    # Model stays alive as long as any reference exists
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    // Wrap in Arc for shared ownership
    let model = Arc::new(Model::load("model.gguf")?);

    // Clone Arc (cheap reference count increment)
    let ctx1 = Context::new(Arc::clone(&model), ContextParams::default())?;
    let ctx2 = Context::new(Arc::clone(&model), ContextParams::default())?;
    let ctx3 = Context::new(Arc::clone(&model), ContextParams::default())?;

    // Model is freed when the last Arc reference is dropped
    ```

=== "CLI"

    ```bash
    # Daemon shares a single model across all connections
    mullama serve --model llama3.2:1b --parallel 3
    ```

## KV Cache Memory

The KV (Key-Value) cache is often the largest memory consumer after model weights. Its size depends on context length, model dimensions, and quantization type.

### Memory Formula

```
KV cache memory = 2 * n_layers * n_ctx * n_embd * bytes_per_element
```

For a 7B model (32 layers, 4096 embedding dim) with 4096 context:

| KV Type | Bytes/Element | KV Cache Size | Total with Model (Q4_K_M) |
|---------|---------------|---------------|---------------------------|
| F32 | 4 | ~4.0 GB | ~8.0 GB |
| F16 | 2 | ~2.0 GB | ~6.0 GB |
| BF16 | 2 | ~2.0 GB | ~6.0 GB |
| Q8_0 | 1 | ~1.0 GB | ~5.0 GB |
| Q4_0 | 0.5 | ~0.5 GB | ~4.5 GB |

### Configuring KV Cache Type

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');

    // F16 (default) -- good balance of quality and memory
    const contextF16 = new Context(model, { kvCacheType: 'f16' });

    // Q8_0 -- 50% less KV memory, minimal quality impact
    const contextQ8 = new Context(model, { kvCacheType: 'q8_0' });

    // Q4_0 -- 75% less KV memory, slight quality impact
    const contextQ4 = new Context(model, { kvCacheType: 'q4_0' });
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")

    # F16 (default)
    context_f16 = Context(model, ContextParams(kv_cache_type="f16"))

    # Q8_0 -- 50% less KV memory
    context_q8 = Context(model, ContextParams(kv_cache_type="q8_0"))

    # Q4_0 -- 75% less KV memory
    context_q4 = Context(model, ContextParams(kv_cache_type="q4_0"))
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams, KvCacheType};

    // F16 (default)
    let params_f16 = ContextParams {
        kv_cache_type: KvCacheType::F16,
        ..Default::default()
    };

    // Q8_0 -- 50% less KV memory
    let params_q8 = ContextParams {
        kv_cache_type: KvCacheType::Q8_0,
        ..Default::default()
    };

    // Q4_0 -- 75% less KV memory
    let params_q4 = ContextParams {
        kv_cache_type: KvCacheType::Q4_0,
        ..Default::default()
    };
    ```

=== "CLI"

    ```bash
    # F16 (default)
    mullama run llama3.2:1b "Hello!" --kv-cache-type f16

    # Q8_0
    mullama run llama3.2:1b "Hello!" --kv-cache-type q8_0

    # Q4_0
    mullama run llama3.2:1b "Hello!" --kv-cache-type q4_0
    ```

!!! tip "Recommended KV Cache Types"
    - **F16**: Default, best quality. Use when memory is not a concern.
    - **Q8_0**: Excellent quality with 50% memory reduction. Recommended for large contexts (8K+).
    - **Q4_0**: Good quality with 75% memory reduction. Use for very large contexts (32K+) or memory-constrained systems.

## Memory-Mapped Model Loading

Memory-mapped (mmap) loading allows the OS to page model data in and out of RAM on demand, reducing initial load time and enabling models larger than available RAM to run (slowly):

=== "Node.js"

    ```javascript
    // mmap is enabled by default
    const model = await Model.loadWithParams('./model.gguf', {
      useMmap: true,   // Default: true
      useMlock: false,  // Default: false
    });

    // mlock keeps all pages in RAM (prevents swapping)
    const lockedModel = await Model.loadWithParams('./model.gguf', {
      useMmap: true,
      useMlock: true,  // Pin all pages in physical RAM
    });
    ```

=== "Python"

    ```python
    from mullama import Model, ModelParams

    # mmap is enabled by default
    model = Model.load_with_params("./model.gguf", ModelParams(
        use_mmap=True,    # Default: True
        use_mlock=False,   # Default: False
    ))

    # mlock keeps all pages in RAM
    locked_model = Model.load_with_params("./model.gguf", ModelParams(
        use_mmap=True,
        use_mlock=True,   # Pin all pages in physical RAM
    ))
    ```

=== "Rust"

    ```rust
    use mullama::ModelBuilder;

    // mmap enabled by default
    let model = ModelBuilder::new("model.gguf")
        .with_use_mmap(true)
        .with_use_mlock(false)
        .build()?;

    // mlock to prevent swapping
    let locked = ModelBuilder::new("model.gguf")
        .with_use_mmap(true)
        .with_use_mlock(true)
        .build()?;
    ```

=== "CLI"

    ```bash
    # mmap enabled by default
    mullama run llama3.2:1b "Hello!" --mmap

    # Lock model in RAM
    mullama run llama3.2:1b "Hello!" --mlock
    ```

| Option | Behavior | Use Case |
|--------|----------|----------|
| `mmap=true, mlock=false` | Pages loaded on demand, can be swapped | Default, good for most use cases |
| `mmap=true, mlock=true` | All pages pinned in RAM | Production servers, predictable latency |
| `mmap=false` | Full model loaded into allocated memory | Maximum performance, no page faults |

!!! warning "mlock Permissions"
    Using `mlock` may require elevated privileges or increased `ulimit` on Linux. Set with `ulimit -l unlimited` or configure in `/etc/security/limits.conf`.

## Context Size vs Memory Usage

The context size directly impacts KV cache memory. Choose the smallest context that fits your use case:

| Context Size | KV Cache (F16, 7B) | Typical Use Case |
|--------------|---------------------|------------------|
| 512 | ~250 MB | Simple completions |
| 2048 | ~1.0 GB | Short conversations |
| 4096 | ~2.0 GB | Standard chat |
| 8192 | ~4.0 GB | Long conversations |
| 16384 | ~8.0 GB | Document analysis |
| 32768 | ~16.0 GB | Very long context |

=== "Node.js"

    ```javascript
    // Use only what you need
    const shortContext = new Context(model, { nCtx: 512 });   // Simple tasks
    const chatContext = new Context(model, { nCtx: 4096 });   // Conversations
    const longContext = new Context(model, { nCtx: 32768 });  // Documents
    ```

=== "Python"

    ```python
    # Use only what you need
    short_context = Context(model, ContextParams(n_ctx=512))     # Simple tasks
    chat_context = Context(model, ContextParams(n_ctx=4096))     # Conversations
    long_context = Context(model, ContextParams(n_ctx=32768))    # Documents
    ```

=== "Rust"

    ```rust
    let short = ContextParams { n_ctx: 512, ..Default::default() };
    let chat = ContextParams { n_ctx: 4096, ..Default::default() };
    let long = ContextParams { n_ctx: 32768, ..Default::default() };
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Quick answer:" --ctx-size 512
    mullama run llama3.2:1b --interactive --ctx-size 4096
    ```

## Batch Size Considerations

The batch size (`n_batch`) affects memory usage during prompt processing:

| Batch Size | Memory Impact | Speed Impact |
|-----------|---------------|--------------|
| 128 | Low | Slower prompt processing |
| 512 | Medium | Balanced (default) |
| 2048 | High | Faster prompt processing |
| 4096 | Very High | Maximum prompt throughput |

=== "Node.js"

    ```javascript
    // Small batch for memory-constrained environments
    const context = new Context(model, { nBatch: 128 });

    // Large batch for fast prompt processing
    const fastContext = new Context(model, { nBatch: 2048 });
    ```

=== "Python"

    ```python
    # Small batch for memory-constrained environments
    context = Context(model, ContextParams(n_batch=128))

    # Large batch for fast prompt processing
    fast_context = Context(model, ContextParams(n_batch=2048))
    ```

=== "Rust"

    ```rust
    // Small batch
    let params = ContextParams { n_batch: 128, ..Default::default() };

    // Large batch
    let fast_params = ContextParams { n_batch: 2048, ..Default::default() };
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" --batch-size 128
    mullama run llama3.2:1b "Hello!" --batch-size 2048
    ```

## Memory Monitoring

Monitor memory usage at runtime to prevent out-of-memory conditions:

=== "Node.js"

    ```javascript
    import { Model, Context, MemoryInfo } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model, { nCtx: 4096 });

    // Check memory usage
    const info = context.memoryInfo();
    console.log(`Model size: ${(info.modelSize / 1e9).toFixed(2)} GB`);
    console.log(`KV cache: ${(info.kvCacheSize / 1e6).toFixed(0)} MB`);
    console.log(`Context usage: ${info.tokenCount}/${info.contextSize} tokens`);
    console.log(`Total allocated: ${(info.totalAllocated / 1e9).toFixed(2)} GB`);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")
    context = Context(model, ContextParams(n_ctx=4096))

    # Check memory usage
    info = context.memory_info()
    print(f"Model size: {info.model_size / 1e9:.2f} GB")
    print(f"KV cache: {info.kv_cache_size / 1e6:.0f} MB")
    print(f"Context usage: {info.token_count}/{info.context_size} tokens")
    print(f"Total allocated: {info.total_allocated / 1e9:.2f} GB")
    ```

=== "Rust"

    ```rust
    let info = context.memory_info()?;
    println!("Model size: {:.2} GB", info.model_size as f64 / 1e9);
    println!("KV cache: {:.0} MB", info.kv_cache_size as f64 / 1e6);
    println!("Context usage: {}/{} tokens", info.token_count, info.context_size);
    println!("Total allocated: {:.2} GB", info.total_allocated as f64 / 1e9);
    ```

=== "CLI"

    ```bash
    # Show memory usage
    mullama daemon status

    # Verbose output includes memory stats
    mullama run llama3.2:1b "Hello!" --verbose
    ```

## Tips for Memory-Constrained Environments

### 1. Use Smaller Quantization

Use more aggressive model quantization to reduce model weight memory:

```
F16 (14 GB) -> Q8_0 (7 GB) -> Q4_K_M (4 GB) -> Q3_K_M (3.5 GB)
```

### 2. Reduce Context Size

Use the minimum context size needed:

=== "Node.js"

    ```javascript
    // Only 512 tokens for simple Q&A
    const context = new Context(model, { nCtx: 512, kvCacheType: 'q4_0' });
    ```

=== "Python"

    ```python
    context = Context(model, ContextParams(n_ctx=512, kv_cache_type="q4_0"))
    ```

=== "Rust"

    ```rust
    let params = ContextParams {
        n_ctx: 512,
        kv_cache_type: KvCacheType::Q4_0,
        ..Default::default()
    };
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" --ctx-size 512 --kv-cache-type q4_0
    ```

### 3. Partial GPU Offloading

Offload only the layers that fit in VRAM:

=== "Node.js"

    ```javascript
    // Offload only 20 of 32 layers to GPU
    const model = await Model.loadWithParams('./model.gguf', {
      nGpuLayers: 20,
    });
    ```

=== "Python"

    ```python
    model = Model.load_with_params("./model.gguf", ModelParams(n_gpu_layers=20))
    ```

=== "Rust"

    ```rust
    let model = ModelBuilder::new("model.gguf")
        .with_n_gpu_layers(20)
        .build()?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" --gpu-layers 20
    ```

### 4. Reuse Contexts

Clear and reuse contexts instead of creating new ones:

=== "Node.js"

    ```javascript
    const context = new Context(model, { nCtx: 4096 });

    for (const prompt of prompts) {
      context.clear();  // Reset without reallocating
      const response = await context.generate(prompt, 200);
      console.log(response);
    }
    ```

=== "Python"

    ```python
    context = Context(model, ContextParams(n_ctx=4096))

    for prompt in prompts:
        context.clear()  # Reset without reallocating
        response = context.generate(prompt, max_tokens=200)
        print(response)
    ```

=== "Rust"

    ```rust
    let mut context = Context::new(model, ContextParams::default())?;

    for prompt in &prompts {
        context.clear()?;  // Reset without reallocating
        let response = context.generate(prompt, 200)?;
        println!("{}", response);
    }
    ```

=== "CLI"

    ```bash
    # Interactive mode reuses the context automatically
    mullama run llama3.2:1b --interactive --ctx-size 4096
    ```

## GPU VRAM Management

When using GPU acceleration, model layers and KV cache consume VRAM:

### Estimating VRAM Requirements

```
VRAM = (offloaded_layers / total_layers) * model_size + kv_cache_size
```

For a 7B Q4_K_M model (~4 GB) with full offload and 4096 context (F16 KV):
```
VRAM = 4 GB (model) + 2 GB (KV cache) = ~6 GB
```

### Handling Out-of-Memory

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    async function loadWithFallback(modelPath) {
      try {
        // Try full GPU offload first
        return await Model.loadWithParams(modelPath, { nGpuLayers: -1 });
      } catch (error) {
        if (error.message.includes('CUDA out of memory') ||
            error.message.includes('insufficient memory')) {
          console.warn('Full GPU offload failed, trying partial...');
          return await Model.loadWithParams(modelPath, { nGpuLayers: 20 });
        }
        throw error;
      }
    }
    ```

=== "Python"

    ```python
    from mullama import Model, ModelParams, MullamaError

    def load_with_fallback(model_path: str) -> Model:
        try:
            # Try full GPU offload first
            return Model.load_with_params(model_path, ModelParams(n_gpu_layers=-1))
        except MullamaError as e:
            if "out of memory" in str(e).lower():
                print("Full GPU offload failed, trying partial...")
                return Model.load_with_params(model_path, ModelParams(n_gpu_layers=20))
            raise
    ```

=== "Rust"

    ```rust
    fn load_with_fallback(path: &str) -> Result<Model, MullamaError> {
        match ModelBuilder::new(path).with_n_gpu_layers(-1).build() {
            Ok(model) => Ok(model),
            Err(MullamaError::GpuMemoryError(_)) => {
                eprintln!("Full GPU offload failed, trying partial...");
                ModelBuilder::new(path).with_n_gpu_layers(20).build()
            }
            Err(e) => Err(e),
        }
    }
    ```

=== "CLI"

    ```bash
    # Monitor GPU memory
    nvidia-smi  # NVIDIA
    rocm-smi    # AMD

    # Start with fewer layers if OOM occurs
    mullama run llama3.2:1b "Hello!" --gpu-layers 20
    ```

## Memory Budget Planning

Use this table to plan your deployment:

| Component | 7B Q4_K_M | 13B Q4_K_M | 70B Q4_K_M |
|-----------|-----------|------------|------------|
| Model weights | 4 GB | 7.5 GB | 40 GB |
| KV cache (4K, F16) | 2 GB | 3 GB | 10 GB |
| KV cache (4K, Q8_0) | 1 GB | 1.5 GB | 5 GB |
| Overhead | ~0.5 GB | ~0.5 GB | ~1 GB |
| **Total (F16 KV)** | **~6.5 GB** | **~11 GB** | **~51 GB** |
| **Total (Q8_0 KV)** | **~5.5 GB** | **~9 GB** | **~46 GB** |

!!! tip "Rule of Thumb"
    For production deployments, reserve 20% more memory than the calculated requirement to account for fragmentation and runtime overhead.

## See Also

- [Loading Models](models.md) -- Model quantization and GPU offloading options
- [Text Generation](generation.md) -- Context parameters affecting memory
- [Sessions & State](sessions.md) -- Session file sizes and KV cache persistence
- [API Reference: Context](../api/context.md) -- Memory-related API documentation
