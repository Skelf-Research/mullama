# Loading Models

Mullama supports loading GGUF format models with flexible configuration options for GPU offloading, memory mapping, and multi-threaded access.

## Basic Model Loading

The simplest way to load a model from a local GGUF file:

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    const model = await Model.load('./path/to/model.gguf');
    console.log(`Loaded model with ${model.vocabSize()} tokens`);
    ```

=== "Python"

    ```python
    from mullama import Model

    model = Model.load("./path/to/model.gguf")
    print(f"Loaded model with {model.vocab_size()} tokens")
    ```

=== "Rust"

    ```rust
    use mullama::Model;
    use std::sync::Arc;

    let model = Arc::new(Model::load("path/to/model.gguf")?);
    println!("Loaded model with {} tokens", model.vocab_size());
    ```

=== "CLI"

    ```bash
    # Load and run directly
    mullama run ./path/to/model.gguf "Hello!"

    # Or use a model alias
    mullama run llama3.2:1b "Hello!"
    ```

!!! note "Shared Ownership"
    Models are designed to be shared between multiple contexts and threads. In Rust, wrap in `Arc`. In Node.js and Python, sharing is handled automatically through reference counting.

## Model Parameters

Configure model loading with `ModelParams` for fine-grained control over GPU offloading, memory mapping, and tensor distribution:

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    const model = await Model.loadWithParams('./model.gguf', {
      nGpuLayers: 35,        // Offload 35 layers to GPU
      useMmap: true,         // Memory-map the model file
      useMlock: false,       // Don't lock in RAM
      vocabOnly: false,      // Load full model, not just vocab
      mainGpu: 0,            // Primary GPU device index
    });
    ```

=== "Python"

    ```python
    from mullama import Model, ModelParams

    model = Model.load_with_params("./model.gguf", ModelParams(
        n_gpu_layers=35,       # Offload 35 layers to GPU
        use_mmap=True,         # Memory-map the model file
        use_mlock=False,       # Don't lock in RAM
        vocab_only=False,      # Load full model, not just vocab
        main_gpu=0,            # Primary GPU device index
    ))
    ```

=== "Rust"

    ```rust
    use mullama::{Model, ModelParams};

    let params = ModelParams {
        n_gpu_layers: 35,
        use_mmap: true,
        use_mlock: false,
        vocab_only: false,
        main_gpu: 0,
        ..Default::default()
    };

    let model = Model::load_with_params("model.gguf", params)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" \
      --gpu-layers 35 \
      --mmap \
      --main-gpu 0
    ```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_gpu_layers` | `i32` | `0` | Number of layers to offload to GPU (-1 for all) |
| `split_mode` | `SplitMode` | `Layer` | How to split model across GPUs |
| `main_gpu` | `i32` | `0` | Primary GPU device index |
| `tensor_split` | `Vec<f32>` | `[]` | Proportional split across GPUs |
| `vocab_only` | `bool` | `false` | Load only vocabulary (for tokenization) |
| `use_mmap` | `bool` | `true` | Enable memory-mapped file loading |
| `use_mlock` | `bool` | `false` | Lock model pages in physical RAM |

## ModelBuilder Pattern

For complex configurations, use the builder pattern which provides a fluent API:

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    const model = await Model.builder('./model.gguf')
      .gpuLayers(35)
      .useMmap(true)
      .useMlock(false)
      .vocabOnly(false)
      .tensorSplit([0.6, 0.4])
      .progressCallback((progress) => {
        console.log(`Loading: ${(progress * 100).toFixed(1)}%`);
        return true;  // Return true to continue
      })
      .build();
    ```

=== "Python"

    ```python
    from mullama import Model

    def on_progress(progress: float) -> bool:
        print(f"Loading: {progress * 100:.1f}%")
        return True  # Return True to continue

    model = (Model.builder("./model.gguf")
        .gpu_layers(35)
        .use_mmap(True)
        .use_mlock(False)
        .vocab_only(False)
        .tensor_split([0.6, 0.4])
        .progress_callback(on_progress)
        .build())
    ```

=== "Rust"

    ```rust
    use mullama::ModelBuilder;

    let model = ModelBuilder::new("model.gguf")
        .with_n_gpu_layers(35)
        .with_use_mmap(true)
        .with_use_mlock(false)
        .with_vocab_only(false)
        .with_tensor_split(&[0.6, 0.4])
        .with_progress_callback(|progress| {
            println!("Loading: {:.1}%", progress * 100.0);
            true  // Return true to continue, false to abort
        })
        .build()?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" \
      --gpu-layers 35 \
      --mmap \
      --tensor-split "0.6,0.4"
    ```

## GPU Layer Offloading

Offload transformer layers to the GPU for significantly faster inference:

| Value | Behavior |
|-------|----------|
| `0` | CPU only -- no GPU acceleration |
| `1` to `N` | Offload N layers to GPU |
| `-1` or large number | Offload all layers to GPU |

=== "Node.js"

    ```javascript
    // Offload all layers to GPU
    const model = await Model.loadWithParams('./model.gguf', {
      nGpuLayers: -1
    });
    ```

=== "Python"

    ```python
    # Offload all layers to GPU
    model = Model.load_with_params("./model.gguf", ModelParams(
        n_gpu_layers=-1
    ))
    ```

=== "Rust"

    ```rust
    // Offload all layers to GPU
    let model = ModelBuilder::new("model.gguf")
        .with_n_gpu_layers(-1)
        .build()?;
    ```

=== "CLI"

    ```bash
    # Offload all layers to GPU
    mullama run llama3.2:1b "Hello!" --gpu-layers -1
    ```

!!! tip "Finding the Right Balance"
    Start with all layers on GPU (`-1`). If you run out of VRAM, reduce the count until the model fits. Monitor GPU memory with:

    - **NVIDIA**: `nvidia-smi`
    - **Apple Silicon**: Activity Monitor (Memory tab)
    - **AMD**: `rocm-smi`

### Multi-GPU Tensor Splitting

For systems with multiple GPUs, distribute model layers across devices:

=== "Node.js"

    ```javascript
    // Split 60% on GPU 0, 40% on GPU 1
    const model = await Model.loadWithParams('./large-model.gguf', {
      nGpuLayers: -1,
      tensorSplit: [0.6, 0.4]
    });
    ```

=== "Python"

    ```python
    # Split 60% on GPU 0, 40% on GPU 1
    model = Model.load_with_params("./large-model.gguf", ModelParams(
        n_gpu_layers=-1,
        tensor_split=[0.6, 0.4]
    ))
    ```

=== "Rust"

    ```rust
    // Split 60% on GPU 0, 40% on GPU 1
    let model = ModelBuilder::new("large-model.gguf")
        .with_n_gpu_layers(-1)
        .with_tensor_split(&[0.6, 0.4])
        .build()?;
    ```

=== "CLI"

    ```bash
    mullama run large-model "Hello!" \
      --gpu-layers -1 \
      --tensor-split "0.6,0.4"
    ```

## Model Introspection

Query model properties after loading to understand its architecture and capabilities:

=== "Node.js"

    ```javascript
    const model = await Model.load('./model.gguf');

    // Architecture information
    console.log(`Embedding dimension: ${model.embeddingDim()}`);
    console.log(`Number of layers: ${model.layerCount()}`);
    console.log(`Training context length: ${model.trainContextLength()}`);

    // Vocabulary information
    console.log(`Vocabulary size: ${model.vocabSize()}`);
    console.log(`BOS token: ${model.bosToken()}`);
    console.log(`EOS token: ${model.eosToken()}`);

    // Model description
    console.log(`Description: ${model.description()}`);
    ```

=== "Python"

    ```python
    model = Model.load("./model.gguf")

    # Architecture information
    print(f"Embedding dimension: {model.embedding_dim()}")
    print(f"Number of layers: {model.layer_count()}")
    print(f"Training context length: {model.train_context_length()}")

    # Vocabulary information
    print(f"Vocabulary size: {model.vocab_size()}")
    print(f"BOS token: {model.bos_token()}")
    print(f"EOS token: {model.eos_token()}")

    # Model description
    print(f"Description: {model.description()}")
    ```

=== "Rust"

    ```rust
    let model = Model::load("model.gguf")?;

    // Architecture information
    println!("Embedding dimension: {}", model.n_embd());
    println!("Number of layers: {}", model.n_layer());
    println!("Training context length: {}", model.n_ctx_train());

    // Vocabulary information
    println!("Vocabulary size: {}", model.vocab_size());
    println!("BOS token: {:?}", model.bos_token());
    println!("EOS token: {:?}", model.eos_token());

    // Model description
    println!("Description: {:?}", model.description());
    ```

=== "CLI"

    ```bash
    # Show model metadata
    mullama show llama3.2:1b

    # Show full modelfile including parameters
    mullama show llama3.2:1b --modelfile
    ```

### Key Metadata Fields

| Method | Returns | Description |
|--------|---------|-------------|
| `vocab_size()` | `usize` | Total vocabulary size |
| `n_embd()` / `embeddingDim()` | `usize` | Embedding/hidden dimension |
| `n_layer()` / `layerCount()` | `usize` | Number of transformer layers |
| `n_ctx_train()` / `trainContextLength()` | `usize` | Maximum trained context length |
| `bos_token()` / `bosToken()` | `Option<TokenId>` | Beginning-of-sequence token |
| `eos_token()` / `eosToken()` | `Option<TokenId>` | End-of-sequence token |
| `description()` | `Option<String>` | Model description from metadata |

## Tokenization and Detokenization

Convert between text and tokens using the model's vocabulary:

=== "Node.js"

    ```javascript
    // Text to tokens
    const tokens = model.tokenize("Hello, world!");
    console.log(`Token IDs: ${tokens}`);
    console.log(`Token count: ${tokens.length}`);

    // Tokens back to text
    const text = model.detokenize(tokens);
    console.log(`Decoded text: ${text}`);

    // Single token to string
    const tokenStr = model.tokenToString(tokens[0]);
    console.log(`First token: '${tokenStr}'`);
    ```

=== "Python"

    ```python
    # Text to tokens
    tokens = model.tokenize("Hello, world!")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Tokens back to text
    text = model.detokenize(tokens)
    print(f"Decoded text: {text}")

    # Single token to string
    token_str = model.token_to_string(tokens[0])
    print(f"First token: '{token_str}'")
    ```

=== "Rust"

    ```rust
    // Text to tokens
    let tokens = model.tokenize("Hello, world!", true, false)?;
    // Arguments: text, add_bos, parse_special_tokens
    println!("Token IDs: {:?}", tokens);
    println!("Token count: {}", tokens.len());

    // Single token to text
    let text = model.token_to_str(tokens[0], 0, false)?;
    println!("First token text: '{}'", text);

    // Detokenize a sequence
    let decoded = model.detokenize(&tokens)?;
    println!("Decoded: {}", decoded);
    ```

=== "CLI"

    ```bash
    # Tokenize text (shows token count)
    mullama tokenize llama3.2:1b "Hello, world!"
    ```

!!! info "Special Tokens"
    By default, tokenization adds the BOS (beginning-of-sequence) token. In Rust, control this with the `add_bos` parameter. In Node.js and Python, use the `addBos`/`add_bos` option.

## Model Aliases

When using the Mullama daemon, you can reference models by aliases instead of file paths:

=== "Node.js"

    ```javascript
    import { Model } from 'mullama';

    // Connect to the daemon and use an alias
    const model = await Model.fromAlias('llama3.2:1b');
    ```

=== "Python"

    ```python
    from mullama import Model

    # Connect to the daemon and use an alias
    model = Model.from_alias("llama3.2:1b")
    ```

=== "Rust"

    ```rust
    use mullama::Model;

    // When using the daemon client
    let model = Model::from_alias("llama3.2:1b")?;
    ```

=== "CLI"

    ```bash
    # List available model aliases
    mullama list

    # Run with alias
    mullama run llama3.2:1b "Hello!"

    # Create a custom alias via Modelfile
    mullama create my-assistant -f ./Modelfile
    ```

## Downloading from HuggingFace

Download GGUF models directly from HuggingFace using the daemon:

=== "Node.js"

    ```javascript
    import { daemon } from 'mullama';

    // Pull a model by name
    await daemon.pull('llama3.2:1b');

    // Pull with progress tracking
    await daemon.pull('llama3.2:1b', (progress) => {
      console.log(`Download: ${(progress * 100).toFixed(1)}%`);
    });
    ```

=== "Python"

    ```python
    from mullama import daemon

    # Pull a model by name
    daemon.pull("llama3.2:1b")

    # Pull with progress tracking
    def on_progress(progress: float):
        print(f"Download: {progress * 100:.1f}%")

    daemon.pull("llama3.2:1b", progress_callback=on_progress)
    ```

=== "Rust"

    ```rust
    use mullama::daemon::DaemonClient;

    let client = DaemonClient::connect().await?;
    client.pull_model("llama3.2:1b").await?;
    ```

=== "CLI"

    ```bash
    # Pull a model
    mullama pull llama3.2:1b

    # Use HuggingFace reference in Modelfile
    # FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF

    # Pin to a specific revision
    # FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d
    ```

## Memory Considerations

GGUF models come in various quantization levels, trading quality for size and speed:

| Quantization | Bits | Quality | Speed | RAM (7B model) |
|--------------|------|---------|-------|----------------|
| F16 | 16 | Best | Slow | ~14 GB |
| Q8_0 | 8 | Excellent | Medium | ~7 GB |
| Q6_K | 6 | Very Good | Fast | ~5.5 GB |
| Q5_K_M | 5 | Good | Fast | ~5 GB |
| Q4_K_M | 4 | Good | Fastest | ~4 GB |
| Q3_K_M | 3 | Acceptable | Fastest | ~3.5 GB |
| Q2_K | 2 | Poor | Fastest | ~3 GB |

!!! tip "Recommended Quantization"
    **Q4_K_M** offers the best balance of quality, speed, and size for most use cases. Use **Q8_0** when quality is paramount, or **Q3_K_M** when memory is extremely limited.

### Estimating Memory Requirements

A rough formula for estimating RAM usage:

```
RAM = (model_parameters * bits_per_weight) / 8 + context_memory
```

For a 7B parameter model with Q4_K_M quantization and 4096 context:

```
Model weights: 7B * 4 bits / 8 = ~3.5 GB
KV cache (F16): 2 * 32 layers * 4096 ctx * 4096 dim * 2 bytes = ~2 GB
Total: ~5.5 GB
```

!!! warning "GPU VRAM"
    When offloading to GPU, the offloaded layers consume VRAM instead of system RAM. Ensure your GPU has sufficient VRAM for the layers you offload.

## Error Handling

=== "Node.js"

    ```javascript
    try {
      const model = await Model.load('./model.gguf');
    } catch (error) {
      if (error.code === 'MODEL_LOAD_ERROR') {
        console.error(`Failed to load: ${error.message}`);
      } else if (error.code === 'FILE_NOT_FOUND') {
        console.error('Model file not found');
      } else {
        console.error(`Unexpected error: ${error.message}`);
      }
    }
    ```

=== "Python"

    ```python
    from mullama import Model, MullamaError

    try:
        model = Model.load("./model.gguf")
    except FileNotFoundError:
        print("Model file not found")
    except MullamaError as e:
        print(f"Failed to load model: {e}")
    ```

=== "Rust"

    ```rust
    use mullama::{Model, MullamaError};

    match Model::load("model.gguf") {
        Ok(model) => {
            println!("Loaded: {} layers", model.n_layer());
        }
        Err(MullamaError::ModelLoadError(msg)) => {
            eprintln!("Failed to load model: {}", msg);
        }
        Err(e) => eprintln!("Unexpected error: {}", e),
    }
    ```

=== "CLI"

    ```bash
    # CLI provides descriptive error messages automatically
    mullama run nonexistent-model "Hello!"
    # Error: model 'nonexistent-model' not found. Run 'mullama list' to see available models.
    ```

## Best Practices

1. **Share models across contexts** -- A single loaded model can serve many concurrent inference contexts
2. **Match quantization to hardware** -- Use Q4_K_M for consumer GPUs, Q8_0 for high-end systems
3. **Enable GPU offloading** -- Significant speedup when VRAM is available
4. **Use mmap for large models** -- Faster loading and better memory efficiency
5. **Cache loaded models** -- Model loading is expensive; reuse loaded models across requests
6. **Start with full GPU offload** -- Use `-1` for gpu_layers, then reduce if OOM occurs

## See Also

- [Text Generation](generation.md) -- Using loaded models for inference
- [Memory Management](memory.md) -- Detailed memory optimization strategies
- [API Reference: Model](../api/model.md) -- Complete Model API documentation
- [Daemon: Model Management](../daemon/model-management.md) -- Managing models with the daemon
