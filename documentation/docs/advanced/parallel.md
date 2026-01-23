# Parallel Processing

Accelerate batch inference and embedding generation with Rayon-powered work-stealing parallelism, NUMA-aware thread pools, and CPU pinning.

!!! info "Feature Gate"
    This feature requires the `parallel` feature flag.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["parallel"] }
    ```

## Overview

The parallel processing module provides:

- **ParallelProcessor** with Rayon work-stealing thread pool
- **ThreadPoolConfig** for fine-tuning thread allocation and NUMA awareness
- **Batch tokenization** for parallel text processing
- **Batch generation** with BatchGenerationConfig
- **GenerationResult** for structured output handling
- **CPU pinning** and NUMA topology support
- **Integration with async** via `spawn_blocking`

---

## ParallelProcessor

The `ParallelProcessor` manages a Rayon thread pool and provides batch processing APIs.

=== "Node.js"

    ```javascript
    const { ParallelProcessor } = require('mullama');

    const processor = new ParallelProcessor({
      model: 'model.gguf',
      numThreads: 8,
      maxBatchSize: 64,
      enableMetrics: true
    });

    const prompts = [
      'Summarize quantum computing:',
      'Explain machine learning:',
      'Describe blockchain:'
    ];

    const results = await processor.batchGenerate(prompts, {
      maxTokens: 150,
      temperature: 0.7
    });

    results.forEach((r, i) => {
      console.log(`[${i}] ${r.text} (${r.tokensGenerated} tokens, ${r.timeMs}ms)`);
    });
    ```

=== "Python"

    ```python
    from mullama import ParallelProcessor, ThreadPoolConfig, BatchGenerationConfig

    processor = ParallelProcessor(
        model="model.gguf",
        num_threads=8,
        max_batch_size=64,
        enable_metrics=True
    )

    prompts = [
        "Summarize quantum computing:",
        "Explain machine learning:",
        "Describe blockchain:"
    ]

    config = BatchGenerationConfig(max_tokens=150, temperature=0.7)
    results = processor.batch_generate(prompts, config)

    for i, r in enumerate(results):
        print(f"[{i}] {r.text} ({r.tokens_generated} tokens, {r.processing_time_ms}ms)")
    ```

=== "Rust"

    ```rust
    use mullama::{ParallelProcessor, ThreadPoolConfig, BatchGenerationConfig, Model};
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);

    let processor = ParallelProcessor::new(model)
        .thread_pool(ThreadPoolConfig::new().num_threads(8))
        .max_batch_size(64)
        .enable_metrics()
        .build()?;

    let prompts = vec![
        "Summarize quantum computing:",
        "Explain machine learning:",
        "Describe blockchain:",
    ];

    let config = BatchGenerationConfig::new()
        .max_tokens(150)
        .temperature(0.7);

    let results = processor.batch_generate(&prompts, &config)?;

    for (i, result) in results.iter().enumerate() {
        println!("[{}] {} ({} tokens, {}ms)",
            i, result.text, result.tokens_generated, result.processing_time_ms);
    }
    ```

### Builder Methods

| Method | Description | Default |
|--------|-------------|---------|
| `thread_pool(config)` | Configure the Rayon thread pool | System default |
| `max_batch_size(n)` | Maximum items per batch | 32 |
| `enable_metrics()` | Enable performance tracking | Disabled |

---

## ThreadPoolConfig

Configure the underlying Rayon thread pool for optimal performance.

```rust
use mullama::ThreadPoolConfig;

let config = ThreadPoolConfig::new()
    .num_threads(8)                    // Worker thread count
    .stack_size(8 * 1024 * 1024)       // 8 MB stack per thread
    .numa_aware(true)                  // Enable NUMA-aware scheduling
    .work_stealing(true);              // Enable work-stealing (default)
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_threads` | Number of worker threads | `num_cpus::get()` |
| `stack_size` | Stack size per thread in bytes | 2 MB |
| `numa_aware` | Enable NUMA-aware thread placement | false |
| `work_stealing` | Enable work-stealing between threads | true |

!!! tip "Thread Count Guidelines"
    - For CPU-bound LLM inference: use `num_cpus` (default)
    - For mixed workloads: use `num_cpus / 2` to leave headroom
    - For GPU-accelerated models: fewer threads needed (2-4) since GPU does the heavy lifting
    - Never exceed physical core count for compute-heavy tasks

---

## BatchGenerationConfig

Control generation parameters for batch operations.

```rust
use mullama::BatchGenerationConfig;

let config = BatchGenerationConfig::new()
    .max_tokens(200)
    .temperature(0.7)
    .top_k(50)
    .top_p(0.9)
    .timeout_ms(30000)      // 30 second timeout per item
    .retry_attempts(2);      // Retry failed generations
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_tokens` | Maximum tokens to generate per prompt | 100 |
| `temperature` | Sampling temperature | 1.0 |
| `top_k` | Top-K sampling parameter | 40 |
| `top_p` | Top-P (nucleus) sampling parameter | 0.9 |
| `timeout_ms` | Timeout per item in milliseconds | None |
| `retry_attempts` | Number of retry attempts on failure | 0 |

---

## Batch Generation

Process multiple prompts in parallel for high throughput.

=== "Node.js"

    ```javascript
    const results = await processor.batchGenerate(prompts, {
      maxTokens: 150,
      temperature: 0.7,
      timeoutMs: 30000,
      retryAttempts: 2
    });

    for (const result of results) {
      if (result.success) {
        console.log(`Output: ${result.text}`);
      } else {
        console.error(`Failed: ${result.error}`);
      }
    }
    ```

=== "Python"

    ```python
    config = BatchGenerationConfig(
        max_tokens=150,
        temperature=0.7,
        timeout_ms=30000,
        retry_attempts=2
    )

    results = processor.batch_generate(prompts, config)

    for result in results:
        if result.success:
            print(f"Output: {result.text}")
        else:
            print(f"Failed: {result.error}")
    ```

=== "Rust"

    ```rust
    let config = BatchGenerationConfig::new()
        .max_tokens(150)
        .temperature(0.7)
        .timeout_ms(30000)
        .retry_attempts(2);

    let results = processor.batch_generate(&prompts, &config)?;

    for result in &results {
        if result.success {
            println!("Output: {}", result.text);
        } else {
            eprintln!("Failed: {:?}", result.error);
        }
    }
    ```

### GenerationResult

```rust
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub processing_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}
```

---

## Batch Tokenization

Tokenize multiple texts in parallel, useful for preprocessing large datasets.

=== "Node.js"

    ```javascript
    const texts = [
      'Hello, world!',
      'The quick brown fox',
      'Machine learning is fascinating',
      'Rust programming language'
    ];

    const tokenResults = await processor.batchTokenize(texts);

    tokenResults.forEach((tokens, i) => {
      console.log(`${texts[i]}: ${tokens.length} tokens`);
    });
    ```

=== "Python"

    ```python
    texts = [
        "Hello, world!",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Rust programming language"
    ]

    token_results = processor.batch_tokenize(texts)

    for text, tokens in zip(texts, token_results):
        print(f"{text}: {len(tokens)} tokens")
    ```

=== "Rust"

    ```rust
    let texts = vec![
        "Hello, world!",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Rust programming language",
    ];

    let token_results = processor.batch_tokenize(&texts)?;

    for (text, tokens) in texts.iter().zip(token_results.iter()) {
        println!("{}: {} tokens", text, tokens.len());
    }
    ```

---

## NUMA-Aware Processing

For multi-socket systems, NUMA-aware scheduling ensures threads access local memory for optimal performance.

```rust
let config = ThreadPoolConfig::new()
    .num_threads(16)
    .numa_aware(true);    // Pin threads to NUMA nodes

let processor = ParallelProcessor::new(model)
    .thread_pool(config)
    .build()?;
```

!!! info "When NUMA Matters"
    NUMA awareness provides the greatest benefit on multi-socket servers (2+ CPUs). On single-socket systems, the overhead of NUMA detection exceeds the benefit. Mullama auto-detects the topology and only applies NUMA pinning when beneficial.

---

## CPU Pinning

Pin worker threads to specific CPU cores for consistent performance.

```rust
let config = ThreadPoolConfig::new()
    .num_threads(8)
    .cpu_affinity(vec![0, 1, 2, 3, 4, 5, 6, 7]);  // Pin to cores 0-7

let processor = ParallelProcessor::new(model)
    .thread_pool(config)
    .build()?;
```

---

## Throughput vs Latency Tradeoffs

| Mode | Threads | Batch Size | Optimizes |
|------|---------|-----------|-----------|
| Low latency | 2-4 | 1-4 | Per-request response time |
| Balanced | 4-8 | 8-16 | Good throughput with reasonable latency |
| Max throughput | 8-16 | 32-64 | Total prompts/second |

!!! tip "When to Use Parallel Processing"
    - **Batch inference** - Processing many prompts offline
    - **Embedding generation** - Generating embeddings for large document collections
    - **Bulk tokenization** - Preprocessing datasets
    - **Evaluation** - Running benchmarks across many inputs

    For single-request latency-sensitive applications (chat, interactive), use async streaming instead.

---

## Integration with Rayon

Access the underlying Rayon pool for custom parallel operations.

```rust
use mullama::{ParallelProcessor, ThreadPoolConfig, Model};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);

let processor = ParallelProcessor::new(model.clone())
    .thread_pool(ThreadPoolConfig::new().num_threads(6))
    .build()?;

// Custom parallel processing
let items: Vec<String> = (0..100)
    .map(|i| format!("Process item {}", i))
    .collect();

let results = processor.parallel_process(
    items,
    |item| {
        let model = model.clone();
        let mut ctx = Context::new(model, ContextParams::default())?;
        ctx.generate(&item, 50)
    }
)?;
```

---

## Combining with Async

Integrate parallel processing with async Tokio applications using `spawn_blocking`.

```rust
use mullama::{ParallelProcessor, BatchGenerationConfig, ThreadPoolConfig, Model};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf")?);

    let processor = Arc::new(
        ParallelProcessor::new(model)
            .thread_pool(ThreadPoolConfig::new().num_threads(4))
            .build()?
    );

    // Process batch in background without blocking the async runtime
    let processor_clone = processor.clone();
    let prompts: Vec<String> = vec!["Hello!".into(); 100];

    let results = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();
        let config = BatchGenerationConfig::new().max_tokens(50);
        processor_clone.batch_generate(&refs, &config)
    }).await??;

    println!("Processed {} prompts", results.len());
    Ok(())
}
```

!!! tip "Async + Parallel Best Practice"
    Use `spawn_blocking` to run `ParallelProcessor` methods from async code. This prevents the Rayon thread pool from interfering with the Tokio runtime's threads.

---

## Memory Considerations

Each thread maintains its own inference context, which consumes memory:

| Context Size | Per-Thread Memory | 8 Threads |
|-------------|-------------------|-----------|
| 2048 tokens | ~200 MB | ~1.6 GB |
| 4096 tokens | ~400 MB | ~3.2 GB |
| 8192 tokens | ~800 MB | ~6.4 GB |

!!! warning "Memory Pressure"
    Monitor system memory when using many threads. If you run out of RAM, reduce `num_threads` or `context_size`.

---

## ProcessorStats

Monitor processor performance.

=== "Node.js"

    ```javascript
    const stats = processor.stats();
    console.log(`Batches processed: ${stats.batchesProcessed}`);
    console.log(`Items processed: ${stats.itemsProcessed}`);
    console.log(`Items/sec: ${stats.itemsPerSecond.toFixed(1)}`);
    ```

=== "Python"

    ```python
    stats = processor.stats()
    print(f"Batches processed: {stats.batches_processed}")
    print(f"Items processed: {stats.items_processed}")
    print(f"Items/sec: {stats.items_per_second:.1f}")
    ```

=== "Rust"

    ```rust
    let stats = processor.stats();

    println!("Total batches processed: {}", stats.batches_processed);
    println!("Total items processed: {}", stats.items_processed);
    println!("Average batch time: {:?}", stats.avg_batch_time);
    println!("Items per second: {:.1}", stats.items_per_second);
    ```

---

## See Also

- [Late Interaction](late-interaction.md) - Parallel scoring for ColBERT retrieval
- [Embeddings Guide](../guide/embeddings.md) - Embedding generation basics
- [Async Integration](../guide/async.md) - Async/await patterns
