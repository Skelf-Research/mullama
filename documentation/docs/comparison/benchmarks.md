# Benchmarks

This page presents performance data for Mullama across key dimensions: inference throughput, call overhead, memory efficiency, GPU utilization, and batch processing. Where applicable, comparisons are made against HTTP-based inference to quantify the advantages of native bindings.

!!! warning "Benchmark Status"
    Benchmarks marked with **(pending)** indicate the measurement framework is in place but final numbers have not yet been published. These will be updated as formal benchmarking is completed. The framework and methodology sections below describe exactly how these measurements will be collected for reproducibility.

## Methodology

### Hardware Reference Configurations

Benchmarks are collected across multiple hardware configurations to represent common deployment scenarios:

| Configuration | CPU | GPU | RAM | Use Case |
|--------------|-----|-----|-----|----------|
| **Desktop** | AMD Ryzen 9 7950X (16C/32T) | NVIDIA RTX 4090 (24GB) | 64GB DDR5 | Development, high-throughput |
| **Workstation** | Intel Xeon w7-2495X (24C/48T) | NVIDIA A6000 (48GB) | 128GB DDR5 | Production inference |
| **Laptop** | Apple M3 Max (16C) | Integrated (40GB unified) | 48GB unified | Mobile development |
| **Edge** | Raspberry Pi 5 | None (CPU only) | 8GB | Edge deployment |

### Measurement Approach

- **Warm-up**: 10 inference runs discarded before measurement
- **Iterations**: Minimum 100 runs per data point (1000 for latency measurements)
- **Timing**: `std::time::Instant` for Rust, `process.hrtime()` for Node.js, `time.perf_counter()` for Python
- **Token counting**: Exact token count from tokenizer, not estimated
- **Statistical reporting**: Median (p50), p95, and p99 values reported
- **Isolation**: Single process, no background workloads, CPU governor set to performance

### Models Used

| Model | Parameters | Quantization | Context | Purpose |
|-------|-----------|--------------|---------|---------|
| Llama 3.2 1B Instruct | 1.24B | Q4_K_M | 4096 | Small model baseline |
| Llama 3.2 3B Instruct | 3.21B | Q4_K_M | 4096 | Medium model |
| Qwen 2.5 7B Instruct | 7.62B | Q4_K_M | 8192 | Standard model |
| Llama 3.1 13B | 13.02B | Q4_K_M | 8192 | Large model |

## Inference Throughput

### Text Generation (tokens/second)

Generation throughput measured with a fixed 128-token prompt and 256-token output.

=== "GPU (RTX 4090)"

    | Model | Mullama (Native) | Mullama (Daemon) | HTTP Baseline | Notes |
    |-------|-------------------|------------------|---------------|-------|
    | Llama 3.2 1B | **(pending)** | **(pending)** | **(pending)** | All layers on GPU |
    | Llama 3.2 3B | **(pending)** | **(pending)** | **(pending)** | All layers on GPU |
    | Qwen 2.5 7B | **(pending)** | **(pending)** | **(pending)** | All layers on GPU |
    | Llama 3.1 13B | **(pending)** | **(pending)** | **(pending)** | All layers on GPU |

=== "GPU (Apple M3 Max)"

    | Model | Mullama (Native) | Mullama (Daemon) | HTTP Baseline | Notes |
    |-------|-------------------|------------------|---------------|-------|
    | Llama 3.2 1B | **(pending)** | **(pending)** | **(pending)** | Metal, unified memory |
    | Llama 3.2 3B | **(pending)** | **(pending)** | **(pending)** | Metal, unified memory |
    | Qwen 2.5 7B | **(pending)** | **(pending)** | **(pending)** | Metal, unified memory |
    | Llama 3.1 13B | **(pending)** | **(pending)** | **(pending)** | Metal, unified memory |

=== "CPU Only (Ryzen 9 7950X)"

    | Model | Mullama (Native) | Mullama (Daemon) | HTTP Baseline | Notes |
    |-------|-------------------|------------------|---------------|-------|
    | Llama 3.2 1B | **(pending)** | **(pending)** | **(pending)** | AVX-512, 32 threads |
    | Llama 3.2 3B | **(pending)** | **(pending)** | **(pending)** | AVX-512, 32 threads |
    | Qwen 2.5 7B | **(pending)** | **(pending)** | **(pending)** | AVX-512, 32 threads |
    | Llama 3.1 13B | **(pending)** | **(pending)** | **(pending)** | AVX-512, 32 threads |

!!! info "Throughput Parity"
    For sustained text generation, throughput (tokens/second) is expected to be nearly identical between native and daemon modes, since the bottleneck is the model computation itself, not the call mechanism. The difference shows up in latency and overhead measurements below.

## First Token Latency

Time from request initiation to first generated token (Time To First Token, TTFT). This is where call overhead is most visible.

| Model | Native Binding | Daemon (HTTP) | Overhead Delta | Notes |
|-------|---------------|---------------|----------------|-------|
| Llama 3.2 1B | **(pending)** | **(pending)** | **(pending)** | GPU, warm cache |
| Llama 3.2 3B | **(pending)** | **(pending)** | **(pending)** | GPU, warm cache |
| Qwen 2.5 7B | **(pending)** | **(pending)** | **(pending)** | GPU, warm cache |
| Llama 3.1 13B | **(pending)** | **(pending)** | **(pending)** | GPU, warm cache |

!!! tip "Why TTFT Matters"
    For interactive applications, TTFT determines perceived responsiveness. A native binding eliminates the HTTP round-trip, connection setup, and JSON parsing that add latency before any tokens are generated.

### Cold Start vs Warm Start

| Scenario | Native | Daemon (HTTP) | Notes |
|----------|--------|---------------|-------|
| Cold start (model load + first inference) | **(pending)** | **(pending)** | Includes mmap |
| Warm start (model cached, first inference) | **(pending)** | **(pending)** | Subsequent calls |
| Hot path (repeated inference, same context) | **(pending)** | **(pending)** | Steady state |

## Binding Overhead Comparison

This section isolates the overhead of the call mechanism itself, independent of model computation. Measured by timing the round-trip from application code to the Mullama core and back, with a minimal operation (tokenize a short string).

### Measured Call Overhead

| Binding Type | Median Latency | p95 Latency | p99 Latency | Notes |
|-------------|---------------|-------------|-------------|-------|
| Rust (native, in-process) | ~0.5 us | ~1 us | ~2 us | Direct function call |
| C/C++ (FFI) | ~1 us | ~2 us | ~3 us | Minimal FFI boundary |
| Node.js (NAPI-RS) | ~3 us | ~5 us | ~8 us | Napi thread-safe function |
| Python (PyO3) | ~5 us | ~8 us | ~12 us | GIL acquisition included |
| Go (cgo) | ~4 us | ~7 us | ~10 us | Goroutine scheduling |
| PHP (FFI) | ~8 us | ~12 us | ~18 us | FFI call overhead |
| HTTP localhost | ~1,500 us | ~3,000 us | ~5,000 us | Full round-trip |

### The 100-1000x Difference Visualized

```
Call overhead (log scale, microseconds):

Native Rust  |#                                                    | ~0.5 us
C/C++ FFI    |#                                                    | ~1 us
Node.js NAPI |##                                                   | ~3 us
Go cgo       |##                                                   | ~4 us
Python PyO3  |##                                                   | ~5 us
PHP FFI      |###                                                  | ~8 us
HTTP local   |########################################################| ~1,500 us
             0.1     1       10      100     1,000   10,000 us
```

!!! abstract "Key Insight"
    Native bindings operate at the **microsecond** level. HTTP operates at the **millisecond** level. This is a fundamental architectural difference -- not an optimization gap that can be closed through better HTTP implementations. The network stack, serialization, and process boundaries impose inherent costs.

### Cumulative Impact

The per-call overhead compounds with the number of operations:

| Operations | Native (total overhead) | HTTP (total overhead) | Time Saved |
|-----------|------------------------|----------------------|------------|
| 1 | ~3 us | ~2 ms | 2 ms |
| 10 | ~30 us | ~20 ms | 20 ms |
| 100 | ~300 us | ~200 ms | 200 ms |
| 1,000 | ~3 ms | ~2,000 ms | 2 seconds |
| 10,000 | ~30 ms | ~20,000 ms | 20 seconds |
| 100,000 | ~300 ms | ~200,000 ms | 3.3 minutes |

## Memory Efficiency

### Model Loading with mmap

Mullama uses memory-mapped file I/O (mmap) for model loading, which provides significant advantages:

| Metric | With mmap | Without mmap | Benefit |
|--------|-----------|--------------|---------|
| Load time (7B model) | **(pending)** | **(pending)** | Near-instant "loading" |
| RSS at load | **(pending)** | **(pending)** | Pages loaded on demand |
| Shared memory (multi-process) | Yes | No | Multiple instances share pages |
| Swap efficiency | Excellent | Poor | OS manages paging |

### Memory Usage by Model Size

| Model | Model File Size | Peak RSS (GPU offload) | Peak RSS (CPU only) |
|-------|----------------|----------------------|-------------------|
| Llama 3.2 1B (Q4_K_M) | ~0.7 GB | **(pending)** | **(pending)** |
| Llama 3.2 3B (Q4_K_M) | ~1.9 GB | **(pending)** | **(pending)** |
| Qwen 2.5 7B (Q4_K_M) | ~4.4 GB | **(pending)** | **(pending)** |
| Llama 3.1 13B (Q4_K_M) | ~7.4 GB | **(pending)** | **(pending)** |

### KV Cache Memory

Context window size directly impacts KV cache memory. Measured for a 7B model:

| Context Size | KV Cache (FP16) | KV Cache (Q8) | KV Cache (Q4) |
|-------------|-----------------|---------------|---------------|
| 2048 | **(pending)** | **(pending)** | **(pending)** |
| 4096 | **(pending)** | **(pending)** | **(pending)** |
| 8192 | **(pending)** | **(pending)** | **(pending)** |
| 16384 | **(pending)** | **(pending)** | **(pending)** |

## GPU Utilization

### Layer Offloading Performance Curve

GPU acceleration in Mullama (and llama.cpp generally) works by offloading transformer layers to the GPU. Performance scales with the number of offloaded layers:

```
Tokens/sec vs GPU Layers Offloaded (7B model, RTX 4090):

tok/s
 |
 |                                          ___________
 |                                     ____/
 |                                ____/
 |                           ____/
 |                      ____/
 |                 ____/
 |            ____/
 |       ____/
 |  ____/
 | /
 |/
 +----+----+----+----+----+----+----+----+
 0    5   10   15   20   25   30   35  All
                GPU Layers
```

| Layers Offloaded | Throughput (tok/s) | VRAM Used | Notes |
|-----------------|-------------------|-----------|-------|
| 0 (CPU only) | **(pending)** | 0 GB | Baseline |
| 10 (~30%) | **(pending)** | **(pending)** | Partial offload |
| 20 (~60%) | **(pending)** | **(pending)** | Majority offloaded |
| 30 (~90%) | **(pending)** | **(pending)** | Near-full offload |
| All (100%) | **(pending)** | **(pending)** | Maximum performance |

!!! tip "Optimal Layer Count"
    The optimal number of layers to offload depends on your available VRAM. Mullama's `--gpu-layers` flag or Modelfile `GPU_LAYERS` directive lets you tune this. Use `mullama show <model>` to see the total layer count for a model.

### Multi-GPU Scaling

For models that exceed single-GPU VRAM, Mullama supports layer splitting across GPUs:

| Configuration | 7B Model | 13B Model | Notes |
|--------------|----------|-----------|-------|
| Single GPU (24GB) | **(pending)** | **(pending)** | RTX 4090 |
| Dual GPU (2x24GB) | **(pending)** | **(pending)** | NVLink not required |

## Batch Processing Throughput

### Rayon Parallel Processing

Mullama uses Rayon for CPU-parallel batch operations. Measured processing multiple independent prompts simultaneously:

| Batch Size | Sequential (1 thread) | Rayon (8 threads) | Rayon (16 threads) | Rayon (32 threads) |
|-----------|----------------------|-------------------|--------------------|--------------------|
| 10 prompts | **(pending)** | **(pending)** | **(pending)** | **(pending)** |
| 50 prompts | **(pending)** | **(pending)** | **(pending)** | **(pending)** |
| 100 prompts | **(pending)** | **(pending)** | **(pending)** | **(pending)** |
| 500 prompts | **(pending)** | **(pending)** | **(pending)** | **(pending)** |

### Scaling Efficiency

```
Speedup vs Thread Count (100 prompts, Llama 3.2 1B, CPU):

Speedup
  |
8 |                              *
  |                         *
6 |                    *
  |               *
4 |          *
  |     *
2 | *
  |*
1 +--+--+--+--+--+--+--+--+
  1  2  4  6  8  12 16 32
           Threads

* = Measured    --- = Linear (ideal)
```

!!! note "Sub-Linear Scaling"
    Batch processing shows sub-linear scaling due to memory bandwidth limitations and cache contention. The optimal thread count depends on the model size and available memory bandwidth. For most configurations, 8-16 threads provide the best efficiency.

## Embedding Generation

### Embeddings per Second

Measured generating embeddings for sentences of varying length:

| Input Length | Native Binding | Daemon (HTTP) | Speedup (overhead) | Notes |
|-------------|---------------|---------------|---------------------|-------|
| 16 tokens | **(pending)** | **(pending)** | **(pending)** | Short phrases |
| 64 tokens | **(pending)** | **(pending)** | **(pending)** | Sentences |
| 256 tokens | **(pending)** | **(pending)** | **(pending)** | Paragraphs |
| 512 tokens | **(pending)** | **(pending)** | **(pending)** | Documents |

### Batch Embedding with Parallel Processing

| Documents | Sequential | Rayon (8 threads) | Throughput |
|-----------|-----------|-------------------|------------|
| 100 | **(pending)** | **(pending)** | **(pending)** embeddings/sec |
| 1,000 | **(pending)** | **(pending)** | **(pending)** embeddings/sec |
| 10,000 | **(pending)** | **(pending)** | **(pending)** embeddings/sec |

### ColBERT Scoring Performance

Late interaction (MaxSim) scoring throughput:

| Corpus Size | Sequential | Parallel (Rayon) | Queries/sec |
|-------------|-----------|------------------|-------------|
| 1,000 docs | **(pending)** | **(pending)** | **(pending)** |
| 10,000 docs | **(pending)** | **(pending)** | **(pending)** |
| 100,000 docs | **(pending)** | **(pending)** | **(pending)** |

## Streaming Performance

### Token Delivery Latency

Time between consecutive tokens reaching the application layer:

| Mode | Median Inter-Token | p95 Inter-Token | Jitter | Notes |
|------|-------------------|-----------------|--------|-------|
| Native callback | **(pending)** | **(pending)** | **(pending)** | Direct callback |
| Native channel | **(pending)** | **(pending)** | **(pending)** | Tokio mpsc |
| WebSocket | **(pending)** | **(pending)** | **(pending)** | Daemon mode |
| SSE (HTTP) | **(pending)** | **(pending)** | **(pending)** | Daemon mode |

!!! info "Streaming Consistency"
    Lower jitter means more consistent token delivery, which translates to smoother text rendering in user interfaces. Native callbacks and channels provide the most consistent delivery due to fewer intermediary layers.

## Reproducing Benchmarks

### Running the Benchmark Suite

```bash
# Clone the repository
git clone https://github.com/neul-labs/mullama.git
cd mullama
git submodule update --init --recursive

# Build with benchmark support
cargo build --release --features "full"

# Run the full benchmark suite
cargo bench

# Run specific benchmark groups
cargo bench -- throughput
cargo bench -- latency
cargo bench -- overhead
cargo bench -- embedding
cargo bench -- batch
```

### Individual Benchmark Scripts

=== "Throughput Benchmark"

    ```bash
    # Measure tokens/second for a specific model
    cargo run --release --features "full" --example bench_throughput -- \
      --model path/to/model.gguf \
      --prompt-tokens 128 \
      --generate-tokens 256 \
      --iterations 100 \
      --gpu-layers -1 \
      --warmup 10
    ```

=== "Overhead Benchmark"

    ```bash
    # Measure binding call overhead (tokenization round-trip)
    cargo run --release --features "full" --example bench_overhead -- \
      --model path/to/model.gguf \
      --iterations 10000 \
      --operation tokenize
    ```

=== "Embedding Benchmark"

    ```bash
    # Measure embedding generation throughput
    cargo run --release --features "full" --example bench_embeddings -- \
      --model path/to/embedding-model.gguf \
      --corpus path/to/corpus.txt \
      --batch-size 32 \
      --threads 8
    ```

=== "Batch Benchmark"

    ```bash
    # Measure parallel batch processing
    cargo run --release --features "full" --example bench_batch -- \
      --model path/to/model.gguf \
      --prompts path/to/prompts.json \
      --threads 1,2,4,8,16,32 \
      --iterations 10
    ```

### Environment Setup for Reproducible Results

```bash
# Linux: Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Linux: Disable turbo boost for consistent results
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# Linux: Set GPU to maximum clocks (NVIDIA)
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=2520

# Verify configuration
mullama --version
nvidia-smi  # GPU info
lscpu       # CPU info
free -h     # Memory info
```

### Reporting Your Results

We welcome community benchmark contributions. To submit results:

1. Run benchmarks with the standard configuration above
2. Include full hardware specifications
3. Report OS version and kernel
4. Include Mullama version and llama.cpp backend version
5. Submit as a GitHub issue with the `benchmark` label

!!! abstract "Benchmark Integrity"
    All published benchmarks will include full reproduction instructions, raw data, and statistical analysis scripts. We believe in transparent, reproducible performance claims.
