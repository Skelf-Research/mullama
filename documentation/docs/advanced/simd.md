# SIMD Optimizations

Mullama automatically leverages SIMD (Single Instruction, Multiple Data) instructions for accelerated sampling operations, processing multiple data elements in parallel at the hardware level.

!!! success "No Feature Gate Required"
    SIMD optimizations are built into the core library and require no feature flags. Capabilities are detected at runtime and the fastest available instruction set is used automatically.

## Overview

The SIMD module provides:

- **Automatic detection** of CPU capabilities (AVX2, AVX-512, NEON)
- **Vectorized operations** for sampling-critical computations
- **Zero configuration** -- optimal path selected at runtime
- **Scalar fallback** when SIMD is unavailable

---

## Why SIMD Matters for LLM Inference

During token sampling, several hot-loop operations process the entire vocabulary (often 32K-128K floats). SIMD processes 4-16 floats per instruction instead of one:

```
Scalar (1 float/cycle):     ################  (128K iterations)
NEON   (4 floats/cycle):    ####              (32K iterations)
AVX2   (8 floats/cycle):    ##                (16K iterations)
AVX512 (16 floats/cycle):   #                 (8K iterations)
```

---

## SimdCapabilities

Detect available SIMD instruction sets at runtime.

=== "Node.js"

    ```javascript
    const { SimdCapabilities } = require('mullama');

    const caps = SimdCapabilities.detect();
    console.log(`AVX2:    ${caps.avx2}`);
    console.log(`AVX-512: ${caps.avx512}`);
    console.log(`NEON:    ${caps.neon}`);
    console.log(`Best:    ${caps.bestAvailable}`);
    ```

=== "Python"

    ```python
    from mullama import SimdCapabilities

    caps = SimdCapabilities.detect()
    print(f"AVX2:    {caps.avx2}")
    print(f"AVX-512: {caps.avx512}")
    print(f"NEON:    {caps.neon}")
    print(f"Best:    {caps.best_available}")
    ```

=== "Rust"

    ```rust
    use mullama::sampling_simd::SimdCapabilities;

    let caps = SimdCapabilities::detect();

    println!("AVX2:    {}", caps.avx2);
    println!("AVX-512: {}", caps.avx512);
    println!("NEON:    {}", caps.neon);
    println!("Best:    {}", caps.best_available());
    ```

=== "CLI"

    ```bash
    # Check SIMD capabilities
    mullama info --simd

    # Output:
    # SIMD Capabilities:
    #   AVX2:    active
    #   AVX-512: detected
    #   NEON:    unavailable
    #   Best:    AVX2 (8 floats/cycle)
    ```

---

## Optimized Operations

### simd_softmax - Vectorized Softmax

The most performance-critical operation for sampling. Computes softmax (probability distribution) over the vocabulary logits.

```rust
use mullama::sampling_simd::simd_softmax;

let mut logits: Vec<f32> = get_model_logits(); // e.g., 128K floats

// Applies softmax in-place using the best available SIMD
simd_softmax(&mut logits);

// logits now contains probabilities summing to 1.0
let sum: f32 = logits.iter().sum();
assert!((sum - 1.0).abs() < 0.001);
```

**Implementation details:**

1. Find maximum value (for numerical stability) using `simd_max_f32`
2. Compute `exp(x - max)` for each element using fast polynomial approximation
3. Sum all exponentiated values using `simd_sum_f32`
4. Divide each element by the total sum

### simd_top_k - Fast Top-K Selection

Efficiently find the K largest values and their indices from the vocabulary.

```rust
use mullama::sampling_simd::simd_top_k;

let logits: Vec<f32> = get_model_logits();

// Find top 50 tokens by logit value
let top_50 = simd_top_k(&logits, 50);

// Returns Vec<(index, value)> sorted by value descending
for (idx, value) in &top_50 {
    println!("Token {}: logit = {:.4}", idx, value);
}
```

### simd_sum_f32 - Fast Vector Summation

Sum all elements in a float slice using SIMD accumulation.

```rust
use mullama::sampling_simd::simd_sum_f32;

let data: Vec<f32> = vec![1.0; 100_000];
let sum = simd_sum_f32(&data);  // = 100000.0
```

### simd_max_f32 - Fast Maximum

Find the maximum value in a float slice.

```rust
use mullama::sampling_simd::simd_max_f32;

let logits: Vec<f32> = get_model_logits();
let max_logit = simd_max_f32(&logits);
```

### simd_argmax - Maximum Index

Find the index of the maximum value (greedy token selection).

```rust
use mullama::sampling_simd::simd_argmax;

let logits: Vec<f32> = get_model_logits();
if let Some(best_token_idx) = simd_argmax(&logits) {
    println!("Greedy token: {}", best_token_idx);
}
```

### simd_select_top_k_tokens - Token Selection

Convenience function combining top-k with token ID conversion.

```rust
use mullama::sampling_simd::simd_select_top_k_tokens;

let logits: Vec<f32> = get_model_logits();

// Returns Vec<(TokenId, f32)>
let top_tokens = simd_select_top_k_tokens(&logits, 40);

for (token_id, logit) in &top_tokens {
    println!("Token ID {}: {:.4}", token_id, logit);
}
```

---

## Operations Accelerated

| Operation | Purpose | SIMD Benefit |
|-----------|---------|-------------|
| `simd_softmax` | Convert logits to probabilities | 2.5-2.8x faster |
| `simd_top_k` | Top-K token selection | 1.5-1.8x faster |
| `simd_sum_f32` | Probability summation | 6-7x faster |
| `simd_max_f32` | Find maximum logit | 6-7x faster |
| `simd_argmax` | Greedy token selection | 6-7x faster |

---

## Performance Comparison

### Softmax Performance (Most Critical)

| Vocab Size | Scalar | AVX2 (8-wide) | Speedup |
|-----------|--------|---------------|---------|
| 32,768 | 0.45ms | 0.18ms | 2.5x |
| 65,536 | 0.90ms | 0.34ms | 2.6x |
| 128,256 | 1.82ms | 0.65ms | 2.8x |

### Top-K Selection (K=50)

| Vocab Size | Scalar | SIMD-assisted | Speedup |
|-----------|--------|---------------|---------|
| 32,768 | 0.12ms | 0.08ms | 1.5x |
| 65,536 | 0.25ms | 0.15ms | 1.7x |
| 128,256 | 0.51ms | 0.28ms | 1.8x |

### Sum/Max Operations

| Vocab Size | Scalar | AVX2 | Speedup |
|-----------|--------|------|---------|
| 32,768 | 0.03ms | 0.005ms | 6x |
| 128,256 | 0.12ms | 0.018ms | 6.7x |

!!! info "Overall Sampling Speedup"
    For a typical sampling step with 128K vocabulary, SIMD provides approximately **20-30% faster** end-to-end sampling time, as softmax dominates the sampling computation.

---

## Platform Support Matrix

| Platform | Instruction Set | Width | Status |
|----------|----------------|-------|--------|
| x86_64 (Intel/AMD) | AVX2 | 8 floats | Fully supported |
| x86_64 (Intel/AMD) | AVX-512 | 16 floats | Detected (uses AVX2 ops) |
| aarch64 (Apple M1/M2, ARM) | NEON | 4 floats | Fully supported |
| Other architectures | Scalar | 1 float | Automatic fallback |

### Architecture-Specific Notes

=== "x86_64 (AVX2)"

    - Processes **8 floats** simultaneously
    - Available on Intel Haswell (2013+) and AMD Excavator (2015+)
    - Uses fast polynomial `exp()` approximation for softmax
    - Horizontal reductions use `_mm256_extractf128_ps` + SSE operations

=== "aarch64 (NEON)"

    - Processes **4 floats** simultaneously
    - Always available on aarch64 (ARM v8+)
    - Includes Apple M1/M2/M3 and Raspberry Pi 4+
    - Uses scalar `exp()` per element (NEON lacks native exp)
    - Vector add/multiply/max fully vectorized

=== "Other / Fallback"

    - Standard scalar operations
    - Fully functional, just slower
    - Includes 32-bit ARM, RISC-V, WebAssembly

---

## How It Works Internally

The SIMD module uses a dispatch pattern that selects the optimal implementation at runtime:

```rust
pub fn simd_softmax(data: &mut [f32]) {
    let max_val = simd_max_f32(data);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { simd_softmax_avx2(data, max_val) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { simd_softmax_neon(data, max_val) };
        return;
    }

    // Scalar fallback
    scalar_softmax(data, max_val);
}
```

The AVX2 softmax uses a two-pass approach:

1. **First pass**: Compute `exp(x - max)` using a polynomial approximation and accumulate the sum
2. **Second pass**: Divide each element by the total sum using SIMD multiply with reciprocal

The polynomial approximation for `exp(x)`:

```
exp(x) = 2^(x * log2(e))

Split into integer (2^n) and fractional (2^f) parts:
  2^n: bit manipulation (shift exponent field)
  2^f: Taylor polynomial (c0 + c1*f + c2*f^2 + c3*f^3)
```

---

## Compile-Time Flags

While SIMD detection is automatic at runtime, you can force specific instruction sets at compile time for additional optimizations:

```bash
# Force AVX2 (skips runtime detection overhead)
RUSTFLAGS="-C target-feature=+avx2" cargo build --release

# Force AVX-512
RUSTFLAGS="-C target-feature=+avx512f" cargo build --release

# Native architecture (uses all available features)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

!!! warning "Portability Tradeoff"
    Using compile-time flags produces a binary that only runs on CPUs with those features. The default runtime detection is recommended for distributed binaries.

---

## When SIMD Matters Most

| Scenario | SIMD Impact | Notes |
|----------|-------------|-------|
| Large vocabulary (128K+) | High | More data per sampling step |
| Batch sampling | High | Multiple softmax operations |
| High token throughput | Medium | Sampling overhead per token |
| Small vocabulary (<8K) | Low | Data fits in L1 cache anyway |
| GPU-accelerated inference | Low | GPU handles most computation |

---

## No Configuration Needed

!!! success "Fully Automatic"
    SIMD optimizations require no user configuration. The library:

    1. Detects CPU capabilities at runtime using `is_x86_feature_detected!` / architecture checks
    2. Selects the fastest available implementation
    3. Falls back to scalar if no SIMD is available
    4. All implementations produce identical results (within floating-point precision)

You benefit from SIMD acceleration simply by using Mullama's sampling functions. The optimization is transparent to your application code.

---

## Verifying SIMD is Active

=== "Node.js"

    ```javascript
    const { SimdCapabilities } = require('mullama');

    const caps = SimdCapabilities.detect();
    console.log('SIMD Status:');
    console.log(`  Best available: ${caps.bestAvailable}`);
    console.log(`  AVX2:    ${caps.avx2 ? 'active' : 'unavailable'}`);
    console.log(`  AVX-512: ${caps.avx512 ? 'detected' : 'unavailable'}`);
    console.log(`  NEON:    ${caps.neon ? 'active' : 'unavailable'}`);
    ```

=== "Python"

    ```python
    from mullama import SimdCapabilities

    caps = SimdCapabilities.detect()
    print("SIMD Status:")
    print(f"  Best available: {caps.best_available}")
    print(f"  AVX2:    {'active' if caps.avx2 else 'unavailable'}")
    print(f"  AVX-512: {'detected' if caps.avx512 else 'unavailable'}")
    print(f"  NEON:    {'active' if caps.neon else 'unavailable'}")
    ```

=== "Rust"

    ```rust
    use mullama::sampling_simd::SimdCapabilities;

    fn main() {
        let caps = SimdCapabilities::detect();
        println!("SIMD Status:");
        println!("  Best available: {}", caps.best_available());
        println!("  AVX2:    {}", if caps.avx2 { "active" } else { "unavailable" });
        println!("  AVX-512: {}", if caps.avx512 { "detected" } else { "unavailable" });
        println!("  NEON:    {}", if caps.neon { "active" } else { "unavailable" });

        if caps.best_available() == "Scalar" {
            println!("\n  Note: Running without SIMD acceleration.");
            println!("  Performance will be reduced for sampling operations.");
        }
    }
    ```

=== "CLI"

    ```bash
    mullama info --simd
    ```

---

## See Also

- [Generation Guide](../guide/generation.md) - Text generation using sampling
- [API: Sampling](../api/sampling.md) - Sampling configuration
- [Parallel Processing](parallel.md) - Thread-level parallelism
