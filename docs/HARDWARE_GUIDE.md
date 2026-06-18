# Mullama Hardware Configuration Guide

This guide provides recommended configurations for every common hardware setup.

## Quick Start

```bash
# Auto-detect your hardware and apply optimal settings
mullama serve --model llama3.2:1b --preset auto

# Or specify a preset explicitly
mullama serve --model llama3.2:3b --preset apple-silicon
mullama serve --model llama3.2:1b --preset gpu-medium
```

## CPU-Only Systems

### 4GB RAM
```bash
mullama serve --model qwen2.5:0.5b \
  --gpu-layers 0 --context-size 2048 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 256
```
**Recommended models:** Q4_K_S quantizations, 0.5B-1B parameter models

### 8GB RAM
```bash
mullama serve --model llama3.2:1b \
  --gpu-layers 0 --context-size 4096 \
  --batch-size 512
```
**Recommended models:** Q4_K_M quantizations, 1B-3B parameter models

### 16GB RAM
```bash
mullama serve --model llama3.2:3b \
  --gpu-layers 0 --context-size 8192 \
  --mlock --batch-size 512
```
**Recommended models:** Q5_K_M quantizations, 3B-7B parameter models

## NVIDIA GPU (CUDA)

### Build with CUDA
```bash
export LLAMA_CUDA=1
cargo build --release --features daemon
```

### 4GB VRAM (GTX 1650, RTX 3050)
```bash
mullama serve --model llama3.2:1b \
  --gpu-layers 20 --context-size 4096 \
  --flash-attn --cache-type-k q8_0 --cache-type-v q8_0
```

### 8GB VRAM (RTX 3060, RTX 4060)
```bash
mullama serve --model llama3.2:3b \
  --gpu-layers 33 --context-size 8192 \
  --flash-attn --cache-type-k q8_0 --cache-type-v q8_0
```

### 16GB VRAM (RTX 4080, A4000)
```bash
mullama serve --model llama3.1:8b \
  --gpu-layers -1 --context-size 16384 \
  --flash-attn --cache-type-k f16 --cache-type-v f16
```

### 24GB VRAM (RTX 4090, A5000)
```bash
mullama serve --model llama3.1:8b \
  --gpu-layers -1 --context-size 32768 \
  --flash-attn --cache-type-k f16 --cache-type-v f16 \
  --mlock
```

### H100 (80GB)
```bash
export LLAMA_CUDA_ARCHS="90;90a"
mullama serve --model llama3.1:70b \
  --gpu-layers -1 --context-size 65536 \
  --flash-attn --batch-size 2048
```

## AMD GPU (Vulkan)

### Build with Vulkan
```bash
export LLAMA_VULKAN=1
cargo build --release --features daemon
```

### Usage
```bash
mullama serve --model llama3.2:3b \
  --gpu-layers -1 --context-size 8192 \
  --flash-attn
```

**Note:** Vulkan works on AMD, NVIDIA, and Intel GPUs. It's the recommended backend for AMD GPUs without ROCm.

## AMD GPU (ROCm)

### Build with ROCm
```bash
export LLAMA_HIPBLAS=1
cargo build --release --features daemon
```

### Usage
```bash
mullama serve --model llama3.2:3b \
  --gpu-layers -1 --context-size 8192 \
  --flash-attn
```

## Intel Arc GPU (SYCL)

### Build with SYCL
```bash
source /opt/intel/oneapi/setvars.sh
export LLAMA_SYCL=1
cargo build --release --features daemon
```

### Usage
```bash
mullama serve --model llama3.2:1b \
  --gpu-layers -1 --context-size 4096 \
  --flash-attn
```

## Apple Silicon

Metal is enabled automatically on Apple Silicon.

### 8GB Unified Memory (M1, M2)
```bash
mullama serve --model llama3.2:1b \
  --gpu-layers -1 --context-size 4096 \
  --flash-attn --cache-type-k q8_0 --cache-type-v q8_0
```

### 16GB Unified Memory (M1 Pro, M2 Pro)
```bash
mullama serve --model llama3.2:3b \
  --gpu-layers -1 --context-size 8192 \
  --flash-attn --cache-type-k q8_0 --cache-type-v q8_0
```

### 32GB+ Unified Memory (M1 Max/Ultra, M2 Max/Ultra, M3 Max)
```bash
mullama serve --model llama3.1:8b \
  --gpu-layers -1 --context-size 16384 \
  --flash-attn --cache-type-k f16 --cache-type-v f16
```

### 64GB+ (M2 Ultra, M3 Ultra)
```bash
mullama serve --model llama3.1:70b-q4_k_m \
  --gpu-layers -1 --context-size 32768 \
  --flash-attn
```

## Multi-Model Setups

### Memory Budgeting
```bash
# Run multiple models with resource limits
mullama serve \
  --model qwen2.5:0.5b --model nomic-embed-text \
  --max-models 4 --max-memory 8G \
  --eviction-policy lru --idle-unload 300 \
  --gpu-layers -1 --flash-attn
```

### Monitor Resource Usage
```bash
# Rich process table with per-model stats
mullama ps

# Detailed daemon status with memory pressure
mullama status

# Prometheus metrics
curl localhost:8080/metrics
```

## Distributed Inference (RPC)

### Build with RPC
```bash
export LLAMA_RPC=1
cargo build --release --features daemon
```

Split large models across multiple machines for inference.

## Portable Builds

For distribution to machines with different CPU capabilities:
```bash
export LLAMA_PORTABLE=1
cargo build --release --features daemon
```

This disables `-march=native` and builds with generic CPU instructions.

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CUDA=1` | OFF | Enable NVIDIA CUDA backend |
| `LLAMA_METAL=1` | Auto on Apple Silicon | Enable Apple Metal backend |
| `LLAMA_VULKAN=1` | OFF | Enable Vulkan backend |
| `LLAMA_HIPBLAS=1` | OFF | Enable AMD ROCm backend |
| `LLAMA_SYCL=1` | OFF | Enable Intel SYCL backend |
| `LLAMA_RPC=1` | OFF | Enable RPC distributed backend |
| `LLAMA_CLBLAST=1` | OFF | Enable OpenCL backend |
| `LLAMA_PORTABLE=1` | OFF | Disable native CPU optimizations |
| `LLAMA_AVX=0` | ON | Disable AVX instructions |
| `LLAMA_AVX2=0` | ON | Disable AVX2 instructions |
| `LLAMA_AVX512=1` | OFF | Enable AVX-512 |
| `LLAMA_FMA=0` | ON | Disable FMA instructions |
| `LLAMA_F16C=0` | ON | Disable F16C instructions |
| `LLAMA_CUDA_ARCHS` | `60;61;70;75;80;86;89;90;90a` | CUDA architectures |
| `LLAMA_BLAS=1` | OFF | Enable BLAS |
| `LLAMA_NO_ACCELERATE=1` | OFF | Disable Apple Accelerate |
| `VULKAN_SDK` | System default | Vulkan SDK path |
