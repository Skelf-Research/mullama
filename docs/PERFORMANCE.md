# Mullama Performance Tuning Guide

## Flash Attention

Flash attention reduces memory usage and improves throughput for long contexts.

```bash
mullama serve --model llama3.2:3b --flash-attn
```

**When to use:** Always, unless you need older GPU compatibility.
**Trade-offs:** Requires GPU compute capability 7.0+ (NVIDIA Volta or newer).

## KV Cache Quantization

The KV cache stores attention state for each token in the context. Quantizing it saves significant memory.

| Type | Memory per token | Quality | Use case |
|------|-----------------|---------|----------|
| `f16` | 100% (baseline) | Best | High VRAM, quality-critical |
| `q8_0` | ~50% | Very good | Recommended default with GPU |
| `q4_0` | ~25% | Good | Memory-constrained systems |

```bash
# Quantized KV cache (saves ~50% KV memory)
mullama serve --model llama3.2:3b \
  --cache-type-k q8_0 --cache-type-v q8_0

# Aggressive quantization (saves ~75% KV memory)
mullama serve --model llama3.2:3b \
  --cache-type-k q4_0 --cache-type-v q4_0
```

## mmap vs mlock

- **mmap** (default): Memory-maps model file. Pages loaded on demand. Good for systems with limited RAM -- unused layers stay on disk.
- **mlock**: Locks all model pages in physical RAM. Prevents swapping. Use when you have enough RAM and want consistent latency.

```bash
# Disable mmap (loads entire model into RAM upfront)
mullama serve --model llama3.2:3b --no-mmap

# Lock model in RAM (prevents swapping)
mullama serve --model llama3.2:3b --mlock
```

**Decision tree:**
- RAM > 2x model size: use `--mlock` for best latency
- RAM ~= model size: use default mmap
- RAM < model size: use default mmap (only loads needed pages)

## Batch Size

Controls how many prompt tokens are processed at once.

```bash
mullama serve --model llama3.2:3b --batch-size 1024
```

- **Larger batches** (1024-2048): Faster prompt processing, more memory
- **Smaller batches** (256-512): Less memory, slower prompt processing
- **Default**: 512 (good balance)

## Thread Count

```bash
mullama serve --model llama3.2:3b --threads 8
```

- Default: half of available CPU cores
- For CPU-only: use all physical cores (not hyperthreads)
- For GPU inference: 4-8 threads is usually sufficient

## Context Pool Size

Controls concurrent request capacity per model.

```bash
mullama serve --model llama3.2:3b --context-pool-size 8
```

- Default: 4 (handles 4 concurrent requests per model)
- Each context uses additional memory proportional to context size
- For high-throughput: increase to 8-16
- For memory-limited: keep at 1-2

## RoPE Scaling for Extended Context

Override RoPE parameters to extend context beyond the model's training length:

```bash
mullama serve --model llama3.2:3b \
  --context-size 16384 \
  --rope-freq-base 500000 \
  --rope-freq-scale 1.0
```

**Warning:** Quality degrades beyond the model's trained context length. Use models specifically trained for long context when possible.

## Quantization Format Selection

| Format | Size | Speed | Quality | Recommended for |
|--------|------|-------|---------|-----------------|
| Q4_K_S | Smallest | Fast | Good | Very limited memory |
| Q4_K_M | Small | Fast | Better | General use, limited VRAM |
| Q5_K_M | Medium | Medium | Very good | Balanced quality/size |
| Q6_K | Large | Slower | Excellent | Quality-focused |
| Q8_0 | Largest | Slow | Near-F16 | Maximum quality |

## Multi-Model Memory Budgeting

When serving multiple models:

```bash
mullama serve \
  --model qwen2.5:0.5b --model llama3.2:1b --model nomic-embed-text \
  --max-models 4 --max-memory 8G \
  --eviction-policy lru --idle-unload 300
```

- Use `--max-memory` to set a hard memory cap
- Use `--eviction-policy lru` to auto-unload least-recently-used models
- Use `--idle-unload 300` to unload models idle for 5 minutes
- Use `mullama ps` to monitor per-model memory and usage

## Memory Estimation

Check estimated memory before loading:
```bash
mullama show llama3.2:3b --estimate --gpu-layers -1 --context-size 8192
```
