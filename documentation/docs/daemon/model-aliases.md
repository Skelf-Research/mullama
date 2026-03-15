---
title: Model Aliases
description: Pre-configured shorthand names for HuggingFace models with 40+ ready-to-use aliases
---

# Model Aliases

Model aliases provide a convenient shorthand for referencing HuggingFace model repositories. Instead of typing full repository paths, you can use simple names like `llama3.2:1b` or `qwen2.5:7b`.

## How Aliases Work

When you use a model alias, Mullama resolves it to a full HuggingFace repository and selects the appropriate GGUF quantization file:

```
llama3.2:1b  -->  bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

The alias system:

1. Looks up the alias in the embedded registry (`configs/models.toml`)
2. Resolves to the HuggingFace repository ID
3. Selects the default quantization file (typically Q4_K_M)
4. Downloads if not already cached

## Using Aliases

```bash
# Run with an alias (downloads automatically on first use)
mullama run llama3.2:1b "Hello!"

# Pull a model by alias
mullama pull qwen2.5:7b

# Start the daemon with aliased models
mullama serve --model deepseek-r1:7b --model llama3.2:1b

# Load into running daemon
mullama load llama3.2:1b -g 35
```

---

## Pre-Configured Aliases

### Llama Family (Meta)

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `llama3.2:1b` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | ~0.8 GB | Q4_K_M | Fast, lightweight chat and tasks |
| `llama3.2:3b` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | ~2.0 GB | Q4_K_M | Balanced size and capability |
| `llama3.1:8b` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | ~4.9 GB | Q4_K_M | High capability general purpose |
| `llama3.1:70b` | `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` | ~40 GB | Q4_K_M | Frontier-class, requires 48+ GB RAM |
| `llama3:8b` | `QuantFactory/Meta-Llama-3-8B-Instruct-GGUF` | ~4.7 GB | Q4_K_M | Llama 3 Instruct (previous gen) |

### Qwen Family (Alibaba)

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `qwen2.5:0.5b` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | ~0.4 GB | Q4_K_M | Ultra-lightweight, edge devices |
| `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | ~1.0 GB | Q4_K_M | Fast general purpose |
| `qwen2.5:3b` | `Qwen/Qwen2.5-3B-Instruct-GGUF` | ~2.0 GB | Q4_K_M | Compact but capable |
| `qwen2.5:7b` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | ~4.7 GB | Q4_K_M | Strong general purpose |
| `qwen2.5:14b` | `Qwen/Qwen2.5-14B-Instruct-GGUF` | ~8.5 GB | Q4_K_M | Advanced reasoning |
| `qwen2.5:32b` | `Qwen/Qwen2.5-32B-Instruct-GGUF` | ~19 GB | Q4_K_M | Expert-level capability |
| `qwen2.5:72b` | `Qwen/Qwen2.5-72B-Instruct-GGUF` | ~42 GB | Q4_K_M | Frontier-class |
| `qwen2.5-coder:7b` | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~4.7 GB | Q4_K_M | Code generation and analysis |
| `qwen2.5-coder:14b` | `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF` | ~8.5 GB | Q4_K_M | Advanced coding tasks |
| `qwen2.5-coder:32b` | `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` | ~19 GB | Q4_K_M | Expert-level code generation |

### DeepSeek Family

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `deepseek-r1:1.5b` | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | ~1.0 GB | Q4_K_M | Fast reasoning tasks |
| `deepseek-r1:7b` | `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` | ~4.9 GB | Q4_K_M | Strong chain-of-thought reasoning |
| `deepseek-r1:14b` | `bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF` | ~8.8 GB | Q4_K_M | Advanced multi-step reasoning |
| `deepseek-r1:32b` | `bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF` | ~19 GB | Q4_K_M | Expert-level reasoning |
| `deepseek-coder:7b` | `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | ~4.1 GB | Q4_K_M | Code generation and completion |
| `deepseek-coder:33b` | `TheBloke/deepseek-coder-33B-instruct-GGUF` | ~19 GB | Q4_K_M | Expert coding, large codebase understanding |

### Phi Family (Microsoft)

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `phi3:mini` | `microsoft/Phi-3-mini-4k-instruct-gguf` | ~2.4 GB | Q4 | Compact powerhouse, fast |
| `phi3:medium` | `bartowski/Phi-3-medium-4k-instruct-GGUF` | ~8.0 GB | Q4_K_M | Medium capability |
| `phi3.5:mini` | `bartowski/Phi-3.5-mini-instruct-GGUF` | ~2.4 GB | Q4_K_M | Latest compact Phi model |
| `phi-4:14b` | `bartowski/phi-4-GGUF` | ~8.5 GB | Q4_K_M | Latest Phi-4, strong reasoning |

### Mistral Family

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `mistral:7b` | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | ~4.1 GB | Q4_K_M | General purpose, fast |
| `mixtral:8x7b` | `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` | ~26 GB | Q4_K_M | MoE architecture, high capability |
| `codestral:22b` | `bartowski/Codestral-22B-v0.1-GGUF` | ~13 GB | Q4_K_M | Mistral's code-focused model |

### Gemma Family (Google)

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `gemma2:2b` | `bartowski/gemma-2-2b-it-GGUF` | ~1.6 GB | Q4_K_M | Compact, efficient |
| `gemma2:9b` | `bartowski/gemma-2-9b-it-GGUF` | ~5.4 GB | Q4_K_M | Strong capability |
| `gemma2:27b` | `bartowski/gemma-2-27b-it-GGUF` | ~16 GB | Q4_K_M | Large, high quality |

### Vision Models (Multimodal)

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `llava:7b` | `mys/ggml_llava-v1.5-7b` | ~4.1 GB | Q4_K | Image understanding and description |
| `llava:13b` | `mys/ggml_llava-v1.5-13b` | ~7.4 GB | Q4_K | Higher quality vision-language |
| `llava-phi3` | `xtuner/llava-phi-3-mini-gguf` | ~2.4 GB | INT4 | Fast vision model |
| `moondream:2b` | `vikhyatk/moondream2` | ~3.5 GB | F16 | Tiny but capable vision model |

!!! note "Vision Models"
    Vision models include an associated multimodal projector (`mmproj`) file that is downloaded automatically alongside the model weights. Use `--mmproj` when loading manually.

### Embedding Models

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `nomic-embed` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | ~0.3 GB | Q4_K_M | High quality text embeddings (768D) |
| `bge:small` | `TaylorAI/bge-small-en-v1.5-gguf` | ~0.1 GB | Q4_K_M | Fast embeddings (384D) |
| `bge:large` | `TaylorAI/bge-large-en-v1.5-gguf` | ~0.7 GB | Q4_K_M | High quality embeddings (1024D) |

### Specialized Models

| Alias | Repository | Size | Quantization | Use Case |
|-------|-----------|------|--------------|----------|
| `starcoder2:3b` | `bartowski/starcoder2-3b-GGUF` | ~2.0 GB | Q4_K_M | Code completion, fill-in-middle |
| `starcoder2:7b` | `bartowski/starcoder2-7b-GGUF` | ~4.4 GB | Q4_K_M | Code completion |
| `starcoder2:15b` | `bartowski/starcoder2-15b-GGUF` | ~9.0 GB | Q4_K_M | Advanced code generation |
| `yi:6b` | `TheBloke/Yi-6B-Chat-GGUF` | ~3.8 GB | Q4_K_M | 01.AI bilingual chat model |
| `yi:34b` | `TheBloke/Yi-34B-Chat-GGUF` | ~20 GB | Q4_K_M | 01.AI large bilingual model |

---

## Quantization Levels Explained

Quantization reduces model size and memory requirements by representing weights with fewer bits. Lower quantization means smaller files but potentially lower quality.

### Quantization Types

| Type | Bits | Size Reduction | Quality | Speed | Recommended For |
|------|------|---------------|---------|-------|-----------------|
| `Q2_K` | 2-bit | ~85% smaller | Lowest | Fastest | Extreme memory constraints |
| `IQ2_M` | 2-bit | ~85% smaller | Low | Fast | Very limited memory |
| `Q3_K_M` | 3-bit | ~75% smaller | Below average | Fast | Memory-constrained |
| `Q4_K_M` | 4-bit | ~65% smaller | Good | Fast | **Best default balance** |
| `Q4_K_S` | 4-bit | ~68% smaller | Good | Fast | Slightly smaller than Q4_K_M |
| `Q5_K_M` | 5-bit | ~55% smaller | Better | Moderate | Quality-sensitive tasks |
| `Q6_K` | 6-bit | ~45% smaller | High | Moderate | High quality requirements |
| `Q8_0` | 8-bit | ~30% smaller | Very High | Slower | Near-lossless quality |
| `F16` | 16-bit | No reduction | Maximum | Slowest | Research, benchmarking |

### Default Preference Order

When auto-selecting a quantization, Mullama prefers: `Q4_K_M > Q4_K_S > Q5_K_M > Q4_0 > Q8_0 > F16`

### Size vs Quality Tradeoffs

For a 7B parameter model:

| Quantization | File Size | RAM Required | Quality (Perplexity) |
|--------------|-----------|-------------|---------------------|
| Q2_K | ~2.8 GB | ~3.5 GB | Noticeable degradation |
| Q3_K_M | ~3.3 GB | ~4.0 GB | Some quality loss |
| **Q4_K_M** | **~4.1 GB** | **~5.0 GB** | **Minimal quality loss** |
| Q5_K_M | ~4.8 GB | ~5.8 GB | Near-original quality |
| Q6_K | ~5.5 GB | ~6.5 GB | Very close to original |
| Q8_0 | ~7.3 GB | ~8.5 GB | Indistinguishable from F16 |
| F16 | ~14 GB | ~16 GB | Original quality |

!!! tip "Choosing a Quantization"
    - **Q4_K_M** is the best default for most users -- it provides excellent quality with manageable memory usage.
    - **Q5_K_M** or **Q6_K** if you have extra RAM and want higher quality.
    - **Q2_K** or **Q3_K_M** for edge devices or when running many models simultaneously.
    - **Q8_0** for evaluation and benchmarking where quality matters most.

### Requesting Specific Quantizations

You can request a specific quantization when pulling:

```bash
# Pull specific quantization via HuggingFace spec
mullama pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q5_K_M.gguf

# Or Q8 for high quality
mullama pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q8_0.gguf
```

---

## Custom Model Paths

You can bypass the alias system entirely and use local GGUF files directly:

```bash
# Local file path
mullama run ./my-model.gguf "Hello"

# Absolute path
mullama serve --model /opt/models/custom.gguf

# With custom alias for the session (alias:path format)
mullama serve --model custom-name:./my-model.gguf

# Load into daemon with alias
mullama load my-alias:/path/to/model.gguf -g 35
```

---

## HuggingFace Direct Paths

For models not in the registry, use the `hf:` prefix:

```bash
# Auto-detect best GGUF file
mullama pull hf:owner/repo-name-GGUF

# Specify exact file
mullama pull hf:owner/repo-name-GGUF:model-file.Q4_K_M.gguf

# Pin to specific commit for reproducibility
mullama pull hf:owner/repo-name-GGUF@commit-hash

# Use directly with serve
mullama serve --model hf:Qwen/Qwen2.5-7B-Instruct-GGUF
```

---

## Adding Custom Aliases

The model registry is defined in `configs/models.toml`. You can add custom aliases by editing this file before building:

```toml
[aliases."my-model:7b"]
repo = "my-username/My-Model-7B-GGUF"
default_file = "my-model-7b.Q4_K_M.gguf"
family = "custom"
description = "My custom fine-tuned model"
tags = ["chat", "instruct", "custom"]
size_hint = "7B"

[aliases."my-embed"]
repo = "my-username/My-Embeddings-GGUF"
default_file = "my-embeddings.Q4_K_M.gguf"
family = "embedding"
description = "Custom embedding model"
tags = ["embedding", "retrieval"]
```

After rebuilding, the alias is embedded in the binary:

```bash
cargo build --release --features daemon
mullama run my-model:7b "Hello from my custom model!"
```

### Registry Entry Format

| Field | Required | Description |
|-------|----------|-------------|
| `repo` | Yes | HuggingFace repository ID (`org/repo-name`) |
| `default_file` | No | Default GGUF filename to download |
| `mmproj` | No | Vision projector filename (for multimodal models) |
| `family` | No | Model family identifier (llama, qwen, deepseek, etc.) |
| `description` | No | Human-readable description |
| `tags` | No | Capability tags for searching/filtering |
| `size_hint` | No | Model size for display (e.g., "7B", "1.5B") |
| `has_thinking` | No | Whether model supports chain-of-thought |
| `has_vision` | No | Whether model supports image input |
| `has_tools` | No | Whether model supports function/tool calling |

---

## Model Selection Guide

### By Use Case

| Use Case | Recommended Alias | RAM Required |
|----------|-------------------|-------------|
| Quick prototyping | `llama3.2:1b` | ~2 GB |
| General chat | `qwen2.5:7b` or `llama3.1:8b` | ~6 GB |
| Code generation | `qwen2.5-coder:7b` | ~6 GB |
| Reasoning / Math | `deepseek-r1:7b` | ~6 GB |
| Image understanding | `llava:7b` | ~6 GB |
| Text embeddings | `nomic-embed` | ~1 GB |
| Edge / Mobile | `qwen2.5:0.5b` | ~1 GB |
| Maximum quality | `qwen2.5:72b` or `llama3.1:70b` | ~48 GB |

### By Available RAM

| Available RAM | Maximum Model Size | Recommended |
|---------------|--------------------|-------------|
| 4 GB | ~1B parameters | `llama3.2:1b`, `qwen2.5:1.5b` |
| 8 GB | ~3-7B parameters | `llama3.2:3b`, `qwen2.5:7b` |
| 16 GB | ~7-14B parameters | `qwen2.5:14b`, `phi-4:14b` |
| 32 GB | ~14-32B parameters | `qwen2.5:32b`, `deepseek-r1:32b` |
| 64 GB | ~70B+ parameters | `llama3.1:70b`, `qwen2.5:72b` |

!!! note "RAM Estimation"
    A rough rule of thumb: Q4_K_M models require approximately `(parameters * 0.6) + 1 GB` of RAM. A 7B model at Q4_K_M needs about 5-6 GB.
