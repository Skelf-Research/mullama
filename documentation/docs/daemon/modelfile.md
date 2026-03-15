---
title: Modelfile Format
description: Custom model configuration using Modelfile and Mullamafile formats
---

# Modelfile Format

Mullama supports creating custom model configurations using two file formats:

- **Modelfile** -- Compatible with Ollama's Modelfile format
- **Mullamafile** -- Extended format with Mullama-specific directives

Both formats use a Dockerfile-like syntax with directives that configure model behavior, parameters, and metadata.

## Quick Example

```dockerfile
FROM llama3.2:1b

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER top_p 0.9

SYSTEM """
You are a helpful coding assistant specialized in Rust programming.
Always provide code examples and explain your reasoning.
"""
```

Create and use this model:

```bash
# Save the above as ./Modelfile, then:
mullama create rust-assistant -f ./Modelfile

# Use it
mullama run rust-assistant "How do I read a file in Rust?"
mullama serve --model rust-assistant
```

---

## Directives Reference

### FROM (Required)

Specifies the base model. This can be an alias, a local path, or a HuggingFace spec.

```dockerfile
# Using a model alias
FROM llama3.2:1b

# Using a local GGUF file
FROM ./models/my-model.gguf

# Using an absolute path
FROM /opt/models/custom-model.gguf

# Using a HuggingFace spec
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF

# Pin to specific commit for reproducibility
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d

# With specific quantization at a commit
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M@abc123def
```

!!! info "Base Model Resolution"
    When using an alias, Mullama resolves it through the model registry. When using `hf:` prefix, it downloads directly from HuggingFace. Local paths must point to valid GGUF files.

### PARAMETER

Set model parameters as key-value pairs. Multiple PARAMETER directives can be specified.

```dockerfile
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
```

**Supported Parameters:**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `temperature` | float | 0.0-2.0 | 0.7 | Sampling temperature. Lower = more deterministic |
| `top_p` | float | 0.0-1.0 | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 1+ | 40 | Top-k sampling candidates |
| `num_ctx` | int | 128+ | 4096 | Context window size in tokens |
| `num_predict` | int | -1+ | 512 | Maximum tokens to generate (-1 = unlimited) |
| `repeat_penalty` | float | 0.0+ | 1.1 | Repetition penalty factor |
| `presence_penalty` | float | -2.0-2.0 | 0.0 | Presence penalty |
| `frequency_penalty` | float | -2.0-2.0 | 0.0 | Frequency penalty |
| `seed` | int | -- | random | Random seed for reproducibility |
| `stop` | string | -- | -- | Stop sequence (can specify multiple) |
| `num_batch` | int | 1+ | 512 | Batch size for prompt processing |
| `num_thread` | int | 1+ | auto | Number of CPU threads |

### SYSTEM

Define the system prompt. Use triple quotes for multi-line content.

```dockerfile
SYSTEM """
You are a helpful assistant. You provide clear, concise answers
and always ask for clarification when a question is ambiguous.
"""
```

Or single-line:

```dockerfile
SYSTEM "You are a helpful coding assistant."
```

!!! tip "System Prompt Best Practices"
    - Keep system prompts focused and specific
    - Define the role, tone, and any constraints
    - Include examples of desired output format if relevant
    - System prompts consume context tokens, so balance detail with token budget

### TEMPLATE

Define the chat template format used for prompt construction. Template variables are wrapped in `{{ }}`.

```dockerfile
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
```

**Template Variables:**

| Variable | Description |
|----------|-------------|
| `{{ .System }}` | The system prompt |
| `{{ .Prompt }}` | The user's message |
| `{{ .Response }}` | The assistant's response (for multi-turn) |
| `{{ if .System }}...{{ end }}` | Conditional rendering |

!!! note "Template Auto-Detection"
    Most models include a chat template in their GGUF metadata. You only need to specify TEMPLATE when overriding the built-in template or using a model without one.

### MESSAGE

Pre-define conversation messages for few-shot examples.

```dockerfile
MESSAGE user "What is 2+2?"
MESSAGE assistant "2+2 equals 4."
MESSAGE user "And 3+3?"
MESSAGE assistant "3+3 equals 6."
```

Few-shot messages are prepended to every conversation, giving the model examples of desired behavior.

### Multiple Stop Sequences

Models can have multiple stop sequences defined:

```dockerfile
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|end|>"
```

---

## Mullamafile Extensions

The Mullamafile format supports all standard Modelfile directives plus Mullama-specific extensions for GPU configuration, adapters, vision, and more.

### GPU_LAYERS

Number of layers to offload to GPU for faster inference.

```dockerfile
GPU_LAYERS 35
```

Set to `99` to offload all layers (will cap at the model's actual layer count):

```dockerfile
GPU_LAYERS 99
```

!!! tip "Finding the Right GPU Layers"
    Start with a high number and reduce if you get out-of-memory errors. A 7B model typically has 32-35 layers. Use `mullama ps` to see current GPU layer allocation.

### FLASH_ATTENTION

Enable flash attention for faster inference and lower memory usage.

```dockerfile
FLASH_ATTENTION true
```

Flash attention is particularly beneficial for longer context lengths and requires GPU offloading.

### ADAPTER

Path to a LoRA adapter file for fine-tuned behavior without modifying the base model.

```dockerfile
ADAPTER ./my-lora-adapter.safetensors
```

Multiple adapters can be stacked:

```dockerfile
ADAPTER ./style-adapter.safetensors
ADAPTER ./domain-adapter.safetensors
```

### VISION_PROJECTOR

Path to the multimodal projector for vision models. Required for models that accept image input.

```dockerfile
VISION_PROJECTOR ./mmproj-model-f16.gguf
```

### DIGEST

SHA256 hash for content-addressed verification of the base model file.

```dockerfile
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

When loading, Mullama verifies the model file matches this hash and raises an error on mismatch. This ensures reproducible deployments.

### THINKING

Configure thinking/reasoning token detection and separation. Used for models like DeepSeek-R1 that output chain-of-thought reasoning.

```dockerfile
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true
```

When enabled, streaming responses include separate `thinking` content in the delta, allowing UIs to display reasoning in a collapsible block.

### TOOLFORMAT

Configure tool/function calling format for models that support structured tool use.

```dockerfile
TOOLFORMAT style qwen
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"
```

**Supported Styles:**

| Style | Models | Description |
|-------|--------|-------------|
| `qwen` | Qwen 2.5 series | Qwen tool calling format |
| `llama` | Llama 3.1+ | Llama function calling format |
| `mistral` | Mistral models | Mistral tool use format |
| `generic` | Any model | Generic XML-based format |

### CAPABILITY

Declare model capabilities. These flags are exposed via the `/v1/models` endpoint for client introspection.

```dockerfile
CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking true
CAPABILITY vision false
CAPABILITY code true
```

### LICENSE

License metadata for the model configuration.

```dockerfile
LICENSE MIT
```

Or multi-line:

```dockerfile
LICENSE """
Apache License 2.0
Copyright (c) 2025 Your Organization
"""
```

### AUTHOR

Author metadata.

```dockerfile
AUTHOR "Your Name <your@email.com>"
```

---

## Complete Directive Reference

| Directive | Format | Modelfile | Mullamafile | Description |
|-----------|--------|:---------:|:-----------:|-------------|
| `FROM` | `FROM <model>` | Yes | Yes | Base model (required) |
| `PARAMETER` | `PARAMETER <key> <value>` | Yes | Yes | Model parameter |
| `SYSTEM` | `SYSTEM """..."""` | Yes | Yes | System prompt |
| `TEMPLATE` | `TEMPLATE """..."""` | Yes | Yes | Chat template |
| `MESSAGE` | `MESSAGE <role> "<text>"` | Yes | Yes | Pre-defined message |
| `GPU_LAYERS` | `GPU_LAYERS <N>` | -- | Yes | GPU layer offloading |
| `FLASH_ATTENTION` | `FLASH_ATTENTION <bool>` | -- | Yes | Flash attention |
| `ADAPTER` | `ADAPTER <path>` | -- | Yes | LoRA adapter path |
| `VISION_PROJECTOR` | `VISION_PROJECTOR <path>` | -- | Yes | Vision projector |
| `DIGEST` | `DIGEST sha256:<hash>` | -- | Yes | SHA256 verification |
| `THINKING` | `THINKING <key> <value>` | -- | Yes | Thinking tokens |
| `TOOLFORMAT` | `TOOLFORMAT <key> <value>` | -- | Yes | Tool calling format |
| `CAPABILITY` | `CAPABILITY <name> <bool>` | -- | Yes | Capability flags |
| `LICENSE` | `LICENSE <text>` | -- | Yes | License metadata |
| `AUTHOR` | `AUTHOR "<name>"` | -- | Yes | Author metadata |

---

## Example Configurations

### General Assistant

```dockerfile
FROM llama3.2:3b

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER num_predict 1024

SYSTEM """
You are a helpful, harmless, and honest assistant. You provide
thoughtful and accurate responses to user questions. When uncertain,
you say so rather than guessing.
"""
```

### Code Assistant

```dockerfile
FROM qwen2.5-coder:7b

PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
PARAMETER stop "<|endoftext|>"

SYSTEM """
You are an expert programming assistant. You write clean, efficient,
and well-documented code. When providing solutions:
1. Explain your approach briefly
2. Write the complete code
3. Add inline comments for complex logic
4. Suggest potential improvements or edge cases
"""

GPU_LAYERS 35
FLASH_ATTENTION true
CAPABILITY tools true
CAPABILITY code true
```

### Reasoning Model

```dockerfile
FROM deepseek-r1:7b

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
PARAMETER num_predict 4096

THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true

CAPABILITY thinking true
CAPABILITY json true

SYSTEM """
You are a reasoning assistant. Think through problems step by step
before providing your final answer. Show your work clearly.
"""

GPU_LAYERS 35
```

### Vision Model

```dockerfile
FROM llava:7b

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER num_predict 1024

VISION_PROJECTOR ./mmproj-model-f16.gguf

CAPABILITY vision true

SYSTEM """
You are a vision-language assistant. Describe images accurately
and answer questions about visual content with detail and precision.
"""

GPU_LAYERS 35
```

### Tool-Calling Model

```dockerfile
FROM qwen2.5:7b

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER stop "<|im_end|>"

TOOLFORMAT style qwen
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"

CAPABILITY tools true
CAPABILITY json true

SYSTEM """
You are a helpful assistant with access to tools. When you need
information that a tool can provide, use the appropriate tool.
Format your tool calls as structured JSON.
"""

GPU_LAYERS 35
```

### Creative Writer

```dockerfile
FROM llama3.1:8b

PARAMETER temperature 1.2
PARAMETER top_p 0.95
PARAMETER top_k 100
PARAMETER repeat_penalty 1.2
PARAMETER num_ctx 8192
PARAMETER num_predict 4096

SYSTEM """
You are a creative writing assistant. You write vivid, engaging prose
with rich descriptions and compelling narratives. You adapt your style
to match the genre and tone requested by the user.
"""
```

### Minimal JSON API

```dockerfile
FROM qwen2.5:7b

PARAMETER temperature 0.1
PARAMETER num_ctx 4096
PARAMETER num_predict 2048
PARAMETER stop "```"

SYSTEM """
You are a JSON API. You respond only with valid JSON. No explanations,
no markdown, no code fences. Just pure JSON output matching the
requested schema.
"""

CAPABILITY json true
```

---

## Revision Pinning

Pin models to specific HuggingFace commits for reproducible deployments:

```dockerfile
# Pin to a specific commit hash
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d

# Pin with specific quantization
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M@abc123def
```

This ensures that the exact same model weights are used regardless of upstream updates. Essential for production deployments.

## Content-Addressed Verification

Verify model integrity using SHA256 digests:

```dockerfile
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Verification happens automatically when loading. If the digest does not match the file on disk, an error is raised:

```
Error: Model digest mismatch
  Expected: sha256:e3b0c44298fc1c149...
  Got:      sha256:7f83b1657ff1fc53b...
  File:     ~/.cache/mullama/models/Qwen/...
```

---

## CLI Commands

```bash
# Create model from Modelfile in current directory
mullama create my-assistant

# Create from specific file
mullama create my-assistant -f ./Modelfile

# Create from Mullamafile
mullama create my-assistant -f ./Mullamafile

# Show model details
mullama show my-assistant

# Show the Modelfile content
mullama show my-assistant --modelfile

# Show parameters only
mullama show my-assistant --parameters

# Copy/rename a model
mullama cp my-assistant my-assistant-v2

# Remove a model
mullama rm my-assistant
```

## File Discovery

When no `-f` flag is provided to `mullama create`, the tool looks for files in this order:

1. `./Mullamafile`
2. `./Modelfile`

If neither exists, an error is returned.

---

## Ollama Compatibility

Mullama's Modelfile format is fully compatible with Ollama's Modelfile specification. You can use existing Ollama Modelfiles without modification:

```bash
# Use an Ollama Modelfile directly
mullama create my-model -f ./Modelfile
```

**Key compatibility notes:**

- All standard Ollama directives (`FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `MESSAGE`) work identically
- The `FROM` directive accepts both Ollama-style model references and Mullama aliases
- Stop sequences use the same syntax: `PARAMETER stop "<token>"`
- Template variables use the same Go-template syntax

**Mullama extensions** (GPU_LAYERS, FLASH_ATTENTION, ADAPTER, VISION_PROJECTOR, THINKING, TOOLFORMAT, CAPABILITY, DIGEST, AUTHOR) are ignored by Ollama if the file is used there.

---

## Comments

Lines starting with `#` are treated as comments:

```dockerfile
# This is my custom model configuration
# Optimized for code generation tasks

FROM qwen2.5-coder:7b

# Use lower temperature for more deterministic code output
PARAMETER temperature 0.3

# Extended context for large code files
PARAMETER num_ctx 16384

SYSTEM """
You are a code assistant.
"""

# GPU acceleration for faster inference
GPU_LAYERS 35
```
