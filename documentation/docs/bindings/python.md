---
title: Python Bindings
description: High-performance Python bindings for Mullama LLM inference, built with PyO3 for native speed with a Pythonic API and NumPy integration.
---

# Python Bindings

High-performance Python bindings for the Mullama LLM library, built with [PyO3](https://pyo3.rs/) for native-speed inference with a Pythonic API. Embeddings are returned as NumPy arrays for seamless integration with the scientific Python ecosystem.

## Installation

```bash
pip install mullama
```

Pre-built wheels are available for:

| Platform | Architecture | Python Versions |
|----------|:------------:|:---------------:|
| Linux | x64 (manylinux2014) | 3.8 - 3.12 |
| Linux | arm64 | 3.8 - 3.12 |
| macOS | x64 (Intel) | 3.8 - 3.12 |
| macOS | arm64 (Apple Silicon) | 3.8 - 3.12 |
| Windows | x64 | 3.8 - 3.12 |

### Building from Source

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd bindings/python
maturin develop --release

# Or build a wheel for distribution
maturin build --release
pip install target/wheels/mullama-*.whl
```

!!! info "Build Requirements"
    - Python >= 3.8
    - Rust toolchain (1.75+)
    - NumPy (for embedding operations)
    - System dependencies (see [Platform Setup](../getting-started/platform-setup.md))
    - For GPU support, set `LLAMA_CUDA=1`, `LLAMA_METAL=1`, etc. before building

### Type Stubs

The mullama package ships with inline type annotations (PEP 561 compatible). Type checkers like mypy, pyright, and Pylance will recognize the types automatically without any additional packages.

```python
# mypy and pyright work out of the box
from mullama import Model, Context, SamplerParams
model: Model = Model.load("./model.gguf")
```

## Quick Start

```python
from mullama import Model, Context, SamplerParams

# Load a model with GPU acceleration
model = Model.load("./llama-3.2-1b.Q4_K_M.gguf", n_gpu_layers=-1)

# Create an inference context
ctx = Context(model, n_ctx=2048)

# Generate text
text = ctx.generate("Once upon a time", max_tokens=100)
print(text)
```

---

## API Reference

### Model

The `Model` class handles model loading and provides access to tokenization, metadata, and model properties.

#### `Model.load(path, ...)`

Load a model from a GGUF file.

```python
@staticmethod
def load(
    path: str,
    n_gpu_layers: int = 0,
    use_mmap: bool = True,
    use_mlock: bool = False,
    vocab_only: bool = False,
) -> Model
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | (required) | Path to the GGUF model file |
| `n_gpu_layers` | `int` | `0` | Layers to offload to GPU (0 = CPU, -1 = all) |
| `use_mmap` | `bool` | `True` | Use memory mapping for loading |
| `use_mlock` | `bool` | `False` | Lock model in memory (prevents paging) |
| `vocab_only` | `bool` | `False` | Only load vocabulary (for tokenization) |

**Returns:** `Model` instance

**Raises:** `RuntimeError` if loading fails

```python
# CPU-only
model = Model.load("./model.gguf")

# Full GPU offloading
model = Model.load("./model.gguf", n_gpu_layers=-1)

# Partial GPU offloading (first 20 layers)
model = Model.load("./model.gguf", n_gpu_layers=20)

# Vocabulary only (for tokenization tasks)
model = Model.load("./model.gguf", vocab_only=True)
```

---

#### `model.tokenize(text, add_bos=True, special=False)`

Convert text to token IDs.

```python
def tokenize(self, text: str, add_bos: bool = True, special: bool = False) -> list[int]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | Text to tokenize |
| `add_bos` | `bool` | `True` | Add beginning-of-sequence token |
| `special` | `bool` | `False` | Parse special tokens |

```python
tokens = model.tokenize("Hello, world!")
print(tokens)  # [1, 10994, 29892, 3186, 29991]
print(f"Token count: {len(tokens)}")
```

---

#### `model.detokenize(tokens, remove_special=False, unparse_special=False)`

Convert token IDs back to text.

```python
def detokenize(
    self,
    tokens: list[int],
    remove_special: bool = False,
    unparse_special: bool = False,
) -> str
```

```python
text = model.detokenize([1, 10994, 29892, 3186, 29991])
print(text)  # "Hello, world!"
```

---

#### `model.apply_chat_template(messages, add_generation_prompt=True)`

Format chat messages using the model's built-in template (e.g., ChatML, Llama-3 format).

```python
def apply_chat_template(
    self,
    messages: list[tuple[str, str]],
    add_generation_prompt: bool = True,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[tuple[str, str]]` | (required) | List of (role, content) tuples |
| `add_generation_prompt` | `bool` | `True` | Add generation prompt at the end |

```python
messages = [
    ("system", "You are a helpful assistant."),
    ("user", "What is Python?"),
]
prompt = model.apply_chat_template(messages)
response = ctx.generate(prompt, max_tokens=300)
```

---

#### `model.metadata()`

Get all model metadata as a dictionary.

```python
def metadata(self) -> dict[str, str]
```

```python
meta = model.metadata()
for key, value in meta.items():
    print(f"{key}: {value}")
```

---

#### `model.token_is_eog(token)`

Check if a token is an end-of-generation token.

```python
def token_is_eog(self, token: int) -> bool
```

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_ctx_train` | `int` | Training context size |
| `n_embd` | `int` | Embedding dimension |
| `n_vocab` | `int` | Vocabulary size |
| `n_layer` | `int` | Number of layers |
| `n_head` | `int` | Number of attention heads |
| `token_bos` | `int` | BOS token ID |
| `token_eos` | `int` | EOS token ID |
| `size` | `int` | Model size in bytes |
| `n_params` | `int` | Number of parameters |
| `description` | `str` | Model description |
| `architecture` | `str \| None` | Architecture name (e.g., "llama") |
| `name` | `str \| None` | Model name from metadata |

```python
model = Model.load("model.gguf")
print(f"Model: {model.name}")
print(f"Architecture: {model.architecture}")
print(f"Parameters: {model.n_params:,}")
print(f"Layers: {model.n_layer}")
print(f"Embedding dim: {model.n_embd}")
print(f"Size: {model.size / 1e9:.2f} GB")
print(repr(model))
# Model(name='Llama-3.2-1B', arch='llama', params=1234567890, size=1234MB)
```

---

### Context

The `Context` class provides the inference context for text generation.

#### `Context(model, ...)`

Create a new inference context.

```python
def __init__(
    self,
    model: Model,
    n_ctx: int = 0,
    n_batch: int = 2048,
    n_threads: int = 0,
    embeddings: bool = False,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | (required) | Model to use |
| `n_ctx` | `int` | `0` | Context size (0 = model default) |
| `n_batch` | `int` | `2048` | Batch size for prompt processing |
| `n_threads` | `int` | `0` | Thread count (0 = auto-detect) |
| `embeddings` | `bool` | `False` | Enable embeddings mode |

```python
# Default context
ctx = Context(model)

# Custom context with 4096 token window
ctx = Context(model, n_ctx=4096, n_batch=512)

# Explicitly limit thread count
ctx = Context(model, n_ctx=2048, n_threads=4)
```

---

#### `ctx.generate(prompt, max_tokens=100, params=None)`

Generate text from a prompt. The prompt can be a string or a list of token IDs.

```python
def generate(
    self,
    prompt: str | list[int],
    max_tokens: int = 100,
    params: SamplerParams | None = None,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| list[int]` | (required) | Text prompt or token IDs |
| `max_tokens` | `int` | `100` | Maximum tokens to generate |
| `params` | `SamplerParams \| None` | `None` | Sampling parameters (None = defaults) |

```python
# From string
text = ctx.generate("Hello, AI!", max_tokens=50)

# From token IDs
tokens = model.tokenize("Hello, AI!")
text = ctx.generate(tokens, max_tokens=50)

# With custom sampling
params = SamplerParams(temperature=0.7, top_p=0.9)
text = ctx.generate("Write a poem:", max_tokens=200, params=params)
```

---

#### `ctx.generate_stream(prompt, max_tokens=100, params=None)`

Generate text and return token pieces as a list. Useful for streaming-style output.

```python
def generate_stream(
    self,
    prompt: str | list[int],
    max_tokens: int = 100,
    params: SamplerParams | None = None,
) -> list[str]
```

```python
pieces = ctx.generate_stream("Once upon a time", max_tokens=200)
for piece in pieces:
    print(piece, end="", flush=True)
print()
```

---

#### `ctx.decode(tokens)`

Process tokens through the model (for advanced use with embeddings).

```python
def decode(self, tokens: list[int]) -> None
```

---

#### `ctx.clear_cache()`

Clear the KV cache. Call this when starting a new conversation or switching between unrelated prompts.

```python
def clear_cache(self) -> None
```

---

#### `ctx.get_embeddings()`

Get embeddings for the last decoded tokens (requires `embeddings=True` in constructor).

```python
def get_embeddings(self) -> numpy.ndarray | None
```

**Returns:** NumPy array of float32 values, or `None` if embeddings are not available.

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_ctx` | `int` | Current context size |
| `n_batch` | `int` | Current batch size |

---

### SamplerParams

The `SamplerParams` class configures text generation sampling behavior.

#### `SamplerParams(...)`

```python
def __init__(
    self,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    penalty_repeat: float = 1.1,
    penalty_freq: float = 0.0,
    penalty_present: float = 0.0,
    penalty_last_n: int = 64,
    seed: int = 0,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `0.8` | Randomness (0.0 = deterministic) |
| `top_k` | `int` | `40` | Top-k sampling (0 = disabled) |
| `top_p` | `float` | `0.95` | Nucleus sampling (1.0 = disabled) |
| `min_p` | `float` | `0.05` | Min-p sampling (0.0 = disabled) |
| `typical_p` | `float` | `1.0` | Typical sampling (1.0 = disabled) |
| `penalty_repeat` | `float` | `1.1` | Repeat penalty (1.0 = disabled) |
| `penalty_freq` | `float` | `0.0` | Frequency penalty |
| `penalty_present` | `float` | `0.0` | Presence penalty |
| `penalty_last_n` | `int` | `64` | Token window for penalties |
| `seed` | `int` | `0` | Random seed (0 = random) |

All properties are readable and writable:

```python
params = SamplerParams(temperature=0.7)
params.top_p = 0.9
params.top_k = 50
print(params.temperature)  # 0.7
```

---

#### Preset Class Methods

```python
@staticmethod
def greedy() -> SamplerParams
```

Create deterministic sampler parameters (temperature=0.0, top_k=1).

```python
@staticmethod
def creative() -> SamplerParams
```

Create high-randomness sampler parameters (temperature=1.2, top_k=100).

```python
@staticmethod
def precise() -> SamplerParams
```

Create low-randomness sampler parameters (temperature=0.3, top_k=20).

```python
# Usage
text = ctx.generate("2+2=", max_tokens=10, params=SamplerParams.greedy())
poem = ctx.generate("Write a poem:", max_tokens=200, params=SamplerParams.creative())
summary = ctx.generate("Summarize:", max_tokens=150, params=SamplerParams.precise())
```

---

### EmbeddingGenerator

The `EmbeddingGenerator` class creates text embeddings using a model. Embeddings are returned as NumPy arrays for seamless integration with scientific Python tools.

#### `EmbeddingGenerator(model, n_ctx=512, normalize=True)`

```python
def __init__(
    self,
    model: Model,
    n_ctx: int = 512,
    normalize: bool = True,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | (required) | Model for embeddings |
| `n_ctx` | `int` | `512` | Context size for embedding computation |
| `normalize` | `bool` | `True` | Normalize embeddings to unit length |

---

#### `gen.embed(text)`

Generate an embedding vector for text.

```python
def embed(self, text: str) -> numpy.ndarray
```

**Returns:** NumPy `ndarray` of `float32` values (shape: `(n_embd,)`)

```python
import numpy as np

embedding = gen.embed("Hello, world!")
print(f"Shape: {embedding.shape}")       # (4096,)
print(f"Dtype: {embedding.dtype}")       # float32
print(f"Norm: {np.linalg.norm(embedding):.4f}")  # ~1.0 if normalized
```

---

#### `gen.embed_batch(texts)`

Generate embeddings for multiple texts efficiently.

```python
def embed_batch(self, texts: list[str]) -> list[numpy.ndarray]
```

**Returns:** List of NumPy `ndarray` embeddings

```python
texts = ["Hello", "World", "Mullama"]
embeddings = gen.embed_batch(texts)
print(f"Count: {len(embeddings)}")       # 3
print(f"Dims: {embeddings[0].shape}")    # (4096,)
```

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_embd` | `int` | Embedding dimension |

---

### Utility Functions

#### `cosine_similarity(a, b)`

Compute cosine similarity between two NumPy embedding vectors.

```python
def cosine_similarity(a: numpy.ndarray, b: numpy.ndarray) -> float
```

**Raises:** `ValueError` if vectors have different lengths.

```python
from mullama import cosine_similarity

sim = cosine_similarity(emb1, emb2)
print(f"Similarity: {sim:.4f}")  # Value between -1 and 1
```

#### `backend_init()`

Initialize the backend (called automatically on first model load).

```python
def backend_init() -> None
```

#### `backend_free()`

Free backend resources.

```python
def backend_free() -> None
```

#### `supports_gpu_offload()`

Check if GPU offloading is available.

```python
def supports_gpu_offload() -> bool
```

#### `system_info()`

Get system information string (CPU features, GPU support).

```python
def system_info() -> str
```

#### `max_devices()`

Get maximum number of compute devices.

```python
def max_devices() -> int
```

---

## Examples

### Streaming Output

```python
from mullama import Model, Context, SamplerParams

model = Model.load("./model.gguf", n_gpu_layers=-1)
ctx = Context(model, n_ctx=4096)

# Stream tokens to console
pieces = ctx.generate_stream(
    "Write a haiku about programming:\n\n",
    max_tokens=50,
    params=SamplerParams.creative(),
)

for piece in pieces:
    print(piece, end="", flush=True)
print()
```

### Embeddings and Semantic Search

```python
import numpy as np
from mullama import Model, EmbeddingGenerator, cosine_similarity

model = Model.load("./nomic-embed-text-v1.5.Q4_K_M.gguf")
gen = EmbeddingGenerator(model)

# Build a document index
documents = [
    "Python is a high-level programming language",
    "Machine learning uses statistical methods to learn from data",
    "The weather forecast predicts rain tomorrow",
    "Neural networks are inspired by biological neurons",
    "JavaScript runs in web browsers and on servers",
    "Docker containers package applications with their dependencies",
]

doc_embeddings = gen.embed_batch(documents)

# Convert to NumPy matrix for efficient operations
embedding_matrix = np.array(doc_embeddings)
print(f"Embedding matrix shape: {embedding_matrix.shape}")  # (6, 4096)

# Query
query_emb = gen.embed("What programming languages are popular?")

# Rank by similarity using NumPy
scores = embedding_matrix @ query_emb  # dot product (embeddings are normalized)

ranked_indices = np.argsort(scores)[::-1]
print("Search results:")
for idx in ranked_indices:
    print(f"  [{scores[idx]:.4f}] {documents[idx]}")
```

### NumPy Integration for Embeddings

```python
import numpy as np
from mullama import Model, EmbeddingGenerator

model = Model.load("./embedding-model.gguf")
gen = EmbeddingGenerator(model)

# Embeddings are already NumPy arrays
emb = gen.embed("Hello, world!")
print(type(emb))         # <class 'numpy.ndarray'>
print(emb.dtype)         # float32
print(emb.shape)         # (4096,)

# Direct NumPy operations
norm = np.linalg.norm(emb)
print(f"L2 norm: {norm:.4f}")  # ~1.0 for normalized embeddings

# Batch: stack into matrix
texts = ["doc1", "doc2", "doc3"]
embeddings = gen.embed_batch(texts)
matrix = np.stack(embeddings)  # shape: (3, 4096)

# Pairwise cosine similarity matrix
similarity_matrix = matrix @ matrix.T
print(f"Similarity matrix shape: {similarity_matrix.shape}")  # (3, 3)

# Save/load embeddings
np.save("embeddings.npy", matrix)
loaded = np.load("embeddings.npy")
```

### Chat Conversations

```python
from mullama import Model, Context, SamplerParams

model = Model.load("./llama-3.2-1b-instruct.gguf", n_gpu_layers=-1)
ctx = Context(model, n_ctx=4096)

messages: list[tuple[str, str]] = [
    ("system", "You are a helpful Python tutor. Be concise and provide code examples."),
]


def chat(user_message: str) -> str:
    """Send a message and get a response."""
    messages.append(("user", user_message))
    prompt = model.apply_chat_template(messages)

    # Check context length
    prompt_tokens = model.tokenize(prompt, add_bos=False)
    if len(prompt_tokens) > ctx.n_ctx - 500:
        # Trim old messages but keep system prompt
        del messages[1:-4]
        prompt = model.apply_chat_template(messages)

    ctx.clear_cache()
    response = ctx.generate(prompt, max_tokens=400, params=SamplerParams.precise())
    messages.append(("assistant", response))
    return response


# Interactive conversation
print(chat("What are list comprehensions?"))
print(chat("Show me an example with filtering."))
print(chat("How about nested list comprehensions?"))
```

### FastAPI Server with Streaming SSE

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mullama import Model, Context, SamplerParams, EmbeddingGenerator
import json

app = FastAPI(title="Mullama API")

# Load model at startup
model = Model.load("./model.gguf", n_gpu_layers=-1)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    stream: bool = False


class EmbedRequest(BaseModel):
    texts: list[str]


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    ctx = Context(model, n_ctx=2048)
    params = SamplerParams(temperature=req.temperature)
    text = ctx.generate(req.prompt, max_tokens=req.max_tokens, params=params)
    tokens = model.tokenize(text, add_bos=False)
    return GenerateResponse(text=text, tokens_generated=len(tokens))


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    def event_stream():
        ctx = Context(model, n_ctx=2048)
        params = SamplerParams(temperature=req.temperature)
        pieces = ctx.generate_stream(
            req.prompt, max_tokens=req.max_tokens, params=params
        )
        for piece in pieces:
            yield f"data: {json.dumps({'token': piece})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/embed")
def embed(req: EmbedRequest):
    gen = EmbeddingGenerator(model)
    embeddings = gen.embed_batch(req.texts)
    return {"embeddings": [emb.tolist() for emb in embeddings]}
```

Run with:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Flask Server

```python
from flask import Flask, request, jsonify, Response
from mullama import Model, Context, SamplerParams, EmbeddingGenerator
import json

app = Flask(__name__)

model = Model.load("./model.gguf", n_gpu_layers=-1)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    ctx = Context(model, n_ctx=2048)
    params = SamplerParams(temperature=data.get("temperature", 0.8))
    text = ctx.generate(
        data["prompt"],
        max_tokens=data.get("max_tokens", 100),
        params=params,
    )
    return jsonify({"text": text})


@app.route("/generate/stream", methods=["POST"])
def generate_stream():
    data = request.json

    def event_stream():
        ctx = Context(model, n_ctx=2048)
        params = SamplerParams(temperature=data.get("temperature", 0.8))
        pieces = ctx.generate_stream(
            data["prompt"],
            max_tokens=data.get("max_tokens", 200),
            params=params,
        )
        for piece in pieces:
            yield f"data: {json.dumps({'token': piece})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    gen = EmbeddingGenerator(model)
    embeddings = gen.embed_batch(data["texts"])
    return jsonify({"embeddings": [emb.tolist() for emb in embeddings]})


if __name__ == "__main__":
    app.run(port=8000)
```

### Django Integration

```python
# myapp/services.py
from mullama import Model, Context, SamplerParams, EmbeddingGenerator
from django.conf import settings

_model = None


def get_model() -> Model:
    """Lazy-load the model as a singleton."""
    global _model
    if _model is None:
        _model = Model.load(
            settings.MULLAMA_MODEL_PATH,
            n_gpu_layers=getattr(settings, "MULLAMA_GPU_LAYERS", -1),
        )
    return _model


def generate_text(prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
    model = get_model()
    ctx = Context(model, n_ctx=2048)
    params = SamplerParams(temperature=temperature)
    return ctx.generate(prompt, max_tokens=max_tokens, params=params)


def get_embeddings(texts: list[str]) -> list:
    model = get_model()
    gen = EmbeddingGenerator(model)
    return gen.embed_batch(texts)
```

```python
# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from .services import generate_text, get_embeddings


@require_POST
def generate_view(request):
    data = json.loads(request.body)
    text = generate_text(
        prompt=data["prompt"],
        max_tokens=data.get("max_tokens", 200),
        temperature=data.get("temperature", 0.8),
    )
    return JsonResponse({"text": text})


@require_POST
def embed_view(request):
    data = json.loads(request.body)
    embeddings = get_embeddings(data["texts"])
    return JsonResponse({"embeddings": [emb.tolist() for emb in embeddings]})
```

### Complete Production Chatbot

```python
from mullama import Model, Context, SamplerParams, supports_gpu_offload

class Chatbot:
    def __init__(
        self,
        model_path: str,
        system_prompt: str = "You are a helpful assistant.",
        max_context: int = 4096,
    ):
        gpu_layers = -1 if supports_gpu_offload() else 0
        self.model = Model.load(model_path, n_gpu_layers=gpu_layers)
        self.ctx = Context(self.model, n_ctx=max_context)
        self.system_prompt = system_prompt
        self.messages: list[tuple[str, str]] = [("system", system_prompt)]
        self.params = SamplerParams.precise()

        print(f"Model: {self.model.name or self.model.description}")
        print(f"Architecture: {self.model.architecture}")
        print(f"Parameters: {self.model.n_params:,}")
        print(f"GPU: {'enabled' if supports_gpu_offload() else 'CPU only'}")
        print(f"Context: {self.ctx.n_ctx} tokens")

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.messages.append(("user", user_message))

        prompt = self.model.apply_chat_template(self.messages)
        prompt_tokens = self.model.tokenize(prompt, add_bos=False)

        # Trim if approaching context limit
        while len(prompt_tokens) > self.ctx.n_ctx - 500 and len(self.messages) > 2:
            self.messages.pop(1)  # Remove oldest non-system message
            prompt = self.model.apply_chat_template(self.messages)
            prompt_tokens = self.model.tokenize(prompt, add_bos=False)

        self.ctx.clear_cache()
        response = self.ctx.generate(prompt, max_tokens=500, params=self.params)
        self.messages.append(("assistant", response))
        return response

    def reset(self):
        """Reset the conversation history."""
        self.messages = [("system", self.system_prompt)]
        self.ctx.clear_cache()

    @property
    def history(self) -> list[tuple[str, str]]:
        """Get conversation history (excluding system prompt)."""
        return self.messages[1:]


def main():
    bot = Chatbot("./llama-3.2-1b-instruct.Q4_K_M.gguf")
    print("---")
    print("Type /quit to exit, /reset to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            bot.reset()
            print("[Conversation reset]\n")
            continue

        response = bot.chat(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
```

### Complete RAG Example with NumPy

```python
import numpy as np
from mullama import (
    Model,
    Context,
    EmbeddingGenerator,
    SamplerParams,
    cosine_similarity,
)


class RAGPipeline:
    def __init__(self, embedding_model_path: str, generation_model_path: str):
        # Separate models for embedding and generation
        self.emb_model = Model.load(embedding_model_path)
        self.gen_model = Model.load(generation_model_path, n_gpu_layers=-1)
        self.embed_gen = EmbeddingGenerator(self.emb_model, n_ctx=512, normalize=True)

        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[str]) -> None:
        """Add documents to the index."""
        new_embeddings = self.embed_gen.embed_batch(docs)
        new_matrix = np.stack(new_embeddings)

        self.documents.extend(docs)
        if self.embeddings is None:
            self.embeddings = new_matrix
        else:
            self.embeddings = np.vstack([self.embeddings, new_matrix])

        print(f"Indexed {len(docs)} documents (total: {len(self.documents)})")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the most relevant documents for a query."""
        if self.embeddings is None:
            return []

        query_emb = self.embed_gen.embed(query)

        # Efficient similarity computation with NumPy
        scores = self.embeddings @ query_emb  # dot product (normalized = cosine sim)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.documents[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def answer(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using retrieved context."""
        relevant = self.retrieve(question, top_k)

        context = "\n".join(
            f"[{i+1}] {r['text']}" for i, r in enumerate(relevant)
        )

        messages = [
            ("system", "Answer the question based only on the provided context. "
                       "Cite sources using [n] notation. If the context does not "
                       "contain the answer, say so."),
            ("user", f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        prompt = self.gen_model.apply_chat_template(messages)
        ctx = Context(self.gen_model, n_ctx=2048)
        answer = ctx.generate(prompt, max_tokens=300, params=SamplerParams.precise())

        return {
            "answer": answer,
            "sources": relevant,
        }

    def save_index(self, path: str) -> None:
        """Save the embedding index to disk."""
        if self.embeddings is not None:
            np.save(path, self.embeddings)

    def load_index(self, path: str, documents: list[str]) -> None:
        """Load a pre-computed embedding index."""
        self.embeddings = np.load(path)
        self.documents = documents


# Usage
rag = RAGPipeline(
    "./nomic-embed-text-v1.5.Q4_K_M.gguf",
    "./llama-3.2-1b-instruct.Q4_K_M.gguf",
)

rag.add_documents([
    "Mullama is a Rust library for running LLMs locally with native performance.",
    "Mullama supports the GGUF model format developed by the llama.cpp project.",
    "GPU offloading accelerates inference by running model layers on the GPU.",
    "The Python bindings use PyO3 for zero-overhead native integration.",
    "Embeddings can be used for semantic search and retrieval-augmented generation.",
    "Streaming generation returns tokens one by one as they are produced.",
    "Models can be quantized to Q4_K_M for the best balance of speed and quality.",
])

result = rag.answer("How do I use GPU acceleration with Mullama?")
print(f"Answer: {result['answer']}")
print("\nSources:")
for src in result["sources"]:
    print(f"  [{src['score']:.4f}] {src['text']}")

# Save for later
rag.save_index("mullama_docs.npy")
```

---

## Async Support

While the core mullama API is synchronous, you can use it with asyncio by running generation in a thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from mullama import Model, Context, SamplerParams

model = Model.load("./model.gguf", n_gpu_layers=-1)
executor = ThreadPoolExecutor(max_workers=4)


async def generate_async(prompt: str, max_tokens: int = 200) -> str:
    """Run generation in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    ctx = Context(model, n_ctx=2048)
    params = SamplerParams(temperature=0.8)
    return await loop.run_in_executor(
        executor,
        lambda: ctx.generate(prompt, max_tokens=max_tokens, params=params),
    )


async def main():
    # Run multiple generations concurrently
    tasks = [
        generate_async("What is Python?"),
        generate_async("What is Rust?"),
        generate_async("What is Go?"),
    ]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result[:100], "...\n")


asyncio.run(main())
```

---

## Error Handling

Python bindings raise standard exceptions:

| Exception | When |
|-----------|------|
| `RuntimeError` | Model loading, context creation, generation, embedding failures |
| `ValueError` | Invalid input (wrong types, mismatched vector lengths) |
| `TypeError` | Wrong argument types |

```python
from mullama import Model, Context, SamplerParams, cosine_similarity
import numpy as np

# Model loading error
try:
    model = Model.load("./nonexistent.gguf")
except RuntimeError as e:
    print(f"Failed to load: {e}")
    # "Failed to load model: No such file or directory: ./nonexistent.gguf"

# Generation error
try:
    model = Model.load("./model.gguf")
    ctx = Context(model)
    result = ctx.generate("", max_tokens=0)
except RuntimeError as e:
    print(f"Generation error: {e}")

# Vector mismatch
try:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    cosine_similarity(a, b)
except ValueError as e:
    print(f"Similarity error: {e}")
    # "Vectors must have the same length"
```

---

## Memory Management

### Context Manager Pattern

```python
from mullama import Model, Context, SamplerParams

# Models persist for the lifetime of the variable
model = Model.load("./model.gguf", n_gpu_layers=-1)

# Contexts can be created and discarded freely
def generate_response(prompt: str) -> str:
    ctx = Context(model, n_ctx=2048)
    result = ctx.generate(prompt, max_tokens=200)
    # ctx is garbage collected here
    return result
```

### Explicit Cleanup

```python
from mullama import backend_free

# For long-running applications, explicitly free resources at shutdown
import atexit
atexit.register(backend_free)
```

### Memory Tips

1. **Model** objects are the largest memory consumers. Load once and share across threads.
2. **Context** objects use memory proportional to `n_ctx`. Use the smallest context size that fits your use case.
3. **Embeddings** are returned as NumPy arrays that can be freed by letting them go out of scope or calling `del`.
4. The Python garbage collector handles all cleanup automatically, but `backend_free()` ensures deterministic release.

---

## Performance Tips

1. **GPU offloading** -- set `n_gpu_layers=-1` to offload all layers for maximum speed.

2. **Reuse models** -- model loading is expensive (seconds). Load once at application startup and create contexts as needed.

3. **NumPy integration** -- embedding results are already NumPy arrays. Use NumPy/SciPy operations directly without conversion for maximum efficiency.

4. **Batch embeddings** -- `embed_batch()` is significantly more efficient than calling `embed()` in a loop due to reduced Python-to-Rust boundary crossings.

5. **Thread count** -- the default `n_threads=0` auto-detects CPU cores. Override only if you need to limit CPU usage for other workloads.

6. **Context size** -- use the smallest `n_ctx` that fits your use case to reduce memory usage.

7. **Memory mapping** -- keep `use_mmap=True` (default) for faster loading and efficient memory sharing between processes.

8. **Thread pool for async** -- wrap synchronous calls in `asyncio.to_thread()` or `ThreadPoolExecutor` when using with async frameworks.

9. **NumPy matrix operations** -- for large-scale similarity search, stack embeddings into a matrix and use matrix multiplication instead of per-vector cosine similarity.
