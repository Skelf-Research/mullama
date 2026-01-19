# Mullama Python Bindings

High-performance Python bindings for the Mullama LLM library, enabling fast local inference with GGUF models.

## Installation

### From PyPI (when published)

```bash
pip install mullama
```

### From Source

Requires Rust and maturin:

```bash
# Install maturin
pip install maturin

# Build and install
cd bindings/python
maturin develop --release
```

## Quick Start

```python
from mullama import Model, Context, SamplerParams

# Load a model
model = Model.load("path/to/model.gguf", n_gpu_layers=32)

# Create a context
ctx = Context(model, n_ctx=2048)

# Generate text
text = ctx.generate("Once upon a time", max_tokens=100)
print(text)

# With custom sampling parameters
params = SamplerParams(temperature=0.7, top_p=0.9)
text = ctx.generate("Hello, AI!", max_tokens=50, params=params)
```

## Features

### Text Generation

```python
from mullama import Model, Context, SamplerParams

model = Model.load("model.gguf")
ctx = Context(model)

# Basic generation
text = ctx.generate("Hello", max_tokens=100)

# With custom parameters
params = SamplerParams(temperature=0.8, top_k=40)
text = ctx.generate("Hello", max_tokens=100, params=params)

# Greedy (deterministic) generation
params = SamplerParams.greedy()
text = ctx.generate("Hello", max_tokens=100, params=params)

# Creative generation
params = SamplerParams.creative()
text = ctx.generate("Hello", max_tokens=100, params=params)
```

### Streaming Generation

```python
# Get tokens as they're generated
tokens = ctx.generate_stream("Once upon a time", max_tokens=100)
for token in tokens:
    print(token, end="", flush=True)
```

### Tokenization

```python
# Tokenize text
tokens = model.tokenize("Hello, world!")
print(f"Tokens: {tokens}")

# Detokenize back to text
text = model.detokenize(tokens)
print(f"Text: {text}")
```

### Embeddings

```python
from mullama import Model, EmbeddingGenerator, cosine_similarity
import numpy as np

model = Model.load("model.gguf")
gen = EmbeddingGenerator(model)

# Generate embeddings
emb1 = gen.embed("Hello, world!")
emb2 = gen.embed("Hi there!")

# Compute similarity
similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity}")

# Batch embedding
texts = ["Hello", "World", "Test"]
embeddings = gen.embed_batch(texts)
```

### Chat Templates

```python
# Format chat messages
messages = [
    ("system", "You are a helpful assistant."),
    ("user", "What is Python?"),
]
prompt = model.apply_chat_template(messages)
text = ctx.generate(prompt, max_tokens=200)
```

### Model Information

```python
model = Model.load("model.gguf")

print(f"Architecture: {model.architecture}")
print(f"Parameters: {model.n_params:,}")
print(f"Embedding dim: {model.n_embd}")
print(f"Vocabulary size: {model.n_vocab}")
print(f"Context size: {model.n_ctx_train}")
print(f"Model size: {model.size / 1e9:.2f} GB")

# Get all metadata
metadata = model.metadata()
for key, value in metadata.items():
    print(f"{key}: {value}")
```

## API Reference

### Model

```python
class Model:
    @staticmethod
    def load(
        path: str,
        n_gpu_layers: int = 0,  # GPU layers (0=CPU, -1=all)
        use_mmap: bool = True,
        use_mlock: bool = False,
        vocab_only: bool = False,
    ) -> Model: ...

    def tokenize(self, text: str, add_bos: bool = True, special: bool = False) -> List[int]: ...
    def detokenize(self, tokens: List[int], remove_special: bool = False) -> str: ...
    def apply_chat_template(self, messages: List[Tuple[str, str]], add_generation_prompt: bool = True) -> str: ...
    def metadata(self) -> Dict[str, str]: ...

    # Properties
    n_ctx_train: int
    n_embd: int
    n_vocab: int
    n_layer: int
    n_head: int
    token_bos: int
    token_eos: int
    size: int
    n_params: int
    description: str
    architecture: Optional[str]
    name: Optional[str]
```

### Context

```python
class Context:
    def __init__(
        self,
        model: Model,
        n_ctx: int = 0,  # 0 = model default
        n_batch: int = 2048,
        n_threads: int = 0,  # 0 = auto
        embeddings: bool = False,
    ): ...

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 100,
        params: Optional[SamplerParams] = None,
    ) -> str: ...

    def generate_stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 100,
        params: Optional[SamplerParams] = None,
    ) -> List[str]: ...

    def clear_cache(self) -> None: ...

    # Properties
    n_ctx: int
    n_batch: int
```

### SamplerParams

```python
class SamplerParams:
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
    ): ...

    @staticmethod
    def greedy() -> SamplerParams: ...

    @staticmethod
    def creative() -> SamplerParams: ...

    @staticmethod
    def precise() -> SamplerParams: ...
```

### EmbeddingGenerator

```python
class EmbeddingGenerator:
    def __init__(
        self,
        model: Model,
        n_ctx: int = 512,
        normalize: bool = True,
    ): ...

    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]: ...

    # Properties
    n_embd: int
```

### Utility Functions

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float: ...
def backend_init() -> None: ...
def backend_free() -> None: ...
def supports_gpu_offload() -> bool: ...
def system_info() -> str: ...
def max_devices() -> int: ...
```

## Development

### Building

```bash
# Install development dependencies
pip install maturin pytest numpy

# Build in development mode
maturin develop

# Build release wheel
maturin build --release
```

### Testing

```bash
# Run tests (without model)
pytest tests/ -v

# Run tests with a model
MULLAMA_TEST_MODEL=/path/to/model.gguf pytest tests/ -v
```

## License

MIT OR Apache-2.0
