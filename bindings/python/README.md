# Mullama for Python — local LLM inference with GGUF models, no Ollama daemon required

[![PyPI](https://img.shields.io/pypi/v/mullama)](https://pypi.org/project/mullama/)
[![Python versions](https://img.shields.io/pypi/pyversions/mullama)](https://pypi.org/project/mullama/)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/cognisoc/mullama/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/cognisoc/mullama/ci.yml?branch=main&label=CI)](https://github.com/cognisoc/mullama/actions)

## What is Mullama for Python?

`mullama` is a Python package that runs local LLMs from GGUF files — Llama 3.2, Qwen 2.5, DeepSeek R1, Mistral, Phi 3, Gemma 2, LLaVA, and anything else in GGUF format — directly inside your Python process. It wraps `llama.cpp` via a Rust core (PyO3 bindings, no `pip install` of `ctypes` glue or C compiler at install time), giving you native llama.cpp throughput without spawning the `ollama` daemon, without HTTP overhead, and without a separate Python subprocess.

Use it as a drop-in alternative to `llama-cpp-python` or `ollama-python` when you want one dependency that covers in-process inference, streaming, embeddings, and chat templating with a single `pip install mullama`.

## Why use `mullama` over `llama-cpp-python` or `ollama-python`?

| | `mullama` | `llama-cpp-python` | `ollama-python` |
|---|:-:|:-:|:-:|
| In-process inference (no daemon) | ✓ | ✓ | ✗ (HTTP to `ollama serve`) |
| Pre-built wheels (linux x86_64/aarch64, macOS x86_64/arm64, Windows) | ✓ | ✓ | n/a |
| GGUF models | ✓ | ✓ | ✓ |
| Streaming token-by-token | ✓ | ✓ | ✓ |
| Built-in embeddings | ✓ | ✓ | ✓ (via HTTP) |
| Vision (LLaVA / Moondream) | ✓ | ✓ | ✓ |
| Memory-safe API (no segfaults from Python) | ✓ (Rust-backed) | partial | n/a |
| Ships an OpenAI-compatible HTTP server too | ✓ (`mullama serve`) | ✗ (separate `llama-cpp-python[server]`) | ✓ |
| GPU backends | CUDA, Metal, ROCm, Vulkan, OpenCL, SYCL | CUDA, Metal, ROCm, OpenCL | depends on daemon build |

## Install

```bash
pip install mullama
```

Pre-built wheels are available for Python 3.8 – 3.12 on Linux x86_64, Linux aarch64, macOS x86_64, macOS aarch64 (Metal), and Windows x86_64. For GPU-accelerated builds, see [GPU acceleration](#gpu-acceleration) below.

## Quick start

```python
from mullama import Model, Context

model = Model.load("./llama3.2-1b.gguf", n_gpu_layers=32)
ctx = Context(model, n_ctx=2048)

print(ctx.generate("What is the capital of France?", max_tokens=100))
```

Don't have a GGUF file yet? Grab one from Hugging Face:

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## Streaming

```python
for token in ctx.generate_stream("Once upon a time", max_tokens=100):
    print(token, end="", flush=True)
```

## Embeddings (RAG-ready)

```python
from mullama import Model, EmbeddingGenerator, cosine_similarity

model = Model.load("./nomic-embed-text-v1.5.Q4_K_M.gguf")
gen = EmbeddingGenerator(model)

vec1 = gen.embed("How do I cook pasta?")
vec2 = gen.embed("What's the best way to boil noodles?")
print(cosine_similarity(vec1, vec2))   # ≈ 0.9

# Batch
vectors = gen.embed_batch(["doc 1", "doc 2", "doc 3"])
```

Works with any GGUF embedding model — `nomic-embed`, `bge:small`, `bge:large`, and others.

## Chat templates

```python
messages = [
    ("system", "You are a helpful Python tutor."),
    ("user", "Explain list comprehensions."),
]
prompt = model.apply_chat_template(messages)
print(ctx.generate(prompt, max_tokens=300))
```

The model's built-in chat template (Llama 3, Qwen, Mistral, Phi, Gemma all supported) is read from the GGUF metadata automatically.

## Tokenization

```python
tokens = model.tokenize("Hello, world!")
print(tokens)                    # [128000, 9906, 11, 1917, 0]
print(model.detokenize(tokens))  # "Hello, world!"
```

## Sampler presets

```python
from mullama import SamplerParams

ctx.generate("Tell me a fact about France.", params=SamplerParams.greedy())     # deterministic
ctx.generate("Write a poem about the moon.", params=SamplerParams.creative())   # high randomness
ctx.generate("Summarize this article.", params=SamplerParams.precise())         # low randomness

# Or pass any combination of fields explicitly
params = SamplerParams(temperature=0.7, top_p=0.9, top_k=40, penalty_repeat=1.1)
ctx.generate("...", params=params)
```

## Model information

```python
print(f"Architecture:    {model.architecture}")
print(f"Parameters:      {model.n_params:,}")
print(f"Embedding dim:   {model.n_embd}")
print(f"Vocabulary:      {model.n_vocab}")
print(f"Trained context: {model.n_ctx_train}")
print(f"Size on disk:    {model.size / 1e9:.2f} GB")
```

## Supported models

Any GGUF file works. Common picks:

- **Llama 3.2** — `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- **Llama 3.1** — `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, 70B variants
- **Qwen 2.5** — `qwen2.5-{0.5b,1.5b,3b,7b,14b,32b,72b}-instruct-q4_k_m.gguf`; coder variants
- **DeepSeek R1** — distilled 1.5B / 7B / 14B / 32B (reasoning)
- **Mistral / Mixtral / Codestral** — 7B Instruct, 8x7B MoE, 22B Codestral
- **Phi 3 / 3.5** — mini, medium
- **Gemma 2** — 2B, 9B, 27B
- **Vision** — LLaVA 1.5 7B/13B, LLaVA-Phi3, Moondream 2B
- **Embeddings** — `nomic-embed-text-v1.5`, `bge-small-en-v1.5`, `bge-large-en-v1.5`

## GPU acceleration

Set the relevant environment variable when building from source (the pre-built wheels are CPU-only — for GPU, install via `pip install mullama --no-binary :all:` with the env var set):

| Backend | Env var | Hardware |
|---|---|---|
| CUDA | `LLAMA_CUDA=1` | NVIDIA |
| Metal | `LLAMA_METAL=1` | Apple Silicon (enabled by default in macOS arm64 wheels) |
| ROCm | `LLAMA_HIPBLAS=1` | AMD |
| Vulkan | `LLAMA_VULKAN=1` | cross-platform |
| SYCL | `LLAMA_SYCL=1` | Intel Arc |

Pass `n_gpu_layers=-1` to offload all layers, or a positive integer for partial offload.

## API reference

### `Model`

```python
class Model:
    @staticmethod
    def load(
        path: str,
        n_gpu_layers: int = 0,   # 0=CPU, -1=all
        use_mmap: bool = True,
        use_mlock: bool = False,
        vocab_only: bool = False,
    ) -> Model: ...

    def tokenize(self, text: str, add_bos: bool = True, special: bool = False) -> List[int]: ...
    def detokenize(self, tokens: List[int], remove_special: bool = False) -> str: ...
    def apply_chat_template(self, messages: List[Tuple[str, str]], add_generation_prompt: bool = True) -> str: ...
    def metadata(self) -> Dict[str, str]: ...

    # Properties: n_ctx_train, n_embd, n_vocab, n_layer, n_head,
    # token_bos, token_eos, size, n_params, description, architecture, name
```

### `Context`

```python
class Context:
    def __init__(
        self,
        model: Model,
        n_ctx: int = 0,        # 0 = model default
        n_batch: int = 2048,
        n_threads: int = 0,    # 0 = auto
        embeddings: bool = False,
    ): ...

    def generate(self, prompt: Union[str, List[int]], max_tokens: int = 100,
                 params: Optional[SamplerParams] = None) -> str: ...
    def generate_stream(self, prompt: Union[str, List[int]], max_tokens: int = 100,
                        params: Optional[SamplerParams] = None) -> Iterator[str]: ...
    def clear_cache(self) -> None: ...
```

### `SamplerParams`

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
        penalize_nl: bool = True,
        ignore_eos: bool = False,
        seed: int = 0,         # 0 = random
    ): ...

    @staticmethod
    def greedy() -> SamplerParams: ...
    @staticmethod
    def creative() -> SamplerParams: ...
    @staticmethod
    def precise() -> SamplerParams: ...
```

### `EmbeddingGenerator`

```python
class EmbeddingGenerator:
    def __init__(self, model: Model, n_ctx: int = 512, normalize: bool = True): ...

    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]: ...

    # Property: n_embd
```

### Utilities

```python
cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
backend_init() -> None
backend_free() -> None
supports_gpu_offload() -> bool
system_info() -> str
max_devices() -> int
```

## FAQ

### Is Mullama production ready?

Yes. The 0.3.x line is used in production at multiple organisations. Memory safety is enforced by the Rust core; the public Python API never exposes raw pointers, and `Model` / `Context` clean up on garbage collection. See [`docs/PRODUCTION.md`](https://docs.cognisoc.com/mullama/deployment/production/) for tuning guidance.

### How does `mullama` compare to `llama-cpp-python`?

Same `llama.cpp` engine underneath. `mullama` adds a Rust-backed safe API, ships a daemon with OpenAI-compatible HTTP server built in, supports streaming embeddings and multimodal inputs out of the box, and is part of a polyglot library so the same models work identically from Node / Go / PHP / Rust. `llama-cpp-python` is a thinner wrapper if you only need the C++ API in Python.

### How does `mullama` compare to `ollama-python`?

`ollama-python` is an HTTP client — it talks to a running `ollama` daemon over `localhost:11434`. `mullama`'s Python package runs inference in-process (no daemon, no HTTP, no Ollama install). If you already have `ollama serve` running, `mullama` can also act as an HTTP client to it via the OpenAI-compatible API on the same port.

### Does Mullama support `asyncio` / `async def`?

The synchronous API is the default and runs in a single GIL-released call (the Rust core releases the GIL during inference, so you can use it from `asyncio.to_thread()` or `run_in_executor` without blocking other tasks). A native async API is on the roadmap for 0.4.

### Can I use Mullama with LangChain or LlamaIndex?

Yes, via two routes: (1) run `mullama serve` and point `ChatOpenAI` / `OpenAILike` at `http://localhost:11434/v1` — the OpenAI-compatible HTTP API works as a drop-in; (2) wrap the in-process `Context` in a custom LLM/Embeddings class — see [the integration guide](https://docs.cognisoc.com/mullama/integrations/langchain/).

### Does it work on Apple Silicon?

Yes, with Metal GPU acceleration enabled by default in the macOS arm64 wheels. M1, M2, M3, M4 all supported.

### How big is the install?

The wheel is roughly 30–60 MB depending on platform (it bundles `libllama.so` / `.dylib` / `.dll`). No `cmake`, no C compiler, no system libraries needed at install time on supported platforms.

### Where do I report bugs?

[GitHub Issues](https://github.com/cognisoc/mullama/issues). For usage questions, [Discussions](https://github.com/cognisoc/mullama/discussions).

## Other language bindings

- [Rust core (`mullama` on crates.io)](https://crates.io/crates/mullama)
- [Node.js / TypeScript (`mullama` on npm)](https://www.npmjs.com/package/mullama) · [README](../node/README.md)
- [Go (`github.com/cognisoc/mullama` on pkg.go.dev)](https://pkg.go.dev/github.com/cognisoc/mullama) · [README](../go/README.md)
- [PHP (`mullama/mullama` on Packagist)](https://packagist.org/packages/mullama/mullama) · [README](../php/README.md)
- [C/C++ (FFI library)](../ffi/README.md)

## Documentation

Full Python docs at **[docs.cognisoc.com/mullama/bindings/python](https://docs.cognisoc.com/mullama/bindings/python/)**.

## Development

```bash
pip install maturin pytest numpy
cd bindings/python
maturin develop --release
pytest tests/ -v
MULLAMA_TEST_MODEL=/path/to/model.gguf pytest tests/ -v
```

## License

MIT OR Apache-2.0
