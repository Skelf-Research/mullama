"""
Mullama - Python bindings for local LLM inference

High-performance Python bindings for the Mullama LLM library,
enabling fast local inference with GGUF models.

Example:
    >>> from mullama import Model, Context
    >>> model = Model.load("model.gguf", n_gpu_layers=32)
    >>> ctx = Context(model)
    >>> text = ctx.generate("Hello, AI!")
    >>> print(text)
"""

from mullama._mullama import (
    Model,
    Context,
    SamplerParams,
    EmbeddingGenerator,
    cosine_similarity,
    backend_init,
    backend_free,
    supports_gpu_offload,
    system_info,
    max_devices,
    __version__,
)

__all__ = [
    "Model",
    "Context",
    "SamplerParams",
    "EmbeddingGenerator",
    "cosine_similarity",
    "backend_init",
    "backend_free",
    "supports_gpu_offload",
    "system_info",
    "max_devices",
    "__version__",
]

# Auto-initialize backend on import
backend_init()
