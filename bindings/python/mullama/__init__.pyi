"""Type stubs for mullama Python bindings."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

__version__: str

class Model:
    """Model class for loading and managing LLM models."""

    @staticmethod
    def load(
        path: str,
        n_gpu_layers: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        vocab_only: bool = False,
    ) -> Model:
        """Load a model from a GGUF file.

        Args:
            path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only, -1 = all)
            use_mmap: Use memory mapping for model loading
            use_mlock: Lock model in memory
            vocab_only: Only load vocabulary (for tokenization only)

        Returns:
            Loaded model instance

        Raises:
            RuntimeError: If model loading fails
        """
        ...

    def tokenize(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> List[int]:
        """Tokenize text into token IDs.

        Args:
            text: Text to tokenize
            add_bos: Whether to add beginning-of-sequence token
            special: Whether to parse special tokens

        Returns:
            List of token IDs
        """
        ...

    def detokenize(
        self,
        tokens: List[int],
        remove_special: bool = False,
        unparse_special: bool = False,
    ) -> str:
        """Detokenize token IDs back to text.

        Args:
            tokens: List of token IDs
            remove_special: Remove special tokens from output
            unparse_special: Include special token text in output

        Returns:
            Decoded text
        """
        ...

    def token_is_eog(self, token: int) -> bool:
        """Check if a token is end-of-generation."""
        ...

    def metadata(self) -> Dict[str, str]:
        """Get all metadata as a dictionary."""
        ...

    def apply_chat_template(
        self,
        messages: List[Tuple[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to format messages.

        Args:
            messages: List of (role, content) tuples
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatted prompt
        """
        ...

    @property
    def n_ctx_train(self) -> int:
        """Get the model's training context size."""
        ...

    @property
    def n_embd(self) -> int:
        """Get the model's embedding dimension."""
        ...

    @property
    def n_vocab(self) -> int:
        """Get the vocabulary size."""
        ...

    @property
    def n_layer(self) -> int:
        """Get the number of layers."""
        ...

    @property
    def n_head(self) -> int:
        """Get the number of attention heads."""
        ...

    @property
    def token_bos(self) -> int:
        """Get the BOS (beginning of sequence) token ID."""
        ...

    @property
    def token_eos(self) -> int:
        """Get the EOS (end of sequence) token ID."""
        ...

    @property
    def size(self) -> int:
        """Get the model size in bytes."""
        ...

    @property
    def n_params(self) -> int:
        """Get the number of parameters."""
        ...

    @property
    def description(self) -> str:
        """Get the model description."""
        ...

    @property
    def architecture(self) -> Optional[str]:
        """Get the model architecture."""
        ...

    @property
    def name(self) -> Optional[str]:
        """Get the model name from metadata."""
        ...


class SamplerParams:
    """Sampler parameters for text generation."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    typical_p: float
    penalty_repeat: float
    penalty_freq: float
    penalty_present: float
    penalty_last_n: int
    seed: int

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
    ) -> None:
        """Create new sampler parameters.

        Args:
            temperature: Randomness (0.0 = deterministic, higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p/nucleus sampling (1.0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            typical_p: Typical sampling (1.0 = disabled)
            penalty_repeat: Repeat penalty (1.0 = disabled)
            penalty_freq: Frequency penalty (0.0 = disabled)
            penalty_present: Presence penalty (0.0 = disabled)
            penalty_last_n: Tokens to consider for penalties
            seed: Random seed (0 = random)
        """
        ...

    @staticmethod
    def greedy() -> SamplerParams:
        """Create greedy (deterministic) sampler params."""
        ...

    @staticmethod
    def creative() -> SamplerParams:
        """Create creative (high randomness) sampler params."""
        ...

    @staticmethod
    def precise() -> SamplerParams:
        """Create precise (low randomness) sampler params."""
        ...


class Context:
    """Context for model inference."""

    def __init__(
        self,
        model: Model,
        n_ctx: int = 0,
        n_batch: int = 2048,
        n_threads: int = 0,
        embeddings: bool = False,
    ) -> None:
        """Create a new context from a model.

        Args:
            model: The model to create context for
            n_ctx: Context size (0 = use model default)
            n_batch: Batch size for prompt processing
            n_threads: Number of threads (0 = auto)
            embeddings: Enable embeddings mode
        """
        ...

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 100,
        params: Optional[SamplerParams] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Text prompt or list of token IDs
            max_tokens: Maximum tokens to generate
            params: Optional sampler parameters

        Returns:
            Generated text
        """
        ...

    def generate_stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 100,
        params: Optional[SamplerParams] = None,
    ) -> List[str]:
        """Generate text with streaming (returns list of token strings).

        Args:
            prompt: Text prompt or list of token IDs
            max_tokens: Maximum tokens to generate
            params: Optional sampler parameters

        Returns:
            List of generated token strings
        """
        ...

    def decode(self, tokens: List[int]) -> None:
        """Decode tokens (process through the model).

        Args:
            tokens: List of token IDs to decode
        """
        ...

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        ...

    def get_embeddings(self) -> Optional[NDArray[np.float32]]:
        """Get embeddings (if embeddings mode is enabled)."""
        ...

    @property
    def n_ctx(self) -> int:
        """Get the context size."""
        ...

    @property
    def n_batch(self) -> int:
        """Get the batch size."""
        ...


class EmbeddingGenerator:
    """Embedding generator for creating text embeddings."""

    def __init__(
        self,
        model: Model,
        n_ctx: int = 512,
        normalize: bool = True,
    ) -> None:
        """Create a new embedding generator.

        Args:
            model: The model to use for embeddings
            n_ctx: Context size (0 = model default)
            normalize: Whether to normalize embeddings
        """
        ...

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    @property
    def n_embd(self) -> int:
        """Get the embedding dimension."""
        ...


def cosine_similarity(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    ...


def backend_init() -> None:
    """Initialize the mullama backend."""
    ...


def backend_free() -> None:
    """Free the mullama backend resources."""
    ...


def supports_gpu_offload() -> bool:
    """Check if GPU offloading is supported."""
    ...


def system_info() -> str:
    """Get system information."""
    ...


def max_devices() -> int:
    """Get the maximum number of supported devices."""
    ...
