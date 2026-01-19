"""
Tests for mullama Python bindings.

These tests cover the core functionality of the mullama library.
Note: Some tests require a model file to be present at the specified path.
"""

import pytest
import numpy as np
import os

# Check if we can import mullama
try:
    import mullama
    MULLAMA_AVAILABLE = True
except ImportError:
    MULLAMA_AVAILABLE = False

# Path to test model (set via environment variable or use default)
TEST_MODEL_PATH = os.environ.get("MULLAMA_TEST_MODEL", "")
MODEL_AVAILABLE = os.path.exists(TEST_MODEL_PATH) if TEST_MODEL_PATH else False


class TestModuleImport:
    """Tests for module import and basic functionality."""

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_import(self):
        """Test that the module can be imported."""
        import mullama
        assert hasattr(mullama, "Model")
        assert hasattr(mullama, "Context")
        assert hasattr(mullama, "SamplerParams")
        assert hasattr(mullama, "EmbeddingGenerator")

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_version(self):
        """Test that version is available."""
        import mullama
        assert hasattr(mullama, "__version__")
        assert isinstance(mullama.__version__, str)
        assert len(mullama.__version__) > 0

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_system_info(self):
        """Test system info function."""
        import mullama
        info = mullama.system_info()
        assert isinstance(info, str)
        # Should contain some hardware info
        assert len(info) > 0

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_supports_gpu_offload(self):
        """Test GPU support detection."""
        import mullama
        result = mullama.supports_gpu_offload()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_max_devices(self):
        """Test max devices function."""
        import mullama
        devices = mullama.max_devices()
        assert isinstance(devices, int)
        assert devices >= 1  # At least one device (CPU)


class TestSamplerParams:
    """Tests for SamplerParams class."""

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_default_params(self):
        """Test default sampler parameters."""
        from mullama import SamplerParams
        params = SamplerParams()
        assert params.temperature == pytest.approx(0.8)
        assert params.top_k == 40
        assert params.top_p == pytest.approx(0.95)

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_custom_params(self):
        """Test custom sampler parameters."""
        from mullama import SamplerParams
        params = SamplerParams(temperature=0.5, top_k=20, top_p=0.8)
        assert params.temperature == pytest.approx(0.5)
        assert params.top_k == 20
        assert params.top_p == pytest.approx(0.8)

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_greedy_params(self):
        """Test greedy sampler parameters."""
        from mullama import SamplerParams
        params = SamplerParams.greedy()
        assert params.temperature == 0.0
        assert params.top_k == 1

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_creative_params(self):
        """Test creative sampler parameters."""
        from mullama import SamplerParams
        params = SamplerParams.creative()
        assert params.temperature > 1.0
        assert params.top_k > 40

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_precise_params(self):
        """Test precise sampler parameters."""
        from mullama import SamplerParams
        params = SamplerParams.precise()
        assert params.temperature < 0.5
        assert params.top_k < 40

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_params_repr(self):
        """Test sampler params string representation."""
        from mullama import SamplerParams
        params = SamplerParams()
        repr_str = repr(params)
        assert "SamplerParams" in repr_str
        assert "temperature" in repr_str

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_params_setters(self):
        """Test sampler params setters."""
        from mullama import SamplerParams
        params = SamplerParams()
        params.temperature = 0.5
        assert params.temperature == pytest.approx(0.5)
        params.top_k = 50
        assert params.top_k == 50


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        from mullama import cosine_similarity
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim - 1.0) < 0.001

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        from mullama import cosine_similarity
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim) < 0.001

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        from mullama import cosine_similarity
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim + 1.0) < 0.001

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_random_vectors(self):
        """Test similarity of random vectors."""
        from mullama import cosine_similarity
        np.random.seed(42)
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


class TestModelLoading:
    """Tests for Model class (requires model file)."""

    @pytest.mark.skipif(not MULLAMA_AVAILABLE, reason="mullama not built")
    def test_load_nonexistent_model(self):
        """Test loading a nonexistent model fails gracefully."""
        from mullama import Model
        with pytest.raises(RuntimeError):
            Model.load("/nonexistent/path/model.gguf")

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_load_model(self):
        """Test loading a model."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)
        assert model is not None
        assert model.n_vocab > 0
        assert model.n_embd > 0

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_model_properties(self):
        """Test model properties."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)

        # Basic properties
        assert model.n_ctx_train > 0
        assert model.n_embd > 0
        assert model.n_vocab > 0
        assert model.n_layer > 0
        assert model.n_head > 0
        assert model.size > 0
        assert model.n_params > 0

        # Special tokens
        assert model.token_bos >= 0
        assert model.token_eos >= 0

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_tokenization(self):
        """Test tokenization."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)

        text = "Hello, world!"
        tokens = model.tokenize(text)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

        # Roundtrip
        decoded = model.detokenize(tokens)
        assert isinstance(decoded, str)

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_tokenize_with_options(self):
        """Test tokenization with options."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)

        text = "Hello"

        # With BOS
        tokens_with_bos = model.tokenize(text, add_bos=True)

        # Without BOS
        tokens_without_bos = model.tokenize(text, add_bos=False)

        # With BOS should have more tokens or same (if model doesn't use BOS)
        assert len(tokens_with_bos) >= len(tokens_without_bos)

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_model_repr(self):
        """Test model string representation."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)
        repr_str = repr(model)
        assert "Model" in repr_str

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_model_metadata(self):
        """Test model metadata."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)
        metadata = model.metadata()
        assert isinstance(metadata, dict)


class TestContext:
    """Tests for Context class (requires model file)."""

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_create_context(self):
        """Test creating a context."""
        from mullama import Model, Context
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model)
        assert ctx is not None
        assert ctx.n_ctx > 0
        assert ctx.n_batch > 0

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_context_with_params(self):
        """Test creating a context with parameters."""
        from mullama import Model, Context
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model, n_ctx=512, n_batch=256)
        assert ctx.n_ctx == 512
        assert ctx.n_batch == 256

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_generate_text(self):
        """Test text generation."""
        from mullama import Model, Context, SamplerParams
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model, n_ctx=256)

        params = SamplerParams.greedy()  # Deterministic
        text = ctx.generate("Hello", max_tokens=10, params=params)

        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_generate_from_tokens(self):
        """Test text generation from tokens."""
        from mullama import Model, Context, SamplerParams
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model, n_ctx=256)

        tokens = model.tokenize("Hello")
        params = SamplerParams.greedy()
        text = ctx.generate(tokens, max_tokens=10, params=params)

        assert isinstance(text, str)

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_generate_stream(self):
        """Test streaming generation."""
        from mullama import Model, Context, SamplerParams
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model, n_ctx=256)

        params = SamplerParams.greedy()
        tokens = ctx.generate_stream("Hello", max_tokens=10, params=params)

        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_clear_cache(self):
        """Test clearing KV cache."""
        from mullama import Model, Context
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model, n_ctx=256)

        # Should not raise
        ctx.clear_cache()

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_context_repr(self):
        """Test context string representation."""
        from mullama import Model, Context
        model = Model.load(TEST_MODEL_PATH)
        ctx = Context(model)
        repr_str = repr(ctx)
        assert "Context" in repr_str


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class (requires model file)."""

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_create_generator(self):
        """Test creating an embedding generator."""
        from mullama import Model, EmbeddingGenerator
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model)
        assert gen is not None
        assert gen.n_embd > 0

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_embed_text(self):
        """Test embedding text."""
        from mullama import Model, EmbeddingGenerator
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model)

        embedding = gen.embed("Hello, world!")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == gen.n_embd

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_embed_batch(self):
        """Test batch embedding."""
        from mullama import Model, EmbeddingGenerator
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model)

        texts = ["Hello", "World", "Test"]
        embeddings = gen.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == gen.n_embd

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        from mullama import Model, EmbeddingGenerator
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model, normalize=True)

        embedding = gen.embed("Hello")

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_embedding_similarity(self):
        """Test embedding similarity."""
        from mullama import Model, EmbeddingGenerator, cosine_similarity
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model)

        emb1 = gen.embed("Hello, world!")
        emb2 = gen.embed("Hello, world!")  # Same text
        emb3 = gen.embed("Goodbye, universe!")  # Different text

        # Same text should have high similarity
        sim_same = cosine_similarity(emb1, emb2)
        assert sim_same > 0.99

        # Different text should have lower similarity
        sim_diff = cosine_similarity(emb1, emb3)
        assert sim_diff < sim_same

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_generator_repr(self):
        """Test generator string representation."""
        from mullama import Model, EmbeddingGenerator
        model = Model.load(TEST_MODEL_PATH)
        gen = EmbeddingGenerator(model)
        repr_str = repr(gen)
        assert "EmbeddingGenerator" in repr_str


class TestChatTemplate:
    """Tests for chat template functionality (requires model file)."""

    @pytest.mark.skipif(
        not MULLAMA_AVAILABLE or not MODEL_AVAILABLE,
        reason="mullama not built or model not available"
    )
    def test_apply_chat_template(self):
        """Test applying chat template."""
        from mullama import Model
        model = Model.load(TEST_MODEL_PATH)

        messages = [
            ("system", "You are a helpful assistant."),
            ("user", "Hello!"),
        ]

        try:
            formatted = model.apply_chat_template(messages)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
        except RuntimeError:
            # Model might not have a chat template
            pytest.skip("Model doesn't support chat templates")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
