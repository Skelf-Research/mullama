"""Pytest configuration and fixtures for mullama tests."""

import os
import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require a model file"
    )


@pytest.fixture(scope="session")
def test_model_path():
    """Get the path to the test model."""
    path = os.environ.get("MULLAMA_TEST_MODEL", "")
    if not path or not os.path.exists(path):
        pytest.skip("Test model not available. Set MULLAMA_TEST_MODEL environment variable.")
    return path


@pytest.fixture(scope="session")
def model(test_model_path):
    """Load a model for testing."""
    try:
        from mullama import Model
        return Model.load(test_model_path)
    except ImportError:
        pytest.skip("mullama not built")
    except Exception as e:
        pytest.skip(f"Failed to load model: {e}")


@pytest.fixture
def context(model):
    """Create a context for testing."""
    from mullama import Context
    return Context(model, n_ctx=256)


@pytest.fixture
def embedding_generator(model):
    """Create an embedding generator for testing."""
    from mullama import EmbeddingGenerator
    return EmbeddingGenerator(model)
