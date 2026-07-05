"""Tests for the embedding model cache."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from open_agent.rag.embedding_cache import (
    _embedding_cache,
    clear_embedding_cache,
    get_embedding_model,
)

# Ensure ``sentence_transformers`` is importable so that
# ``patch("sentence_transformers.SentenceTransformer")`` can resolve its target
# even when the real package is not installed in the test environment.
if "sentence_transformers" not in sys.modules:
    _stub = types.ModuleType("sentence_transformers")
    _stub.SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"] = _stub


class TestEmbeddingCache:
    def setup_method(self):
        clear_embedding_cache()

    def teardown_method(self):
        clear_embedding_cache()

    def test_returns_same_instance_for_same_model(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance

            model1 = get_embedding_model("test-model")
            model2 = get_embedding_model("test-model")

            assert model1 is model2
            assert mock_st.call_count == 1

    def test_returns_different_instances_for_different_models(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.side_effect = [MagicMock(), MagicMock()]

            model1 = get_embedding_model("model-a")
            model2 = get_embedding_model("model-b")

            assert model1 is not model2
            assert mock_st.call_count == 2

    def test_clear_cache_forces_reload(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()

            get_embedding_model("test-model")
            assert mock_st.call_count == 1

            clear_embedding_cache()
            get_embedding_model("test-model")
            assert mock_st.call_count == 2

    def test_cache_is_empty_after_clear(self):
        with patch("sentence_transformers.SentenceTransformer"):
            get_embedding_model("test-model")
            assert len(_embedding_cache) == 1
            clear_embedding_cache()
            assert len(_embedding_cache) == 0
