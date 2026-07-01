"""Tests for RAG reranker implementations."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from open_agent.rag.reranker import BGEReranker, NoOpReranker, build_reranker


def _ensure_fake_sentence_transformers() -> MagicMock:
    """Create a fake sentence_transformers module so BGEReranker can be tested."""
    fake_mod = types.ModuleType("sentence_transformers")
    fake_cross = MagicMock()
    fake_mod.CrossEncoder = fake_cross
    sys.modules["sentence_transformers"] = fake_mod
    return fake_cross


def test_noop_reranker_preserves_order() -> None:
    docs = [
        {"id": "a", "document": "first", "score": 0.9},
        {"id": "b", "document": "second", "score": 0.5},
    ]
    ranked = NoOpReranker().rank("query", docs)
    assert [d["id"] for d in ranked] == ["a", "b"]
    assert ranked[0]["rerank_score"] == 0.9


def test_build_reranker_noop() -> None:
    assert isinstance(build_reranker(None), NoOpReranker)
    assert isinstance(build_reranker("none"), NoOpReranker)
    assert isinstance(build_reranker(""), NoOpReranker)


def test_bge_reranker() -> None:
    fake_cross = _ensure_fake_sentence_transformers()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]
    fake_cross.return_value = mock_model

    reranker = BGEReranker("BAAI/bge-reranker-v2-m3")
    docs = [
        {"id": "a", "document": "less relevant"},
        {"id": "b", "document": "very relevant"},
    ]
    ranked = reranker.rank("query", docs)
    assert [d["id"] for d in ranked] == ["b", "a"]
    assert ranked[0]["rerank_score"] == 0.9
    mock_model.predict.assert_called_once()
    pairs = mock_model.predict.call_args[0][0]
    assert pairs == [("query", "less relevant"), ("query", "very relevant")]


def test_noop_reranker_empty() -> None:
    reranker = NoOpReranker()
    assert reranker.rank("query", []) == []
