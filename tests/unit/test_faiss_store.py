"""Tests for the FAISSStore vector store."""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- Stub the faiss module so tests run without faiss-cpu installed ----------
class _FakeIndex:
    """Minimal IndexFlatIP stand-in: inner product over stored vectors."""

    def __init__(self, dim: int) -> None:
        self.d = dim
        self._vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = np.array(arr, dtype=np.float32)
        else:
            self._vectors = np.vstack([self._vectors, arr])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or len(self._vectors) == 0:
            return (
                np.zeros((1, 0), dtype=np.float32),
                np.full((1, k), -1, dtype=np.int64),
            )
        scores = np.dot(self._vectors, query[0])
        k = min(k, len(scores))
        top_idx = np.argsort(-scores)[:k]
        top_scores = scores[top_idx]
        return (
            np.array([top_scores], dtype=np.float32),
            np.array([top_idx], dtype=np.int64),
        )


def _fake_normalize_l2(arr: np.ndarray) -> None:
    """L2-normalize ``arr`` in place (1-D or 2-D)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        norm = float(np.linalg.norm(a))
        if norm > 0:
            arr /= norm
    else:
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        a /= norms


def _fake_write_index(index: _FakeIndex, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)


def _fake_read_index(path: str) -> _FakeIndex:
    with open(path, "rb") as f:
        return pickle.load(f)


if "faiss" not in sys.modules:
    _faiss_stub = types.ModuleType("faiss")
    _faiss_stub.IndexFlatIP = _FakeIndex
    _faiss_stub.normalize_L2 = _fake_normalize_l2
    _faiss_stub.write_index = _fake_write_index
    _faiss_stub.read_index = _fake_read_index
    sys.modules["faiss"] = _faiss_stub

from open_agent.rag.stores.faiss_store import FAISSStore  # noqa: E402


def _embed_texts(texts: list[str], dim: int = 8) -> np.ndarray:
    """Deterministic fake embeddings keyed by text content (MD5 hash).

    The same text always yields the same vector, so querying with a stored
    document's text returns it as the top hit.
    """
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.md5(text.encode("utf-8")).digest()
        for j in range(dim):
            embeddings[i, j] = (digest[j % len(digest)] / 255.0) - 0.5
    return embeddings


def _make_mock_model(dim: int = 8) -> MagicMock:
    """Build a fake SentenceTransformer with a fixed embedding dimension."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def encode(texts: list[str], **_kwargs: Any) -> np.ndarray:
        return _embed_texts(texts, dim)

    model.encode = encode
    return model


@pytest.fixture
def mock_model():
    """Patch ``get_embedding_model`` so FAISSStore uses a fake model."""
    model = _make_mock_model(dim=8)
    with patch(
        "open_agent.rag.stores.faiss_store.get_embedding_model",
        return_value=model,
    ):
        yield model


@pytest.fixture
def store(mock_model):
    """Return an empty FAISSStore backed by the mock embedding model."""
    return FAISSStore()


# ---------- initialization ---------------------------------------------------


def test_default_embedding_model_is_bge_small_zh():
    """The project requires the default embedding model to be BAAI/bge-small-zh-v1.5."""
    model = _make_mock_model(dim=8)
    with patch(
        "open_agent.rag.stores.faiss_store.get_embedding_model",
        return_value=model,
    ) as mock_get:
        FAISSStore()
        mock_get.assert_called_once_with("BAAI/bge-small-zh-v1.5")


def test_custom_embedding_model_passed_through():
    model = _make_mock_model(dim=8)
    with patch(
        "open_agent.rag.stores.faiss_store.get_embedding_model",
        return_value=model,
    ) as mock_get:
        FAISSStore(embedding_model="custom-model")
        mock_get.assert_called_once_with("custom-model")


def test_store_starts_empty(store):
    assert store._ids == []
    assert store._documents == []
    assert store._metadatas == []


# ---------- add --------------------------------------------------------------


async def test_add_texts_stores_ids_documents_and_metadata(store):
    ids = ["a", "b", "c"]
    docs = ["apple", "banana", "cherry"]
    metas = [{"n": 1}, {"n": 2}, {"n": 3}]
    await store.add(ids, docs, metas)
    assert store._ids == ["a", "b", "c"]
    assert store._documents == ["apple", "banana", "cherry"]
    assert store._metadatas == [{"n": 1}, {"n": 2}, {"n": 3}]


async def test_add_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        await store.add(["a", "b"], ["only one doc"])


async def test_add_default_metadata_when_none(store):
    await store.add(["a"], ["doc"])
    assert store._metadatas == [{}]


# ---------- search -----------------------------------------------------------


async def test_search_returns_top_k_with_scores(store):
    ids = ["a", "b", "c"]
    docs = ["apple pie", "banana split", "cherry tart"]
    await store.add(ids, docs)
    results = await store.query("apple pie", n_results=2)
    assert len(results) == 2
    for r in results:
        assert {"id", "document", "score", "metadata"} <= r.keys()
    # The exact-match query should rank "apple pie" first.
    assert results[0]["id"] == "a"


async def test_search_on_empty_store_returns_empty(store):
    assert await store.query("anything", n_results=5) == []


async def test_search_n_results_zero_returns_empty(store):
    await store.add(["a"], ["doc"])
    assert await store.query("doc", n_results=0) == []


# ---------- delete -----------------------------------------------------------


async def test_delete_removes_documents_by_id(store):
    ids = ["a", "b", "c"]
    docs = ["alpha", "beta", "gamma"]
    await store.add(ids, docs)
    await store.delete(["b"])
    assert store._ids == ["a", "c"]
    assert store._documents == ["alpha", "gamma"]


async def test_delete_all_leaves_empty_index(store):
    await store.add(["a", "b"], ["x", "y"])
    await store.delete(["a", "b"])
    assert store._ids == []
    assert await store.count() == 0
    # Empty store can still be queried.
    assert await store.query("x", n_results=5) == []


async def test_delete_unknown_id_is_noop(store):
    await store.add(["a"], ["doc"])
    await store.delete(["nonexistent"])
    assert store._ids == ["a"]


# ---------- count ------------------------------------------------------------


async def test_count_returns_number_of_documents(store):
    await store.add(["a", "b", "c"], ["x", "y", "z"])
    assert await store.count() == 3
    await store.delete(["a"])
    assert await store.count() == 2


# ---------- save / load ------------------------------------------------------


async def test_save_and_load_round_trip(store, tmp_path):
    ids = ["a", "b"]
    docs = ["first doc", "second doc"]
    metas = [{"src": 1}, {"src": 2}]
    await store.add(ids, docs, metas)

    index_path = str(tmp_path / "index.faiss")
    await store.save(index_path)

    # Load into a new store (patch is still active via the `store` fixture).
    loaded = FAISSStore(index_path=index_path)
    assert loaded._ids == ["a", "b"]
    assert loaded._documents == ["first doc", "second doc"]
    assert loaded._metadatas == [{"src": 1}, {"src": 2}]
    # Loaded store can search and returns the correct top hit.
    results = await loaded.query("first doc", n_results=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


# ---------- async lock -------------------------------------------------------


async def test_concurrent_adds_no_deadlock(store):
    """Multiple concurrent add calls should all complete (asyncio.Lock
    serializes them without deadlock)."""
    await asyncio.gather(
        store.add(["a"], ["apple"]),
        store.add(["b"], ["banana"]),
        store.add(["c"], ["cherry"]),
    )
    assert sorted(store._ids) == ["a", "b", "c"]
