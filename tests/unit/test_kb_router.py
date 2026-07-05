"""Tests for the KnowledgeBaseRouter (semantic query routing)."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np

# Stub faiss so importing kb_router (which imports faiss_store) works even
# when faiss-cpu is not installed. Tests mock the KB objects entirely, so the
# stub only needs to make ``import faiss`` succeed.
if "faiss" not in sys.modules:
    _faiss_stub = types.ModuleType("faiss")
    _faiss_stub.IndexFlatIP = MagicMock
    _faiss_stub.normalize_L2 = MagicMock()
    _faiss_stub.write_index = MagicMock()
    _faiss_stub.read_index = MagicMock()
    sys.modules["faiss"] = _faiss_stub

from open_agent.rag.kb_router import KnowledgeBaseRouter  # noqa: E402


def _make_fake_kb(
    name: str,
    routing_vec: list[float],
    retrieve_results: list[dict[str, Any]] | None = None,
    query_vec: list[float] | None = None,
) -> MagicMock:
    """Build a fake KnowledgeBase with controllable routing + retrieval.

    ``embed_texts`` returns ``query_vec`` (default ``[1, 0, 0, 0]``) so the
    routing cosine similarity is ``dot(query_vec, routing_vec)``.
    """
    kb = MagicMock()
    kb.name = name
    kb.routing_embedding = AsyncMock(
        return_value=np.array([routing_vec], dtype=np.float32)
    )
    kb.embed_texts = AsyncMock(
        return_value=np.array(
            [query_vec or [1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
    )
    kb.retrieve = AsyncMock(return_value=list(retrieve_results or []))
    return kb


# ---------- registry ---------------------------------------------------------


def test_add_and_list_kbs():
    router = KnowledgeBaseRouter()
    router.add_kb(_make_fake_kb("alpha", [1.0, 0.0, 0.0, 0.0]))
    router.add_kb(_make_fake_kb("beta", [0.0, 1.0, 0.0, 0.0]))
    assert set(router.list_kbs()) == {"alpha", "beta"}


def test_remove_kb():
    router = KnowledgeBaseRouter()
    router.add_kb(_make_fake_kb("alpha", [1.0, 0.0, 0.0, 0.0]))
    router.remove_kb("alpha")
    assert router.list_kbs() == []


def test_remove_absent_kb_is_noop():
    router = KnowledgeBaseRouter()
    router.remove_kb("nonexistent")  # should not raise
    assert router.list_kbs() == []


# ---------- routing ----------------------------------------------------------


async def test_route_single_kb_shortcuts_without_embedding():
    """With one KB, routing returns [name] without calling embed_texts."""
    kb = _make_fake_kb("only", [1.0, 0.0, 0.0, 0.0])
    router = KnowledgeBaseRouter()
    router.add_kb(kb)
    routed = await router.route("query")
    assert routed == ["only"]
    kb.embed_texts.assert_not_called()


async def test_route_ranks_kbs_by_cosine_similarity():
    """The KB whose routing vector is closest to the query vector ranks first."""
    # query_vec = [1.0, 0.0, 0.0, 0.0] for all KBs (default).
    kb_alpha = _make_fake_kb("alpha", [1.0, 0.0, 0.0, 0.0])  # dot = 1.0
    kb_beta = _make_fake_kb("beta", [0.5, 0.5, 0.0, 0.0])  # dot = 0.5
    kb_gamma = _make_fake_kb("gamma", [0.0, 1.0, 0.0, 0.0])  # dot = 0.0
    router = KnowledgeBaseRouter(max_kbs=3)
    for kb in (kb_alpha, kb_beta, kb_gamma):
        router.add_kb(kb)
    routed = await router.route("query")
    assert routed[0] == "alpha"
    assert routed[1] == "beta"
    assert routed[2] == "gamma"


async def test_route_truncates_to_max_kbs():
    """Only the top ``max_kbs`` KBs are returned."""
    kbs = [
        _make_fake_kb(f"kb{i}", [1.0 - i * 0.1, 0.0, 0.0, 0.0])
        for i in range(5)
    ]
    router = KnowledgeBaseRouter(max_kbs=2)
    for kb in kbs:
        router.add_kb(kb)
    routed = await router.route("query")
    assert len(routed) == 2
    assert routed[0] == "kb0"


async def test_route_empty_kbs_returns_empty():
    router = KnowledgeBaseRouter()
    assert await router.route("query") == []


# ---------- retrieve ---------------------------------------------------------


async def test_retrieve_merges_results_from_routed_kbs():
    kb_alpha = _make_fake_kb(
        "alpha",
        [1.0, 0.0, 0.0, 0.0],
        retrieve_results=[
            {"id": "a1", "document": "alpha doc", "score": 0.9, "metadata": {}},
        ],
    )
    kb_beta = _make_fake_kb(
        "beta",
        [0.0, 1.0, 0.0, 0.0],
        retrieve_results=[
            {"id": "b1", "document": "beta doc", "score": 0.5, "metadata": {}},
        ],
    )
    router = KnowledgeBaseRouter(max_kbs=2, top_k_per_kb=3)
    router.add_kb(kb_alpha)
    router.add_kb(kb_beta)
    results = await router.retrieve("query", top_k=5)
    assert len(results) == 2
    # Merged results are sorted by score descending.
    assert results[0]["id"] == "a1"
    assert results[1]["id"] == "b1"
    # Each result's metadata includes the source KB name.
    assert results[0]["metadata"]["kb_name"] == "alpha"
    assert results[1]["metadata"]["kb_name"] == "beta"


async def test_retrieve_empty_kbs_returns_empty():
    router = KnowledgeBaseRouter()
    assert await router.retrieve("query", top_k=5) == []


async def test_retrieve_with_precomputed_routed_skips_routing():
    """Passing ``routed`` bypasses route() and queries only the named KBs."""
    kb_alpha = _make_fake_kb(
        "alpha",
        [1.0, 0.0, 0.0, 0.0],
        retrieve_results=[
            {"id": "a1", "document": "doc", "score": 0.8, "metadata": {}},
        ],
    )
    kb_beta = _make_fake_kb(
        "beta",
        [0.0, 1.0, 0.0, 0.0],
        retrieve_results=[
            {"id": "b1", "document": "doc", "score": 0.9, "metadata": {}},
        ],
    )
    router = KnowledgeBaseRouter()
    router.add_kb(kb_alpha)
    router.add_kb(kb_beta)
    # Only query alpha even though beta has a higher-scored doc.
    results = await router.retrieve("query", top_k=5, routed=["alpha"])
    assert len(results) == 1
    assert results[0]["id"] == "a1"
    kb_beta.retrieve.assert_not_called()


async def test_retrieve_routed_returns_scores_and_results():
    kb_alpha = _make_fake_kb(
        "alpha",
        [1.0, 0.0, 0.0, 0.0],
        retrieve_results=[
            {"id": "a1", "document": "doc", "score": 0.9, "metadata": {}},
        ],
    )
    kb_beta = _make_fake_kb(
        "beta",
        [0.0, 1.0, 0.0, 0.0],
        retrieve_results=[],
    )
    router = KnowledgeBaseRouter(max_kbs=2)
    router.add_kb(kb_alpha)
    router.add_kb(kb_beta)
    out = await router.retrieve_routed("query")
    assert out["routed_kbs"][0] == "alpha"
    assert "alpha" in out["routing_scores"]
    assert "beta" in out["routing_scores"]
    assert out["routing_scores"]["alpha"] > out["routing_scores"]["beta"]
    assert len(out["results"]) == 1
    assert out["results"][0]["id"] == "a1"


# ---------- constructor / config --------------------------------------------


def test_constructor_clamps_max_kbs_to_at_least_one():
    assert KnowledgeBaseRouter(max_kbs=0).max_kbs == 1


def test_constructor_clamps_top_k_per_kb_to_at_least_one():
    assert KnowledgeBaseRouter(top_k_per_kb=0).top_k_per_kb == 1
