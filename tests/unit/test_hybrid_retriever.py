"""Tests for the HybridRetriever (RRF fusion of vector + BM25 search)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from open_agent.rag.hybrid_retriever import HybridRetriever


class MockVectorStore:
    """Minimal async vector store stub used by HybridRetriever tests.

    Exposes the ``_ids``/``_documents``/``_metadatas`` parallel lists that
    ``HybridRetriever._get_corpus`` reads, plus async ``count`` and ``query``
    methods. ``query_results`` is independent of the corpus so tests can
    simulate a vector index that returns hits even when the BM25 corpus is
    empty (and vice versa).
    """

    def __init__(
        self,
        ids: list[str] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        query_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ids = list(ids or [])
        self._documents = list(documents or [])
        self._metadatas = list(metadatas or [{} for _ in self._ids])
        self._query_results = list(query_results or [])

    async def count(self) -> int:
        return len(self._ids)

    async def query(
        self, query_text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        return list(self._query_results[:n_results])


def _vec_result(doc_id: str, score: float, document: str = "") -> dict[str, Any]:
    return {
        "id": doc_id,
        "document": document or doc_id,
        "score": score,
        "metadata": {"src": doc_id},
    }


# ---------- _fuse (RRF) tests -------------------------------------------------


def test_fuse_rrf_combines_vector_and_keyword_ranks():
    """RRF with k=60: a doc ranked high in both lists should win overall."""
    retriever = HybridRetriever(
        MockVectorStore(),
        keyword_weight=0.5,
        vector_weight=0.5,
        reranker=MagicMock(),
    )
    vector_results = [
        _vec_result("A", 0.9),
        _vec_result("B", 0.8),
        _vec_result("C", 0.7),
    ]
    keyword_results = [
        _vec_result("B", 5.0),
        _vec_result("C", 4.0),
        _vec_result("D", 3.0),
    ]
    fused = retriever._fuse(vector_results, keyword_results)
    ids = [d["id"] for d in fused]
    # B is rank 1 in vector, rank 0 in keyword -> highest combined RRF.
    assert ids[0] == "B"
    # D appears only in keyword results (rank 2) -> lowest combined score.
    assert ids[-1] == "D"
    # All four unique ids are present.
    assert set(ids) == {"A", "B", "C", "D"}


def test_fuse_rrf_score_uses_k_equals_60():
    """Verify the exact RRF formula with k=60 for a vector-only document."""
    retriever = HybridRetriever(
        MockVectorStore(),
        keyword_weight=0.5,
        vector_weight=0.5,
        reranker=MagicMock(),
    )
    vector_results = [_vec_result("A", 0.9)]  # rank 0
    fused = retriever._fuse(vector_results, [])
    # score = 0.5 * 1/(60 + 0) = 0.5/60
    assert pytest.approx(fused[0]["score"], rel=1e-9) == 0.5 / 60.0


def test_fuse_weighted_vector_contributes_more():
    """With vector_weight=0.7, a vector-only doc beats a keyword-only doc
    even when both are rank 0 (because 0.7/60 > 0.3/60)."""
    retriever = HybridRetriever(
        MockVectorStore(),
        keyword_weight=0.3,
        vector_weight=0.7,
        reranker=MagicMock(),
    )
    vector_results = [_vec_result("V", 0.9)]  # rank 0, vector only
    keyword_results = [_vec_result("K", 5.0)]  # rank 0, keyword only
    fused = retriever._fuse(vector_results, keyword_results)
    assert fused[0]["id"] == "V"
    assert fused[1]["id"] == "K"


def test_fuse_weighting_flips_order_when_swapped():
    """Swapping the weights flips the order of vector-only vs keyword-only docs."""
    vector_results = [_vec_result("V", 0.9)]
    keyword_results = [_vec_result("K", 5.0)]

    r1 = HybridRetriever(
        MockVectorStore(),
        keyword_weight=0.3,
        vector_weight=0.7,
        reranker=MagicMock(),
    )
    fused1 = r1._fuse(vector_results, keyword_results)
    assert fused1[0]["id"] == "V"

    r2 = HybridRetriever(
        MockVectorStore(),
        keyword_weight=0.7,
        vector_weight=0.3,
        reranker=MagicMock(),
    )
    fused2 = r2._fuse(vector_results, keyword_results)
    assert fused2[0]["id"] == "K"


def test_fuse_empty_inputs_returns_empty():
    retriever = HybridRetriever(MockVectorStore(), reranker=MagicMock())
    assert retriever._fuse([], []) == []


def test_fuse_preserves_document_and_metadata():
    retriever = HybridRetriever(MockVectorStore(), reranker=MagicMock())
    vector_results = [
        {
            "id": "A",
            "document": "hello world",
            "score": 0.9,
            "metadata": {"src": "doc1"},
        }
    ]
    fused = retriever._fuse(vector_results, [])
    assert fused[0]["document"] == "hello world"
    assert fused[0]["metadata"] == {"src": "doc1"}


# ---------- retrieve (end-to-end with mocks) --------------------------------


async def test_retrieve_returns_empty_when_top_k_le_zero():
    retriever = HybridRetriever(MockVectorStore(), top_k=5, reranker=MagicMock())
    assert await retriever.retrieve("query", top_k=0) == []


async def test_retrieve_bm25_fallback_vector_only():
    """When the corpus is empty (no BM25 index), retrieval still returns
    vector results from the store."""
    store = MockVectorStore(
        ids=[],  # empty corpus -> _kw_index is None
        documents=[],
        query_results=[_vec_result("A", 0.9, "doc A")],
    )
    # NoOpReranker (built by default) keeps order.
    retriever = HybridRetriever(store, top_k=5)
    results = await retriever.retrieve("query")
    assert len(results) == 1
    assert results[0]["id"] == "A"
    assert results[0]["document"] == "doc A"


async def test_retrieve_uses_reranker_output():
    """The reranker's output replaces the fused list; the final score is
    the rerank_score."""
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        {
            "id": "Z",
            "document": "best doc",
            "metadata": {"src": "z"},
            "score": 0.5,
            "rerank_score": 0.99,
        }
    ]
    store = MockVectorStore(
        ids=["A"],
        documents=["doc A"],
        query_results=[_vec_result("A", 0.5, "doc A")],
    )
    retriever = HybridRetriever(store, top_k=5, reranker=mock_reranker)
    results = await retriever.retrieve("query")
    mock_reranker.rank.assert_called_once()
    # The query and candidates are passed positionally.
    call_args = mock_reranker.rank.call_args
    assert call_args[0][0] == "query"
    assert isinstance(call_args[0][1], list)
    # Result is the reranker's output, with score = rerank_score.
    assert len(results) == 1
    assert results[0]["id"] == "Z"
    assert results[0]["score"] == 0.99


async def test_retrieve_empty_results_skips_reranker():
    """No vector results and no keyword matches -> empty list, reranker
    not called."""
    mock_reranker = MagicMock()
    store = MockVectorStore(ids=[], documents=[], query_results=[])
    retriever = HybridRetriever(store, top_k=5, reranker=mock_reranker)
    results = await retriever.retrieve("nothing matches")
    assert results == []
    mock_reranker.rank.assert_not_called()


async def test_retrieve_combines_vector_and_keyword_results():
    """With a real corpus, both vector and keyword search contribute."""
    store = MockVectorStore(
        ids=["d1", "d2", "d3"],
        documents=[
            "Python is a programming language.",
            "The cat sat on the mat.",
            "Python is great for data science.",
        ],
        query_results=[
            _vec_result("d1", 0.9, "Python is a programming language."),
            _vec_result("d3", 0.7, "Python is great for data science."),
        ],
    )
    retriever = HybridRetriever(store, top_k=5)
    results = await retriever.retrieve("Python programming")
    ids = [r["id"] for r in results]
    # d1 and d3 match both vector and keyword; d2 doesn't match either.
    assert "d1" in ids
    assert "d3" in ids
    assert "d2" not in ids


async def test_retrieve_with_scores_includes_per_method_fields():
    """retrieve_with_scores returns vector_score, keyword_score and ranks."""
    store = MockVectorStore(
        ids=["d1", "d2"],
        documents=["apple fruit", "banana split"],
        query_results=[
            _vec_result("d1", 0.9, "apple fruit"),
            _vec_result("d2", 0.5, "banana split"),
        ],
    )
    retriever = HybridRetriever(store, top_k=5)
    results = await retriever.retrieve_with_scores("apple")
    assert len(results) >= 1
    top = results[0]
    assert "vector_score" in top
    assert "keyword_score" in top
    assert "vector_rank" in top
    assert "keyword_rank" in top
    # d1 is rank 0 in the vector results.
    assert top["vector_rank"] == 0


# ---------- constructor validation ------------------------------------------


def test_constructor_rejects_negative_weights():
    with pytest.raises(ValueError):
        HybridRetriever(MockVectorStore(), keyword_weight=-0.1, vector_weight=0.7)


def test_constructor_defaults():
    retriever = HybridRetriever(MockVectorStore())
    assert retriever.keyword_weight == 0.3
    assert retriever.vector_weight == 0.7
    assert retriever.top_k == 5
    assert retriever.rerank_k == 20


# ---------- failure isolation ------------------------------------------------


async def test_hybrid_retriever_one_leg_fails_returns_other():
    """If the vector search leg raises, keyword results are still returned.

    Regression for ``asyncio.gather`` without ``return_exceptions=True``:
    a failing leg used to cancel the whole retrieval. With the fix the
    failure is logged and treated as empty, so the keyword leg's results
    still come back.
    """
    store = MockVectorStore(
        ids=["d1", "d2"],
        documents=["apple fruit", "banana split"],
        query_results=[],  # not used because query() raises
    )

    async def failing_query(
        query_text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        raise RuntimeError("vector store down")

    store.query = failing_query  # type: ignore[assignment]

    retriever = HybridRetriever(store, top_k=5)
    results = await retriever.retrieve("apple")
    # Keyword search still returns d1 (matches "apple"); d2 does not match.
    ids = [r["id"] for r in results]
    assert "d1" in ids
    assert "d2" not in ids
