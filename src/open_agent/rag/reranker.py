"""Reranking models for RAG retrieval.

A reranker takes the candidate documents returned by the initial retrieval
stage and scores each ``(query, document)`` pair, producing a more accurate
final ranking. This is especially useful when combining vector and keyword
retrieval because the fused results may contain good documents in low positions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rank(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score and reorder ``documents`` for ``query``.

        Each document dict must contain ``id`` and ``document`` keys. The
        returned list is the same dicts with an added ``rerank_score`` key,
        sorted from highest to lowest score.
        """


class NoOpReranker(Reranker):
    """Pass-through reranker that keeps the input order unchanged."""

    def rank(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for d in documents:
            d["rerank_score"] = d.get("score", 0.0)
        return documents


class BGEReranker(Reranker):
    """Cross-encoder reranker using a BGE reranking model.

    Recommended models for Chinese/English mixed text:

    - ``BAAI/bge-reranker-v2-m3`` (default, small, fast)
    - ``BAAI/bge-reranker-base`` (slightly stronger, larger)

    The underlying ``CrossEncoder`` is loaded lazily on the first
    :meth:`rank` call rather than in ``__init__``. Constructing a
    ``BGEReranker`` therefore does not load the (multi-hundred-MB) model and
    does not block the event loop; :meth:`rank` is expected to be invoked via
    :func:`asyncio.to_thread`, so the lazy load also happens off the loop.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> None:
        """Lazily load the CrossEncoder on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for BGEReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(self.model_name, max_length=512)

    def rank(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        self._ensure_model()
        pairs = [(query, str(d.get("document", ""))) for d in documents]
        scores = self._model.predict(pairs)
        scored = [
            {**doc, "rerank_score": float(score)}
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored


def build_reranker(model_name: str | None) -> Reranker:
    """Factory for creating a reranker from a model name.

    Returns :class:`NoOpReranker` when ``model_name`` is empty or ``"none"``.
    """
    if not model_name or model_name.lower() == "none":
        return NoOpReranker()
    return BGEReranker(model_name)
