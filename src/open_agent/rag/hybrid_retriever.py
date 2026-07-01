"""Hybrid retriever combining vector similarity and keyword matching with RRF fusion.

Runs a vector similarity search (via the supplied :class:`FAISSStore` or
:class:`ChromaStore`) and a BM25 keyword search over the same corpus in
parallel, then fuses the two ranked lists with weighted Reciprocal Rank
Fusion (RRF). RRF is rank-based, so it tolerates the different score scales
produced by each backend (e.g. cosine similarity vs. ChromaDB distance).
"""
from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from typing import Any

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False

from open_agent.rag.reranker import Reranker, build_reranker


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, used for both BM25 indexing and queries."""
    return [tok for tok in re.findall(r"\w+", text.lower()) if tok]


class _BM25:
    """Minimal in-process BM25 implementation.

    Used as a fallback when the optional ``rank_bm25`` package is not
    installed. Mirrors the ``BM25Okapi`` interface used here (a ``corpus``
    of token lists plus :meth:`get_scores`).
    """

    def __init__(
        self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0.0
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        n = len(corpus)
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        avgdl = self.avgdl or 1.0
        scores = [0.0] * len(self.corpus)
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            denom_base = self.k1 * (1 - self.b + self.b * (self.doc_len[i] / avgdl))
            s = 0.0
            for term in query:
                if term not in self.idf:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                s += self.idf[term] * (f * (self.k1 + 1)) / (f + denom_base)
            scores[i] = s
        return scores


class HybridRetriever:
    """Fuse vector and keyword search via weighted Reciprocal Rank Fusion.

    Args:
        vector_store: A :class:`FAISSStore` or :class:`ChromaStore` providing
            ``query`` and ``count``. The keyword index is built from the
            store's document corpus.
        keyword_weight: Weight applied to the keyword (BM25) RRF contribution.
        vector_weight: Weight applied to the vector RRF contribution.
        top_k: Default number of fused results to return per query.
    """

    def __init__(
        self,
        vector_store: Any,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_k: int = 20,
    ) -> None:
        if keyword_weight < 0 or vector_weight < 0:
            raise ValueError("weights must be non-negative")
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.top_k = top_k
        self.reranker = reranker or build_reranker(None)
        self.rerank_k = rerank_k
        # Cached keyword index over the store's corpus; rebuilt when the
        # store's document count changes.
        self._kw_index: Any = None
        self._kw_ids: list[str] = []
        self._kw_docs: list[str] = []
        self._kw_metas: list[dict[str, Any]] = []
        self._kw_count: int = -1

    async def _get_corpus(
        self,
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Fetch the full (ids, documents, metadatas) corpus from the store."""
        store = self.vector_store
        # FAISSStore keeps parallel lists directly.
        if hasattr(store, "_documents") and hasattr(store, "_ids"):
            return (
                list(store._ids),
                list(store._documents),
                list(getattr(store, "_metadatas", [])),
            )
        # ChromaStore wraps a collection; use .get() to fetch everything.
        collection = getattr(store, "_collection", None)
        if collection is not None:
            raw = await asyncio.to_thread(collection.get)
            ids = list(raw.get("ids", []))
            docs = list(raw.get("documents", []))
            metas = [
                m if m is not None else {}
                for m in raw.get("metadatas", [])
            ]
            return ids, docs, metas
        raise TypeError(
            f"Unsupported vector store type {type(store).__name__!r}: "
            "cannot fetch document corpus"
        )

    async def _ensure_keyword_index(self) -> None:
        """Build/refresh the BM25 index when the store's document count changes."""
        count = await self.vector_store.count()
        if self._kw_index is not None and count == self._kw_count:
            return
        ids, docs, metas = await self._get_corpus()
        tokenized = [_tokenize(d) for d in docs]
        if tokenized:
            self._kw_index = (
                BM25Okapi(tokenized) if _HAS_RANK_BM25 else _BM25(tokenized)
            )
        else:
            self._kw_index = None
        self._kw_ids = ids
        self._kw_docs = docs
        self._kw_metas = metas
        self._kw_count = count

    async def _keyword_search(
        self, query: str, n: int
    ) -> list[dict[str, Any]]:
        await self._ensure_keyword_index()
        if self._kw_index is None or not self._kw_docs:
            return []
        scores = self._kw_index.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self._kw_docs)), key=lambda i: scores[i], reverse=True
        )
        results: list[dict[str, Any]] = []
        for i in ranked[: max(n, 0)]:
            s = float(scores[i])
            if s <= 0:
                continue
            results.append(
                {
                    "id": self._kw_ids[i],
                    "document": self._kw_docs[i],
                    "score": s,
                    "metadata": self._kw_metas[i]
                    if i < len(self._kw_metas)
                    else {},
                }
            )
        return results

    async def _vector_search(
        self, query: str, n: int
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = await self.vector_store.query(
            query, n_results=n
        )
        return results

    async def _retrieve_both(
        self, query: str, top_k: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch a pooled set of vector and keyword results in parallel."""
        fetch_n = max(top_k * 4, self.rerank_k, 10)
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(query, fetch_n),
            self._keyword_search(query, fetch_n),
        )
        return vector_results, keyword_results

    def _fuse(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse the two ranked lists with weighted RRF (k=60)."""
        k = 60
        v_rank: dict[str, tuple[int, float]] = {}
        for rank, r in enumerate(vector_results):
            v_rank.setdefault(r["id"], (rank, float(r.get("score", 0.0))))
        kw_rank: dict[str, tuple[int, float]] = {}
        for rank, r in enumerate(keyword_results):
            kw_rank.setdefault(r["id"], (rank, float(r.get("score", 0.0))))

        # Preserve best-known document text and metadata for each id.
        docs_map: dict[str, dict[str, Any]] = {}
        for r in vector_results:
            docs_map.setdefault(r["id"], r)
        for r in keyword_results:
            docs_map.setdefault(r["id"], r)

        fused: list[dict[str, Any]] = []
        for doc_id, info in docs_map.items():
            score = 0.0
            v_score = 0.0
            kw_score = 0.0
            v_r: int | None = None
            kw_r: int | None = None
            if doc_id in v_rank:
                v_r, v_score = v_rank[doc_id]
                score += self.vector_weight * (1.0 / (k + v_r))
            if doc_id in kw_rank:
                kw_r, kw_score = kw_rank[doc_id]
                score += self.keyword_weight * (1.0 / (k + kw_r))
            fused.append(
                {
                    "id": doc_id,
                    "document": info.get("document", ""),
                    "metadata": info.get("metadata", {}),
                    "score": score,
                    "vector_score": v_score,
                    "keyword_score": kw_score,
                    "vector_rank": v_r,
                    "keyword_rank": kw_r,
                }
            )
        fused.sort(key=lambda d: d["score"], reverse=True)
        return fused

    async def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` fused results for ``query``.

        Each result is a dict with keys ``id``, ``document``, ``metadata`` and
        ``score`` (the combined RRF score, higher is better).
        """
        k = top_k or self.top_k
        if k <= 0:
            return []
        vector_results, keyword_results = await self._retrieve_both(query, k)
        fused = self._fuse(vector_results, keyword_results)

        # Optional cross-encoder reranking over the top fused candidates.
        if fused:
            candidates = fused[: max(self.rerank_k, k)]
            fused = self.reranker.rank(query, candidates)

        return [
            {
                "id": d["id"],
                "document": d["document"],
                "metadata": d["metadata"],
                "score": d.get("rerank_score", d["score"]),
            }
            for d in fused[:k]
        ]

    async def retrieve_with_scores(self, query: str) -> list[dict[str, Any]]:
        """Return fused results including per-method scores and ranks.

        Each result dict contains: ``id``, ``document``, ``metadata``,
        ``score`` (combined RRF), ``vector_score``, ``keyword_score``,
        ``vector_rank`` and ``keyword_rank``. ``vector_rank``/``keyword_rank``
        are ``None`` when the document was not returned by that method.
        """
        vector_results, keyword_results = await self._retrieve_both(
            query, self.top_k
        )
        return self._fuse(vector_results, keyword_results)[: self.top_k]
