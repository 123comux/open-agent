"""Multi-document knowledge base router with semantic query routing."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "numpy is required for the knowledge base router. Install it with: "
        "pip install numpy"
    ) from exc

from open_agent.rag.hybrid_retriever import HybridRetriever
from open_agent.rag.indexer import Indexer
from open_agent.rag.stores.faiss_store import FAISSStore


class KnowledgeBase:
    """A single named knowledge base backed by FAISS and hybrid retrieval.

    Documents are chunked with :class:`Indexer`, stored as L2-normalized
    embeddings in a :class:`FAISSStore`, and retrieved via
    :class:`HybridRetriever` (vector + BM25 keyword fusion).

    Args:
        name: Unique identifier for this knowledge base.
        description: Natural-language description used by the router to match
            incoming queries to this knowledge base.
        documents: Optional initial documents to index at construction time.
        embedding_model: Sentence-transformers model name used for embeddings.
        index_path: Optional path to a persisted FAISS index. If the file
            exists it is loaded; otherwise a new empty index is created and
            will be persisted when :meth:`FAISSStore.save` is called.
    """

    def __init__(
        self,
        name: str,
        description: str,
        documents: list[str] | None = None,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._documents: list[str] = list(documents) if documents else []
        self._store = FAISSStore(
            embedding_model=embedding_model, index_path=index_path
        )
        self._indexer = Indexer(chunk_size=1, chunk_overlap=0)
        self._retriever = HybridRetriever(vector_store=self._store, top_k=5)
        self._doc_seq = 0
        self._routing_embedding: np.ndarray[Any, np.dtype[Any]] | None = None

    async def add_documents(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> None:
        """Chunk and index ``texts`` into this knowledge base."""
        if not texts:
            return
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas must match texts in length")
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            doc_id = f"{self.name}:doc:{self._doc_seq}"
            self._doc_seq += 1
            base_meta: dict[str, Any] = dict(metadatas[i])
            base_meta.setdefault("source_kb", self.name)
            base_meta.setdefault("doc_index", i)
            chunks = self._indexer.index_text(doc_id, text, base_meta)
            for chunk in chunks:
                if not chunk.text.strip():
                    continue
                ids.append(str(uuid.uuid4()))
                docs.append(chunk.text)
                metas.append(chunk.metadata)
        if ids:
            await self._store.add(ids, docs, metas)
        self._documents.extend(texts)
        # Invalidate cached routing embedding since the sample documents changed.
        self._routing_embedding = None

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve up to ``top_k`` relevant chunks from this knowledge base."""
        return await self._retriever.retrieve(query, top_k=top_k)

    async def count(self) -> int:
        """Return the number of indexed chunks in this knowledge base."""
        return await self._store.count()

    async def embed_texts(
        self, texts: list[str]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Embed ``texts`` with this KB's model, returning L2-normalized vectors.

        The returned array has shape ``(len(texts), dim)``; because vectors are
        normalized, inner products equal cosine similarities.
        """
        return await asyncio.to_thread(self._store._embed_sync, texts)

    def routing_text(self) -> str:
        """Return the text used by the router to represent this KB.

        Combines the description with up to three sample documents (truncated)
        so routing reflects both intent and content.
        """
        parts: list[str] = [self.description] if self.description else []
        for sample in self._documents[:3]:
            snippet = sample.strip()
            if snippet:
                parts.append(snippet[:500])
        return "\n".join(parts) if parts else self.name

    async def routing_embedding(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Return a cached embedding of :meth:`routing_text` (shape ``(1, dim)``)."""
        if self._routing_embedding is None:
            self._routing_embedding = await self.embed_texts([self.routing_text()])
        return self._routing_embedding


class KnowledgeBaseRouter:
    """Route queries to the most relevant registered knowledge bases.

    Routing embeds the query and each knowledge base's :meth:`routing_text`
    and ranks knowledge bases by cosine similarity (vectors are L2-normalized,
    so the inner product is the cosine). The top ``max_kbs`` knowledge bases
    are queried per request and their results merged and re-ranked by score.

    Args:
        top_k_per_kb: Chunks to fetch from each selected KB before merging.
        max_kbs: Maximum number of KBs to query per request.
    """

    def __init__(self, top_k_per_kb: int = 3, max_kbs: int = 3) -> None:
        self.top_k_per_kb = max(1, top_k_per_kb)
        self.max_kbs = max(1, max_kbs)
        self._kbs: dict[str, KnowledgeBase] = {}

    def add_kb(self, kb: KnowledgeBase) -> None:
        """Register a knowledge base with the router."""
        self._kbs[kb.name] = kb

    def remove_kb(self, name: str) -> None:
        """Remove a registered knowledge base by name (no-op if absent)."""
        self._kbs.pop(name, None)

    def list_kbs(self) -> list[str]:
        """Return the names of all registered knowledge bases."""
        return list(self._kbs.keys())

    async def _route_with_scores(
        self, query: str
    ) -> tuple[list[str], dict[str, float]]:
        """Return ``(ranked KB names, full score map)`` for ``query``."""
        if not self._kbs:
            return [], {}
        # Skip routing when there is only one KB: query it directly.
        if len(self._kbs) == 1:
            name = next(iter(self._kbs.keys()))
            return [name], {name: 1.0}
        # All KBs share the default embedding model, so any KB can embed the
        # query; the resulting dimension matches every KB's routing embedding.
        any_kb = next(iter(self._kbs.values()))
        query_vec = await any_kb.embed_texts([query])  # shape (1, dim)
        scores: dict[str, float] = {}
        for name, kb in self._kbs.items():
            kb_vec = await kb.routing_embedding()  # shape (1, dim)
            scores[name] = float(np.dot(query_vec[0], kb_vec[0]))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top = [name for name, _ in ranked[: self.max_kbs]]
        return top, scores

    async def route(self, query: str) -> list[str]:
        """Return the names of the most relevant KBs for ``query``, ranked."""
        names, _ = await self._route_with_scores(query)
        return names

    async def _retrieve_from(
        self, query: str, routed: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        """Retrieve from ``routed`` KBs in parallel, merge and re-rank."""
        if not routed:
            return []
        per_kb = min(self.top_k_per_kb, max(top_k, 1))
        batches = await asyncio.gather(
            *(self._kbs[name].retrieve(query, top_k=per_kb) for name in routed)
        )
        merged: list[dict[str, Any]] = []
        for name, results in zip(routed, batches):
            for r in results:
                meta = dict(r.get("metadata", {}))
                meta["kb_name"] = name
                merged.append(
                    {
                        "id": r.get("id"),
                        "document": r.get("document", ""),
                        "metadata": meta,
                        "score": r.get("score", 0.0),
                    }
                )
        merged.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    async def retrieve(
        self, query: str, top_k: int = 5, *, routed: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Route ``query`` to the top KBs and return merged, re-ranked results.

        Each result's ``metadata`` includes a ``kb_name`` field identifying the
        source knowledge base. Pass ``routed`` to inject a pre-computed routing
        decision and avoid re-routing.
        """
        if routed is None:
            routed = await self.route(query)
        return await self._retrieve_from(query, routed, top_k)

    async def retrieve_routed(self, query: str) -> dict[str, Any]:
        """Return both the routing decision and the merged retrieval results.

        The result dict contains:
            - ``routed_kbs``: ranked list of queried KB names.
            - ``results``: merged and re-ranked chunks (each with ``kb_name``).
            - ``routing_scores``: ``{kb_name: cosine_similarity}`` for all KBs.
        """
        routed, scores = await self._route_with_scores(query)
        results = await self._retrieve_from(query, routed, top_k=5)
        return {
            "routed_kbs": routed,
            "results": results,
            "routing_scores": scores,
        }
