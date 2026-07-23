"""ChromaDB-backed vector store for RAG persistence.

Wraps a ChromaDB collection to provide async-friendly add/query/delete/count
operations. When ``persist_path`` is ``None`` the store runs entirely in
memory; otherwise documents are persisted to disk via
:class:`chromadb.PersistentClient`.

The synchronous ChromaDB client calls are offloaded to a worker thread with
:func:`asyncio.to_thread` so the event loop is never blocked.
"""
from __future__ import annotations

import asyncio
from typing import Any

try:
    import chromadb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ChromaDB is required for ChromaStore. Install it with: "
        "pip install 'open-agent[rag]'"
    ) from exc


class ChromaStore:
    """Async wrapper around a ChromaDB collection.

    Args:
        collection_name: Name of the ChromaDB collection to use.
        persist_path: Directory for persistent storage. ``None`` runs the
            store in-memory (data is lost when the process exits).
    """

    def __init__(
        self,
        collection_name: str = "open_agent",
        persist_path: str | None = None,
    ) -> None:
        if persist_path is None:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the collection."""
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    async def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Return the ``n_results`` most similar documents for ``query_text``.

        Each result is a dict with keys ``id``, ``document``, ``score`` and
        ``metadata``. The ``score`` is converted from the raw ChromaDB distance
        via ``1.0 - distance`` so that higher values indicate higher similarity,
        matching FAISSStore's cosine-similarity semantics (higher = better).
        """
        if n_results <= 0:
            return []
        raw = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query_text],
            n_results=n_results,
        )
        ids_raw = raw.get("ids") or [[]]
        docs_raw = raw.get("documents") or [[]]
        dists_raw = raw.get("distances") or [[]]
        meta_raw = raw.get("metadatas") or [[]]
        ids_batch = ids_raw[0] if ids_raw else []
        docs_batch = docs_raw[0] if docs_raw else []
        dists_batch = dists_raw[0] if dists_raw else []
        meta_batch = meta_raw[0] if meta_raw else []
        results: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids_batch):
            results.append(
                {
                    "id": doc_id,
                    "document": docs_batch[i] if i < len(docs_batch) else "",
                    # ChromaDB returns distances (lower = more similar); convert
                    # to a similarity-like score so higher = better.
                    "score": (1.0 - float(dists_batch[i])) if i < len(dists_batch) else 0.0,
                    "metadata": meta_batch[i] if i < len(meta_batch) else {},
                }
            )
        return results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by id."""
        await asyncio.to_thread(self._collection.delete, ids=ids)

    async def count(self) -> int:
        """Return the number of documents in the collection."""
        return await asyncio.to_thread(self._collection.count)
