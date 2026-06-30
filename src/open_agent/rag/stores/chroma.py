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
        self._collection = self._client.get_or_create_collection(name=collection_name)

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents to the collection."""
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    async def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return the ``n_results`` most similar documents for ``query_text``.

        Each result is a dict with keys ``id``, ``document``, ``score`` and
        ``metadata``. The ``score`` is the raw ChromaDB distance, so lower
        values indicate higher similarity.
        """
        if n_results <= 0:
            return []
        raw = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query_text],
            n_results=n_results,
        )
        ids_batch = raw.get("ids", [[]])[0]
        docs_batch = raw.get("documents", [[]])[0]
        dists_batch = raw.get("distances", [[]])[0]
        meta_batch = raw.get("metadatas", [[]])[0]
        results: list[dict] = []
        for i, doc_id in enumerate(ids_batch):
            results.append(
                {
                    "id": doc_id,
                    "document": docs_batch[i] if i < len(docs_batch) else "",
                    "score": float(dists_batch[i]) if i < len(dists_batch) else 0.0,
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
