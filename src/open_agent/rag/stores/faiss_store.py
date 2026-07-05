"""FAISS vector store for efficient similarity search.

Wraps a FAISS ``IndexFlatIP`` over L2-normalized sentence embeddings so that
inner product equals cosine similarity. Document texts and metadata are kept in
parallel Python lists keyed by their position in the index. The synchronous
sentence-transformers / FAISS calls are offloaded to a worker thread with
:func:`asyncio.to_thread` so the event loop is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

try:
    import faiss  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required for FAISSStore. Install it with: "
        "pip install faiss-cpu"
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "numpy is required for FAISSStore. Install it with: pip install numpy"
    ) from exc

from open_agent.rag.embedding_cache import get_embedding_model


class FAISSStore:
    """Async FAISS-backed vector store using normalized embeddings.

    Vectors are L2-normalized and stored in a ``IndexFlatIP`` (inner product)
    index, so the returned scores are cosine similarities in the range
    ``[-1, 1]`` where higher means more similar.

    Args:
        embedding_model: Name of the sentence-transformers model used to embed
            documents and queries.
        index_path: Optional path to a persisted FAISS index file. If the file
            exists at construction time the index and its sidecar metadata file
            are loaded; otherwise an empty index is created.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        index_path: str | None = None,
    ) -> None:
        self._model = get_embedding_model(embedding_model)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._index: faiss.Index | None = None
        self._index_path: str | None = index_path
        self._lock = asyncio.Lock()
        if index_path is not None and os.path.exists(index_path):
            self._load_sync(index_path)
        else:
            self._index = faiss.IndexFlatIP(self._dim)

    @staticmethod
    def _meta_path(path: str) -> str:
        return path + ".meta.json"

    def _normalize(
        self, vectors: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Return an L2-normalized float32 copy of ``vectors``."""
        copy = np.asarray(vectors, dtype=np.float32)
        if copy.ndim == 1:
            copy = copy.reshape(1, -1)
        faiss.normalize_L2(copy)
        return copy

    def _embed_sync(
        self, texts: list[str]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return self._normalize(np.asarray(embeddings, dtype=np.float32))

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Embed and add documents to the index."""
        if not ids:
            return
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        if metadatas is None:
            metadatas = [{} for _ in ids]
        elif len(metadatas) != len(ids):
            raise ValueError("metadatas must match ids in length")
        async with self._lock:
            vectors = await asyncio.to_thread(self._embed_sync, documents)
            if self._index is None:
                self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)
            self._ids.extend(ids)
            self._documents.extend(documents)
            self._metadatas.extend(metadatas)

    async def query(
        self, query_text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Return the ``n_results`` most similar documents for ``query_text``.

        Each result is a dict with keys ``id``, ``document``, ``score`` and
        ``metadata``. ``score`` is the cosine similarity (higher is more
        similar) because vectors are L2-normalized and the index uses inner
        product.

        The query vector is computed outside the lock so concurrent
        ``add``/``delete`` calls can proceed while the embedding runs; the
        search itself and the parallel ``_ids``/``_documents``/``_metadatas``
        reads are performed under ``self._lock`` to avoid racing with a
        concurrent ``delete`` that rebuilds the index and the parallel lists.
        """
        if n_results <= 0:
            return []
        vector = await asyncio.to_thread(self._embed_sync, [query_text])
        async with self._lock:
            if self._index is None or not self._ids:
                return []
            k = min(n_results, len(self._ids))
            if k <= 0:
                return []
            index = self._index
            distances, indices = await asyncio.to_thread(index.search, vector, k)
            results: list[dict[str, Any]] = []
            ids = self._ids
            documents = self._documents
            metadatas = self._metadatas
            for rank, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                i = int(idx)
                if i < 0 or i >= len(ids):
                    continue
                results.append(
                    {
                        "id": ids[i],
                        "document": documents[i],
                        "score": float(distances[0][rank]),
                        "metadata": metadatas[i],
                    }
                )
            return results

    def _save_sync(self, path: str) -> None:
        if self._index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self._index, path)
        meta = {
            "ids": self._ids,
            "documents": self._documents,
            "metadatas": self._metadatas,
            "dim": self._dim,
        }
        with open(self._meta_path(path), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    async def save(self, path: str) -> None:
        """Persist the FAISS index and its metadata to disk.

        The FAISS index is written to ``path`` and the parallel id/document/
        metadata lists are written to a sidecar file at ``path + ".meta.json"``.
        """
        await asyncio.to_thread(self._save_sync, path)

    def _load_sync(self, path: str) -> None:
        index = faiss.read_index(path)
        if index.d != self._dim:
            raise RuntimeError(
                f"Stored FAISS index dimension ({index.d}) does not match the "
                f"current embedding model dimension ({self._dim}). The knowledge "
                f"base was created with a different embedding model. Re-index "
                f"the knowledge base or restore the original model."
            )
        self._index = index
        self._dim = index.d
        meta_path = self._meta_path(path)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta: dict[str, Any] = json.load(f)
            self._ids = list(meta.get("ids", []))
            self._documents = list(meta.get("documents", []))
            self._metadatas = list(meta.get("metadatas", []))
            if meta.get("dim"):
                self._dim = int(meta["dim"])

    async def load(self, path: str) -> None:
        """Load a FAISS index and its metadata from disk."""
        await asyncio.to_thread(self._load_sync, path)

    async def delete(self, ids: list[str]) -> None:
        """Remove documents by id, rebuilding the index without them."""
        if not ids or self._index is None:
            return
        async with self._lock:
            to_delete = set(ids)
            keep = [i for i, doc_id in enumerate(self._ids) if doc_id not in to_delete]
            new_ids = [self._ids[i] for i in keep]
            new_documents = [self._documents[i] for i in keep]
            new_metadatas = [self._metadatas[i] for i in keep]
            new_index = faiss.IndexFlatIP(self._dim)
            if new_documents:
                vectors = await asyncio.to_thread(self._embed_sync, new_documents)
                new_index.add(vectors)
            self._index = new_index
            self._ids = new_ids
            self._documents = new_documents
            self._metadatas = new_metadatas

    async def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._ids)
