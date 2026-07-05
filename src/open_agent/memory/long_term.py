"""Long-term memory backed by a FAISS vector store.

Stores text memories as dense embeddings so the agent can recall relevant
information across sessions. The store is persisted to disk and auto-saved on
add. It is intentionally lightweight: each memory is a short text snippet
(typically a user/assistant exchange or an explicit fact) with optional
metadata.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single long-term memory entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class LongTermMemory:
    """Persistent semantic memory using FAISS + sentence-transformers.

    Args:
        embedding_model: Sentence-transformers model name used to embed
            memories and queries. Should match the RAG embedding model for
            consistent behaviour.
        storage_dir: Directory where the FAISS index and metadata are stored.
            Created automatically if it does not exist.
        index_name: Base filename for the persisted index.
        top_k: Default number of memories to retrieve.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        storage_dir: str = ".open_agent_long_term",
        index_name: str = "long_term",
        top_k: int = 3,
    ) -> None:
        from open_agent.rag.stores.faiss_store import FAISSStore

        self._embedding_model = embedding_model
        self._storage_dir = storage_dir
        self._index_name = index_name
        self._top_k = top_k
        self._store: FAISSStore | None = None

        os.makedirs(storage_dir, exist_ok=True)
        self._index_path = os.path.join(storage_dir, f"{index_name}.faiss")
        try:
            self._store = FAISSStore(
                embedding_model=embedding_model, index_path=self._index_path
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Long-term memory requires faiss, numpy and sentence-transformers. "
                "Install them with: pip install faiss-cpu numpy sentence-transformers"
            ) from exc

    async def add(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> MemoryEntry:
        """Store a memory entry and persist the index."""
        if self._store is None:
            raise RuntimeError("Long-term memory store is not available")
        if not text or not text.strip():
            raise ValueError("Memory text cannot be empty")

        entry = MemoryEntry(text=text.strip(), metadata=metadata or {})
        stored_metadata = {
            "id": entry.id,
            "timestamp": entry.timestamp,
            **entry.metadata,
        }
        await self._store.add(
            ids=[entry.id],
            documents=[entry.text],
            metadatas=[stored_metadata],
        )
        await self._persist()
        return entry

    async def search(self, query: str, k: int | None = None) -> list[MemoryEntry]:
        """Return up to ``k`` memories relevant to ``query``.

        Results are ordered by cosine similarity (highest first).
        """
        if self._store is None:
            return []
        if not query or not query.strip():
            return []

        results = await self._store.query(query_text=query, n_results=k or self._top_k)
        entries: list[MemoryEntry] = []
        for result in results:
            meta = dict(result.get("metadata", {}))
            entry_id = str(result.get("id", meta.pop("id", str(uuid.uuid4()))))
            timestamp = float(meta.pop("timestamp", 0.0))
            entries.append(
                MemoryEntry(
                    id=entry_id,
                    text=str(result.get("document", "")),
                    metadata=meta,
                    timestamp=timestamp,
                )
            )
        return entries

    async def add_exchange(
        self,
        user_input: str,
        assistant_response: str,
        session_id: str = "default",
    ) -> MemoryEntry:
        """Convenience helper to store a user/assistant exchange as one memory."""
        text = f"User: {user_input.strip()}\nAssistant: {assistant_response.strip()}"
        return await self.add(
            text=text,
            metadata={"type": "exchange", "session_id": session_id},
        )

    async def delete(self, entry_id: str) -> bool:
        """Remove a memory by id. Returns True if the id existed."""
        if self._store is None:
            return False
        before = await self._store.count()
        await self._store.delete([entry_id])
        after = await self._store.count()
        if after < before:
            await self._persist()
            return True
        return False

    async def clear(self) -> None:
        """Remove all memories and persist the empty index."""
        if self._store is None:
            return
        ids = list(getattr(self._store, "_ids", []))
        if ids:
            await self._store.delete(ids)
            await self._persist()

    async def count(self) -> int:
        """Return the number of stored memories."""
        if self._store is None:
            return 0
        return await self._store.count()

    async def _persist(self) -> None:
        """Save the FAISS index and metadata to disk."""
        if self._store is None:
            return
        await self._store.save(self._index_path)
