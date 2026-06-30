"""Long-term memory stub backed by an in-memory list.

This will eventually be backed by a vector store for semantic recall across
sessions. For now it performs simple case-insensitive substring matching over
stored entries, ranked by the number of query-term occurrences.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single long-term memory entry."""

    text: str
    metadata: dict = Field(default_factory=dict)


class LongTermMemory:
    """Persistent memory store (stub: in-memory list + text matching)."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Store a memory entry."""
        self._entries.append(MemoryEntry(text=text, metadata=metadata or {}))

    def search(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Return up to ``k`` entries whose text contains ``query`` (case-insensitive).

        Results are ranked by how many times the query appears in each entry.
        """
        if not query:
            return []
        lowered = query.lower()
        scored = [
            (entry, entry.text.lower().count(lowered))
            for entry in self._entries
            if lowered in entry.text.lower()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in scored[:k]]

    def all_entries(self) -> list[MemoryEntry]:
        """Return all stored entries."""
        return list(self._entries)

    def clear(self) -> None:
        """Remove all stored entries."""
        self._entries.clear()
