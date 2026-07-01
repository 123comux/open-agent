"""Document indexer: splits text into chunks for retrieval.

The current implementation uses simple paragraph-based chunking with an
optional overlap. A future version will compute embeddings and persist them
into ChromaDB for vector similarity search.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A source document to be indexed."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexed chunk of a document."""

    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class Indexer:
    """Split documents into chunks and store them for retrieval.

    Args:
        chunk_size: Size of each chunk. When ``split_unit`` is ``"paragraph"``
            this is the number of paragraphs per chunk; when ``split_unit`` is
            ``"char"`` this is the number of characters per chunk.
        chunk_overlap: Overlap between consecutive chunks. Interpreted as
            paragraphs or characters depending on ``split_unit``.
        split_unit: ``"paragraph"`` (legacy default) or ``"char"`` (recommended
            for RAG, especially Chinese text).
    """

    def __init__(
        self,
        chunk_size: int = 1,
        chunk_overlap: int = 0,
        split_unit: str = "paragraph",
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.split_unit = split_unit
        self._chunks: list[Chunk] = []

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]

    def _split_text(self, text: str) -> list[str]:
        """Split text into units (paragraphs or characters)."""
        if self.split_unit == "char":
            # Character-level sliding window with overlap handled externally.
            return [text]
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        return paragraphs

    def _make_windows(self, units: list[str]) -> list[str]:
        """Build overlapping windows from the given units."""
        if self.split_unit == "char":
            text = units[0] if units else ""
            if not text:
                return []
            step = max(1, self.chunk_size - self.chunk_overlap)
            windows: list[str] = []
            i = 0
            while i < len(text):
                window = text[i : i + self.chunk_size]
                if window.strip():
                    windows.append(window)
                i += step
                if not window:
                    break
            return windows
        step = max(1, self.chunk_size - self.chunk_overlap)
        windows = []
        i = 0
        while i < len(units):
            window = units[i : i + self.chunk_size]
            if window:
                windows.append("\n\n".join(window))
            i += step
        return windows

    def index(self, documents: list[Document]) -> list[Chunk]:
        """Chunk and store the given documents; returns the new chunks."""
        new_chunks: list[Chunk] = []
        for doc in documents:
            units = self._split_text(doc.text)
            windows = self._make_windows(units)
            for idx, window in enumerate(windows):
                metadata = {**doc.metadata}
                if self.split_unit == "paragraph":
                    metadata["paragraph_start"] = idx * max(
                        1, self.chunk_size - self.chunk_overlap
                    )
                else:
                    metadata["char_start"] = idx * max(
                        1, self.chunk_size - self.chunk_overlap
                    )
                new_chunks.append(
                    Chunk(
                        document_id=doc.id,
                        text=window,
                        metadata=metadata,
                    )
                )
        self._chunks.extend(new_chunks)
        return new_chunks

    def index_text(
        self, doc_id: str, text: str, metadata: dict | None = None
    ) -> list[Chunk]:
        """Convenience helper to index a single text document."""
        return self.index([Document(id=doc_id, text=text, metadata=metadata or {})])

    def get_chunks(self) -> list[Chunk]:
        """Return all stored chunks."""
        return list(self._chunks)

    def clear(self) -> None:
        """Remove all stored chunks."""
        self._chunks.clear()
