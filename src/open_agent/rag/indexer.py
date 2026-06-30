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
        chunk_size: Number of paragraphs per chunk.
        chunk_overlap: Number of paragraphs to overlap between consecutive
            chunks (0 means no overlap).
    """

    def __init__(self, chunk_size: int = 1, chunk_overlap: int = 0) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self._chunks: list[Chunk] = []

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]

    def index(self, documents: list[Document]) -> list[Chunk]:
        """Chunk and store the given documents; returns the new chunks."""
        new_chunks: list[Chunk] = []
        for doc in documents:
            paragraphs = self._split_paragraphs(doc.text)
            if not paragraphs:
                paragraphs = [doc.text.strip()] if doc.text.strip() else []
            step = max(1, self.chunk_size - self.chunk_overlap)
            i = 0
            while i < len(paragraphs):
                window = paragraphs[i : i + self.chunk_size]
                if window:
                    new_chunks.append(
                        Chunk(
                            document_id=doc.id,
                            text="\n\n".join(window),
                            metadata={**doc.metadata, "paragraph_start": i},
                        )
                    )
                i += step
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
