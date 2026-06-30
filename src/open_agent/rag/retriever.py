"""Retriever: returns relevant chunks for a query.

The current implementation uses keyword-overlap scoring between the query and
each chunk. A future version will use ChromaDB-backed vector similarity search
over embeddings produced by a sentence-transformers model.
"""
from __future__ import annotations

from open_agent.rag.indexer import Chunk, Indexer


class Retriever:
    """Retrieve relevant chunks from an :class:`Indexer` via keyword matching.

    Args:
        indexer: The indexer holding the chunks to search.
        top_k: Default number of chunks to return per query.
    """

    def __init__(self, indexer: Indexer, top_k: int = 5) -> None:
        self.indexer = indexer
        self.top_k = top_k

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {tok for tok in text.lower().split() if tok}

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``.

        Relevance is measured by the number of query tokens that appear in a
        chunk. Chunks with zero overlap are excluded.
        """
        k = top_k or self.top_k
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[Chunk, int]] = []
        for chunk in self.indexer.get_chunks():
            chunk_tokens = self._tokenize(chunk.text)
            overlap = len(query_tokens & chunk_tokens)
            if overlap > 0:
                scored.append((chunk, overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [chunk for chunk, _ in scored[:k]]

    async def aretrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Async variant of :meth:`retrieve`."""
        return self.retrieve(query, top_k)
