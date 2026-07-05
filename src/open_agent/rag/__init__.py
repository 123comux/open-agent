"""RAG package: document indexing and retrieval."""
from __future__ import annotations

from open_agent.rag.indexer import Chunk, Document, Indexer
from open_agent.rag.retriever import Retriever

__all__ = [
    "Chunk",
    "Document",
    "Indexer",
    "Retriever",
]

# Optional imports requiring faiss/sentence-transformers
try:
    from open_agent.rag.stores.faiss_store import FAISSStore  # noqa: F401

    __all__.append("FAISSStore")
except ImportError:
    pass

try:
    from open_agent.rag.hybrid_retriever import HybridRetriever  # noqa: F401

    __all__.append("HybridRetriever")
except ImportError:
    pass

try:
    from open_agent.rag.kb_router import (  # noqa: F401
        KnowledgeBase,
        KnowledgeBaseRouter,
    )

    __all__.extend(["KnowledgeBase", "KnowledgeBaseRouter"])
except ImportError:
    pass

try:
    from open_agent.rag.kb_manager import KBManager  # noqa: F401

    __all__.append("KBManager")
except ImportError:
    pass

try:
    from open_agent.rag.evaluation import RAGEvaluator, RAGTestCase  # noqa: F401

    __all__.extend(["RAGEvaluator", "RAGTestCase"])
except ImportError:
    pass
