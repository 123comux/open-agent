"""Knowledge base tool: enables the agent to query indexed documents via RAG.

Wraps :class:`~open_agent.rag.kb_manager.KBManager` so the agent can search
indexed documents (route -> retrieve -> context) as part of its normal tool
loop. The RAG stack (numpy/faiss/sentence-transformers) is optional at import
time: the import of ``KBManager`` is guarded so this module loads even when
those dependencies are absent, and the tool reports a friendly error at call
time when no knowledge base is wired up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from open_agent.tools.base import Tool

if TYPE_CHECKING:
    from open_agent.rag.kb_manager import KBManager

try:  # The RAG stack is optional at tool import time.
    from open_agent.rag.kb_manager import KBManager  # type: ignore[no-redef]

    _HAS_KB_MANAGER = True
except ImportError:  # pragma: no cover - depends on optional deps being present
    _HAS_KB_MANAGER = False


class KnowledgeBaseTool(Tool):
    """Query indexed documents through the RAG knowledge base.

    Delegates to a :class:`~open_agent.rag.kb_manager.KBManager` instance which
    routes the query to the most relevant knowledge base(s), retrieves and
    re-ranks chunks, and returns them formatted as text passages.
    """

    name = "knowledge_base"
    description = (
        "Search the indexed knowledge base for relevant information. Use this "
        "for questions about documents, policies, manuals, or any static "
        "knowledge that has been indexed. Returns relevant text passages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        kb_manager: KBManager | None = None,
        top_k: int = 5,
    ) -> None:
        self.kb_manager = kb_manager
        self.top_k = top_k

    async def execute(self, **kwargs: object) -> str:
        query = str(kwargs.get("query", ""))
        if not query:
            return "Error: no query provided."

        kb_manager = self.kb_manager
        if kb_manager is None or not kb_manager.list_kbs():
            return (
                "No knowledge base available. Index documents first using "
                "'open-agent index' command."
            )

        top_k = int(kwargs.get("top_k", self.top_k))
        result: dict[str, Any] = await kb_manager.query(query, top_k=top_k)
        chunks: list[dict[str, Any]] = result.get("chunks", [])
        if not chunks:
            return "No relevant documents found in the knowledge base."

        blocks: list[str] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            kb_name = metadata.get("kb_name", "unknown")
            score = chunk.get("score", 0.0)
            document = chunk.get("document", "")
            blocks.append(
                f"Source: {kb_name}\n"
                f"Relevance: {score}\n"
                f"Content: {document}\n"
                "---"
            )
        return "\n".join(blocks)
