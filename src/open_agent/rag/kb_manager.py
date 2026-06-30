"""Knowledge base manager for indexing documents and serving RAG queries."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from open_agent.rag.kb_router import KnowledgeBase, KnowledgeBaseRouter

# Text file extensions indexed by :meth:`KBManager.index_directory`.
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".text"}


class KBManager:
    """High-level manager tying together knowledge bases, routing, and RAG.

    Creates and registers knowledge bases, indexes files/directories into them,
    and serves end-to-end RAG queries (route -> retrieve -> context).

    Args:
        storage_dir: Optional directory for persisting FAISS indexes. When set,
            each created knowledge base is backed by
            ``<storage_dir>/<name>.faiss`` (loaded if present, created
            otherwise).
    """

    def __init__(self, storage_dir: str | None = None) -> None:
        self.storage_dir = storage_dir
        self._router = KnowledgeBaseRouter()
        self._kbs: dict[str, KnowledgeBase] = {}

    async def create_kb(self, name: str, description: str) -> KnowledgeBase:
        """Create and register a new knowledge base, returning it."""
        index_path: str | None = None
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            index_path = os.path.join(self.storage_dir, f"{name}.faiss")
        kb = KnowledgeBase(
            name=name, description=description, index_path=index_path
        )
        self._kbs[name] = kb
        self._router.add_kb(kb)
        return kb

    def get_kb(self, name: str) -> KnowledgeBase | None:
        """Return a registered knowledge base by name, or ``None``."""
        return self._kbs.get(name)

    def list_kbs(self) -> list[str]:
        """Return the names of all registered knowledge bases."""
        return self._router.list_kbs()

    async def index_file(self, file_path: str, kb_name: str) -> int:
        """Index a single text file into a knowledge base; return chunk count.

        If the knowledge base does not exist it is created with ``kb_name`` as
        its description.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            kb = await self.create_kb(kb_name, description=kb_name)
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        before = await kb.count()
        await kb.add_documents([text], metadatas=[{"source": file_path}])
        after = await kb.count()
        return after - before

    async def index_directory(
        self, dir_path: str, kb_name: str, description: str = ""
    ) -> int:
        """Index all text files in a directory into a knowledge base.

        Text files are those with one of the extensions {``.txt``, ``.md``,
        ``.rst``, ``.text``}. Only the top level of ``dir_path`` is scanned.
        Returns the number of chunks indexed.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            kb = await self.create_kb(kb_name, description=description)
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(dir_path)
        before = await kb.count()
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            await kb.add_documents([text], metadatas=[{"source": str(path)}])
        after = await kb.count()
        return after - before

    async def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Run a full RAG query: route -> retrieve -> build context.

        Returns a dict with:
            - ``routed_kbs``: KB names the query was routed to.
            - ``chunks``: merged, re-ranked chunks (each with ``kb_name``).
            - ``context_text``: chunks joined by blank lines for prompting.
        """
        routed = await self._router.route(question)
        chunks = await self._router.retrieve(
            question, top_k=top_k, routed=routed
        )
        context_text = "\n\n".join(str(c.get("document", "")) for c in chunks)
        return {
            "routed_kbs": routed,
            "chunks": chunks,
            "context_text": context_text,
        }
