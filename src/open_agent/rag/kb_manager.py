"""Knowledge base manager for indexing documents and serving RAG queries."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from open_agent.rag.document_loaders import SUPPORTED_EXTENSIONS, load_file
from open_agent.rag.kb_router import KnowledgeBase, KnowledgeBaseRouter


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

    def __init__(
        self,
        storage_dir: str | None = None,
        *,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        split_unit: str = "char",
        top_k: int = 5,
        reranker_model: str | None = None,
        rerank_k: int = 20,
    ) -> None:
        self.storage_dir = storage_dir
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._split_unit = split_unit
        self._top_k = top_k
        self._reranker_model = reranker_model
        self._rerank_k = rerank_k
        self._router = KnowledgeBaseRouter()
        self._kbs: dict[str, KnowledgeBase] = {}

    async def create_kb(self, name: str, description: str) -> KnowledgeBase:
        """Create and register a new knowledge base, returning it."""
        index_path: str | None = None
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            index_path = os.path.join(self.storage_dir, f"{name}.faiss")
        kb = KnowledgeBase(
            name=name,
            description=description,
            index_path=index_path,
            embedding_model=self._embedding_model,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            split_unit=self._split_unit,
            top_k=self._top_k,
            reranker_model=self._reranker_model,
            rerank_k=self._rerank_k,
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
        """Index a single document file into a knowledge base; return chunk count.

        The file is loaded via :func:`open_agent.rag.document_loaders.load_file`,
        which auto-detects the format. If the knowledge base does not exist it
        is created with ``kb_name`` as its description.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            kb = await self.create_kb(kb_name, description=kb_name)
        loaded = load_file(file_path)
        before = await kb.count()
        await kb.add_documents([loaded.text], metadatas=[{"source": file_path}])
        after = await kb.count()
        return after - before

    async def index_directory(
        self, dir_path: str, kb_name: str, description: str = ""
    ) -> int:
        """Index all supported document files in a directory into a knowledge base.

        Supported files are those whose extension is in
        :data:`open_agent.rag.document_loaders.SUPPORTED_EXTENSIONS`
        (``.txt``, ``.md``, ``.rst``, ``.pdf``, ``.docx``, ``.csv``, ``.json``,
        ``.html``). Only the top level of ``dir_path`` is scanned. Returns the
        number of chunks indexed.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            kb = await self.create_kb(kb_name, description=description)
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(dir_path)
        before = await kb.count()
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                loaded = load_file(str(path))
            except OSError:
                continue
            if not loaded.text.strip():
                continue
            await kb.add_documents([loaded.text], metadatas=[{"source": str(path)}])
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
