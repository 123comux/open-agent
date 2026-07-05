"""Knowledge base manager for indexing documents and serving RAG queries."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from open_agent.rag.document_loaders import SUPPORTED_EXTENSIONS, load_file
from open_agent.rag.kb_router import KnowledgeBase, KnowledgeBaseRouter

logger = logging.getLogger(__name__)


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
        # Serializes the get-or-create critical section so two concurrent
        # index_file/index_directory calls targeting the same new KB name do
        # not both create it (which would orphan the first KB's index).
        self._kb_lock = asyncio.Lock()

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

    async def _get_or_create_kb(
        self, kb_name: str, description: str = ""
    ) -> KnowledgeBase:
        """Return the named KB, creating it if missing.

        Holds ``self._kb_lock`` so two concurrent callers targeting the same
        new KB name do not both invoke :meth:`create_kb` (which would orphan
        the first KB's FAISS index and any documents already added to it).
        """
        async with self._kb_lock:
            kb = self._kbs.get(kb_name)
            if kb is None:
                kb = await self.create_kb(kb_name, description=description)
            return kb

    def list_kbs(self) -> list[str]:
        """Return the names of all registered knowledge bases."""
        return self._router.list_kbs()

    async def _persist_kb(self, kb: KnowledgeBase) -> None:
        """Persist the KB's FAISS index to disk if an index path is set."""
        store = kb._store
        index_path = getattr(store, "_index_path", None)
        if index_path:
            await store.save(index_path)
        else:
            logger.debug(
                "Skipping index persist: no index_path set for KB '%s'", kb.name
            )

    async def index_file(self, file_path: str, kb_name: str) -> int:
        """Index a single document file into a knowledge base; return chunk count.

        The file is loaded via :func:`open_agent.rag.document_loaders.load_file`,
        which auto-detects the format. If the knowledge base does not exist it
        is created with ``kb_name`` as its description.
        """
        kb = await self._get_or_create_kb(kb_name, description=kb_name)
        loaded = await asyncio.to_thread(load_file, file_path)
        before = await kb.count()
        await kb.add_documents([loaded.text], metadatas=[{"source": file_path}])
        await self._persist_kb(kb)
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
        kb = await self._get_or_create_kb(kb_name, description=description)
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(dir_path)
        before = await kb.count()
        all_texts: list[str] = []
        all_metadatas: list[dict[str, str]] = []
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                loaded = await asyncio.to_thread(load_file, str(path))
            except OSError:
                continue
            if not loaded.text.strip():
                continue
            all_texts.append(loaded.text)
            all_metadatas.append({"source": str(path)})
        if all_texts:
            await kb.add_documents(all_texts, metadatas=all_metadatas)
        await self._persist_kb(kb)
        after = await kb.count()
        return after - before

    def list_documents(self, kb_name: str) -> list[dict[str, Any]]:
        """Return the documents indexed in ``kb_name``.

        Raises :class:`KeyError` if the knowledge base does not exist.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            raise KeyError(f"Knowledge base '{kb_name}' not found")
        return kb.list_documents()

    async def delete_document(self, kb_name: str, source: str) -> int:
        """Delete all chunks from ``source`` in ``kb_name``.

        Raises :class:`KeyError` if the knowledge base does not exist.
        """
        kb = self._kbs.get(kb_name)
        if kb is None:
            raise KeyError(f"Knowledge base '{kb_name}' not found")
        return await kb.delete_by_source(source)

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
