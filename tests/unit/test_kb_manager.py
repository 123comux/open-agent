"""Tests for the KBManager (knowledge base lifecycle and indexing)."""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock

# Stub faiss so importing kb_manager (which imports kb_router -> faiss_store)
# works even when faiss-cpu is not installed.
if "faiss" not in sys.modules:
    _faiss_stub = types.ModuleType("faiss")
    _faiss_stub.IndexFlatIP = MagicMock
    _faiss_stub.normalize_L2 = MagicMock()
    _faiss_stub.write_index = MagicMock()
    _faiss_stub.read_index = MagicMock()
    sys.modules["faiss"] = _faiss_stub

from open_agent.rag.kb_manager import KBManager  # noqa: E402

# ---------- get-or-create race ----------------------------------------------


async def test_kb_manager_concurrent_create_same_kb() -> None:
    """Two concurrent _get_or_create_kb calls for the same new KB name
    must create exactly one KB and return the same instance.

    Regression for the race where two concurrent ``index_file`` calls
    targeting a new kb_name both saw ``None`` and both invoked
    ``create_kb``, with the second overwriting the first and orphaning its
    FAISS index and already-indexed documents. With the fix, the
    get-or-create critical section is serialized by ``self._kb_lock``.
    """
    manager = KBManager()

    create_count = 0

    async def fake_create(name: str, description: str) -> Any:
        nonlocal create_count
        create_count += 1
        # Sleep to force the race window: without the lock the second
        # caller would also enter create_kb here.
        await asyncio.sleep(0.01)
        kb = MagicMock()
        kb.name = name
        manager._kbs[name] = kb
        manager._router.add_kb(kb)
        return kb

    manager.create_kb = fake_create  # type: ignore[assignment]

    kb1, kb2 = await asyncio.gather(
        manager._get_or_create_kb("x", "desc"),
        manager._get_or_create_kb("x", "desc"),
    )
    assert kb1 is kb2
    assert create_count == 1
    assert manager.list_kbs() == ["x"]


async def test_kb_manager_get_or_create_returns_existing() -> None:
    """When the KB already exists, _get_or_create_kb returns it without
    creating a new one."""
    manager = KBManager()

    create_count = 0

    async def fake_create(name: str, description: str) -> Any:
        nonlocal create_count
        create_count += 1
        kb = MagicMock()
        kb.name = name
        manager._kbs[name] = kb
        manager._router.add_kb(kb)
        return kb

    manager.create_kb = fake_create  # type: ignore[assignment]

    first = await manager._get_or_create_kb("x", "desc")
    second = await manager._get_or_create_kb("x", "other desc")
    assert first is second
    assert create_count == 1
