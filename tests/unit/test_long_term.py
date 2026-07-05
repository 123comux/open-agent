"""Tests for :class:`LongTermMemory` edge cases."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_agent.memory.long_term import LongTermMemory


@pytest.fixture
def mock_faiss_store():
    """Patch ``FAISSStore`` so ``LongTermMemory`` can be constructed without
    the optional faiss/sentence-transformers dependencies.

    Mirrors the fixture in ``tests/unit/test_memory.py`` so the None-metadata
    regression test is self-contained.
    """
    store = MagicMock()
    store.add = AsyncMock()
    store.query = AsyncMock(return_value=[])
    store.delete = AsyncMock()
    store.save = AsyncMock()
    store.count = AsyncMock(return_value=0)
    store._ids = []

    fake_module = MagicMock()
    fake_module.FAISSStore = MagicMock(return_value=store)

    with patch.dict(
        "sys.modules", {"open_agent.rag.stores.faiss_store": fake_module}
    ):
        yield store


@pytest.mark.asyncio
async def test_long_term_search_with_none_metadata(mock_faiss_store) -> None:
    """A backend result with ``metadata: None`` must not crash search.

    Regression test for M14: ``dict(result.get("metadata", {}))`` raised
    ``TypeError`` when the key was present but ``None``; the fix uses
    ``dict(result.get("metadata") or {})`` which falls back to an empty dict.
    """
    mem = LongTermMemory()
    mock_faiss_store.query.return_value = [
        {
            "id": "id-1",
            "document": "Python note",
            "score": 0.9,
            "metadata": None,
        }
    ]

    results = await mem.search("python")

    assert len(results) == 1
    assert results[0].text == "Python note"
    assert results[0].id == "id-1"
    assert results[0].metadata == {}
