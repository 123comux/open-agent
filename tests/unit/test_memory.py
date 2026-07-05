"""Tests for short-term and long-term memory."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_agent.memory.long_term import LongTermMemory, MemoryEntry
from open_agent.memory.session_manager import SessionManager
from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message


def test_short_term_add_and_get_history_preserves_order():
    mem = ShortTermMemory()
    m1 = Message(role="user", content="hi")
    m2 = Message(role="assistant", content="hello")
    mem.add(m1)
    mem.add(m2)

    history = mem.get_history()
    assert len(history) == 2
    assert history[0] is m1
    assert history[1] is m2


def test_short_term_max_messages_evicts_oldest():
    mem = ShortTermMemory(max_messages=3)
    for i in range(5):
        mem.add(Message(role="user", content=f"msg {i}"))

    history = mem.get_history()
    assert len(history) == 3
    assert len(mem) == 3
    # Oldest two messages evicted; the window keeps messages 2, 3, 4.
    assert history[0].content == "msg 2"
    assert history[-1].content == "msg 4"


def test_short_term_default_max_messages():
    mem = ShortTermMemory()
    assert mem.max_messages == 20


def test_short_term_clear():
    mem = ShortTermMemory()
    mem.add(Message(role="user", content="hi"))
    mem.clear()
    assert mem.get_history() == []
    assert len(mem) == 0


def test_short_term_get_history_returns_copy():
    mem = ShortTermMemory()
    mem.add(Message(role="user", content="hi"))
    first = mem.get_history()
    first.append(Message(role="user", content="extra"))
    second = mem.get_history()
    # Mutating the returned list must not affect internal state.
    assert len(second) == 1


@pytest.fixture
def mock_faiss_store():
    """Return a patched FAISSStore usable for LongTermMemory tests.

    Avoids importing the real faiss_store module (which requires faiss) by
    injecting a fake submodule into ``sys.modules``.
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
async def test_long_term_add_calls_store_and_persists(mock_faiss_store):
    mem = LongTermMemory()
    entry = await mem.add("Python is great.", metadata={"tag": "py"})

    assert isinstance(entry, MemoryEntry)
    assert entry.text == "Python is great."
    assert entry.metadata == {"tag": "py"}
    mock_faiss_store.add.assert_awaited_once()
    call_args = mock_faiss_store.add.call_args
    assert call_args.kwargs["documents"] == ["Python is great."]
    assert call_args.kwargs["metadatas"][0]["tag"] == "py"
    mock_faiss_store.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_term_add_exchange_formats_text(mock_faiss_store):
    mem = LongTermMemory()
    entry = await mem.add_exchange("hello", "hi there", session_id="s1")

    assert "User: hello" in entry.text
    assert "Assistant: hi there" in entry.text
    assert entry.metadata.get("type") == "exchange"
    assert entry.metadata.get("session_id") == "s1"


@pytest.mark.asyncio
async def test_long_term_search_returns_entries(mock_faiss_store):
    mem = LongTermMemory()
    mock_faiss_store.query.return_value = [
        {
            "id": "id-1",
            "document": "Python note",
            "score": 0.9,
            "metadata": {"timestamp": 123.0},
        }
    ]

    results = await mem.search("python")

    assert len(results) == 1
    assert results[0].text == "Python note"
    assert results[0].id == "id-1"
    mock_faiss_store.query.assert_awaited_once_with(query_text="python", n_results=3)


@pytest.mark.asyncio
async def test_long_term_search_empty_query_returns_nothing(mock_faiss_store):
    mem = LongTermMemory()
    assert await mem.search("") == []
    assert await mem.search("   ") == []
    mock_faiss_store.query.assert_not_called()


@pytest.mark.asyncio
async def test_long_term_add_empty_text_raises(mock_faiss_store):
    mem = LongTermMemory()
    with pytest.raises(ValueError):
        await mem.add("")
    with pytest.raises(ValueError):
        await mem.add("   ")


@pytest.mark.asyncio
async def test_long_term_delete_removes_existing_entry(mock_faiss_store):
    mem = LongTermMemory()
    mock_faiss_store.count.side_effect = [1, 0]

    deleted = await mem.delete("id-1")

    assert deleted is True
    mock_faiss_store.delete.assert_awaited_once_with(["id-1"])
    mock_faiss_store.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_term_delete_missing_entry_returns_false(mock_faiss_store):
    mem = LongTermMemory()
    mock_faiss_store.count.side_effect = [1, 1]

    deleted = await mem.delete("missing")

    assert deleted is False


@pytest.mark.asyncio
async def test_long_term_clear_deletes_all(mock_faiss_store):
    mem = LongTermMemory()
    mock_faiss_store._ids = ["a", "b"]

    await mem.clear()

    mock_faiss_store.delete.assert_awaited_once_with(["a", "b"])
    mock_faiss_store.save.assert_awaited_once()


def test_session_manager_rename_updates_memory_and_file(tmp_path):
    manager = SessionManager(storage_dir=str(tmp_path))
    manager.add_message("old-id", Message(role="user", content="hello"))

    manager.rename_session("old-id", "new-id")

    assert "old-id" not in manager.list_sessions()
    assert "new-id" in manager.list_sessions()
    assert len(manager.get_history("new-id")) == 1
    assert not (tmp_path / "old-id.json").exists()
    assert (tmp_path / "new-id.json").exists()


def test_session_manager_rename_raises_on_duplicate(tmp_path):
    manager = SessionManager(storage_dir=str(tmp_path))
    manager.add_message("a", Message(role="user", content="hello"))
    manager.add_message("b", Message(role="user", content="world"))

    with pytest.raises(ValueError):
        manager.rename_session("a", "b")


def test_session_manager_search_matches_id_and_content():
    manager = SessionManager()
    manager.add_message("session-alpha", Message(role="user", content="Python tips"))
    manager.add_message("session-beta", Message(role="user", content="JavaScript tips"))

    results = manager.search_sessions("python")

    assert len(results) == 1
    assert results[0]["session_id"] == "session-alpha"
    assert results[0]["matches"] == 1


def test_session_manager_export_json():
    manager = SessionManager()
    manager.add_message("s1", Message(role="user", content="hi"))
    manager.add_message("s1", Message(role="assistant", content="hello"))

    content = manager.export_session("s1", fmt="json")
    data = json.loads(content)

    assert data["session_id"] == "s1"
    assert len(data["messages"]) == 2


def test_session_manager_export_markdown():
    manager = SessionManager()
    manager.add_message("s1", Message(role="user", content="hi"))

    content = manager.export_session("s1", fmt="md")

    assert "# Session: s1" in content
    assert "## User" in content
    assert "hi" in content
