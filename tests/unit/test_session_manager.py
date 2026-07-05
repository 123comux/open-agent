"""Tests for :class:`SessionManager.rename_session` edge cases.

Covers the data-loss regression (C2) where a disk-only session was clobbered
by an empty in-memory ``ShortTermMemory`` during rename, and the rollback
behaviour when the target id already exists on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_agent.memory.session_manager import SessionManager
from open_agent.models.base import Message


def test_rename_session_disk_only_preserves_history(tmp_path: Path) -> None:
    """A session that exists only on disk is renamed without losing history.

    Regression test for C2: previously ``rename_session`` created an empty
    ``ShortTermMemory`` when ``old_id`` was absent from ``_sessions`` and then
    called ``_save``, overwriting the just-renamed file with empty history.
    The fix leaves the renamed file untouched.
    """
    manager = SessionManager(storage_dir=str(tmp_path))
    # Write a session file directly to disk without going through the manager
    # so it is NOT present in ``_sessions`` (disk-only session).
    (tmp_path / "disk-only.json").write_text(
        json.dumps(
            {
                "session_id": "disk-only",
                "messages": [
                    {"role": "user", "content": "msg1"},
                    {"role": "assistant", "content": "msg2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Sanity: not loaded into memory.
    assert "disk-only" not in manager._sessions

    manager.rename_session("disk-only", "renamed")

    # Old disk file gone, new disk file present.
    assert not (tmp_path / "disk-only.json").exists()
    assert (tmp_path / "renamed.json").exists()
    assert "disk-only" not in manager.list_sessions()
    assert "renamed" in manager.list_sessions()
    # History survived the rename (not clobbered by an empty memory).
    history = manager.get_history("renamed")
    assert len(history) == 2
    assert history[0].content == "msg1"
    assert history[1].content == "msg2"


def test_rename_session_rollback_on_target_exists(tmp_path: Path) -> None:
    """When the target id already exists on disk, rename raises ``ValueError``
    and leaves the source session intact in both memory and on disk.

    The disk pre-check runs before any in-memory mutation, so ``_sessions``
    and the source file are untouched on failure.
    """
    manager = SessionManager(storage_dir=str(tmp_path))
    manager.add_message("old-id", Message(role="user", content="hello"))
    # Create the target file on disk without going through the manager so it
    # is not in ``_sessions`` (exercises the disk pre-check, not the memory
    # duplicate check).
    (tmp_path / "target-id.json").write_text(
        json.dumps({"session_id": "target-id", "messages": []}),
        encoding="utf-8",
    )
    assert "target-id" not in manager._sessions

    with pytest.raises(ValueError):
        manager.rename_session("old-id", "target-id")

    # Source session still intact in memory and on disk.
    assert "old-id" in manager._sessions
    assert "old-id" in manager.list_sessions()
    assert (tmp_path / "old-id.json").exists()
    assert len(manager.get_history("old-id")) == 1
    # Target file untouched.
    assert (tmp_path / "target-id.json").exists()


def test_rename_session_missing_raises_keyerror(tmp_path: Path) -> None:
    """Renaming a session that exists neither in memory nor on disk raises."""
    manager = SessionManager(storage_dir=str(tmp_path))
    with pytest.raises(KeyError):
        manager.rename_session("nope", "new")


def test_rename_session_memory_only_creates_disk_file(tmp_path: Path) -> None:
    """A session that exists only in memory is persisted to disk on rename."""
    manager = SessionManager(storage_dir=str(tmp_path))
    manager.add_message("mem-only", Message(role="user", content="hi"))
    # Remove the disk file so the session is memory-only.
    (tmp_path / "mem-only.json").unlink()
    assert not (tmp_path / "mem-only.json").exists()

    manager.rename_session("mem-only", "moved")

    assert "moved" in manager._sessions
    assert len(manager.get_history("moved")) == 1
    # _save was invoked to materialize the new disk file.
    assert (tmp_path / "moved.json").exists()
    assert not (tmp_path / "mem-only.json").exists()
