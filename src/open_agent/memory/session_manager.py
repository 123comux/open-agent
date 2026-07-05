"""Session-based conversation memory with optional file persistence."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, cast

from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


class SessionManager:
    """Manages per-session conversation memory with optional persistence.

    Args:
        max_messages: Maximum messages per session (sliding window).
        storage_dir: Optional directory for JSON persistence. When set,
            sessions are auto-loaded on access and auto-saved on mutation.

    Note:
        Public methods are synchronous for backwards compatibility. Callers
        running in an asyncio event loop who need non-blocking disk I/O can
        wrap the file-backed calls (``add_message``, ``get_history`` when
        storage_dir is set) with ``asyncio.to_thread(...)``. The internal
        ``_save``/``_load`` helpers perform atomic writes (temp file +
        ``os.replace``) so a crash mid-write cannot corrupt the session file.
    """

    def __init__(
        self,
        max_messages: int = 20,
        storage_dir: str | None = None,
    ) -> None:
        self.max_messages = max_messages
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
        self._sessions: dict[str, ShortTermMemory] = {}

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject session IDs that could escape the storage directory.

        Only ``[A-Za-z0-9_\\-.]`` are permitted; this blocks path separators
        and ``..`` traversal so a crafted session_id cannot read or overwrite
        arbitrary files on disk.
        """
        if not session_id or not _SESSION_ID_RE.match(session_id):
            raise ValueError(
                f"Invalid session_id: {session_id!r}. Only letters, digits, "
                " '_', '-', and '.' are allowed."
            )

    def get_session(self, session_id: str) -> ShortTermMemory:
        """Get or create a memory for the given session."""
        self._validate_session_id(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory(self.max_messages)
            if self.storage_dir:
                self._load(session_id)
        return self._sessions[session_id]

    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session and persist if configured."""
        self._validate_session_id(session_id)
        mem = self.get_session(session_id)
        mem.add(message)
        if self.storage_dir:
            self._save(session_id)

    def get_history(self, session_id: str) -> list[Message]:
        """Get conversation history for a session."""
        self._validate_session_id(session_id)
        return self.get_session(session_id).get_history()

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        self._validate_session_id(session_id)
        if session_id in self._sessions:
            self._sessions[session_id].clear()
        if self.storage_dir:
            path = self._session_path(session_id)
            if path.exists():
                path.unlink()

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        sessions = set(self._sessions.keys())
        if self.storage_dir:
            for f in Path(self.storage_dir).glob("*.json"):
                sessions.add(f.stem)
        return sorted(sessions)

    def rename_session(self, old_id: str, new_id: str) -> None:
        """Rename a session, updating both memory and persisted file."""
        self._validate_session_id(old_id)
        self._validate_session_id(new_id)
        if old_id == new_id:
            return
        if new_id in self._sessions:
            raise ValueError(f"Session '{new_id}' already exists")
        mem = self._sessions.pop(old_id, ShortTermMemory(self.max_messages))
        self._sessions[new_id] = mem
        if self.storage_dir:
            old_path = self._session_path(old_id)
            new_path = self._session_path(new_id)
            if new_path.exists():
                raise ValueError(f"Session '{new_id}' already exists")
            if old_path.exists():
                old_path.rename(new_path)
            self._save(new_id)

    def search_sessions(self, query: str) -> list[dict[str, Any]]:
        """Search sessions by id and message content.

        Returns a list of dicts with ``session_id`` and ``matches`` (number of
        matching messages).
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for session_id in self.list_sessions():
            matches = 0
            if query_lower in session_id.lower():
                matches += 1
            for msg in self.get_history(session_id):
                if query_lower in msg.content.lower():
                    matches += 1
            if matches > 0:
                results.append({"session_id": session_id, "matches": matches})
        results.sort(key=lambda r: r["matches"], reverse=True)
        return results

    def export_session(self, session_id: str, fmt: str = "json") -> str:
        """Export a session's history as JSON or Markdown."""
        self._validate_session_id(session_id)
        history = self.get_history(session_id)
        if fmt == "json":
            data = {
                "session_id": session_id,
                "exported_at": time.time(),
                "messages": [
                    {"role": m.role, "content": m.content} for m in history
                ],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)
        lines: list[str] = [f"# Session: {session_id}\n"]
        for m in history:
            role_label = "User" if m.role == "user" else "Assistant"
            lines.append(f"## {role_label}\n\n{m.content}\n")
        return "\n".join(lines)

    def _session_path(self, session_id: str) -> Path:
        # Callers guard on `self.storage_dir` before invoking this helper;
        # cast away the Optional so mypy accepts Path(...).
        return Path(cast(str, self.storage_dir)) / f"{session_id}.json"

    def _save(self, session_id: str) -> None:
        """Persist session history to JSON atomically.

        Writes to a ``.tmp`` sibling file and then ``os.replace``s it onto the
        final path so a crash mid-write leaves the previous file intact rather
        than a truncated/partial JSON file.
        """
        if not self.storage_dir:
            return
        mem = self._sessions.get(session_id)
        if not mem:
            return
        data = {
            "session_id": session_id,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": time.time()}
                for m in mem.get_history()
            ],
        }
        path = self._session_path(session_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)

    def _load(self, session_id: str) -> None:
        """Load session history from JSON."""
        if not self.storage_dir:
            return
        path = self._session_path(session_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for msg in data.get("messages", []):
                self._sessions[session_id].add(
                    Message(role=msg["role"], content=msg["content"])
                )
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, start fresh
