"""Session-based conversation memory with optional file persistence."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message


class SessionManager:
    """Manages per-session conversation memory with optional persistence.

    Args:
        max_messages: Maximum messages per session (sliding window).
        storage_dir: Optional directory for JSON persistence. When set,
            sessions are auto-loaded on access and auto-saved on mutation.
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

    def get_session(self, session_id: str) -> ShortTermMemory:
        """Get or create a memory for the given session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory(self.max_messages)
            if self.storage_dir:
                self._load(session_id)
        return self._sessions[session_id]

    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session and persist if configured."""
        mem = self.get_session(session_id)
        mem.add(message)
        if self.storage_dir:
            self._save(session_id)

    def get_history(self, session_id: str) -> list[Message]:
        """Get conversation history for a session."""
        return self.get_session(session_id).get_history()

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
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

    def _session_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / f"{session_id}.json"

    def _save(self, session_id: str) -> None:
        """Persist session history to JSON."""
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
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

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
