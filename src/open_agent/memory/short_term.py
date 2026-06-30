"""Short-term conversation memory using a bounded deque.

Keeps the most recent ``max_messages`` messages (default 20) so the agent has
conversational continuity without an unbounded context window. Older messages
are evicted automatically once the capacity is reached.
"""
from __future__ import annotations

from collections import deque

from open_agent.models.base import Message


class ShortTermMemory:
    """Sliding-window conversation history backed by a deque."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, message: Message) -> None:
        """Append a message, evicting the oldest if at capacity."""
        self._messages.append(message)

    def get_history(self) -> list[Message]:
        """Return the current history as a list (oldest first)."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all stored messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
