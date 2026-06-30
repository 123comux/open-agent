"""Shared pytest fixtures for the open-agent test suite."""
from __future__ import annotations

import pytest

from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolSchema,
)
from open_agent.tools.builtin.file import FileTool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.builtin.shell import ShellTool
from open_agent.tools.registry import ToolRegistry


class MockModel(ModelInterface):
    """A fake :class:`ModelInterface` that returns queued responses in order.

    Tests append :class:`ModelResponse` objects via :meth:`queue` (or pass them
    to the constructor); each ``chat`` call pops and returns the next one. When
    the queue is empty a default textual response is returned. Every call is
    recorded in ``self.calls`` for assertions.
    """

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses: list[ModelResponse] = list(responses or [])
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return ModelResponse(content="mock default response")
        return self.responses.pop(0)

    def queue(self, response: ModelResponse) -> None:
        """Append a response to the end of the queue."""
        self.responses.append(response)


@pytest.fixture
def mock_model() -> MockModel:
    """Return a fresh :class:`MockModel` with an empty response queue."""
    return MockModel()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Return a :class:`ToolRegistry` with the builtin shell/python/file tools."""
    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(PythonTool())
    registry.register(FileTool())
    return registry


@pytest.fixture
def tmp_file(tmp_path) -> str:
    """Create a temp file containing ``hello world`` and return its path."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world", encoding="utf-8")
    return str(file_path)
