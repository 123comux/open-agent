"""Tests for the :class:`ToolRegistry`."""
from __future__ import annotations

import pytest

from open_agent.tools.base import Tool, ToolResult
from open_agent.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the provided text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: object) -> str:
        return f"echo: {kwargs.get('text', '')}"


class BoomTool(Tool):
    name = "boom"
    description = "Always raises an error."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: object) -> str:
        raise RuntimeError("kaboom")


def test_register_and_get():
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.get("echo") is tool


def test_get_unknown_raises_keyerror():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")


def test_list_tools():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(BoomTool())
    names = registry.list_tools()
    assert set(names) == {"echo", "boom"}


def test_list_tools_empty():
    registry = ToolRegistry()
    assert registry.list_tools() == []


def test_schemas_returns_tool_schemas():
    registry = ToolRegistry()
    registry.register(EchoTool())
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"
    assert schemas[0]["description"] == "Echo back the provided text."
    assert schemas[0]["parameters"]["required"] == ["text"]


def test_duplicate_registration_overwrites():
    registry = ToolRegistry()
    first = EchoTool()
    second = EchoTool()
    registry.register(first)
    registry.register(second)
    assert registry.get("echo") is second
    assert len(registry.list_tools()) == 1


async def test_execute_known_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = await registry.execute("echo", text="hello")
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "echo: hello"
    assert result.error is None


async def test_execute_unknown_tool_returns_unsuccessful_result():
    registry = ToolRegistry()
    result = await registry.execute("missing", x=1)
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.output == ""
    assert "missing" in (result.error or "")


async def test_execute_captures_exceptions():
    registry = ToolRegistry()
    registry.register(BoomTool())
    result = await registry.execute("boom")
    assert result.success is False
    assert result.output == ""
    assert "RuntimeError" in (result.error or "")
    assert "kaboom" in (result.error or "")
