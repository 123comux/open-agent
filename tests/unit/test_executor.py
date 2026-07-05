"""Tests for the :class:`ToolExecutor`."""
from __future__ import annotations

import asyncio

from open_agent.agent.executor import Observation, ToolExecutor
from open_agent.tools.base import Tool
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


async def test_execute_known_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry)
    obs = await executor.execute("echo", {"text": "hello"})
    assert isinstance(obs, Observation)
    assert obs.is_error is False
    assert obs.text == "echo: hello"
    assert str(obs) == "echo: hello"


async def test_execute_unknown_tool_returns_error_observation():
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    obs = await executor.execute("missing", {"x": 1})
    assert isinstance(obs, Observation)
    assert obs.is_error is True
    assert "missing" in obs.text


async def test_execute_failing_tool_returns_error_observation():
    registry = ToolRegistry()
    registry.register(BoomTool())
    executor = ToolExecutor(registry)
    obs = await executor.execute("boom", {})
    assert obs.is_error is True
    assert "RuntimeError" in obs.text
    assert "kaboom" in obs.text


def test_observation_str_and_defaults():
    obs = Observation("some text")
    assert obs.text == "some text"
    assert obs.is_error is False
    assert str(obs) == "some text"

    err_obs = Observation("boom", is_error=True)
    assert err_obs.is_error is True


async def test_executor_timeout_returns_error_observation():
    """Tools that exceed the executor timeout return an error Observation."""

    class SlowTool(Tool):
        name = "slow"
        description = "A tool that sleeps"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs: object) -> str:
            await asyncio.sleep(10)
            return "done"

    registry = ToolRegistry()
    registry.register(SlowTool())
    executor = ToolExecutor(registry, timeout=0.05)
    obs = await executor.execute("slow", {})
    assert obs.is_error
    assert "timed out" in obs.text.lower()
