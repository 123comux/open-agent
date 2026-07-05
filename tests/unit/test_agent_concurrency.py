"""Tests for Agent concurrency (run lock serialization)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from open_agent.agent.core import Agent
from open_agent.models.base import Message, ModelInterface, ModelResponse, ToolSchema
from open_agent.tools.registry import ToolRegistry


class _ConcurrencyModel(ModelInterface):
    """Tracks how many ``chat`` calls run concurrently."""

    def __init__(self) -> None:
        self._current = 0
        self.max_concurrent = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        self._current += 1
        self.max_concurrent = max(self.max_concurrent, self._current)
        # Simulate a small amount of work so that without a lock two
        # concurrent calls would overlap here.
        await asyncio.sleep(0.02)
        self._current -= 1
        return ModelResponse(content="done")

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        yield "done"


async def test_agent_run_serializes_concurrent_calls():
    """Two concurrent ``run`` calls on the same Agent are serialized."""
    model = _ConcurrencyModel()
    agent = Agent(model=model, tool_registry=ToolRegistry(), max_steps=1)

    await asyncio.gather(agent.run("task 1"), agent.run("task 2"))

    # With the run lock, ``chat`` calls never overlap. Without the lock the
    # two sleeps would overlap and ``max_concurrent`` would be 2.
    assert model.max_concurrent == 1
