"""Integration tests for the agent ReAct loop."""
from __future__ import annotations

import pytest

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import ModelResponse
from open_agent.models.base import ToolCall as MToolCall
from open_agent.tools.builtin import PythonTool
from open_agent.tools.registry import ToolRegistry


class MockFlowModel:
    """Mock model that simulates a tool call then a direct response."""

    def __init__(self):
        self._call_count = 0

    async def chat(self, messages, tools=None):
        self._call_count += 1
        if self._call_count == 1:
            return ModelResponse(
                content="I need to calculate this",
                tool_calls=[MToolCall(name="python", arguments={"code": "print(6*7)"})],
            )
        return ModelResponse(content="The answer is 42.", tool_calls=[])

    async def stream_chat(self, messages, tools=None):
        yield "The answer is 42."


@pytest.mark.asyncio
async def test_agent_tool_call_flow():
    """Test that agent can call a tool and use the result."""
    model = MockFlowModel()
    registry = ToolRegistry()
    registry.register(PythonTool())
    agent = Agent(model=model, tool_registry=registry, max_steps=5)

    output = await agent.run("What is 6*7?")

    assert isinstance(output, AgentOutput)
    assert output.steps == 2  # one tool call + one direct response
    assert len(output.tool_calls_made) == 1
    assert output.tool_calls_made[0]["name"] == "python"
    assert "42" in output.response


@pytest.mark.asyncio
async def test_agent_streaming_flow():
    """Test that agent streaming yields the right events."""
    model = MockFlowModel()
    registry = ToolRegistry()
    registry.register(PythonTool())
    agent = Agent(model=model, tool_registry=registry, max_steps=5)

    events = []
    async for event in agent.run_stream("What is 6*7?"):
        events.append(event)

    # Should have: thought, tool_start, tool_end, thought, token(s), done
    event_types = [e["type"] for e in events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert "token" in event_types
    assert "done" in event_types
    assert events[-1]["type"] == "done"
