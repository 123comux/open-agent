"""Tests for the :class:`Agent` ReAct loop."""
from __future__ import annotations

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import ModelResponse, ToolCall
from open_agent.tools.base import Tool
from open_agent.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo back text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: object) -> str:
        return f"echo: {kwargs.get('text', '')}"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


async def test_direct_response_no_tool_calls(mock_model):
    mock_model.queue(ModelResponse(content="The answer is 42."))
    agent = Agent(model=mock_model, tool_registry=_make_registry())
    output = await agent.run("What is the answer?")

    assert isinstance(output, AgentOutput)
    assert output.response == "The answer is 42."
    assert output.steps == 1
    assert output.tool_calls_made == []
    # Only one model call should have happened for a direct response.
    assert len(mock_model.calls) == 1


async def test_react_loop_one_tool_call_then_returns(mock_model):
    mock_model.queue(
        ModelResponse(
            content="Let me echo that.",
            tool_calls=[ToolCall(name="echo", arguments={"text": "hello"})],
        )
    )
    mock_model.queue(ModelResponse(content="Done echoing."))

    agent = Agent(model=mock_model, tool_registry=_make_registry())
    output = await agent.run("Echo hello for me.")

    assert output.response == "Done echoing."
    assert output.steps == 2
    assert len(output.tool_calls_made) == 1

    call = output.tool_calls_made[0]
    assert call["step"] == 1
    assert call["name"] == "echo"
    assert call["arguments"] == {"text": "hello"}
    assert call["observation"] == "echo: hello"
    assert call["is_error"] is False
    # Two model calls: first produces the tool call, second the final answer.
    assert len(mock_model.calls) == 2


async def test_max_steps_enforcement(mock_model):
    # The model keeps requesting a tool call; the loop should stop at max_steps
    # and then ask the model for a final summary.
    tool_call_resp = ModelResponse(
        content="Calling echo again.",
        tool_calls=[ToolCall(name="echo", arguments={"text": "again"})],
    )
    mock_model.queue(tool_call_resp)
    mock_model.queue(tool_call_resp)
    mock_model.queue(ModelResponse(content="Best final answer after max steps."))

    agent = Agent(model=mock_model, tool_registry=_make_registry(), max_steps=2)
    output = await agent.run("Keep echoing.")

    assert output.steps == 2
    assert output.response == "Best final answer after max steps."
    assert len(output.tool_calls_made) == 2
    # Every recorded tool call hit the echo tool successfully.
    for entry in output.tool_calls_made:
        assert entry["name"] == "echo"
        assert entry["is_error"] is False
    # 2 loop iterations + 1 final summary call.
    assert len(mock_model.calls) == 3


async def test_agent_uses_memory(mock_model):
    from open_agent.memory.short_term import ShortTermMemory
    from open_agent.models.base import Message

    memory = ShortTermMemory()
    memory.add(Message(role="user", content="prior question"))
    memory.add(Message(role="assistant", content="prior answer"))

    mock_model.queue(ModelResponse(content="final answer"))
    agent = Agent(model=mock_model, tool_registry=_make_registry(), memory=memory)
    await agent.run("next question")

    # The conversation history passed to the model must include prior messages.
    first_call_messages = mock_model.calls[0]["messages"]
    contents = [m.content for m in first_call_messages]
    assert "prior question" in contents
    assert "prior answer" in contents
    # After the run, the new exchange should be stored in memory.
    assert len(memory) == 4


async def test_agent_uses_long_term_memory(mock_model):
    """Relevant long-term memories are injected and the exchange is saved."""
    from unittest.mock import AsyncMock, MagicMock

    from open_agent.memory.long_term import MemoryEntry

    ltm = MagicMock()
    ltm.search = AsyncMock(
        return_value=[MemoryEntry(text="User likes Python.", metadata={})]
    )
    ltm.add_exchange = AsyncMock()

    mock_model.queue(ModelResponse(content="final answer"))
    agent = Agent(
        model=mock_model, tool_registry=_make_registry(), long_term_memory=ltm
    )
    output = await agent.run("What language should I use?")

    assert output.response == "final answer"
    ltm.search.assert_awaited_once_with("What language should I use?")
    # The memory content should appear in the messages sent to the model.
    first_call_messages = mock_model.calls[0]["messages"]
    contents = [m.content for m in first_call_messages]
    assert any("User likes Python" in c for c in contents)
    ltm.add_exchange.assert_awaited_once_with(
        user_input="What language should I use?",
        assistant_response="final answer",
        session_id="default",
    )
