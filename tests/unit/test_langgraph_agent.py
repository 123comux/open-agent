"""Tests for the :class:`LangGraphAgent` multi-node agentic graph.

These tests exercise the LangGraph agent in two complementary ways:

* **Node-level isolation tests** -- an :class:`AgentState` is constructed
  directly and a single node method (``_intent_classifier_node``,
  ``_planner_node``, ``_agent_node``, ``_tools_node``, ``_reflector_node``)
  is awaited, asserting on the partial state it returns. The language model
  is driven by the shared :class:`~tests.conftest.MockModel` fixture (wrapped
  automatically by :class:`LangChainModelAdapter`), so no real LLM is called.
* **End-to-end tests** via :meth:`LangGraphAgent.run`, queueing a scripted
  sequence of :class:`ModelResponse` objects to drive the whole graph through
  intent classification, planning, tool execution, reflection and termination.

Routing helpers (``_route_after_*``) and static parsing helpers
(``_parse_intent``, ``_parse_sub_tasks``, ``_reflection_decision``) are also
covered directly where they do not require model calls. ``unittest.mock``
is used to inject scripted model responses and mock tools.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from open_agent.agent.langgraph_agent import (
    MAX_CONSECUTIVE_ERRORS,
    AgentOutput,
    AgentState,
    LangGraphAgent,
)
from open_agent.models.base import ModelInterface, ModelResponse, ToolCall
from open_agent.tools.base import Tool

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


class EchoTool(Tool):
    """A simple tool that echoes its input -- used across the tests."""

    name = "echo"
    description = "Echo back the given text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo."}},
        "required": ["text"],
    }

    async def execute(self, **kwargs: object) -> str:
        return f"echo: {kwargs.get('text', '')}"


class FailingTool(Tool):
    """A tool that always raises -- used to exercise tool-error handling."""

    name = "failing"
    description = "A tool that always fails."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: object) -> str:
        raise RuntimeError("boom")


class FailingModel(ModelInterface):
    """A model provider that always raises -- used for error-path tests."""

    async def chat(self, messages, tools=None):
        raise ConnectionError("model unavailable")

    async def stream_chat(self, messages, tools=None):
        raise ConnectionError("model unavailable")
        yield ""  # pragma: no cover - make it a generator


def _make_state(messages=None, **overrides) -> AgentState:
    """Build a complete :class:`AgentState` with sensible defaults."""
    state: AgentState = {
        "messages": messages
        if messages is not None
        else [SystemMessage(content="system"), HumanMessage(content="hello")],
        "tool_results": [],
        "thoughts": [],
        "steps": 0,
        "intent": "",
        "sub_tasks": [],
        "current_sub_task": 0,
        "reflection": "",
        "reflections": [],
        "error_count": 0,
    }
    state.update(overrides)
    return state


def _build_agent(mock_model, tools=None, max_steps=10) -> LangGraphAgent:
    """Create a :class:`LangGraphAgent` backed by ``mock_model``."""
    return LangGraphAgent(model=mock_model, tools=tools or [], max_steps=max_steps)


def _tool_result(step, name, observation, is_error=False, arguments=None) -> dict:
    """Build a tool-result dict matching the shape stored in ``AgentState``."""
    return {
        "step": step,
        "name": name,
        "arguments": arguments or {},
        "observation": observation,
        "is_error": is_error,
    }

# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #


def test_init_default_max_steps(mock_model):
    """An agent with no tools defaults to ``max_steps == 10``."""
    agent = _build_agent(mock_model)
    assert agent.max_steps == 10
    assert agent._tools == []
    # No tools means ``_bound_model`` is the plain model (no tool binding).
    assert agent._bound_model is agent._model


def test_init_custom_params(mock_model):
    """Custom ``max_steps`` and tools are honored; tools are normalized."""
    agent = _build_agent(mock_model, tools=[EchoTool()], max_steps=3)
    assert agent.max_steps == 3
    assert len(agent._tools) == 1
    assert agent._tools[0].name == "echo"
    assert "echo" in agent._tool_map
    # With tools the bound model is a distinct (bound) runnable.
    assert agent._bound_model is not agent._model


def test_init_invalid_max_steps_raises(mock_model):
    """``max_steps < 1`` is rejected."""
    with pytest.raises(ValueError, match="max_steps"):
        LangGraphAgent(model=mock_model, tools=[], max_steps=0)


def test_init_invalid_model_type_raises():
    """A model that is neither ``BaseChatModel`` nor ``ModelInterface`` raises."""
    with pytest.raises(TypeError, match="model must be"):
        LangGraphAgent(model="not-a-model", tools=[])


def test_init_invalid_tool_type_raises(mock_model):
    """A tool that is neither ``Tool`` nor ``BaseTool`` raises."""
    with pytest.raises(TypeError, match="tools must be"):
        LangGraphAgent(model=mock_model, tools=[123])


def test_init_wraps_model_interface(mock_model):
    """A ``ModelInterface`` is wrapped transparently in the LangChain adapter."""
    from open_agent.models.langchain_adapter import LangChainModelAdapter

    agent = _build_agent(mock_model)
    assert isinstance(agent._model, LangChainModelAdapter)
    assert agent._model.wrapped_model is mock_model

# --------------------------------------------------------------------------- #
# Static helper methods
# --------------------------------------------------------------------------- #


def test_parse_intent_recognizes_categories():
    assert LangGraphAgent._parse_intent("This is KNOWLEDGE") == "knowledge"
    assert LangGraphAgent._parse_intent("realtime query") == "realtime"
    assert LangGraphAgent._parse_intent("a complex task") == "complex"
    assert LangGraphAgent._parse_intent("direct") == "direct"
    assert LangGraphAgent._parse_intent("computation needed") == "computation"


def test_parse_intent_defaults_to_direct():
    assert LangGraphAgent._parse_intent("nonsense reply") == "direct"
    assert LangGraphAgent._parse_intent("") == "direct"


def test_parse_sub_tasks_strips_numbering():
    text = "1. Search the web\n2) Summarize findings\n3: Write the answer"
    assert LangGraphAgent._parse_sub_tasks(text) == [
        "Search the web",
        "Summarize findings",
        "Write the answer",
    ]


def test_parse_sub_tasks_empty():
    assert LangGraphAgent._parse_sub_tasks("") == []
    assert LangGraphAgent._parse_sub_tasks("   \n  ") == []


def test_reflection_decision():
    assert LangGraphAgent._reflection_decision("SUFFICIENT info") == "sufficient"
    assert LangGraphAgent._reflection_decision("NEEDS_MORE data") == "needs_more"
    # Ambiguous reflections default to ``sufficient`` to avoid infinite loops.
    assert LangGraphAgent._reflection_decision("maybe") == "sufficient"
    assert LangGraphAgent._reflection_decision("") == "sufficient"


def test_summarize_tool_results_formats_entries():
    results = [
        _tool_result(1, "echo", "echo: hi"),
        _tool_result(2, "failing", "RuntimeError: boom", is_error=True),
    ]
    summary = LangGraphAgent._summarize_tool_results(results)
    assert "echo" in summary
    assert "OK" in summary
    assert "ERROR" in summary
    assert LangGraphAgent._summarize_tool_results([]) == "(no tool results yet)"


def test_summarize_tool_results_truncates_long_observations():
    long_obs = "x" * 1000
    results = [_tool_result(1, "t", long_obs)]
    summary = LangGraphAgent._summarize_tool_results(results)
    assert "..." in summary
    assert len(summary) < len(long_obs)

# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_route_after_intent(mock_model):
    agent = _build_agent(mock_model)
    assert agent._route_after_intent(_make_state(intent="direct")) == END
    assert agent._route_after_intent(_make_state(intent="complex")) == "planner"
    assert agent._route_after_intent(_make_state(intent="knowledge")) == "agent"
    assert agent._route_after_intent(_make_state(intent="realtime")) == "agent"
    assert agent._route_after_intent(_make_state(intent="computation")) == "agent"


def test_route_after_agent(mock_model):
    agent = _build_agent(mock_model, max_steps=3)
    # Over budget -> END (steps > max_steps).
    over = _make_state(
        steps=4, messages=[HumanMessage("q"), AIMessage(content="done")]
    )
    assert agent._route_after_agent(over) == END
    # AIMessage with tool calls -> tools.
    with_tools = _make_state(
        steps=1,
        messages=[
            HumanMessage("q"),
            AIMessage(
                content="go",
                tool_calls=[{"name": "echo", "args": {}, "id": "c"}],
            ),
        ],
    )
    assert agent._route_after_agent(with_tools) == "tools"
    # AIMessage without tool calls -> END.
    no_tools = _make_state(
        steps=1, messages=[HumanMessage("q"), AIMessage(content="done")]
    )
    assert agent._route_after_agent(no_tools) == END


def test_route_after_reflector(mock_model):
    agent = _build_agent(mock_model)
    # Sufficient reflection -> END.
    assert (
        agent._route_after_reflector(
            _make_state(reflection="SUFFICIENT", error_count=0)
        )
        == END
    )
    # Needs more -> agent.
    assert (
        agent._route_after_reflector(
            _make_state(reflection="NEEDS_MORE", error_count=0)
        )
        == "agent"
    )
    # Too many errors -> END even if needs more.
    assert (
        agent._route_after_reflector(
            _make_state(reflection="NEEDS_MORE", error_count=MAX_CONSECUTIVE_ERRORS)
        )
        == END
    )

# --------------------------------------------------------------------------- #
# Node isolation tests
# --------------------------------------------------------------------------- #


async def test_intent_classifier_node_direct(mock_model):
    """A 'direct' intent produces an inline answer and finishes."""
    mock_model.queue(ModelResponse(content="direct"))
    mock_model.queue(ModelResponse(content="Hi! How can I help?"))
    agent = _build_agent(mock_model)
    state = _make_state(
        messages=[SystemMessage(content="sys"), HumanMessage(content="hello")]
    )

    result = await agent._intent_classifier_node(state)

    assert result["intent"] == "direct"
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Hi! How can I help?"
    assert any("[intent]" in t for t in result["thoughts"])


async def test_intent_classifier_node_routes_non_direct(mock_model):
    """A 'knowledge' intent is recorded without producing an inline answer."""
    mock_model.queue(ModelResponse(content="knowledge"))
    agent = _build_agent(mock_model)
    state = _make_state(messages=[HumanMessage(content="what is RAG?")])

    result = await agent._intent_classifier_node(state)

    assert result["intent"] == "knowledge"
    # No second model call for non-direct intents: messages are unchanged.
    assert len(result["messages"]) == 1


async def test_intent_classifier_model_failure_defaults_to_direct():
    """If the classifier model call fails, the agent degrades to 'direct'."""
    agent = _build_agent(FailingModel())
    state = _make_state(messages=[HumanMessage(content="hi")])

    result = await agent._intent_classifier_node(state)

    assert result["intent"] == "direct"
    assert any("classification failed" in t for t in result["thoughts"])


async def test_planner_node_parses_subtasks(mock_model):
    """The planner decomposes the request and feeds the plan back into messages."""
    mock_model.queue(ModelResponse(content="1. Search the web\n2. Summarize results"))
    agent = _build_agent(mock_model)
    state = _make_state(messages=[HumanMessage(content="research RAG")])

    result = await agent._planner_node(state)

    assert result["sub_tasks"] == ["Search the web", "Summarize results"]
    assert result["current_sub_task"] == 0
    # A HumanMessage with the plan is appended.
    plan_msg = result["messages"][-1]
    assert isinstance(plan_msg, HumanMessage)
    assert "[Plan]" in plan_msg.content


async def test_planner_node_model_failure_returns_empty_plan():
    """If planning fails, no sub-tasks are produced and no plan is appended."""
    agent = _build_agent(FailingModel())
    state = _make_state(messages=[HumanMessage(content="complex task")])

    result = await agent._planner_node(state)

    assert result["sub_tasks"] == []
    assert any("planning failed" in t for t in result["thoughts"])


async def test_agent_node_records_tool_selection(mock_model):
    """The agent node increments steps and records tool-selection reasoning."""
    mock_model.queue(
        ModelResponse(
            content="I'll echo that.",
            tool_calls=[ToolCall(name="echo", arguments={"text": "hi"})],
        )
    )
    agent = _build_agent(mock_model, tools=[EchoTool()])
    state = _make_state(steps=0)

    result = await agent._agent_node(state)

    assert result["steps"] == 1
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].tool_calls  # non-empty
    assert any("Tool selection" in t for t in result["thoughts"])


async def test_agent_node_model_failure_records_error():
    """A model failure in the agent node yields an error AIMessage, not a crash."""
    agent = _build_agent(FailingModel(), tools=[EchoTool()])
    state = _make_state(steps=0)

    result = await agent._agent_node(state)

    assert result["steps"] == 1
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Model call failed" in last.content


async def test_agent_node_with_async_mock_model(mock_model):
    """``unittest.mock.AsyncMock`` can drive the bound model for the agent node."""
    agent = _build_agent(mock_model, tools=[EchoTool()])
    agent._bound_model = AsyncMock()
    agent._bound_model.ainvoke.return_value = AIMessage(
        content="mocked response",
        tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "call_0"}],
    )
    state = _make_state(steps=0)

    result = await agent._agent_node(state)

    agent._bound_model.ainvoke.assert_awaited_once()
    assert result["steps"] == 1
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "mocked response"

async def test_tools_node_executes_tool(mock_model):
    """The tools node runs the requested tool and records a ToolMessage."""
    agent = _build_agent(mock_model, tools=[EchoTool()])
    state = _make_state(
        steps=1,
        messages=[
            HumanMessage(content="echo hi"),
            AIMessage(
                content="calling echo",
                tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "call_0"}],
            ),
        ],
    )

    result = await agent._tools_node(state)

    assert len(result["tool_results"]) == 1
    tr = result["tool_results"][0]
    assert tr["name"] == "echo"
    assert tr["arguments"] == {"text": "hi"}
    assert tr["observation"] == "echo: hi"
    assert tr["is_error"] is False
    assert tr["step"] == 1
    # A ToolMessage is appended to the conversation.
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].content == "echo: hi"


async def test_tools_node_unknown_tool_is_error(mock_model):
    """A request for a missing tool produces an error result, not a crash."""
    agent = _build_agent(mock_model, tools=[EchoTool()])
    state = _make_state(
        steps=1,
        messages=[
            HumanMessage(content="q"),
            AIMessage(
                content="calling missing",
                tool_calls=[{"name": "nope", "args": {}, "id": "call_0"}],
            ),
        ],
    )

    result = await agent._tools_node(state)

    assert result["tool_results"][0]["is_error"] is True
    assert "not available" in result["tool_results"][0]["observation"]


async def test_tools_node_tool_failure_is_recorded(mock_model):
    """A tool that raises is recorded as an error result with the exception."""
    agent = _build_agent(mock_model, tools=[FailingTool()])
    state = _make_state(
        steps=1,
        messages=[
            HumanMessage(content="q"),
            AIMessage(
                content="calling failing",
                tool_calls=[{"name": "failing", "args": {}, "id": "call_0"}],
            ),
        ],
    )

    result = await agent._tools_node(state)

    assert result["tool_results"][0]["is_error"] is True
    assert "RuntimeError" in result["tool_results"][0]["observation"]
    assert "boom" in result["tool_results"][0]["observation"]


async def test_tools_node_with_mocked_basetool(mock_model):
    """A mocked LangChain ``BaseTool`` is executed via ``ainvoke``."""
    agent = _build_agent(mock_model, tools=[])
    mock_tool = MagicMock()
    mock_tool.name = "mock_tool"
    mock_tool.ainvoke = AsyncMock(return_value="mocked output")
    agent._tool_map["mock_tool"] = mock_tool
    state = _make_state(
        steps=2,
        messages=[
            HumanMessage(content="q"),
            AIMessage(
                content="go",
                tool_calls=[{"name": "mock_tool", "args": {"x": 1}, "id": "c1"}],
            ),
        ],
    )

    result = await agent._tools_node(state)

    mock_tool.ainvoke.assert_awaited_once_with({"x": 1})
    assert result["tool_results"][0]["observation"] == "mocked output"
    assert result["tool_results"][0]["is_error"] is False


async def test_reflector_node_sufficient_synthesizes_answer(mock_model):
    """A SUFFICIENT reflection triggers a final synthesis model call."""
    mock_model.queue(ModelResponse(content="SUFFICIENT. We have enough info."))
    mock_model.queue(ModelResponse(content="Here is the final answer."))
    agent = _build_agent(mock_model)
    state = _make_state(steps=1, tool_results=[_tool_result(1, "echo", "echo: hi")])

    result = await agent._reflector_node(state)

    assert "SUFFICIENT" in result["reflection"]
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Here is the final answer."
    assert result["error_count"] == 0


async def test_reflector_node_needs_more_loops_back(mock_model):
    """A NEEDS_MORE reflection feeds the reflection back without synthesizing."""
    mock_model.queue(ModelResponse(content="NEEDS_MORE. Search again."))
    agent = _build_agent(mock_model)
    state = _make_state(steps=1, tool_results=[_tool_result(1, "echo", "echo: hi")])

    result = await agent._reflector_node(state)

    assert "NEEDS_MORE" in result["reflection"]
    # No synthesis: the last message is a HumanMessage with the reflection.
    assert isinstance(result["messages"][-1], HumanMessage)
    assert "[Self-reflection]" in result["messages"][-1].content


async def test_reflector_node_consecutive_errors_force_terminate(mock_model):
    """After MAX_CONSECUTIVE_ERRORS the reflector gives up and finalizes."""
    mock_model.queue(ModelResponse(content="Best-effort answer due to errors."))
    agent = _build_agent(mock_model)
    # error_count one short of the threshold; a fresh error pushes it over.
    state = _make_state(
        steps=2,
        error_count=MAX_CONSECUTIVE_ERRORS - 1,
        tool_results=[
            _tool_result(2, "failing", "RuntimeError: boom", is_error=True)
        ],
    )

    result = await agent._reflector_node(state)

    assert result["error_count"] == MAX_CONSECUTIVE_ERRORS
    assert "forced termination" in result["reflection"]
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Best-effort answer due to errors."


async def test_reflector_node_reflection_failure_terminates_safely():
    """If the reflection model call itself fails, the agent terminates safely."""
    agent = _build_agent(FailingModel())
    state = _make_state(steps=1, tool_results=[_tool_result(1, "echo", "ok")])

    result = await agent._reflector_node(state)

    # Reflection failure defaults to SUFFICIENT to avoid an infinite loop, then
    # a synthesis call is attempted -- which also fails, yielding an error
    # AIMessage rather than crashing.
    assert "SUFFICIENT" in result["reflection"]
    assert "reflection failed" in result["reflection"]
    assert isinstance(result["messages"][-1], AIMessage)

# --------------------------------------------------------------------------- #
# End-to-end run() tests
# --------------------------------------------------------------------------- #


async def test_run_direct_intent(mock_model):
    """A direct intent answers immediately without entering the agent node."""
    mock_model.queue(ModelResponse(content="direct"))
    mock_model.queue(ModelResponse(content="Hello! I am Open Agent."))
    agent = _build_agent(mock_model)

    output = await agent.run("Hi, what can you do?")

    assert isinstance(output, AgentOutput)
    assert output.intent == "direct"
    assert output.response == "Hello! I am Open Agent."
    assert output.steps == 0  # direct intent never enters the agent node
    assert output.tool_calls_made == []


async def test_run_knowledge_intent_direct_answer(mock_model):
    """A knowledge intent with no tool calls answers from the agent node."""
    mock_model.queue(ModelResponse(content="knowledge"))
    mock_model.queue(ModelResponse(content="RAG is retrieval-augmented generation."))
    agent = _build_agent(mock_model)

    output = await agent.run("What is RAG?")

    assert output.intent == "knowledge"
    assert output.response == "RAG is retrieval-augmented generation."
    assert output.steps == 1
    assert output.tool_calls_made == []


async def test_run_complex_intent_uses_planner(mock_model):
    """A complex intent routes through the planner before the agent answers."""
    mock_model.queue(ModelResponse(content="complex"))
    mock_model.queue(ModelResponse(content="1. Define RAG\n2. Give an example"))
    mock_model.queue(ModelResponse(content="RAG combines retrieval with generation."))
    agent = _build_agent(mock_model)

    output = await agent.run("Explain RAG in depth.")

    assert output.intent == "complex"
    assert output.sub_tasks == ["Define RAG", "Give an example"]
    assert output.response == "RAG combines retrieval with generation."
    assert output.steps == 1


async def test_run_with_tool_execution_and_reflection(mock_model):
    """A full ReAct loop: classify -> call tool -> reflect -> synthesize."""
    mock_model.queue(ModelResponse(content="knowledge"))
    mock_model.queue(
        ModelResponse(
            content="I'll echo that.",
            tool_calls=[ToolCall(name="echo", arguments={"text": "hello"})],
        )
    )
    mock_model.queue(ModelResponse(content="SUFFICIENT. The echo result is enough."))
    mock_model.queue(ModelResponse(content="The tool returned: echo: hello"))
    agent = _build_agent(mock_model, tools=[EchoTool()])

    output = await agent.run("Echo the word hello.")

    assert output.intent == "knowledge"
    assert output.steps == 1
    assert len(output.tool_calls_made) == 1
    call = output.tool_calls_made[0]
    assert call["name"] == "echo"
    assert call["arguments"] == {"text": "hello"}
    assert call["observation"] == "echo: hello"
    assert call["is_error"] is False
    assert output.response == "The tool returned: echo: hello"
    assert len(output.reflections) == 1


async def test_run_max_steps_enforcement(mock_model):
    """The loop finalizes once the step budget is exhausted."""
    tool_call = ModelResponse(
        content="calling echo",
        tool_calls=[ToolCall(name="echo", arguments={"text": "again"})],
    )
    mock_model.queue(ModelResponse(content="knowledge"))
    mock_model.queue(tool_call)  # step 1
    mock_model.queue(ModelResponse(content="NEEDS_MORE. Try again."))
    mock_model.queue(tool_call)  # step 2
    mock_model.queue(ModelResponse(content="NEEDS_MORE. Once more."))
    mock_model.queue(ModelResponse(content="Best final answer after max steps."))

    agent = _build_agent(mock_model, tools=[EchoTool()], max_steps=2)
    output = await agent.run("Keep echoing.")

    assert output.steps == 2  # capped at max_steps
    assert output.response == "Best final answer after max steps."
    assert len(output.tool_calls_made) == 2
