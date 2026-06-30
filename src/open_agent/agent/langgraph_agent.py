"""LangGraph-based agent with ReAct loop and thinking chain visibility.

This module provides :class:`LangGraphAgent`, an alternative to
:class:`~open_agent.agent.core.Agent` that uses LangGraph's
:class:`~langgraph.graph.StateGraph` for orchestration. The graph has two nodes:

* ``agent`` -- calls the language model (with tools bound) and records the
  reasoning/thought for that step.
* ``tools`` -- executes any tool calls the model requested and records the
  observations.

A conditional edge leaves the ``agent`` node: if the model emitted tool calls and
the step budget has not been exhausted the flow continues to ``tools``, otherwise
it terminates. The ``tools`` node always loops back to ``agent`` so the model can
reason over the observations (multi-step ReAct). When ``max_steps`` is reached
the agent makes one final, tool-free call to produce a best-effort answer.

The agent accepts either a native LangChain ``BaseChatModel`` or any of our own
:class:`~open_agent.models.base.ModelInterface` providers (wrapped automatically
by :class:`~open_agent.models.langchain_adapter.LangChainModelAdapter`).
"""
from __future__ import annotations

from typing import Any, Sequence, TypedDict, Union

from pydantic import BaseModel, Field

from open_agent.models.base import ModelInterface
from open_agent.tools.base import Tool

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import BaseTool
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langgraph and langchain-core are required for LangGraphAgent. "
        "Install them with: pip install langgraph langchain-core"
    ) from exc

from open_agent.agent.langchain_tools import to_langchain_tool
from open_agent.models.langchain_adapter import LangChainModelAdapter


SYSTEM_PROMPT = """\
You are Open Agent, a general-purpose autonomous work assistant.
You solve the user's task by reasoning step by step and, when helpful, calling
the tools available to you. When you have enough information, respond directly
with the final answer.
"""

# Prompt appended when the step budget is exhausted so the model wraps up.
FINALIZE_PROMPT = "Maximum steps reached. Provide your best final answer now."


class AgentOutput(BaseModel):
    """Final result of a LangGraph agent run."""

    response: str
    steps: int
    tool_calls_made: list[dict] = Field(default_factory=list)
    thoughts: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    """Mutable state threaded through the LangGraph agent graph."""

    messages: list
    tool_results: list
    thoughts: list
    steps: int


# Type alias for the accepted model input.
AcceptedModel = Union[BaseChatModel, ModelInterface]


class LangGraphAgent:
    """ReAct agent orchestrated by a LangGraph ``StateGraph``.

    Args:
        model: A LangChain ``BaseChatModel`` or an open-agent
            :class:`ModelInterface`. Our interface is wrapped transparently.
        tools: A sequence of open-agent :class:`Tool` or LangChain
            :class:`BaseTool` instances the agent may call.
        max_steps: Maximum number of reasoning steps before a forced
            finalization. The finalize call does not count towards the reported
            step count.
    """

    def __init__(
        self,
        model: AcceptedModel,
        tools: Sequence[Union[Tool, BaseTool]],
        max_steps: int = 10,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self._model: BaseChatModel = self._coerce_model(model)

        # Normalize the supplied tools into LangChain BaseTool instances.
        self._tools: list[BaseTool] = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                self._tools.append(tool)
            elif isinstance(tool, Tool):
                self._tools.append(to_langchain_tool(tool))
            else:
                raise TypeError(
                    "tools must be Tool or BaseTool instances, got "
                    f"{type(tool).__name__}"
                )

        self.max_steps = max_steps
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in self._tools}

        if self._tools:
            self._bound_model = self._model.bind_tools(self._tools)
        else:
            self._bound_model = self._model

        self._graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_model(model: AcceptedModel) -> BaseChatModel:
        """Wrap our :class:`ModelInterface` if a native LangChain model was not given."""
        if isinstance(model, ModelInterface):
            return LangChainModelAdapter(model=model)
        if isinstance(model, BaseChatModel):
            return model
        raise TypeError(
            "model must be a BaseChatModel or ModelInterface, got "
            f"{type(model).__name__}"
        )

    def _build_graph(self) -> Any:
        """Compile the two-node ReAct state graph."""
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Graph nodes
    # ------------------------------------------------------------------ #

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Call the model, append the assistant message, record the thought."""
        steps = state["steps"]
        messages: list[BaseMessage] = list(state["messages"])
        thoughts: list[str] = list(state["thoughts"])

        if steps >= self.max_steps:
            # Budget exhausted: ask for a best-effort answer without tools.
            messages = messages + [HumanMessage(content=FINALIZE_PROMPT)]
            response = await self._model.ainvoke(messages)
            ai_message = self._ensure_ai(response)
            messages.append(ai_message)
            thoughts.append(
                f"[finalize] Reasoning step budget exhausted; final answer: "
                f"{self._extract_text(ai_message)}"
            )
            return {
                "messages": messages,
                "thoughts": thoughts,
                "steps": steps + 1,
                "tool_results": list(state["tool_results"]),
            }

        response = await self._bound_model.ainvoke(messages)
        ai_message = self._ensure_ai(response)
        messages.append(ai_message)

        new_steps = steps + 1
        thoughts.append(self._build_thought(new_steps, ai_message))
        return {
            "messages": messages,
            "thoughts": thoughts,
            "steps": new_steps,
            "tool_results": list(state["tool_results"]),
        }

    async def _tools_node(self, state: AgentState) -> AgentState:
        """Execute the tool calls requested in the last assistant message."""
        messages: list[BaseMessage] = list(state["messages"])
        tool_results: list[dict] = list(state["tool_results"])
        thoughts: list[str] = list(state["thoughts"])

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        current_step = state["steps"]

        for call in tool_calls:
            name = self._tc_field(call, "name", "")
            args = self._tc_field(call, "args", {}) or {}
            call_id = self._tc_field(call, "id", "") or ""

            tool = self._tool_map.get(name)
            if tool is None:
                output = f"Error: tool '{name}' is not available."
                tool_results.append(
                    self._record_tool_result(current_step, name, args, output, True)
                )
                messages.append(ToolMessage(content=output, tool_call_id=call_id))
                thoughts.append(f"[step {current_step}] Observation({name}): {output}")
                continue

            try:
                raw_output = await tool.ainvoke(args)
                output = raw_output if isinstance(raw_output, str) else str(raw_output)
                is_error = False
            except Exception as exc:  # noqa: BLE001 - mirror ToolExecutor behavior
                output = f"{type(exc).__name__}: {exc}"
                is_error = True

            tool_results.append(
                self._record_tool_result(current_step, name, args, output, is_error)
            )
            messages.append(ToolMessage(content=output, tool_call_id=call_id))
            thoughts.append(f"[step {current_step}] Observation({name}): {output}")

        return {
            "messages": messages,
            "thoughts": thoughts,
            "steps": state["steps"],
            "tool_results": tool_results,
        }

    def _should_continue(self, state: AgentState) -> str:
        """Route out of the agent node: continue to tools or finish."""
        # A finalize call increments steps past max_steps -- always stop then.
        if state["steps"] > self.max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self, user_input: str, session_id: str = "default") -> AgentOutput:
        """Run the ReAct loop for a single user request.

        Args:
            user_input: The user's task or question.
            session_id: Reserved for future per-session memory isolation.

        Returns:
            An :class:`AgentOutput` with the final response, the number of
            reasoning steps taken, the tool calls made and the full thinking
            chain.
        """
        del session_id  # reserved for future per-session memory isolation

        initial: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ],
            "tool_results": [],
            "thoughts": [],
            "steps": 0,
        }

        final_state = await self._graph.ainvoke(initial)
        return self._build_output(final_state)

    # ------------------------------------------------------------------ #
    # Output assembly & helpers
    # ------------------------------------------------------------------ #

    def _build_output(self, state: AgentState) -> AgentOutput:
        last = state["messages"][-1] if state["messages"] else None
        response_text = self._extract_text(last) if last is not None else ""

        tool_calls_made: list[dict] = []
        for result in state["tool_results"]:
            tool_calls_made.append(
                {
                    "step": result["step"],
                    "name": result["name"],
                    "arguments": result["arguments"],
                    "observation": result["observation"],
                    "is_error": result["is_error"],
                }
            )

        # The finalize call (if any) increments steps past max_steps; cap the
        # reported step count at max_steps so it reflects actual reasoning.
        reported_steps = min(state["steps"], self.max_steps)

        return AgentOutput(
            response=response_text,
            steps=reported_steps,
            tool_calls_made=tool_calls_made,
            thoughts=list(state["thoughts"]),
        )

    @staticmethod
    def _record_tool_result(
        step: int,
        name: str,
        arguments: dict,
        observation: str,
        is_error: bool,
    ) -> dict:
        return {
            "step": step,
            "name": name,
            "arguments": arguments,
            "observation": observation,
            "is_error": is_error,
        }

    @staticmethod
    def _ensure_ai(message: BaseMessage) -> AIMessage:
        """Coerce any model response into an :class:`AIMessage`."""
        if isinstance(message, AIMessage):
            return message
        return AIMessage(content=LangGraphAgent._extract_text(message))

    @staticmethod
    def _extract_text(message: BaseMessage) -> str:
        content = message.content
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _build_thought(step: int, response: AIMessage) -> str:
        text = LangGraphAgent._extract_text(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            calls = ", ".join(
                f"{LangGraphAgent._tc_field(tc, 'name', '?')}("
                f"{LangGraphAgent._tc_field(tc, 'args', {})})"
                for tc in tool_calls
            )
            reasoning = text or "(no explicit reasoning)"
            return f"[step {step}] Reasoning: {reasoning} -> calling {calls}"
        return f"[step {step}] Answer: {text}"

    @staticmethod
    def _tc_field(call: Any, key: str, default: Any = None) -> Any:
        """Read a field from a tool call that may be a dict or an object."""
        if isinstance(call, dict):
            return call.get(key, default)
        return getattr(call, key, default)


__all__ = ["AgentOutput", "AgentState", "LangGraphAgent"]
