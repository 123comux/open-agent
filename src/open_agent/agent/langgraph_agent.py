"""LangGraph-based agent with intent classification, planning and reflection.

This module provides :class:`LangGraphAgent`, an alternative to
:class:`~open_agent.agent.core.Agent` that uses LangGraph's
:class:`~langgraph.graph.StateGraph` for orchestration. The graph is a
multi-node pipeline with advanced agentic capabilities:

* ``intent_classifier`` -- classifies the user's question into one of
  ``knowledge``, ``realtime``, ``computation``, ``complex`` or ``direct``.
  Direct questions are answered immediately; the others are routed to the
  agent (``complex`` goes through the planner first).
* ``planner`` -- for ``complex`` intents, decomposes the request into ordered
  sub-tasks and feeds the plan back into the conversation.
* ``agent`` -- calls the language model with tools bound (ReAct step) and
  records the tool-selection reasoning.
* ``tools`` -- executes the tool calls requested by the agent.
* ``reflector`` -- after each tool execution, evaluates whether the results are
  sufficient, whether errors occurred, and whether to retry. It loops back to
  the agent when more work is needed and synthesizes the final answer when the
  information is sufficient (or the error/step budget is exhausted).

Graph flow::

    START -> intent_classifier
    intent_classifier -"direct"-> END
    intent_classifier -"knowledge"/"realtime"/"computation"-> agent
    intent_classifier -"complex"-> planner -> agent
    agent -has_tool_calls-> tools -> reflector
    reflector -needs_more-> agent
    reflector -sufficient-> END
    agent -no_tool_calls-> END

The agent accepts either a native LangChain ``BaseChatModel`` or any of our own
:class:`~open_agent.models.base.ModelInterface` providers (wrapped automatically
by :class:`~open_agent.models.langchain_adapter.LangChainModelAdapter`).
"""
from __future__ import annotations

import re
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
You are Open Agent, a general-purpose autonomous work assistant with Agentic RAG capabilities.

You solve user tasks by:
1. Understanding the intent of the question
2. Planning what tools and steps are needed
3. Executing tools to gather information
4. Reflecting on results to verify correctness
5. Iterating if information is insufficient or errors occur
6. Providing a comprehensive final answer

Available tools:
{tool_descriptions}

Tool Selection Guidelines:
- For static knowledge questions: use knowledge_base tool first
- For real-time/current information: use web_search tool
- For calculations and data processing: use python tool
- For file operations: use file tool
- For system commands: use shell tool
- Multiple tools can be called in sequence for complex tasks

Always explain your reasoning before calling a tool. After receiving tool results, verify the information is correct and sufficient before giving your final answer.
"""

INTENT_PROMPT = (
    "Classify the user's question into one of: knowledge, realtime, computation, "
    "complex, direct. Reply with ONLY the category name."
)

PLANNER_PROMPT = (
    "Break down the user's request into ordered sub-tasks. Each sub-task should be "
    "a specific action. Return as a numbered list."
)

REFLECT_PROMPT = (
    "Given the user's question and the tool results so far, evaluate: "
    "1) Is the information sufficient to answer? "
    "2) Are there errors or contradictions? "
    "3) What additional steps are needed? "
    "Reply with SUFFICIENT or NEEDS_MORE and a brief explanation."
)

# Prompt appended when the step budget is exhausted so the model wraps up.
FINALIZE_PROMPT = "Maximum steps reached. Provide your best final answer now."

SUFFICIENT_PROMPT = (
    "The gathered information is sufficient. Provide your final comprehensive "
    "answer to the user's original question."
)

ERROR_FINALIZE_PROMPT = (
    "Multiple tool errors occurred. Provide a best-effort final answer, "
    "explaining what information is missing or what went wrong."
)

# Number of consecutive tool-error iterations after which the reflector gives up.
MAX_CONSECUTIVE_ERRORS = 3


class AgentOutput(BaseModel):
    """Final result of a LangGraph agent run."""

    response: str
    steps: int
    intent: str = ""
    sub_tasks: list[str] = Field(default_factory=list)
    tool_calls_made: list[dict] = Field(default_factory=list)
    thoughts: list[str] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    """Mutable state threaded through the LangGraph agent graph."""

    messages: list
    tool_results: list
    thoughts: list
    steps: int
    intent: str  # classified intent
    sub_tasks: list[str]  # decomposed sub-tasks
    current_sub_task: int  # index of current sub-task
    reflection: str  # latest reflection result
    reflections: list[str]  # accumulated reflection history
    error_count: int  # track consecutive errors


# Type alias for the accepted model input.
AcceptedModel = Union[BaseChatModel, ModelInterface]


class LangGraphAgent:
    """Multi-node agentic LangGraph agent.

    Args:
        model: A LangChain ``BaseChatModel`` or an open-agent
            :class:`ModelInterface`. Our interface is wrapped transparently.
        tools: A sequence of open-agent :class:`Tool` or LangChain
            :class:`BaseTool` instances the agent may call.
        max_steps: Maximum number of agent reasoning steps before a forced
            finalization. Classification, planning and reflection calls do not
            count towards this budget; only the ``agent`` node does.
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

        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools
        ) or "(no tools available)"
        self._system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

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
        """Compile the multi-node agentic state graph."""
        graph = StateGraph(AgentState)
        graph.add_node("intent_classifier", self._intent_classifier_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("reflector", self._reflector_node)

        graph.add_edge(START, "intent_classifier")
        graph.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {"planner": "planner", "agent": "agent", END: END},
        )
        graph.add_edge("planner", "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "reflector")
        graph.add_conditional_edges(
            "reflector",
            self._route_after_reflector,
            {"agent": "agent", END: END},
        )
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Graph nodes
    # ------------------------------------------------------------------ #

    async def _intent_classifier_node(self, state: AgentState) -> AgentState:
        """Classify the user's intent. Direct questions are answered inline."""
        messages: list[BaseMessage] = list(state["messages"])
        thoughts: list[str] = list(state["thoughts"])
        user_question = self._extract_user_question(messages)

        try:
            response = await self._model.ainvoke(
                [
                    SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=user_question),
                ]
            )
            raw = self._extract_text(response)
            intent = self._parse_intent(raw)
            reason = raw
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            intent = "direct"
            reason = (
                f"classification failed ({type(exc).__name__}: {exc}); "
                "defaulting to direct"
            )

        thoughts.append(f"[intent] Classified as '{intent}' because {reason}")

        # For direct (simple chat) intent, produce the answer here and finish.
        if intent == "direct":
            try:
                response = await self._model.ainvoke(messages)
                ai = self._ensure_ai(response)
            except Exception as exc:  # noqa: BLE001
                ai = AIMessage(
                    content=(
                        "I encountered an issue responding: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            messages.append(ai)
            thoughts.append(f"[direct] Direct response: {self._extract_text(ai)}")

        return {"messages": messages, "thoughts": thoughts, "intent": intent}

    async def _planner_node(self, state: AgentState) -> AgentState:
        """Decompose a complex request into ordered sub-tasks."""
        messages: list[BaseMessage] = list(state["messages"])
        thoughts: list[str] = list(state["thoughts"])
        user_question = self._extract_user_question(messages)

        try:
            response = await self._model.ainvoke(
                [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=user_question),
                ]
            )
            raw = self._extract_text(response)
            sub_tasks = self._parse_sub_tasks(raw)
        except Exception as exc:  # noqa: BLE001
            raw = f"planning failed ({type(exc).__name__}: {exc})"
            sub_tasks = []

        thoughts.append(f"[decompose] Sub-tasks: {sub_tasks if sub_tasks else raw}")

        # Feed the plan back into the conversation so the agent can follow it.
        if sub_tasks:
            plan_text = "\n".join(
                f"{i + 1}. {t}" for i, t in enumerate(sub_tasks)
            )
            messages.append(
                HumanMessage(
                    content=(
                        "[Plan] Decomposed sub-tasks:\n"
                        f"{plan_text}\n"
                        "Work through these steps using the available tools."
                    )
                )
            )

        return {
            "messages": messages,
            "thoughts": thoughts,
            "sub_tasks": sub_tasks,
            "current_sub_task": 0,
        }

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Call the model (with tools bound) and record tool-selection reasoning."""
        messages: list[BaseMessage] = list(state["messages"])
        thoughts: list[str] = list(state["thoughts"])
        steps = state["steps"]

        if steps >= self.max_steps:
            # Budget exhausted: ask for a best-effort answer without tools.
            messages.append(HumanMessage(content=FINALIZE_PROMPT))
            try:
                response = await self._model.ainvoke(messages)
                ai = self._ensure_ai(response)
            except Exception as exc:  # noqa: BLE001
                ai = AIMessage(
                    content=f"Unable to finalize: {type(exc).__name__}: {exc}"
                )
            messages.append(ai)
            thoughts.append(
                "[finalize] Reasoning step budget exhausted; final answer: "
                f"{self._extract_text(ai)}"
            )
            return {"messages": messages, "thoughts": thoughts, "steps": steps + 1}

        try:
            response = await self._bound_model.ainvoke(messages)
            ai = self._ensure_ai(response)
        except Exception as exc:  # noqa: BLE001
            ai = AIMessage(content=f"Model call failed: {type(exc).__name__}: {exc}")
        messages.append(ai)

        new_steps = steps + 1
        thoughts.append(self._build_agent_thought(new_steps, ai))
        return {"messages": messages, "thoughts": thoughts, "steps": new_steps}

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
            "tool_results": tool_results,
        }

    async def _reflector_node(self, state: AgentState) -> AgentState:
        """Verify tool results; loop back to the agent or synthesize the answer."""
        messages: list[BaseMessage] = list(state["messages"])
        thoughts: list[str] = list(state["thoughts"])
        reflections: list[str] = list(state.get("reflections", []))
        tool_results: list[dict] = list(state["tool_results"])
        error_count = state["error_count"]
        current_step = state["steps"]
        user_question = self._extract_user_question(messages)

        # Track consecutive errors across iterations; reset on success.
        recent = [r for r in tool_results if r["step"] == current_step]
        has_error = any(r["is_error"] for r in recent) if recent else False
        error_count = error_count + 1 if has_error else 0
        forced_terminate = error_count >= MAX_CONSECUTIVE_ERRORS

        if forced_terminate:
            reflection_text = (
                f"NEEDS_MORE -> forced termination after {error_count} consecutive "
                f"tool errors. Giving up on retrying."
            )
        else:
            try:
                tool_summary = self._summarize_tool_results(tool_results)
                response = await self._model.ainvoke(
                    [
                        SystemMessage(content=REFLECT_PROMPT),
                        HumanMessage(
                            content=(
                                f"User question: {user_question}\n\n"
                                f"Tool results so far:\n{tool_summary}"
                            )
                        ),
                    ]
                )
                reflection_text = self._extract_text(response)
            except Exception as exc:  # noqa: BLE001
                # Default to terminating safely to avoid an infinite loop.
                reflection_text = (
                    f"SUFFICIENT (reflection failed: {type(exc).__name__}: {exc}; "
                    "terminating safely)."
                )

        thoughts.append(f"[reflect] {reflection_text}")
        reflections.append(reflection_text)

        decision = self._reflection_decision(reflection_text)

        if forced_terminate or decision == "sufficient":
            # Synthesize the final answer from the gathered information.
            prompt = ERROR_FINALIZE_PROMPT if forced_terminate else SUFFICIENT_PROMPT
            messages.append(HumanMessage(content=prompt))
            try:
                response = await self._model.ainvoke(messages)
                ai = self._ensure_ai(response)
            except Exception as exc:  # noqa: BLE001
                ai = AIMessage(
                    content=(
                        "Unable to produce a final answer: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            messages.append(ai)
            return {
                "messages": messages,
                "thoughts": thoughts,
                "reflection": reflection_text,
                "reflections": reflections,
                "error_count": error_count,
            }

        # NEEDS_MORE: feed the reflection back to the agent so it can correct
        # course (retry with different args, try a different tool, etc.).
        messages.append(
            HumanMessage(
                content=(
                    f"[Self-reflection] {reflection_text} "
                    "Adjust your next action accordingly."
                )
            )
        )
        return {
            "messages": messages,
            "thoughts": thoughts,
            "reflection": reflection_text,
            "reflections": reflections,
            "error_count": error_count,
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _route_after_intent(self, state: AgentState) -> str:
        """Route out of the intent classifier."""
        intent = state.get("intent", "direct")
        if intent == "direct":
            return END
        if intent == "complex":
            return "planner"
        # knowledge, realtime, computation
        return "agent"

    def _route_after_agent(self, state: AgentState) -> str:
        """Route out of the agent node: continue to tools or finish."""
        # A finalize call increments steps past max_steps -- always stop then.
        if state["steps"] > self.max_steps:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    def _route_after_reflector(self, state: AgentState) -> str:
        """Route out of the reflector: iterate or finish."""
        if state["error_count"] >= MAX_CONSECUTIVE_ERRORS:
            return END
        if self._reflection_decision(state["reflection"]) == "sufficient":
            return END
        return "agent"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self, user_input: str, session_id: str = "default") -> AgentOutput:
        """Run the agentic loop for a single user request.

        Args:
            user_input: The user's task or question.
            session_id: Reserved for future per-session memory isolation.

        Returns:
            An :class:`AgentOutput` with the final response, the number of
            reasoning steps taken, the classified intent, decomposed sub-tasks,
            the tool calls made, the thinking chain and the reflection history.
        """
        del session_id  # reserved for future per-session memory isolation

        initial: AgentState = {
            "messages": [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_input),
            ],
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

        final_state = await self._graph.ainvoke(initial)
        return self._build_output(final_state)

    # ------------------------------------------------------------------ #
    # Output assembly & helpers
    # ------------------------------------------------------------------ #

    def _build_output(self, state: AgentState) -> AgentOutput:
        # Prefer the last assistant message as the final response.
        response_text = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                response_text = self._extract_text(m)
                break
        if not response_text and state["messages"]:
            response_text = self._extract_text(state["messages"][-1])

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
            intent=state.get("intent", ""),
            sub_tasks=list(state.get("sub_tasks", [])),
            tool_calls_made=tool_calls_made,
            thoughts=list(state["thoughts"]),
            reflections=list(state.get("reflections", [])),
        )

    # ------------------------------------------------------------------ #
    # Parsing / formatting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_intent(text: str) -> str:
        """Extract a valid intent category from the model's reply."""
        lowered = (text or "").lower().strip()
        for category in ("knowledge", "realtime", "computation", "complex", "direct"):
            if category in lowered:
                return category
        return "direct"

    @staticmethod
    def _parse_sub_tasks(text: str) -> list[str]:
        """Parse a numbered list of sub-tasks from the planner's reply."""
        tasks: list[str] = []
        for line in (text or "").strip().splitlines():
            cleaned = re.sub(r"^\s*\d+[\.\)\:]\s*", "", line.strip())
            if cleaned:
                tasks.append(cleaned)
        return tasks

    @staticmethod
    def _reflection_decision(text: str) -> str:
        """Return ``'sufficient'`` or ``'needs_more'`` from a reflection reply.

        Defaults to ``'sufficient'`` (i.e. terminate) when neither keyword is
        present, to avoid infinite loops on ambiguous reflections.
        """
        upper = (text or "").upper()
        if "NEEDS_MORE" in upper:
            return "needs_more"
        if "SUFFICIENT" in upper:
            return "sufficient"
        return "sufficient"

    @staticmethod
    def _summarize_tool_results(tool_results: list[dict]) -> str:
        if not tool_results:
            return "(no tool results yet)"
        lines: list[str] = []
        for r in tool_results:
            status = "ERROR" if r["is_error"] else "OK"
            obs = r["observation"]
            if len(obs) > 500:
                obs = obs[:500] + "..."
            lines.append(
                f"- [step {r['step']}] {r['name']}({r['arguments']}) "
                f"-> {status}: {obs}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_user_question(messages: Sequence[BaseMessage]) -> str:
        """Return the content of the first user message in ``messages``."""
        for m in messages:
            if isinstance(m, HumanMessage):
                return LangGraphAgent._extract_text(m)
        return ""

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
    def _build_agent_thought(step: int, response: AIMessage) -> str:
        """Build the thinking-chain entry, including tool-selection reasoning."""
        text = LangGraphAgent._extract_text(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            reasoning = text or "(no explicit reasoning provided)"
            selections = []
            for tc in tool_calls:
                name = LangGraphAgent._tc_field(tc, "name", "?")
                selections.append(f"chose {name} because {reasoning}")
            return f"[step {step}] Tool selection: " + "; ".join(selections)
        return f"[step {step}] Answer: {text}"

    @staticmethod
    def _tc_field(call: Any, key: str, default: Any = None) -> Any:
        """Read a field from a tool call that may be a dict or an object."""
        if isinstance(call, dict):
            return call.get(key, default)
        return getattr(call, key, default)


__all__ = ["AgentOutput", "AgentState", "LangGraphAgent"]
