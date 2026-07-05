"""Agent core: the ReAct (Reason + Act) loop.

The :class:`Agent` orchestrates a language model, a tool registry and a
short-term memory. For each user request it builds a system prompt describing
the available tools, then loops: call the model -> if it requests a tool,
execute it and feed the observation back -> repeat until the model returns a
direct answer or ``max_steps`` is reached.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from open_agent.agent.context_window import truncate_messages
from open_agent.agent.executor import Observation, ToolExecutor
from open_agent.agent.planner import DirectResponse, ParsedPlan, Planner, ToolCall
from open_agent.memory.long_term import LongTermMemory
from open_agent.memory.session_manager import SessionManager
from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message, ModelInterface, ModelResponse, ToolSchema
from open_agent.observability.tracer import NoOpTracer, Trace, Tracer, TraceSpan
from open_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentOutput(BaseModel):
    """Final result of an agent run."""

    response: str
    steps: int
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None


SYSTEM_PROMPT_TEMPLATE = """\
You are Open Agent, a general-purpose autonomous work assistant.
You solve the user's task by reasoning step by step and, when helpful, calling
the tools available to you. When you have enough information, respond directly
with the final answer.

Current date and time: {current_date}

Important:
- When searching for real-time or current information, do NOT invent a specific
  date. Use broad terms like "latest" or "today" unless the user explicitly
  provides a date.
- For web_search, write the query in the SAME language as the user's question.
  Do NOT mix languages (e.g., avoid "today news 2026"). For Chinese news use
  queries like "今日新闻 最新消息" or "2026年7月2日 新闻" (use the real current
  date from above, not a made-up one).
- Do NOT call file or shell tools unless the user explicitly asks for file or
  system operations.

Available tools:
{tool_descriptions}

When you want to use a tool, request it through the provided tool-calling
interface. After each tool call you will receive an observation; use it to
continue reasoning. If a tool fails, decide whether to retry, use another tool,
or answer the user.
"""


class Agent:
    """ReAct agent that orchestrates a model, tools, and memory."""

    def __init__(
        self,
        model: ModelInterface,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        max_context_tokens: int = 8000,
        memory: ShortTermMemory | None = None,
        session_manager: SessionManager | None = None,
        long_term_memory: LongTermMemory | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.max_context_tokens = max_context_tokens
        self.memory = memory or ShortTermMemory()
        self.session_manager = session_manager
        self.long_term_memory = long_term_memory
        self.tracer = tracer or NoOpTracer()
        self.planner = Planner()
        self.executor = ToolExecutor(tool_registry)

    def _apply_context_window(self, messages: list[Message]) -> list[Message]:
        """Truncate ``messages`` to fit within ``max_context_tokens``.

        Applied before every LLM call so the conversation never exceeds
        the configured context window. Returns a new list; the input is
        not mutated.
        """
        if self.max_context_tokens <= 0:
            return messages
        truncated = truncate_messages(
            [m.model_dump() for m in messages],
            max_tokens=self.max_context_tokens,
            preserve_system=True,
            preserve_recent=2,
            model=str(getattr(self.model, "model", "") or ""),
        )
        return [Message(**d) for d in truncated]

    def _system_prompt(self) -> str:
        descriptions: list[str] = []
        for name in self.tool_registry.list_tools():
            tool = self.tool_registry.get(name)
            descriptions.append(
                f"- {tool.name}: {tool.description}\n  parameters: {tool.parameters}"
            )
        joined = "\n".join(descriptions) if descriptions else "(no tools registered)"
        current_date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=joined, current_date=current_date
        )

    async def _recall_long_term_memory(self, user_input: str) -> list[Message]:
        """Retrieve relevant long-term memories for the user input."""
        if not self.long_term_memory or not user_input.strip():
            return []
        try:
            entries = await self.long_term_memory.search(user_input)
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("long-term memory recall failed: %s", exc, exc_info=True)
            return []
        if not entries:
            return []
        fragments = []
        for entry in entries:
            fragments.append(f"- {entry.text.replace(chr(10), ' ')}")
        content = (
            "The following memories from previous conversations may be relevant.\n"
            "Use them only if they help answer the user's current request:\n"
            + "\n".join(fragments)
        )
        return [Message(role="system", content=content)]

    async def _remember_exchange(
        self, user_input: str, final_text: str, session_id: str
    ) -> None:
        """Persist the current exchange to long-term memory if configured."""
        if not self.long_term_memory:
            return
        try:
            await self.long_term_memory.add_exchange(
                user_input=user_input,
                assistant_response=final_text,
                session_id=session_id,
            )
        except Exception as exc:  # pragma: no cover - optional backend
            logger.debug("long-term memory remember exchange failed: %s", exc, exc_info=True)

    def _tool_schemas(self) -> list[ToolSchema]:
        schemas: list[ToolSchema] = []
        for name in self.tool_registry.list_tools():
            tool = self.tool_registry.get(name)
            schemas.append(
                ToolSchema(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return schemas

    async def _prepare_run(
        self, user_input: str, session_id: str, *, trace_name: str = "agent.run"
    ) -> tuple[Trace, TraceSpan, list[Message], list[ToolSchema]]:
        """Initialize trace, assemble messages, and build tool schemas."""
        trace = self.tracer.start_trace(
            name=trace_name,
            input_data={"user_input": user_input, "session_id": session_id},
        )
        root = self.tracer.start_span(
            trace, None, "agent", "react_loop", input_data={"user_input": user_input}
        )
        trace.root_span = root

        if self.session_manager:
            history = self.session_manager.get_history(session_id)
        else:
            history = self.memory.get_history()

        messages: list[Message] = [Message(role="system", content=self._system_prompt())]
        messages.extend(await self._recall_long_term_memory(user_input))
        messages.extend(history)
        messages.append(Message(role="user", content=user_input))
        return trace, root, messages, self._tool_schemas()

    async def _llm_step(
        self,
        trace: Trace,
        root: TraceSpan,
        step: int,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
    ) -> tuple[ModelResponse, ParsedPlan]:
        """Call the model for one reasoning step, with tracing, and parse the plan."""
        llm_start = time.monotonic()
        llm_span = self.tracer.start_span(
            trace,
            root,
            "llm",
            f"step_{step}.llm",
            input_data={
                "messages": [m.model_dump() for m in messages],
                "tools": [t.model_dump() for t in tool_schemas],
            },
        )
        response = await self.model.chat(messages, tools=tool_schemas)
        llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
        self.tracer.end_span(
            llm_span,
            output_data=response.model_dump(),
            metrics={"latency_ms": llm_latency_ms},
        )
        root.children.append(llm_span)
        plan = self.planner.parse(response)
        return response, plan
    async def _execute_tool_call(
        self,
        trace: Trace,
        root: TraceSpan,
        step: int,
        call: ToolCall,
        response: ModelResponse,
        messages: list[Message],
        tool_calls_made: list[dict[str, Any]],
    ) -> Observation:
        """Append assistant intent, execute tool, record, and feed observation back."""
        messages.append(
            Message(
                role="assistant",
                content=response.content or f"Calling tool {call.name}.",
            )
        )
        tool_start = time.monotonic()
        tool_span = self.tracer.start_span(
            trace, root, "tool", call.name, input_data=call.arguments
        )
        observation: Observation = await self.executor.execute(
            call.name, call.arguments
        )
        tool_latency_ms = int((time.monotonic() - tool_start) * 1000)
        self.tracer.end_span(
            tool_span,
            output_data={"observation": observation.text},
            metrics={"latency_ms": tool_latency_ms, "is_error": observation.is_error},
            status="error" if observation.is_error else "ok",
        )
        root.children.append(tool_span)
        tool_calls_made.append(
            {
                "step": step,
                "name": call.name,
                "arguments": call.arguments,
                "observation": observation.text,
                "is_error": observation.is_error,
            }
        )
        messages.append(
            Message(role="user", content=f"Observation: {observation.text}")
        )
        return observation

    def _persist_exchange(
        self, user_input: str, final_text: str, session_id: str
    ) -> None:
        """Record the user/assistant exchange in session or short-term memory."""
        if self.session_manager:
            self.session_manager.add_message(
                session_id, Message(role="user", content=user_input)
            )
            self.session_manager.add_message(
                session_id, Message(role="assistant", content=final_text)
            )
        else:
            self.memory.add(Message(role="user", content=user_input))
            self.memory.add(Message(role="assistant", content=final_text))

    def _end_exhausted_trace(
        self,
        trace: Trace,
        root: TraceSpan,
        final_text: str,
        tool_calls_made: list[dict[str, Any]],
    ) -> None:
        """Finalize trace when the step budget is exhausted."""
        self.tracer.end_span(
            root,
            output_data={"response": final_text, "steps": self.max_steps},
            status="ok" if tool_calls_made else "incomplete",
        )
        self.tracer.end_trace(
            trace,
            output_data={"response": final_text, "steps": self.max_steps},
        )
    async def run(self, user_input: str, session_id: str = "default") -> AgentOutput:
        """Run the ReAct loop for a single user request.

        Returns an AgentOutput with the final response, the number of reasoning
        steps taken, and a log of tool calls made.
        """
        trace, root, messages, tool_schemas = await self._prepare_run(
            user_input, session_id
        )

        tool_calls_made: list[dict[str, Any]] = []
        trace_ended = False
        try:
            for step in range(1, self.max_steps + 1):
                messages = self._apply_context_window(messages)
                response, plan = await self._llm_step(
                    trace, root, step, messages, tool_schemas
                )

                if isinstance(plan, DirectResponse):
                    final_text = plan.text or response.content
                    self._persist_exchange(user_input, final_text, session_id)
                    await self._remember_exchange(user_input, final_text, session_id)
                    self.tracer.end_span(
                        root, output_data={"response": final_text, "steps": step}
                    )
                    self.tracer.end_trace(
                        trace, output_data={"response": final_text, "steps": step}
                    )
                    trace_ended = True
                    return AgentOutput(
                        response=final_text,
                        steps=step,
                        tool_calls_made=tool_calls_made,
                        trace_id=trace.id,
                    )

                call: ToolCall = plan
                await self._execute_tool_call(
                    trace, root, step, call, response, messages, tool_calls_made
                )

            # Exhausted steps: ask the model for a final summary based on context.
            messages.append(
                Message(
                    role="user",
                    content="Maximum steps reached. Provide your best final answer now.",
                )
            )
            messages = self._apply_context_window(messages)
            final = await self.model.chat(messages, tools=None)
            final_text = final.content or (
                "I was unable to complete the task within the step limit."
            )
            self._persist_exchange(user_input, final_text, session_id)
            await self._remember_exchange(user_input, final_text, session_id)
            self._end_exhausted_trace(trace, root, final_text, tool_calls_made)
            trace_ended = True
            return AgentOutput(
                response=final_text,
                steps=self.max_steps,
                tool_calls_made=tool_calls_made,
                trace_id=trace.id,
            )
        finally:
            if not trace_ended:
                self.tracer.end_span(root, status="error")
                self.tracer.end_trace(trace)

    async def run_stream(
        self, user_input: str, session_id: str = "default"
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the agent run, yielding events as they occur.

        Yields dicts with a ``type`` field:

        - ``{"type": "thought", "content": ..., "step": int}``
        - ``{"type": "tool_start", "name": ..., "arguments": ...}``
        - ``{"type": "tool_end", "name": ..., "observation": ..., "is_error": bool}``
        - ``{"type": "token", "content": "chunk"}``
        - ``{"type": "done", "response": ..., "steps": ...,
          "tool_calls_made": [...], "trace_id": ...}``

        The ReAct loop uses :meth:`chat` for tool-call detection. When the model
        returns a direct answer (or the step budget is exhausted), the final
        response is produced through :meth:`stream_chat` so its tokens can be
        streamed to the caller as they arrive.
        """
        trace, root, messages, tool_schemas = await self._prepare_run(
            user_input, session_id, trace_name="agent.run_stream"
        )

        tool_calls_made: list[dict[str, Any]] = []
        trace_ended = False
        try:
            for step in range(1, self.max_steps + 1):
                messages = self._apply_context_window(messages)
                response, plan = await self._llm_step(
                    trace, root, step, messages, tool_schemas
                )

                if isinstance(plan, DirectResponse):
                    final_text = ""
                    stream_span = self.tracer.start_span(
                        trace, root, "llm_stream", "final_stream"
                    )
                    stream_start = time.monotonic()
                    async for chunk in self.model.stream_chat(messages, tools=tool_schemas):
                        final_text += chunk
                        yield {"type": "token", "content": chunk}
                    stream_latency_ms = int((time.monotonic() - stream_start) * 1000)
                    if not final_text:
                        final_text = plan.text or response.content
                        yield {"type": "token", "content": final_text}
                    self.tracer.end_span(
                        stream_span,
                        output_data={"response": final_text},
                        metrics={"latency_ms": stream_latency_ms},
                    )
                    root.children.append(stream_span)
                    self._persist_exchange(user_input, final_text, session_id)
                    await self._remember_exchange(user_input, final_text, session_id)
                    self.tracer.end_span(
                        root, output_data={"response": final_text, "steps": step}
                    )
                    self.tracer.end_trace(
                        trace, output_data={"response": final_text, "steps": step}
                    )
                    trace_ended = True
                    yield {
                        "type": "done",
                        "response": final_text,
                        "steps": step,
                        "tool_calls_made": tool_calls_made,
                        "trace_id": trace.id,
                    }
                    return
                call: ToolCall = plan
                yield {
                    "type": "thought",
                    "content": f"Step {step}: Deciding to use {call.name} to gather information",
                    "step": step,
                }
                yield {
                    "type": "tool_start",
                    "name": call.name,
                    "arguments": call.arguments,
                }
                observation = await self._execute_tool_call(
                    trace, root, step, call, response, messages, tool_calls_made
                )
                yield {
                    "type": "tool_end",
                    "name": call.name,
                    "observation": observation.text,
                    "is_error": observation.is_error,
                }
                yield {
                    "type": "thought",
                    "content": f"Step {step}: Received result from {call.name}, analyzing...",
                    "step": step,
                }

            # Exhausted steps: stream a final summary based on the context so far.
            messages.append(
                Message(
                    role="user",
                    content="Maximum steps reached. Provide your best final answer now.",
                )
            )
            final_text = ""
            messages = self._apply_context_window(messages)
            async for chunk in self.model.stream_chat(messages, tools=None):
                final_text += chunk
                yield {"type": "token", "content": chunk}
            if not final_text:
                final_text = "I was unable to complete the task within the step limit."
                yield {"type": "token", "content": final_text}
            self._persist_exchange(user_input, final_text, session_id)
            await self._remember_exchange(user_input, final_text, session_id)
            self._end_exhausted_trace(trace, root, final_text, tool_calls_made)
            trace_ended = True
            yield {
                "type": "done",
                "response": final_text,
                "steps": self.max_steps,
                "tool_calls_made": tool_calls_made,
                "trace_id": trace.id,
            }
        finally:
            if not trace_ended:
                self.tracer.end_span(root, status="error")
                self.tracer.end_trace(trace)
