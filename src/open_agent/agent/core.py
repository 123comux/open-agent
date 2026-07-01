"""Agent core: the ReAct (Reason + Act) loop.

The :class:`Agent` orchestrates a language model, a tool registry and a
short-term memory. For each user request it builds a system prompt describing
the available tools, then loops: call the model -> if it requests a tool,
execute it and feed the observation back -> repeat until the model returns a
direct answer or ``max_steps`` is reached.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from open_agent.agent.executor import Observation, ToolExecutor
from open_agent.agent.planner import DirectResponse, Planner, ToolCall
from open_agent.memory.session_manager import SessionManager
from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message, ModelInterface, ToolSchema
from open_agent.tools.registry import ToolRegistry


class AgentOutput(BaseModel):
    """Final result of an agent run."""

    response: str
    steps: int
    tool_calls_made: list[dict] = Field(default_factory=list)


SYSTEM_PROMPT_TEMPLATE = """\
You are Open Agent, a general-purpose autonomous work assistant.
You solve the user's task by reasoning step by step and, when helpful, calling
the tools available to you. When you have enough information, respond directly
with the final answer.

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
        memory: ShortTermMemory | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.memory = memory or ShortTermMemory()
        self.session_manager = session_manager
        self.planner = Planner()
        self.executor = ToolExecutor(tool_registry)

    def _system_prompt(self) -> str:
        descriptions: list[str] = []
        for name in self.tool_registry.list_tools():
            tool = self.tool_registry.get(name)
            descriptions.append(
                f"- {tool.name}: {tool.description}\n  parameters: {tool.parameters}"
            )
        joined = "\n".join(descriptions) if descriptions else "(no tools registered)"
        return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=joined)

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

    async def run(self, user_input: str, session_id: str = "default") -> AgentOutput:
        """Run the ReAct loop for a single user request.

        Returns an :class:`AgentOutput` with the final response, the number of
        reasoning steps taken, and a log of tool calls made.
        """
        # Use session manager if available, otherwise fall back to single memory.
        if self.session_manager:
            history = self.session_manager.get_history(session_id)
        else:
            history = self.memory.get_history()

        messages: list[Message] = [Message(role="system", content=self._system_prompt())]
        # Load recent conversation history for continuity.
        messages.extend(history)
        messages.append(Message(role="user", content=user_input))

        tool_calls_made: list[dict] = []
        tool_schemas = self._tool_schemas()

        for step in range(1, self.max_steps + 1):
            response = await self.model.chat(messages, tools=tool_schemas)
            plan = self.planner.parse(response)

            if isinstance(plan, DirectResponse):
                final_text = plan.text or response.content
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
                return AgentOutput(
                    response=final_text,
                    steps=step,
                    tool_calls_made=tool_calls_made,
                )

            # ToolCall branch: record the assistant intent, then execute.
            call: ToolCall = plan
            messages.append(
                Message(
                    role="assistant",
                    content=response.content or f"Calling tool {call.name}.",
                )
            )
            observation: Observation = await self.executor.execute(
                call.name, call.arguments
            )
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

        # Exhausted steps: ask the model for a final summary based on context.
        messages.append(
            Message(
                role="user",
                content="Maximum steps reached. Provide your best final answer now.",
            )
        )
        final = await self.model.chat(messages, tools=None)
        final_text = final.content or (
            "I was unable to complete the task within the step limit."
        )
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
        return AgentOutput(
            response=final_text,
            steps=self.max_steps,
            tool_calls_made=tool_calls_made,
        )

    async def run_stream(
        self, user_input: str, session_id: str = "default"
    ) -> AsyncIterator[dict]:
        """Stream the agent run, yielding events as they occur.

        Yields dicts with a ``type`` field:

        - ``{'type': 'tool_start', 'name': ..., 'arguments': ...}``
        - ``{'type': 'tool_end', 'name': ..., 'observation': ..., 'is_error': bool}``
        - ``{'type': 'token', 'content': 'chunk'}``
        - ``{'type': 'done', 'response': ..., 'steps': ..., 'tool_calls_made': [...]}``

        The ReAct loop uses :meth:`chat` for tool-call detection. When the model
        returns a direct answer (or the step budget is exhausted), the final
        response is produced through :meth:`stream_chat` so its tokens can be
        streamed to the caller as they arrive.
        """
        # Use session manager if available, otherwise fall back to single memory.
        if self.session_manager:
            history = self.session_manager.get_history(session_id)
        else:
            history = self.memory.get_history()

        messages: list[Message] = [Message(role="system", content=self._system_prompt())]
        messages.extend(history)
        messages.append(Message(role="user", content=user_input))

        tool_calls_made: list[dict] = []
        tool_schemas = self._tool_schemas()

        for step in range(1, self.max_steps + 1):
            response = await self.model.chat(messages, tools=tool_schemas)
            plan = self.planner.parse(response)

            if isinstance(plan, DirectResponse):
                final_text = ""
                async for chunk in self.model.stream_chat(messages, tools=tool_schemas):
                    final_text += chunk
                    yield {"type": "token", "content": chunk}
                if not final_text:
                    final_text = plan.text or response.content
                    yield {"type": "token", "content": final_text}
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
                yield {
                    "type": "done",
                    "response": final_text,
                    "steps": step,
                    "tool_calls_made": tool_calls_made,
                }
                return

            # ToolCall branch: announce, execute, then feed observation back.
            call: ToolCall = plan
            yield {
                "type": "tool_start",
                "name": call.name,
                "arguments": call.arguments,
            }
            messages.append(
                Message(
                    role="assistant",
                    content=response.content or f"Calling tool {call.name}.",
                )
            )
            observation: Observation = await self.executor.execute(
                call.name, call.arguments
            )
            tool_calls_made.append(
                {
                    "step": step,
                    "name": call.name,
                    "arguments": call.arguments,
                    "observation": observation.text,
                    "is_error": observation.is_error,
                }
            )
            yield {
                "type": "tool_end",
                "name": call.name,
                "observation": observation.text,
                "is_error": observation.is_error,
            }
            messages.append(
                Message(role="user", content=f"Observation: {observation.text}")
            )

        # Exhausted steps: stream a final summary based on the context so far.
        messages.append(
            Message(
                role="user",
                content="Maximum steps reached. Provide your best final answer now.",
            )
        )
        final_text = ""
        async for chunk in self.model.stream_chat(messages, tools=None):
            final_text += chunk
            yield {"type": "token", "content": chunk}
        if not final_text:
            final_text = "I was unable to complete the task within the step limit."
            yield {"type": "token", "content": final_text}
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
        yield {
            "type": "done",
            "response": final_text,
            "steps": self.max_steps,
            "tool_calls_made": tool_calls_made,
        }
