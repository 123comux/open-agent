"""Adapter to use open-agent's ModelInterface with LangChain/LangGraph.

The :class:`LangChainModelAdapter` wraps any open-agent
:class:`~open_agent.models.base.ModelInterface` provider (OpenAI, Anthropic,
Ollama, ...) so it can be used wherever a LangChain
:class:`~langchain_core.language_models.chat_models.BaseChatModel` is expected
-- in particular by the LangGraph agent.

The adapter converts LangChain messages to our :class:`Message` format, LangChain
tool bindings to our :class:`ToolSchema`, and our :class:`ModelResponse` back to
a LangChain :class:`~langchain_core.outputs.ChatResult` (with
:class:`~langchain_core.messages.AIMessage` and tool-call metadata).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, List, Optional, Sequence

from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall as OAToolCall,
    ToolSchema,
)

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import BaseTool
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langchain-core is required for LangChainModelAdapter. "
        "Install it with: pip install langchain-core"
    ) from exc


# Maps LangChain message types to our normalized ``role`` strings.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


def _coerce_content(content: Any) -> str:
    """LangChain message content may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _lc_message_to_our_message(message: BaseMessage) -> Message:
    """Convert a LangChain :class:`BaseMessage` to our :class:`Message`."""
    role = _ROLE_MAP.get(message.type, message.type or "user")
    return Message(role=role, content=_coerce_content(message.content))


class LangChainModelAdapter(BaseChatModel):
    """Expose an open-agent :class:`ModelInterface` as a LangChain ``BaseChatModel``.

    The wrapped provider is stored on ``wrapped_model``. Tool bindings supplied
    via :meth:`bind_tools` are forwarded to the provider as
    :class:`ToolSchema` objects on each generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped_model: ModelInterface

    def __init__(self, model: ModelInterface, **kwargs: Any) -> None:
        super().__init__(wrapped_model=model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "open-agent-model"

    # ------------------------------------------------------------------ #
    # Tool conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_one_tool(tool: Any) -> ToolSchema:
        """Convert a LangChain tool / dict / open-agent tool to ``ToolSchema``."""
        # Our ToolAdapter attaches the original JSON-schema parameters.
        raw_params = getattr(tool, "raw_parameters", None)
        if raw_params:
            return ToolSchema(
                name=getattr(tool, "name", ""),
                description=getattr(tool, "description", ""),
                parameters=raw_params,
            )
        if isinstance(tool, BaseTool):
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                params: dict = args_schema.model_json_schema()
            else:
                params = {"type": "object", "properties": dict(tool.args)}
            return ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=params,
            )
        if isinstance(tool, dict):
            # OpenAI-style {"type": "function", "function": {...}} wrapper.
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                fn = tool["function"]
                return ToolSchema(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}) or {},
                )
            return ToolSchema(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {}) or {},
            )
        # Fall back to duck-typing our own ``Tool`` interface.
        return ToolSchema(
            name=getattr(tool, "name", ""),
            description=getattr(tool, "description", ""),
            parameters=getattr(tool, "parameters", {"type": "object", "properties": {}}),
        )

    def _convert_tools(
        self, tools: Optional[Sequence[Any]]
    ) -> Optional[list[ToolSchema]]:
        if not tools:
            return None
        return [self._convert_one_tool(tool) for tool in tools]

    # ------------------------------------------------------------------ #
    # Message / response conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_messages(messages: Sequence[BaseMessage]) -> list[Message]:
        return [_lc_message_to_our_message(m) for m in messages]

    def _build_result(self, response: ModelResponse) -> ChatResult:
        """Build a LangChain ``ChatResult`` from our :class:`ModelResponse`."""
        tool_calls: list[dict] = []
        for idx, call in enumerate(response.tool_calls):
            tool_calls.append(
                {
                    "name": call.name,
                    "args": call.arguments,
                    "id": f"call_{idx}",
                }
            )
        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync generation -- bridges to the async provider.

        Our :class:`ModelInterface` is async-only, so we run the coroutine on an
        event loop. If this is called from within a running loop (e.g. an async
        stack that fell back to the sync API) we execute the coroutine on a
        worker thread with its own loop to avoid blocking the caller's loop.
        """
        our_messages = self._convert_messages(messages)
        our_tools = self._convert_tools(kwargs.get("tools"))
        coro = self.wrapped_model.chat(our_messages, tools=our_tools)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(asyncio.run, coro).result()
        elif loop is not None:
            response = loop.run_until_complete(coro)
        else:
            response = asyncio.run(coro)
        return self._build_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation -- the primary path used by the LangGraph agent."""
        our_messages = self._convert_messages(messages)
        our_tools = self._convert_tools(kwargs.get("tools"))
        response = await self.wrapped_model.chat(our_messages, tools=our_tools)
        return self._build_result(response)

    # ------------------------------------------------------------------ #
    # Tool binding
    # ------------------------------------------------------------------ #

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Bind ``tools`` so they are passed to the provider on each call.

        Returns a ``Runnable`` (a ``RunnableBinding``) that forwards the tool
        list to :meth:`_generate` / :meth:`_agenerate` via ``kwargs``.
        """
        return self.bind(tools=list(tools), **kwargs)


__all__ = ["LangChainModelAdapter"]
