"""Abstract model interface and shared pydantic schemas.

Defines the data structures used to communicate with language model providers
and the :class:`ModelInterface` abstract base class that all providers
implement. Keeping a single normalized interface lets the agent swap providers
without touching the core loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message in a conversation."""

    role: str
    content: str


class ToolSchema(BaseModel):
    """JSON-schema description of a tool exposed to the model."""

    name: str
    description: str
    parameters: dict


class ToolCall(BaseModel):
    """A parsed tool call requested by the model."""

    name: str
    arguments: dict


class ModelResponse(BaseModel):
    """Normalized response returned by a model provider.

    ``content`` holds any free-form text the model produced. ``tool_calls``
    lists structured tool invations the model requested (may be empty).
    """

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelInterface(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        """Generate a chat completion for the given messages.

        Args:
            messages: The conversation so far, oldest first.
            tools: Optional tool schemas the model is allowed to call.

        Returns:
            A normalized :class:`ModelResponse`.
        """
        raise NotImplementedError
