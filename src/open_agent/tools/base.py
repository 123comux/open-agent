"""Abstract Tool base class and result schema.

A :class:`Tool` is a unit of capability the agent can invoke. Each tool
declares a JSON-schema ``parameters`` dict describing its inputs and implements
:meth:`execute` to perform the action and return a textual result.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Outcome of a tool execution.

    ``success`` indicates whether the tool ran without error. ``output`` holds
    the textual result; ``error`` is populated only when ``success`` is False.
    """

    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Abstract base class for agent tools.

    Subclasses must set the ``name``, ``description`` and ``parameters`` class
    attributes and implement :meth:`execute`.
    """

    name: str
    description: str
    parameters: dict

    @abstractmethod
    async def execute(self, **kwargs: object) -> str:
        """Run the tool with the given keyword arguments.

        Returns:
            The textual result of the tool (stdout, computed value, etc.).
        """
        raise NotImplementedError

    def to_schema(self) -> dict:
        """Return a JSON-schema dict describing this tool (for model binding)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
