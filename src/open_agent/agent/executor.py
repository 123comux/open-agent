"""ToolExecutor: routes parsed tool calls to the registry and returns observations.

Each execution runs in its own exception domain: any failure (missing tool,
bad arguments, runtime error) is converted into an error :class:`Observation`
so the ReAct loop can keep going and let the model decide how to recover.
"""
from __future__ import annotations

from open_agent.tools.registry import ToolRegistry


class Observation:
    """A string wrapper representing the result of a tool execution.

    Carrying a dedicated type keeps the ReAct message history readable and lets
    callers distinguish error observations from successful ones.
    """

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.text = text
        self.is_error = is_error

    def __str__(self) -> str:
        return self.text


class ToolExecutor:
    """Execute tool calls via a :class:`ToolRegistry`."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(self, name: str, arguments: dict) -> Observation:
        """Execute the named tool with ``arguments`` and return an Observation.

        Any exception raised during lookup or execution is converted into an
        error observation instead of being propagated.
        """
        result = await self.registry.execute(name, **arguments)
        if result.success:
            return Observation(result.output)
        return Observation(result.error or "Unknown error", is_error=True)
