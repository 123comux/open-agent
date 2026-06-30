"""Tool registry: registers tools by name and dispatches execution.

The :class:`ToolRegistry` is the central directory of tools available to an
agent. Tools are registered under a unique name and looked up by that name when
the agent requests an execution. Execution errors are captured and converted
into :class:`ToolResult` objects so a single failing tool never crashes the
agent loop.
"""
from __future__ import annotations

from open_agent.tools.base import Tool, ToolResult


class ToolRegistry:
    """Registry mapping tool names to :class:`Tool` instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance under its ``name``.

        Re-registering a name overwrites the previous tool.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """Return JSON-schema descriptions for all registered tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: object) -> ToolResult:
        """Execute the named tool with ``kwargs`` and return a :class:`ToolResult`.

        Both lookup failures and execution exceptions are captured and returned
        as unsuccessful :class:`ToolResult` objects rather than raised.
        """
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        try:
            output = await tool.execute(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
            )
