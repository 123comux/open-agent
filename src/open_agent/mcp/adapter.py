"""Adapter that wraps MCP tools as Open Agent ``Tool`` instances."""
from __future__ import annotations

from typing import Any

from mcp.types import Tool as MCPTool

from open_agent.mcp.client import MCPClient
from open_agent.tools.base import Tool


class MCPToolAdapter(Tool):
    """Wrap an MCP tool so it can be registered in Open Agent's ToolRegistry.

    The tool name is namespaced as ``server_name/tool_name`` to avoid clashes
    across multiple MCP servers or with built-in tools.
    """

    def __init__(self, namespaced_name: str, mcp_tool: MCPTool, client: MCPClient) -> None:
        self.name = namespaced_name
        self.description = mcp_tool.description or f"MCP tool {namespaced_name}"
        self.parameters = mcp_tool.inputSchema or {
            "type": "object",
            "properties": {},
        }
        self._client = client
        self._raw_tool = mcp_tool

    async def execute(self, **kwargs: object) -> str:
        return await self._client.call_tool(self.name, kwargs)


def adapt_mcp_tools(client: MCPClient) -> list[Tool]:
    """Convert all tools discovered by ``client`` into Open Agent tools."""
    return [
        MCPToolAdapter(name, tool, client)
        for name, tool in client.list_tools().items()
    ]
