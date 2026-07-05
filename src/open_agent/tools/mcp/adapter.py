"""MCP tool adapter: wraps MCP server tools as internal Tool instances.

The :class:`MCPToolAdapter` bridges a tool exposed by an MCP server into the
:class:`open_agent.tools.base.Tool` interface so it can be registered in a
:class:`ToolRegistry` and dispatched like any builtin tool. The
:class:`MCPToolRegistry` manages the lifecycle of multiple MCP servers and
their adapted tools.
"""
from __future__ import annotations

from typing import Any

from open_agent.tools.base import Tool
from open_agent.tools.mcp.client import MCPClient
from open_agent.tools.registry import ToolRegistry


class MCPToolAdapter(Tool):
    """Adapt an MCP server tool to the internal :class:`Tool` interface.

    The adapter stores a reference to the owning :class:`MCPClient` and
    delegates :meth:`execute` calls to :meth:`MCPClient.call_tool`.
    """

    def __init__(
        self,
        client: MCPClient,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        server_name: str = "",
    ) -> None:
        self._client = client
        # Retain the original tool name for forwarding calls to the MCP server.
        self._original_name = tool_name
        # Prefix the registered name with the server name to avoid collisions
        # when multiple MCP servers expose tools with the same name.
        self.name = f"{server_name}__{tool_name}" if server_name else tool_name
        self.description = tool_description
        self.parameters = tool_schema

    async def execute(self, **kwargs: object) -> str:
        """Delegate execution to the MCP server via the owning client."""
        return await self._client.call_tool(self._original_name, kwargs)


class MCPToolRegistry:
    """Manage multiple MCP clients and their adapted tools.

    Servers are connected with :meth:`connect_server`, which discovers the
    server's tools and wraps each one in an :class:`MCPToolAdapter`. The
    collected tools can then be inspected via :meth:`get_tools` or pushed into
    a :class:`ToolRegistry` with :meth:`register_into`.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._tools: list[MCPToolAdapter] = []

    async def connect_server(self, name: str, **config: Any) -> None:
        """Connect to an MCP server and register all the tools it exposes.

        ``config`` is forwarded to the :class:`MCPClient` constructor and must
        contain either ``command`` (stdio) or ``url`` (SSE), optionally with
        ``args``.

        If the server connects but tool discovery fails partway, the client is
        disconnected before re-raising so no connection is leaked.
        """
        client = MCPClient(**config)
        try:
            await client.connect()
            self._clients[name] = client
            for tool in await client.list_tools():
                adapter = MCPToolAdapter(
                    client=client,
                    tool_name=tool.get("name", ""),
                    tool_description=tool.get("description", ""),
                    tool_schema=tool.get("inputSchema", {}),
                    server_name=name,
                )
                self._tools.append(adapter)
        except Exception:
            # Tear down the partially-connected client so a failed
            # connect_server does not leak a subprocess or HTTP connection.
            try:
                await client.disconnect()
            except Exception:
                pass
            # Remove the client if it was registered before the failure.
            self._clients.pop(name, None)
            raise

    async def disconnect_all(self) -> None:
        """Disconnect every connected MCP server and clear the tool list.

        Each client is disconnected in its own try/except so one failing
        disconnect cannot skip the remaining ones.
        """
        for client in list(self._clients.values()):
            try:
                await client.disconnect()
            except Exception:
                pass
        self._clients.clear()
        self._tools.clear()

    def get_tools(self) -> list[MCPToolAdapter]:
        """Return all adapted tools collected from connected servers."""
        return list(self._tools)

    def register_into(self, registry: ToolRegistry) -> None:
        """Register every adapted tool into the given :class:`ToolRegistry`."""
        for tool in self._tools:
            registry.register(tool)
