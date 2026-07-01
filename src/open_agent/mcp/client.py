"""MCP client for connecting to external MCP servers and exposing their tools.

Currently supports stdio-based MCP servers. Each server is started as a
subprocess and its tools are discovered via the MCP protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent, Tool


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection.

    Args:
        name: Logical name for this server (used for namespacing tools).
        command: Executable to run (e.g. ``"npx"``, ``"python"``).
        args: Command-line arguments for the executable.
        env: Optional environment variables for the subprocess.
        cwd: Optional working directory for the subprocess.
    """

    name: str
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None

    def to_stdio_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.command,
            args=list(self.args or []),
            env=self.env,
            cwd=self.cwd,
        )


class MCPClient:
    """Manage connections to one or more MCP servers.

    Usage::

        client = MCPClient([MCPServerConfig(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])])
        async with client:
            tools = await client.list_tools()
            result = await client.call_tool("fs/read_file", {"path": "README.md"})
    """

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self._servers = servers
        self._sessions: dict[str, ClientSession] = {}
        self._stdio_contexts: dict[str, Any] = {}
        self._tools: dict[str, Tool] = {}
        self._server_by_tool: dict[str, str] = {}

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Start all configured servers and initialize MCP sessions."""
        for cfg in self._servers:
            params = cfg.to_stdio_params()
            cm = stdio_client(params)
            read_stream, write_stream = await cm.__aenter__()
            self._stdio_contexts[cfg.name] = cm
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            self._sessions[cfg.name] = session

            result = await session.list_tools()
            for tool in result.tools:
                namespaced = f"{cfg.name}/{tool.name}"
                self._tools[namespaced] = tool
                self._server_by_tool[namespaced] = cfg.name

    async def close(self) -> None:
        """Close all MCP sessions and terminate subprocesses."""
        for session in self._sessions.values():
            await session.__aexit__(None, None, None)
        self._sessions.clear()
        for cm in self._stdio_contexts.values():
            await cm.__aexit__(None, None, None)
        self._stdio_contexts.clear()
        self._tools.clear()
        self._server_by_tool.clear()

    def list_tools(self) -> dict[str, Tool]:
        """Return all discovered tools keyed by ``server_name/tool_name``."""
        return dict(self._tools)

    async def call_tool(self, namespaced_name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return its text output.

        Raises:
            KeyError: If ``namespaced_name`` is not a known tool.
        """
        if namespaced_name not in self._tools:
            raise KeyError(f"Unknown MCP tool: {namespaced_name}")
        server_name = self._server_by_tool[namespaced_name]
        session = self._sessions[server_name]
        tool_name = namespaced_name.split("/", 1)[1]

        result: CallToolResult = await session.call_tool(tool_name, arguments)
        return self._format_result(result)

    @staticmethod
    def _format_result(result: CallToolResult) -> str:
        """Extract text from an MCP tool result, including error details."""
        if result.isError:
            parts = ["MCP tool error:"]
            for item in result.content:
                if isinstance(item, TextContent):
                    parts.append(item.text)
            return "\n".join(parts) if len(parts) > 1 else parts[0]

        texts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                texts.append(item.text)
        return "\n".join(texts) if texts else "(no text content)"
