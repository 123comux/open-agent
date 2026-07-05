"""MCP (Model Context Protocol) integration for Open Agent.

Allows Open Agent to discover and call tools exposed by external MCP servers.
"""
from __future__ import annotations

try:
    from open_agent.mcp.adapter import MCPToolAdapter, adapt_mcp_tools
    from open_agent.mcp.client import MCPClient, MCPServerConfig
    from open_agent.mcp.loader import load_mcp_servers
except ImportError:  # pragma: no cover - optional mcp dependency
    MCPToolAdapter = None  # type: ignore[assignment, misc]
    adapt_mcp_tools = None  # type: ignore[assignment]
    MCPClient = None  # type: ignore[assignment, misc]
    MCPServerConfig = None  # type: ignore[assignment, misc]
    load_mcp_servers = None  # type: ignore[assignment]

__all__ = ["MCPClient", "MCPServerConfig", "MCPToolAdapter", "adapt_mcp_tools", "load_mcp_servers"]
