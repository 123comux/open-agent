"""Tests for MCP integration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_agent.mcp.adapter import MCPToolAdapter, adapt_mcp_tools
from open_agent.mcp.client import MCPServerConfig
from open_agent.mcp.loader import load_mcp_servers


@pytest.fixture
def servers_file(tmp_path: Path) -> Path:
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "fs",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                        "env": {"FOO": "bar"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_mcp_servers(servers_file: Path) -> None:
    servers = load_mcp_servers(servers_file)
    assert len(servers) == 1
    cfg = servers[0]
    assert isinstance(cfg, MCPServerConfig)
    assert cfg.name == "fs"
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "@modelcontextprotocol/server-filesystem", "."]
    assert cfg.env == {"FOO": "bar"}
    assert cfg.cwd is None


def test_load_mcp_servers_missing_file(tmp_path: Path) -> None:
    assert load_mcp_servers(tmp_path / "missing.json") == []


def test_mcp_tool_adapter() -> None:
    mcp_tool = MagicMock()
    mcp_tool.description = "Read a file"
    mcp_tool.inputSchema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    client = MagicMock()
    client.call_tool = AsyncMock(return_value="file contents")

    adapter = MCPToolAdapter("fs/read_file", mcp_tool, client)
    assert adapter.name == "fs/read_file"
    assert adapter.description == "Read a file"
    assert adapter.parameters == mcp_tool.inputSchema


def test_adapt_mcp_tools() -> None:
    mcp_tool = MagicMock()
    mcp_tool.description = "List files"
    mcp_tool.inputSchema = {"type": "object", "properties": {}}

    client = MagicMock()
    client.list_tools.return_value = {"fs/list_files": mcp_tool}

    tools = adapt_mcp_tools(client)
    assert len(tools) == 1
    assert tools[0].name == "fs/list_files"
