"""Load MCP server configurations from JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from open_agent.mcp.client import MCPServerConfig


def load_mcp_servers(path: str | Path) -> list[MCPServerConfig]:
    """Load MCP server definitions from a JSON file.

    Expected schema::

        {
          "servers": [
            {
              "name": "filesystem",
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
              "env": {"optional": "value"},
              "cwd": "/optional/workdir"
            }
          ]
        }
    """
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    servers: list[MCPServerConfig] = []
    for item in data.get("servers", []):
        servers.append(
            MCPServerConfig(
                name=str(item["name"]),
                command=str(item["command"]),
                args=[str(a) for a in item.get("args", [])],
                env={str(k): str(v) for k, v in item.get("env", {}).items()}
                if item.get("env")
                else None,
                cwd=str(item["cwd"]) if item.get("cwd") else None,
            )
        )
    return servers
