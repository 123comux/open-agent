"""MCP integration: connect an MCP server and use its tools through the agent.

This example shows how to:
  * Load MCP server definitions from a JSON file.
  * Connect with :class:`MCPClient` and adapt MCP tools to Open Agent tools.
  * Register the adapted tools in a :class:`ToolRegistry`.

A real MCP server is required to actually run this example. The snippet below
uses a filesystem server as an illustration; adjust the command/args to match a
server you have installed.

Run with:  python examples/mcp_integration.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.mcp import adapt_mcp_tools, load_mcp_servers
from open_agent.mcp.client import MCPClient
from open_agent.tools.registry import ToolRegistry


def create_sample_config(path: Path) -> None:
    """Write a sample MCP servers config for the example."""
    config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd())],
        }
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


async def main() -> None:
    # 1. Create a temporary MCP servers config.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        config_path = Path(tmp.name)
    create_sample_config(config_path)
    print(f"Sample MCP config written to: {config_path}")

    # 2. Load servers and connect.
    servers = load_mcp_servers(str(config_path))
    if not servers:
        print("No MCP servers configured.")
        return

    client = MCPClient(servers)
    await client.connect()

    try:
        # 3. Adapt MCP tools and register them.
        registry = ToolRegistry()
        for tool in adapt_mcp_tools(client):
            registry.register(tool)

        print("Registered MCP tools:", registry.list_tools())

        # 4. Optionally call an MCP tool directly.
        if "filesystem/read_file" in registry.list_tools():
            result = await registry.execute(
                "filesystem/read_file", {"path": "README.md"}
            )
            print("\n=== Read README via MCP ===")
            print(result[:500])
    finally:
        await client.disconnect()
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
