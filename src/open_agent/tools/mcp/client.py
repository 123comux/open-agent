"""MCP client: connects to an MCP server via stdio or SSE transport.

Speaks JSON-RPC 2.0 over the chosen transport and exposes the basic MCP
methods (``initialize``, ``tools/list``, ``tools/call``) needed by the
:mod:`open_agent.tools.mcp.adapter` module.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

try:
    import mcp  # noqa: F401  -- capability gate; ensures the mcp extra is installed
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required for MCP client support. "
        "Install it with: pip install 'open-agent[mcp]'"
    ) from exc

import httpx


class MCPClient:
    """Client for a Model Context Protocol (MCP) server.

    Connect either over stdio (by passing ``command`` and optional ``args``)
    or over SSE/HTTP (by passing ``url``). The MCP protocol is JSON-RPC 2.0;
    this client implements the ``initialize``, ``tools/list`` and
    ``tools/call`` methods.
    """

    def __init__(
        self,
        command: str | None = None,
        url: str | None = None,
        args: list[str] | None = None,
    ) -> None:
        if not command and not url:
            raise ValueError("Either 'command' (stdio) or 'url' (SSE) must be provided.")
        if command and url:
            raise ValueError("Provide either 'command' or 'url', not both.")
        self._command = command
        self._url = url
        self._args = list(args) if args else []
        self._transport: str = "stdio" if command else "sse"
        self._process: asyncio.subprocess.Process | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._initialized = False

    async def connect(self) -> None:
        """Establish the transport connection and complete the MCP handshake."""
        if self._transport == "stdio":
            await self._connect_stdio()
        else:
            await self._connect_sse()
        await self._initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tools advertised by the MCP server."""
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke ``name`` on the MCP server with ``arguments``; return text output."""
        result = await self._send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        return self._extract_text(result)

    async def disconnect(self) -> None:
        """Close the transport connection."""
        if self._transport == "stdio":
            if self._process is not None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
                self._process = None
        else:
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None
        self._initialized = False

    # -- transport setup -------------------------------------------------

    async def _connect_stdio(self) -> None:
        assert self._command is not None
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _connect_sse(self) -> None:
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    # -- MCP handshake ---------------------------------------------------

    async def _initialize(self) -> None:
        """Send the ``initialize`` request and the ``initialized`` notification."""
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "open-agent-mcp-client", "version": "0.1.0"},
            },
        )
        await self._send_notification("notifications/initialized", {})
        self._initialized = True

    # -- JSON-RPC plumbing ----------------------------------------------

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        if self._transport == "stdio":
            return await self._send_request_stdio(request)
        return await self._send_request_sse(request)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if self._transport == "stdio":
            await self._send_notification_stdio(notification)
        else:
            await self._send_notification_sse(notification)

    async def _send_request_stdio(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        payload = json.dumps(request) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()
        response_line = await self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed the connection (stdio).")
        message = json.loads(response_line.decode())
        return self._unwrap(message)

    async def _send_notification_stdio(self, notification: dict[str, Any]) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        payload = json.dumps(notification) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

    async def _send_request_sse(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._http_client is not None
        assert self._url is not None
        async with self._http_client.stream("POST", self._url, json=request) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                async for line in response.aiter_lines():
                    message = self._parse_sse_line(line)
                    if message is None:
                        continue
                    if message.get("id") != request["id"]:
                        continue
                    return self._unwrap(message)
                raise RuntimeError("MCP server closed the SSE stream without responding.")
            body = await response.aread()
            message = json.loads(body.decode())
            return self._unwrap(message)

    async def _send_notification_sse(self, notification: dict[str, Any]) -> None:
        assert self._http_client is not None
        assert self._url is not None
        resp = await self._http_client.post(self._url, json=notification)
        resp.raise_for_status()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _parse_sse_line(line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data:"):
            data = line[5:].lstrip()
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _unwrap(message: dict[str, Any]) -> dict[str, Any]:
        if "error" in message:
            err = message["error"]
            raise RuntimeError(f"MCP error ({err.get('code')}): {err.get('message')}")
        result = message.get("result", {})
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _extract_text(result: dict[str, Any]) -> str:
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
        text = result.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(result)
