"""Ollama local model provider.

Implements :class:`ModelInterface` by calling a local Ollama server's
``/api/chat`` endpoint. Streaming is disabled so a single JSON response is
returned, which keeps the interface consistent with the other providers.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from open_agent.models._http import request_with_retry
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolSchema,
)


class OllamaModel(ModelInterface):
    """Language model provider backed by a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Optional auth headers for remote Ollama-compatible endpoints that
        # require a bearer token / API key (e.g. a hosted gateway). Empty for
        # the default localhost deployment.
        self.headers: dict[str, str] = dict(headers) if headers else {}
        self._client: httpx.AsyncClient | None = None

    @property
    def _async_client(self) -> httpx.AsyncClient:
        """Return a shared async client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        return payload

    @staticmethod
    def _parse_arguments(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {"_raw": raw_args}
            except json.JSONDecodeError:
                return {"_raw": raw_args}
        if isinstance(raw_args, dict):
            return raw_args
        return {}

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        message = data.get("message") or {}
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            func = raw.get("function") or {}
            tool_calls.append(
                ToolCall(
                    name=func.get("name") or "",
                    arguments=self._parse_arguments(func.get("arguments")),
                )
            )
        return ModelResponse(content=content, tool_calls=tool_calls)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        """Call the local Ollama ``/api/chat`` endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(messages, tools)
        response = await request_with_retry(
            self._async_client,
            "POST",
            url,
            headers=self.headers or None,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion via Ollama's newline-delimited JSON.

        Ollama returns one JSON object per line when ``stream`` is true; each
        object carries a ``message.content`` chunk. The final object has
        ``done`` set to true.
        """
        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(messages, tools)
        payload["stream"] = True
        # NOTE: Streaming bypasses ``request_with_retry`` because a full retry
        # would require buffering already-yielded chunks to resume cleanly. As
        # a partial mitigation we retry ONCE on a connection error
        # (``httpx.TransportError``) raised while opening the stream; HTTP
        # errors that occur mid-stream are surfaced immediately because the
        # response has already started.
        for attempt in range(2):
            try:
                async with self._async_client.stream(
                    "POST", url, headers=self.headers or None, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        message = data.get("message") or {}
                        content = message.get("content")
                        if content:
                            yield content
                        if data.get("done"):
                            break
                return
            except httpx.TransportError:
                if attempt >= 1:
                    raise
