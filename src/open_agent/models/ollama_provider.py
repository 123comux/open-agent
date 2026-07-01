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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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
    def _parse_arguments(raw_args: Any) -> dict:
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
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
