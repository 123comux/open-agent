"""OpenAI-compatible chat completion provider.

Implements :class:`ModelInterface` by calling any OpenAI-compatible
``/v1/chat/completions`` endpoint via ``httpx``. Works with the official OpenAI
API as well as self-hosted compatible servers (vLLM, LM Studio, etc.) by
pointing ``base_url`` at the right host.
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


class OpenAIModel(ModelInterface):
    """Language model provider backed by the OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def _async_client(self) -> httpx.AsyncClient:
        """Return a shared async client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=10, max_keepalive_connections=5
                ),
            )
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
        choices = data.get("choices") or []
        if not choices:
            return ModelResponse(content="")
        message = choices[0].get("message") or {}
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
        """Call the OpenAI-compatible ``/v1/chat/completions`` endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(messages, tools)
        response = await request_with_retry(
            self._async_client, "POST", url, headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion via SSE, yielding text chunks as they arrive."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(messages, tools)
        payload["stream"] = True
        # NOTE: Streaming bypasses ``request_with_retry`` because a full retry
        # would require buffering already-yielded chunks to resume cleanly. As
        # a partial mitigation we retry ONCE on a connection error
        # (``httpx.TransportError``) raised while opening the stream; HTTP
        # errors that occur mid-stream are surfaced immediately because the
        # response has already started.
        yielded_any = False
        for attempt in range(2):
            try:
                async with self._async_client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                                        yielded_any = True
                            except json.JSONDecodeError:
                                continue
                return
            except httpx.TransportError:
                if attempt >= 1 or yielded_any:
                    raise
