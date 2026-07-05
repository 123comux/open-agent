"""Anthropic Claude chat provider.

Implements :class:`ModelInterface` by calling the Anthropic ``/v1/messages``
endpoint. System messages are extracted and sent in the top-level ``system``
field, per the Anthropic API contract. Tool use is mapped to/from Anthropic's
``tool_use``/``input_schema`` content blocks.
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


class AnthropicModel(ModelInterface):
    """Language model provider backed by the Anthropic API."""

    API_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
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
        system_parts: list[str] = []
        convo: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                convo.append({"role": m.role, "content": m.content})
        # Anthropic requires strictly alternating user/assistant turns. Merge
        # consecutive messages with the same role by joining their content
        # with a newline so we never violate that contract.
        merged: list[dict[str, Any]] = []
        for entry in convo:
            if merged and merged[-1]["role"] == entry["role"]:
                prev_content = merged[-1]["content"]
                merged[-1]["content"] = f"{prev_content}\n{entry['content']}"
            else:
                merged.append({"role": entry["role"], "content": entry["content"]})
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": merged,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        return payload

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        content_blocks = data.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )
        return ModelResponse(content="".join(text_parts), tool_calls=tool_calls)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        """Call the Anthropic ``/v1/messages`` endpoint."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload = self._build_payload(messages, tools)
        response = await request_with_retry(
            self._async_client, "POST", self.API_URL, headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion via Anthropic SSE, yielding text chunks.

        Anthropic streams Server-Sent Events; text tokens arrive on
        ``content_block_delta`` events whose ``delta.type`` is ``text_delta``.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
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
                    "POST", self.API_URL, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                text = delta.get("text")
                                if text:
                                    yield text
                                    yielded_any = True
                return
            except httpx.TransportError:
                if attempt >= 1 or yielded_any:
                    raise
