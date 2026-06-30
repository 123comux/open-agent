"""Anthropic Claude chat provider.

Implements :class:`ModelInterface` by calling the Anthropic ``/v1/messages``
endpoint. System messages are extracted and sent in the top-level ``system``
field, per the Anthropic API contract. Tool use is mapped to/from Anthropic's
``tool_use``/``input_schema`` content blocks.
"""
from __future__ import annotations

from typing import Any

import httpx

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
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        convo: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                convo.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": convo,
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return self._parse_response(data)
