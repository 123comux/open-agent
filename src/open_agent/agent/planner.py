"""Planner: decides whether a model response is a direct answer or a tool call.

The planner prefers structured ``tool_calls`` produced by the model provider.
When none are present it falls back to best-effort parsing of a JSON tool-call
block embedded in the text content, and otherwise treats the content as a
direct textual response.
"""
from __future__ import annotations

import json
import re
from typing import Union

from pydantic import BaseModel

from open_agent.models.base import ModelResponse, ToolCall


class DirectResponse(BaseModel):
    """The model produced a final textual answer."""

    text: str


# A parsed plan is either a direct response or a single tool call. When the
# model requests multiple tool calls, only the first is returned here; the
# remaining calls remain accessible via ``ModelResponse.tool_calls``.
ParsedPlan = Union[DirectResponse, ToolCall]


class Planner:
    """Parse :class:`ModelResponse` objects into actionable plans."""

    def parse(self, response: ModelResponse) -> ParsedPlan:
        """Return a :class:`DirectResponse` or :class:`ToolCall`.

        Preference order:
        1. Structured ``tool_calls`` on the response (preferred).
        2. A JSON tool-call block embedded in the content (best-effort).
        3. Otherwise, treat the content as a direct textual response.
        """
        if response.tool_calls:
            first = response.tool_calls[0]
            return ToolCall(name=first.name, arguments=first.arguments)

        parsed = self._parse_text(response.content)
        if parsed is not None:
            return parsed

        return DirectResponse(text=response.content)

    @staticmethod
    def _parse_text(content: str) -> ParsedPlan | None:
        """Attempt to extract a tool call from raw model text."""
        if not content:
            return None
        # Fenced JSON block: ```json\n{...}\n``` (or bare ```).
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fence_match:
            candidate = Planner._try_json(fence_match.group(1))
            if candidate is not None:
                return candidate
        # Bare JSON object on its own.
        stripped = content.strip()
        if stripped.startswith("{"):
            candidate = Planner._try_json(stripped)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _try_json(raw: str) -> ToolCall | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "name" in data:
            arguments = data.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            return ToolCall(name=str(data["name"]), arguments=arguments)
        return None
