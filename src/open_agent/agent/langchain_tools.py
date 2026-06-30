"""Adapt open-agent's built-in tools to LangChain Tool interface.

A :class:`~open_agent.tools.base.Tool` is the project's own async tool abstraction.
LangGraph / LangChain expect :class:`~langchain_core.tools.BaseTool` objects, so
:func:`to_langchain_tool` wraps each tool while preserving its ``name``,
``description`` and JSON-schema ``parameters``. The original parameters schema
is retained on the adapter as ``raw_parameters`` so the model-facing schema stays
faithful (enums, defaults, required fields) rather than being lossily rebuilt
from a pydantic model.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr, create_model

from open_agent.tools.base import Tool
from open_agent.tools.registry import ToolRegistry

try:
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langchain-core is required for the LangChain tool adapters. "
        "Install it with: pip install langchain-core"
    ) from exc


_JSON_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_type_to_py_type(spec: Any) -> Any:
    """Map a JSON-schema property spec to a Python type."""
    if not isinstance(spec, dict):
        return str
    json_type = spec.get("type", "string")
    # Nullable types like ["string", "null"]: pick the first non-null member.
    if isinstance(json_type, list):
        json_type = next((t for t in json_type if t != "null"), "string")
    return _JSON_TYPE_MAP.get(json_type, str)


def _parameters_to_pydantic(tool: Tool) -> Type[BaseModel]:
    """Build a pydantic args model from the tool's JSON-schema ``parameters``.

    Used as the LangChain ``args_schema`` for input validation when the tool is
    invoked. The original schema is also kept verbatim on ``raw_parameters`` so
    the model-facing description stays lossless.
    """
    params = tool.parameters or {}
    properties = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])
    fields: dict[str, Any] = {}
    for name, spec in properties.items():
        spec_dict = spec if isinstance(spec, dict) else {}
        py_type = _json_type_to_py_type(spec_dict)
        description = spec_dict.get("description")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            default = spec_dict.get("default")
            fields[name] = (py_type, Field(default=default, description=description))
    model_name = f"{tool.name.replace('-', '_').title()}Args"
    return create_model(model_name, **fields) if fields else create_model(model_name)


class ToolAdapter(BaseTool):
    """A LangChain ``BaseTool`` that delegates execution to an open-agent ``Tool``.

    ``raw_parameters`` holds the tool's original JSON-schema ``parameters`` dict
    so downstream consumers (e.g. :class:`LangChainModelAdapter`) can expose the
    exact schema to the model without rebuilding it from the pydantic model.
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    raw_parameters: dict = Field(default_factory=dict)

    _oa_tool: Tool = PrivateAttr()

    def __init__(self, tool: Tool, **kwargs: Any) -> None:
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=_parameters_to_pydantic(tool),
            raw_parameters=dict(tool.parameters or {}),
            **kwargs,
        )
        self._oa_tool = tool

    def _run(self, **kwargs: Any) -> str:
        """Sync execution -- bridges to the underlying async ``Tool.execute``."""
        coro = self._oa_tool.execute(**kwargs)
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        if running:
            # Called from within a running event loop; run the coroutine on a
            # worker thread with its own loop to avoid blocking the caller.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    async def _arun(self, **kwargs: Any) -> str:
        """Async execution -- the primary path used by the LangGraph agent."""
        return await self._oa_tool.execute(**kwargs)


def to_langchain_tool(tool: Tool) -> BaseTool:
    """Wrap an open-agent :class:`Tool` as a LangChain :class:`BaseTool`."""
    return ToolAdapter(tool)


def to_langchain_tools(registry: ToolRegistry) -> list[BaseTool]:
    """Convert every tool in ``registry`` to a LangChain :class:`BaseTool`."""
    return [to_langchain_tool(registry.get(name)) for name in registry.list_tools()]


__all__ = ["ToolAdapter", "to_langchain_tool", "to_langchain_tools"]
