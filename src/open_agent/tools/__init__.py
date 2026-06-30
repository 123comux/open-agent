"""Tools package: tool base class, registry, and builtin tools.

Importing this package only loads the abstract :class:`Tool` and the
:class:`ToolRegistry`; builtin tools live in :mod:`open_agent.tools.builtin`
and are imported explicitly by consumers (CLI/server) to keep the core light.
"""
from __future__ import annotations

from open_agent.tools.base import Tool, ToolResult
from open_agent.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolResult", "ToolRegistry"]
