"""Builtin tools shipped with open-agent.

Importing this package registers nothing automatically; consumers instantiate
the tools they want and register them on a :class:`ToolRegistry`.
"""
from __future__ import annotations

from open_agent.tools.builtin.file import FileTool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.builtin.shell import ShellTool
from open_agent.tools.builtin.web_search import WebSearchTool

__all__ = ["FileTool", "PythonTool", "ShellTool", "WebSearchTool"]
