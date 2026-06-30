"""Builtin file tool: read, write, and list files.

Exposes a single ``file`` tool whose ``action`` parameter selects between
``read``, ``write`` and ``list``. All filesystem I/O is performed off the event
loop via :func:`asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
import os

from open_agent.tools.base import Tool


class FileTool(Tool):
    """Perform filesystem read/write/list operations."""

    name = "file"
    description = (
        "Read from, write to, or list files on the local filesystem. "
        "Specify 'action' as 'read', 'write', or 'list'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list"],
                "description": "The file operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "File path for read/write, or directory for list.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (only for the 'write' action).",
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs: object) -> str:
        action = str(kwargs.get("action", "")).lower()
        path = str(kwargs.get("path", ""))
        if not action:
            return "Error: 'action' is required (read|write|list)."
        if not path:
            return "Error: 'path' is required."
        if action == "read":
            return await asyncio.to_thread(self._read, path)
        if action == "write":
            content = str(kwargs.get("content", ""))
            return await asyncio.to_thread(self._write, path, content)
        if action == "list":
            return await asyncio.to_thread(self._list, path)
        return f"Error: unknown action '{action}'."

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except OSError as exc:
            return f"Error reading file: {exc}"

    @staticmethod
    def _write(path: str, content: str) -> str:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return f"Wrote {len(content)} characters to {path}."
        except OSError as exc:
            return f"Error writing file: {exc}"

    @staticmethod
    def _list(path: str) -> str:
        try:
            entries = sorted(os.listdir(path))
        except FileNotFoundError:
            return f"Error: directory not found: {path}"
        except NotADirectoryError:
            return f"Error: not a directory: {path}"
        except OSError as exc:
            return f"Error listing directory: {exc}"
        if not entries:
            return f"(empty) {path}"
        return "\n".join(entries)
