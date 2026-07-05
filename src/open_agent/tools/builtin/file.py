"""Builtin file tool: read, write, and list files.

Exposes a single ``file`` tool whose ``action`` parameter selects between
``read``, ``write`` and ``list``. All filesystem I/O is performed off the event
loop via :func:`asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from open_agent.tools.base import Tool
from open_agent.tools.sandbox import check_path


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
        blocked = check_path(path)
        if blocked:
            return blocked
        if action == "read":
            return await asyncio.to_thread(self._read, path)
        if action == "write":
            content = kwargs.get("content")
            if content is None:
                return "Error: 'content' is required for write action."
            return await asyncio.to_thread(self._write, path, str(content))
        if action == "list":
            return await asyncio.to_thread(self._list, path)
        return f"Error: unknown action '{action}'."

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except OSError as exc:
            return f"Error reading file: {exc}"

    @staticmethod
    def _write(path: str, content: str) -> str:
        try:
            # Atomic write: write to a temp file in the same directory then
            # os.replace onto the target. A crash mid-write leaves the temp
            # file (not the target) partially written, so the target is never
            # truncated. os.replace is atomic on both POSIX and Windows.
            target = Path(path)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, target)
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
