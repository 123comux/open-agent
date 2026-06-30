"""Builtin shell tool: executes shell commands via asyncio subprocess.

The command string is split with :func:`shlex.split` and executed with
:func:`asyncio.create_subprocess_exec` (no intermediate shell). A per-call
timeout kills the process if it exceeds the limit. Combined stdout and stderr
are returned.
"""
from __future__ import annotations

import asyncio
import shlex

from open_agent.tools.base import Tool


class ShellTool(Tool):
    """Execute shell commands and return combined stdout/stderr."""

    name = "shell"
    description = (
        "Execute a shell command and return its combined stdout and stderr. "
        "Use for running system commands, scripts, and build tools."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Maximum execution time in seconds.",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: object) -> str:
        command = str(kwargs.get("command", ""))
        if not command:
            return "Error: no command provided."
        timeout = float(kwargs.get("timeout", 30))
        # Split the command so we can use create_subprocess_exec (no shell).
        args = shlex.split(command, posix=True)
        if not args:
            return "Error: could not parse command."
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return f"Error: command not found: {args[0]}"
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return f"Error: command timed out after {timeout}s."

        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        combined = stdout
        if stderr:
            combined = f"{stdout}\n[stderr]\n{stderr}" if stdout else stderr
        return_code = process.returncode
        if return_code and return_code != 0:
            combined = f"{combined}\n[exit code: {return_code}]"
        return combined.strip()
