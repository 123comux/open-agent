"""Builtin shell tool: executes shell commands via asyncio subprocesses.

Runs a command (optionally with a timeout) and returns its captured stdout.
Dangerous commands are filtered by the sandbox guards before execution; when
the sandbox blocks a command the blocking message is returned instead of
raising, so the agent can report the issue to the model.
"""
from __future__ import annotations

import asyncio

from open_agent.tools.base import Tool
from open_agent.tools.sandbox import check_shell_safety, parse_shell_command


class ShellTool(Tool):
    """Execute a shell command and return its output."""

    name = "shell"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: object) -> str:
        command = str(kwargs.get("command", ""))
        if not command:
            return "Error: no command provided."
        timeout_raw = kwargs.get("timeout", 30)
        timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float, str)) else 30.0
        try:
            check_shell_safety(command)
        except PermissionError as exc:
            return str(exc)
        tokens = parse_shell_command(command)
        if not tokens or (len(tokens) == 1 and not tokens[0]):
            return "Error: no command provided."
        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return f"Error: command not found: {tokens[0]}"
        except OSError as exc:
            return f"Error spawning command: {exc}"
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return f"Error: command timed out after {timeout}s."
        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        if proc.returncode == 0:
            return stdout
        stderr_msg = f"stderr: {stderr}" if stderr else "stderr: (empty)"
        return (
            f"Error: command failed with exit code {proc.returncode}.\n"
            f"stdout: {stdout}\n{stderr_msg}"
        )
