"""Builtin Python code interpreter tool.

Executes Python source code via :func:`exec` inside a restricted namespace and
captures anything written to stdout with :func:`contextlib.redirect_stdout`.
A timeout is enforced by running the code in a worker thread.

Note: running in a thread means a timeout cannot truly interrupt a hung
computation; for hard isolation a subprocess-based interpreter would be needed.
This implementation favours simplicity and zero extra dependencies.
"""
from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
from typing import Any

from open_agent.tools.base import Tool


class PythonTool(Tool):
    """Execute Python code in a restricted namespace and return captured stdout."""

    name = "python"
    description = (
        "Execute Python source code in a restricted namespace and return "
        "anything printed to stdout. Useful for calculations and data wrangling."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python source code to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Maximum execution time in seconds.",
                "default": 10,
            },
        },
        "required": ["code"],
    }

    def _run_sync(self, code: str) -> str:
        buffer = io.StringIO()
        globals_ns: dict[str, Any] = {
            "__name__": "__python_tool__",
            "__builtins__": vars(builtins),
        }
        locals_ns: dict[str, Any] = {}
        with contextlib.redirect_stdout(buffer):
            try:
                # Try to compile as a single expression first (like Jupyter)
                try:
                    compiled = compile(code, "<python_tool>", "eval")
                    result = eval(compiled, globals_ns, locals_ns)  # noqa: S307
                    if result is not None:
                        print(repr(result) if not isinstance(result, str) else result)
                except SyntaxError:
                    # Multi-line code: exec all but try to eval the last line
                    lines = code.rstrip().split("\n")
                    # Check if the last non-empty line is an expression
                    last_line = lines[-1].strip() if lines else ""
                    if last_line and not last_line.endswith((":", "=")) and "import" not in last_line:
                        # Split: exec everything before, eval the last line
                        prefix = "\n".join(lines[:-1])
                        if prefix.strip():
                            exec(prefix, globals_ns, locals_ns)  # noqa: S102
                        try:
                            result = eval(last_line, globals_ns, locals_ns)  # noqa: S307
                            if result is not None:
                                print(repr(result) if not isinstance(result, str) else result)
                        except SyntaxError:
                            exec(code, globals_ns, locals_ns)  # noqa: S102
                    else:
                        exec(code, globals_ns, locals_ns)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                return f"Error during execution: {type(exc).__name__}: {exc}"
        return buffer.getvalue()

    async def execute(self, **kwargs: object) -> str:
        code = str(kwargs.get("code", ""))
        if not code:
            return "Error: no code provided."
        timeout = float(kwargs.get("timeout", 10))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, code), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"Error: execution timed out after {timeout}s."
        return result.strip()
