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
from open_agent.tools.sandbox import check_python, sandbox_enabled


def _safe_builtins() -> dict[str, Any]:
    """Return a whitelist of safe builtin functions for sandboxed execution.

    Dangerous builtins such as ``__import__``, ``open``, ``eval``, ``exec``,
    ``compile``, ``globals``, ``locals`` and ``vars`` are deliberately excluded
    so sandboxed code cannot escape the namespace to reach the filesystem, the
    import system, or arbitrary code execution. The attribute-access builtins
    ``getattr``, ``setattr`` and ``delattr`` and ``super`` are also excluded to
    hinder reflection-based sandbox escapes (e.g. reaching ``__subclasses__``).
    """
    safe_names = {
        # Constants
        "True", "False", "None",
        # Type conversion
        "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset",
        "int", "list", "set", "str", "tuple", "type",
        # Numeric
        "abs", "bin", "divmod", "hex", "oct", "ord", "pow", "round", "sum",
        # Iteration
        "enumerate", "filter", "iter", "map", "next", "range", "reversed", "zip",
        # Logic
        "all", "any",
        # Comparison
        "max", "min",
        # Object
        "callable", "classmethod", "hasattr", "hash",
        "id", "isinstance", "issubclass", "object", "property", "repr",
        "slice", "staticmethod",
        # String
        "ascii", "chr", "format",
        # Sorting
        "sorted",
        # I/O (safe)
        "print",
        # Memory
        "memoryview",
        # Exceptions
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
        "ZeroDivisionError", "ArithmeticError", "LookupError", "OverflowError",
        "FileNotFoundError", "NameError", "OSError", "BufferError",
        "BlockingIOError", "ChildProcessError", "ConnectionError",
        "BrokenPipeError", "ConnectionAbortedError", "ConnectionRefusedError",
        "ConnectionResetError", "InterruptedError", "IsADirectoryError",
        "NotADirectoryError", "PermissionError", "ProcessLookupError",
        "TimeoutError", "EOFError", "ImportError", "ModuleNotFoundError",
        "RecursionError", "ReferenceError", "SystemError", "TabError",
        "UnboundLocalError", "UnicodeError", "UnicodeDecodeError",
        "UnicodeEncodeError", "UnicodeTranslateError",
    }
    return {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}


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

    def _run_sync(self, code: str, sandbox: bool = False) -> str:
        buffer = io.StringIO()
        # When the sandbox is enabled, restrict builtins to a safe whitelist so
        # the executed code cannot import modules, open files, or call
        # eval/exec. When disabled, full builtins are available.
        builtin_ns = _safe_builtins() if sandbox else vars(builtins)
        globals_ns: dict[str, Any] = {
            "__name__": "__python_tool__",
            "__builtins__": builtin_ns,
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
                    if (
                        last_line
                        and not last_line.endswith((":", "="))
                        and "import" not in last_line
                    ):
                        # Split: exec everything before, eval the last line
                        prefix = "\n".join(lines[:-1])
                        if prefix.strip():
                            try:
                                exec(prefix, globals_ns, locals_ns)  # noqa: S102
                            except SyntaxError:
                                # prefix is incomplete (e.g. a continuation
                                # line like ``result = (1 +``); fall back to
                                # exec'ing the whole code block.
                                exec(code, globals_ns, locals_ns)  # noqa: S102
                                return buffer.getvalue()
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
        blocked = check_python(code)
        if blocked:
            return blocked
        timeout_raw = kwargs.get("timeout", 10)
        timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float, str)) else 10.0
        sandbox = sandbox_enabled()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, code, sandbox), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"Error: execution timed out after {timeout}s."
        return result.strip()
