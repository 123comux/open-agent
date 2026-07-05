"""Unit tests for :class:`open_agent.tools.builtin.python.PythonTool`.

Covers the multiline-continuation fallback (M7) and the tightened safe
builtins (M6).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from open_agent.tools.builtin.python import PythonTool, _safe_builtins

# ---------------------------------------------------------------------------
# Multiline continuation fallback (M7)
# ---------------------------------------------------------------------------


async def test_python_multiline_continuation_fallback():
    """A prefix that is itself a SyntaxError must fall back to exec'ing the
    full block instead of crashing the whole tool.

    Here ``"x = 1\\nprint((x +"`` is an incomplete statement (the ``print((``
    paren is never closed on those lines), so ``exec(prefix)`` raises
    SyntaxError. Before the fix this surfaced as
    ``"Error during execution: SyntaxError: ..."``; after the fix the full
    block is exec'd and the trailing expression completes, printing ``2``.
    """
    tool = PythonTool()
    code = "x = 1\nprint((x +\n1))"
    result = await tool.execute(code=code)
    assert "Error" not in result
    assert result.strip() == "2"


async def test_python_continuation_prefix_syntax_error_does_not_crash():
    """Even when the fallback produces no stdout, it must not return an error."""
    tool = PythonTool()
    # prefix "x = (1 +" is incomplete; full block assigns x = 3 (no print).
    code = "x = (1 +\n2)"
    result = await tool.execute(code=code)
    assert "Error" not in result
    # Assignment only — nothing printed.
    assert result == ""


async def test_python_normal_multiline_still_evaluates_last_line():
    """Regression guard: normal multiline code still eval's the last line."""
    tool = PythonTool()
    result = await tool.execute(code="x = 40\nx + 2")
    assert result.strip() == "42"


async def test_python_simple_expression_still_works():
    tool = PythonTool()
    result = await tool.execute(code="2 + 3")
    assert result.strip() == "5"


async def test_python_runtime_error_returned_as_string():
    tool = PythonTool()
    result = await tool.execute(code="raise ValueError('boom')")
    assert "Error" in result
    assert "ValueError" in result
    assert "boom" in result


# ---------------------------------------------------------------------------
# Safe builtins tightening (M6)
# ---------------------------------------------------------------------------


def test_safe_builtins_excludes_attribute_access_helpers():
    """getattr/setattr/delattr/super must not be in the safe builtins."""
    safe = _safe_builtins()
    for name in ("getattr", "setattr", "delattr", "super"):
        assert name not in safe, f"{name} should be excluded from safe builtins"


def test_safe_builtins_keeps_essential_types():
    """type/object are retained (removing them would break normal class defs)."""
    safe = _safe_builtins()
    assert "type" in safe
    assert "object" in safe
    assert "isinstance" in safe


async def test_python_sandbox_blocks_getattr_escape_attempt():
    """With the sandbox on, getattr-based escapes are harder; verify a basic
    sandboxed run still works but attribute-builtin reflection is unavailable.
    """
    tool = PythonTool()
    settings = MagicMock()
    settings.enable_tool_sandbox = True
    settings.sandbox_allowed_paths = []
    settings.sandbox_blocked_paths = []
    with patch("open_agent.tools.sandbox.get_settings", return_value=settings):
        # getattr is not in safe_builtins, so this NameError surfaces as an error.
        result = await tool.execute(code="getattr({}, 'x', None)")
    assert "Error" in result
