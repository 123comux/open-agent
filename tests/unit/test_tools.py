"""Tests for the builtin tools (shell, python, file)."""
from __future__ import annotations

import sys

from open_agent.tools.builtin.file import FileTool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.builtin.shell import ShellTool


def _echo_command() -> str:
    """Return a platform-appropriate command that prints ``hello``."""
    # On Windows ``echo`` is a cmd.exe builtin, so it must be run via ``cmd /c``.
    # ``ShellTool`` uses ``create_subprocess_exec`` (no intermediate shell), so a
    # bare ``echo`` would raise FileNotFoundError on Windows.
    if sys.platform == "win32":
        return "cmd /c echo hello"
    return "echo hello"


async def test_shell_tool_simple_command():
    tool = ShellTool()
    result = await tool.execute(command=_echo_command())
    assert "hello" in result


async def test_shell_tool_no_command():
    tool = ShellTool()
    result = await tool.execute(command="")
    assert "Error" in result


async def test_shell_tool_unknown_command():
    tool = ShellTool()
    result = await tool.execute(command="this-command-does-not-exist-xyz arg")
    assert "Error" in result
    assert "not found" in result


async def test_shell_tool_to_schema():
    tool = ShellTool()
    schema = tool.to_schema()
    assert schema["name"] == "shell"
    assert "command" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["command"]


async def test_python_tool_simple_calculation():
    tool = PythonTool()
    result = await tool.execute(code="print(2 + 3)")
    assert result.strip() == "5"


async def test_python_tool_captures_multiple_prints():
    tool = PythonTool()
    result = await tool.execute(code="print('hello')\nprint('world')")
    assert "hello" in result
    assert "world" in result


async def test_python_tool_no_code():
    tool = PythonTool()
    result = await tool.execute(code="")
    assert "Error" in result


async def test_python_tool_handles_runtime_error():
    tool = PythonTool()
    result = await tool.execute(code="raise ValueError('boom')")
    assert "Error" in result
    assert "ValueError" in result
    assert "boom" in result


async def test_python_tool_to_schema():
    tool = PythonTool()
    schema = tool.to_schema()
    assert schema["name"] == "python"
    assert schema["parameters"]["required"] == ["code"]


async def test_file_tool_write_read_list(tmp_path):
    tool = FileTool()
    target = tmp_path / "sample.txt"

    # write
    write_result = await tool.execute(
        action="write", path=str(target), content="hello world"
    )
    assert "Wrote" in write_result
    assert target.read_text(encoding="utf-8") == "hello world"

    # read
    read_result = await tool.execute(action="read", path=str(target))
    assert read_result == "hello world"

    # list
    list_result = await tool.execute(action="list", path=str(tmp_path))
    assert "sample.txt" in list_result


async def test_file_tool_read_existing_tmp_file(tmp_file):
    tool = FileTool()
    result = await tool.execute(action="read", path=tmp_file)
    assert result == "hello world"


async def test_file_tool_read_missing_returns_error():
    tool = FileTool()
    result = await tool.execute(action="read", path="nonexistent_file_xyz_123.txt")
    assert "Error" in result


async def test_file_tool_list_missing_directory_returns_error(tmp_path):
    tool = FileTool()
    missing = tmp_path / "does_not_exist"
    result = await tool.execute(action="list", path=str(missing))
    assert "Error" in result


async def test_file_tool_unknown_action_returns_error(tmp_path):
    tool = FileTool()
    result = await tool.execute(action="delete", path=str(tmp_path))
    assert "Error" in result


async def test_file_tool_missing_action_returns_error(tmp_path):
    tool = FileTool()
    result = await tool.execute(action="", path=str(tmp_path))
    assert "Error" in result


async def test_file_tool_missing_path_returns_error():
    tool = FileTool()
    result = await tool.execute(action="read", path="")
    assert "Error" in result


async def test_file_tool_to_schema():
    tool = FileTool()
    schema = tool.to_schema()
    assert schema["name"] == "file"
    assert "action" in schema["parameters"]["properties"]
    assert "path" in schema["parameters"]["properties"]
