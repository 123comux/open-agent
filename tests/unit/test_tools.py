"""Tests for the builtin tools (shell, python, file)."""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from open_agent.tools.builtin.browser import BrowserTool
from open_agent.tools.builtin.file import FileTool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.builtin.shell import ShellTool
from open_agent.tools.builtin.web_search import WebSearchTool


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


# ---------------------------------------------------------------------------
# WebSearchTool — Bing HTML parsing regression tests
# (real Bing HTML uses <h2 class=""> with attributes and <p class="b_lineclampN">)
# ---------------------------------------------------------------------------

BING_HTML_SAMPLE = """
<ol id="b_results" class="">
<li class="b_algo" data-id iid=SERP.5331>
<link rel="stylesheet" href="https://r.bing.com/rs/x.css" type="text/css"/>
<h2 class=""><a target="_blank" href="https://example.com/news1" h="ID=SERP,5127.2">
<strong>AI</strong> News | <strong>Latest News</strong> | Insights</a></h2>
<div class="b_caption"><p class="b_lineclamp2">2 天之前 · AI News delivers the latest updates.</p>
</div>
</li>
<li class="b_algo" data-id iid=SERP.5332>
<h2><a href="https://example.com/news2">Second Result</a></h2>
<p class="b_lineclamp4">Snippet for second result here.</p>
</li>
<li class="b_algo" data-id iid=SERP.5333>
<h2 class="b_title"><a href="https://example.com/news3">Third Result</a></h2>
<div class="b_caption"><p>Third snippet without b_lineclamp class.</p></div>
</li>
</ol>
"""


def test_parse_bing_extracts_title_url_snippet():
    """Regression: <h2 class=""> with attributes must be parsed (not just bare <h2>)."""
    tool = WebSearchTool()
    results = tool._parse_bing(BING_HTML_SAMPLE, max_results=5)
    assert len(results) == 3
    assert results[0]["title"] == "AI News | Latest News | Insights"
    assert results[0]["url"] == "https://example.com/news1"
    assert "latest updates" in results[0]["snippet"]
    assert results[1]["title"] == "Second Result"
    assert results[1]["url"] == "https://example.com/news2"
    assert results[2]["title"] == "Third Result"


def test_parse_bing_respects_max_results():
    tool = WebSearchTool()
    results = tool._parse_bing(BING_HTML_SAMPLE, max_results=2)
    assert len(results) == 2


def test_parse_bing_empty_html():
    tool = WebSearchTool()
    assert tool._parse_bing("<html></html>", max_results=5) == []


def test_web_search_tool_schema():
    tool = WebSearchTool()
    schema = tool.to_schema()
    assert schema["name"] == "web_search"
    assert "query" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["query"]


async def test_web_search_no_query_returns_error():
    tool = WebSearchTool()
    result = await tool.execute(query="")
    assert "Error" in result


# ---------------------------------------------------------------------------
# BrowserTool — static actions and Playwright interaction scaffolding
# ---------------------------------------------------------------------------

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a <strong>test</strong> paragraph.</p>
<a href="https://example.com">Example</a>
<a href="/relative">Relative</a>
<script>alert('ignored');</script>
</body>
</html>
"""


def test_browser_tool_to_schema():
    tool = BrowserTool()
    schema = tool.to_schema()
    assert schema["name"] == "browser"
    actions = schema["parameters"]["properties"]["action"]["enum"]
    assert "screenshot" in actions
    assert "click" in actions
    assert "fill" in actions


async def test_browser_tool_fetch_truncates_html():
    tool = BrowserTool()
    with patch.object(tool, "_fetch", new=AsyncMock(return_value=SAMPLE_HTML)):
        result = await tool.execute(action="fetch", url="https://example.com")
    assert result == SAMPLE_HTML[: tool.MAX_HTML_CHARS]


async def test_browser_tool_extract_text():
    tool = BrowserTool()
    with patch.object(tool, "_fetch", new=AsyncMock(return_value=SAMPLE_HTML)):
        result = await tool.execute(action="extract_text", url="https://example.com")
    assert "Hello World" in result
    assert "test paragraph" in result
    assert "alert" not in result


async def test_browser_tool_extract_text_with_selector():
    tool = BrowserTool()
    with patch.object(tool, "_fetch", new=AsyncMock(return_value=SAMPLE_HTML)):
        result = await tool.execute(
            action="extract_text", url="https://example.com", selector="p"
        )
    assert "test paragraph" in result
    assert "Hello World" not in result


async def test_browser_tool_extract_links():
    tool = BrowserTool()
    with patch.object(tool, "_fetch", new=AsyncMock(return_value=SAMPLE_HTML)):
        result = await tool.execute(action="extract_links", url="https://example.com")
    links = json.loads(result)
    assert len(links) == 2
    assert links[0]["href"] == "https://example.com"
    assert links[0]["text"] == "Example"


async def test_browser_tool_missing_action_returns_error():
    tool = BrowserTool()
    result = await tool.execute(action="", url="https://example.com")
    assert "Error" in result


async def test_browser_tool_missing_url_returns_error():
    tool = BrowserTool()
    result = await tool.execute(action="fetch", url="")
    assert "Error" in result


async def test_browser_tool_unknown_action_returns_error():
    tool = BrowserTool()
    result = await tool.execute(action="unknown", url="https://example.com")
    assert "Error" in result


async def test_browser_tool_static_fetch_http_error():
    tool = BrowserTool()
    with patch.object(
        tool, "_fetch", new=AsyncMock(side_effect=httpx.HTTPError("network down"))
    ):
        result = await tool.execute(action="fetch", url="https://example.com")
    assert "Error fetching URL" in result


async def test_browser_tool_interactive_requires_playwright():
    """Interactive actions return a clear error when Playwright is unavailable."""
    tool = BrowserTool()
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", False):
        result = await tool.execute(action="click", url="https://example.com", selector="button")
    assert "Playwright is required" in result


def _build_mock_playwright_page(
    title: str = "Mock Page", text: str = "", links: list | None = None
):
    """Return a mocked async_playwright context for BrowserTool tests."""
    page = MagicMock()
    page.title = AsyncMock(return_value=title)
    page.inner_text = AsyncMock(return_value=text)
    page.evaluate = AsyncMock(return_value={"ok": True})
    page.eval_on_selector_all = AsyncMock(return_value=links or [])
    page.wait_for_load_state = AsyncMock(return_value=None)
    page.goto = AsyncMock(return_value=None)
    page.screenshot = AsyncMock(return_value=b"pngbytes")

    locator = MagicMock()
    locator.wait_for = AsyncMock(return_value=None)
    locator.click = AsyncMock(return_value=None)
    locator.fill = AsyncMock(return_value=None)
    locator.inner_text = AsyncMock(return_value=text)
    locator.screenshot = AsyncMock(return_value=b"pngbytes")
    page.locator = MagicMock(return_value=locator)

    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock(return_value=None)

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock(return_value=None)

    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(return_value=browser)
    playwright.stop = AsyncMock(return_value=None)

    runner = MagicMock()
    runner.start = AsyncMock(return_value=playwright)
    return runner, page


async def test_browser_tool_navigate():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page(title="Example")
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(action="navigate", url="https://example.com")
    assert "Navigated to https://example.com" in result
    assert "Example" in result
    page.goto.assert_awaited_once()


async def test_browser_tool_click():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page(title="After Click")
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(
            action="click", url="https://example.com", selector="#btn"
        )
    assert "Clicked '#btn'" in result
    assert "After Click" in result


async def test_browser_tool_fill():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page()
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(
            action="fill",
            url="https://example.com",
            selector="input[name=q]",
            text="hello",
        )
    assert "Filled 'input[name=q]'" in result


async def test_browser_tool_get_text():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page(text="Rendered text")
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(action="get_text", url="https://example.com")
    assert result == "Rendered text"


async def test_browser_tool_get_links():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page(links=[{"href": "https://x.com", "text": "X"}])
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(action="get_links", url="https://example.com")
    assert "https://x.com" in result
    assert "X" in result


async def test_browser_tool_evaluate():
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page()
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(
            action="evaluate", url="https://example.com", script="return {ok:true}"
        )
    assert '"ok": true' in result


async def test_browser_tool_screenshot_to_path(tmp_path):
    tool = BrowserTool()
    runner, page = _build_mock_playwright_page()
    output = tmp_path / "shot.png"
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(
            action="screenshot",
            url="https://example.com",
            output_path=str(output),
            full_page=True,
        )
    assert "Screenshot saved" in result
    assert str(output) in result


# ---------------------------------------------------------------------------
# Sandbox guards
# ---------------------------------------------------------------------------


def _patch_sandbox(enabled: bool = True, allowed: list[str] | None = None):
    settings = MagicMock()
    settings.enable_tool_sandbox = enabled
    settings.sandbox_allowed_paths = allowed or []
    settings.sandbox_blocked_paths = []
    return patch("open_agent.tools.sandbox.get_settings", return_value=settings)


async def test_shell_sandbox_blocks_dangerous_command():
    with _patch_sandbox():
        tool = ShellTool()
        result = await tool.execute(command="rm -rf /")
    assert "Sandbox blocked" in result


async def test_shell_sandbox_allows_safe_command():
    with _patch_sandbox():
        tool = ShellTool()
        result = await tool.execute(command=_echo_command())
    assert "Sandbox blocked" not in result


async def test_python_sandbox_blocks_import_os():
    with _patch_sandbox():
        tool = PythonTool()
        result = await tool.execute(code="import os")
    assert "Sandbox blocked" in result


async def test_python_sandbox_allows_safe_code():
    with _patch_sandbox():
        tool = PythonTool()
        result = await tool.execute(code="print(2 + 2)")
    assert result == "4"


async def test_file_sandbox_blocks_outside_allowed_path(tmp_path):
    allowed = str(tmp_path / "sandbox")
    with _patch_sandbox(allowed=[allowed]):
        tool = FileTool()
        result = await tool.execute(action="read", path=str(tmp_path / "outside.txt"))
    assert "Sandbox blocked" in result


async def test_file_sandbox_allows_inside_allowed_path(tmp_path):
    allowed = tmp_path / "sandbox"
    allowed.mkdir()
    file_path = allowed / "test.txt"
    file_path.write_text("hello", encoding="utf-8")
    with _patch_sandbox(allowed=[str(allowed)]):
        tool = FileTool()
        result = await tool.execute(action="read", path=str(file_path))
    assert result == "hello"


async def test_sandbox_disabled_by_default(tmp_path):
    with _patch_sandbox(enabled=False):
        tool = FileTool()
        result = await tool.execute(action="read", path="/etc/hosts")
    assert "Sandbox blocked" not in result
