"""Unit tests for :class:`open_agent.tools.builtin.browser.BrowserTool`.

Focuses on the security guards (SSRF prevention, screenshot path traversal)
and the resource-cleanup refactor. Playwright-dependent paths are exercised
through mocks so the suite runs without a real browser.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from open_agent.tools.builtin.browser import BrowserTool

# ---------------------------------------------------------------------------
# SSRF protection (M5)
# ---------------------------------------------------------------------------


async def test_browser_fetch_ssrf_blocks_cloud_metadata_endpoint():
    """The cloud metadata IP 169.254.169.254 must be blocked before fetch."""
    tool = BrowserTool()
    result = await tool.execute(
        action="fetch", url="http://169.254.169.254/latest/meta-data/"
    )
    assert "Error" in result
    assert "blocked" in result.lower()
    # 169.254.169.254 is link-local; the message should mention it.
    assert "169.254.169.254" in result


async def test_browser_fetch_ssrf_blocks_loopback():
    tool = BrowserTool()
    result = await tool.execute(action="fetch", url="http://127.0.0.1:8080/admin")
    assert "Error" in result
    assert "blocked" in result.lower()


async def test_browser_fetch_ssrf_blocks_non_http_scheme():
    tool = BrowserTool()
    result = await tool.execute(action="fetch", url="file:///etc/passwd")
    assert "Error" in result
    assert "scheme" in result.lower()


async def test_browser_fetch_ssrf_allows_public_ip():
    """A public IP literal must pass validation and reach _fetch."""
    tool = BrowserTool()
    with patch.object(tool, "_fetch", new=AsyncMock(return_value="<html>ok</html>")):
        result = await tool.execute(action="fetch", url="http://1.1.1.1")
    assert result == "<html>ok</html>"


async def test_browser_fetch_ssrf_blocks_private_ten():
    tool = BrowserTool()
    result = await tool.execute(action="fetch", url="http://10.0.0.1/internal")
    assert "Error" in result
    assert "blocked" in result.lower()


# ---------------------------------------------------------------------------
# Screenshot path traversal (M4)
# ---------------------------------------------------------------------------


def _enable_sandbox():
    settings = MagicMock()
    settings.enable_tool_sandbox = True
    settings.sandbox_allowed_paths = []
    settings.sandbox_blocked_paths = []
    return patch("open_agent.tools.sandbox.get_settings", return_value=settings)


async def test_browser_screenshot_path_traversal_blocked():
    """_screenshot must run output_path through check_path and refuse traversal."""
    tool = BrowserTool()
    with _enable_sandbox():
        result = await tool._screenshot(
            "https://example.com", {"output_path": "../evil.png"}
        )
    assert "Sandbox blocked" in result
    # No playwright resources should have been touched (early return).
    # (If playwright had been started, this test would require it installed.)


async def test_browser_screenshot_path_traversal_allowed_when_sandbox_off(tmp_path):
    """When the sandbox is off, check_path returns None and the path is used.

    We mock _with_page so no real Playwright is needed; we only assert that
    the path-traversal guard does NOT fire when the sandbox is disabled.
    """
    tool = BrowserTool()
    output = tmp_path / "shot.png"

    fake_page = MagicMock()
    fake_page.screenshot = AsyncMock(return_value=b"png")

    class _FakeCtx:
        async def __aenter__(self):
            return fake_page

        async def __aexit__(self, *exc):
            return False

    with patch.object(tool, "_with_page", return_value=_FakeCtx()), patch(
        "open_agent.tools.sandbox._sandbox_enabled", return_value=False
    ):
        result = await tool._screenshot(
            "https://example.com", {"output_path": str(output)}
        )
    assert "Screenshot saved" in result


# ---------------------------------------------------------------------------
# _with_page resource cleanup (H2 / H7)
# ---------------------------------------------------------------------------


def _build_mock_playwright_chain():
    """Build a mock playwright runner + browser/context for cleanup tests."""
    page = MagicMock()
    page.goto = AsyncMock(return_value=None)
    page.title = AsyncMock(return_value="Title")

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
    return runner, playwright, browser, context, page


async def test_browser_with_page_cleans_up_on_success():
    tool = BrowserTool()
    runner, playwright, browser, context, page = _build_mock_playwright_chain()
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        async with tool._with_page("https://example.com") as pg:
            assert pg is page
    # All three resources closed after the context exits.
    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    playwright.stop.assert_awaited_once()


async def test_browser_with_page_cleans_up_when_launch_fails():
    """H2: if chromium.launch raises, playwright must still be stopped."""
    tool = BrowserTool()
    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(side_effect=RuntimeError("launch failed"))
    playwright.stop = AsyncMock(return_value=None)
    runner = MagicMock()
    runner.start = AsyncMock(return_value=playwright)

    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        with pytest.raises(RuntimeError, match="launch failed"):
            async with tool._with_page("https://example.com") as _:
                pass
    # playwright was started, launch failed, so playwright.stop must still run.
    playwright.stop.assert_awaited_once()


async def test_browser_with_page_continues_cleanup_if_one_close_raises():
    """H7: a failing close must not skip the remaining cleanups."""
    tool = BrowserTool()
    page = MagicMock()
    page.goto = AsyncMock(return_value=None)

    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock(side_effect=RuntimeError("context close boom"))

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock(return_value=None)

    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(return_value=browser)
    playwright.stop = AsyncMock(return_value=None)

    runner = MagicMock()
    runner.start = AsyncMock(return_value=playwright)

    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        # The body should not see the cleanup error; cleanup is in __aexit__.
        async with tool._with_page("https://example.com") as _:
            pass
    # context.close raised, but browser.close and playwright.stop still ran.
    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    playwright.stop.assert_awaited_once()


async def test_browser_navigate_uses_context_manager():
    """Smoke test: _navigate works with the refactored _with_page."""
    tool = BrowserTool()
    runner, playwright, browser, context, page = _build_mock_playwright_chain()
    with patch("open_agent.tools.builtin.browser._PLAYWRIGHT_AVAILABLE", True), patch(
        "open_agent.tools.builtin.browser.async_playwright", return_value=runner
    ):
        result = await tool.execute(action="navigate", url="http://1.1.1.1")
    assert "Navigated" in result
    page.goto.assert_awaited_once()


async def test_browser_static_fetch_http_error_still_surfaces():
    """Regression: a mocked _fetch HTTP error still returns an error string."""
    tool = BrowserTool()
    with patch.object(
        tool, "_fetch", new=AsyncMock(side_effect=httpx.HTTPError("down"))
    ):
        result = await tool.execute(action="fetch", url="http://1.1.1.1")
    assert "Error fetching URL" in result
