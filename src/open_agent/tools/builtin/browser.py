"""Builtin browser tool with httpx fallback and optional Playwright actions.

The ``browser`` tool supports lightweight static fetching through :mod:`httpx`
and real headless-browser interactions through Playwright. When Playwright is
installed, actions such as ``screenshot``, ``click``, and ``fill`` execute in a
real browser context; otherwise they return an error prompting for installation.
Static actions (``fetch``, ``extract_links``, ``extract_text``) always work.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import re
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from open_agent.tools.base import Tool
from open_agent.tools.sandbox import check_path

# Playwright is an optional dependency.
try:
    from playwright.async_api import async_playwright

    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover
    async_playwright = None  # type: ignore[assignment]
    _PLAYWRIGHT_AVAILABLE = False


async def _safe_close(resource: Any) -> None:
    """Close a Playwright resource, ignoring errors during cleanup."""
    try:
        await resource.close()
    except Exception:
        pass


async def _safe_stop(playwright: Any) -> None:
    """Stop a Playwright instance, ignoring errors during cleanup."""
    try:
        await playwright.stop()
    except Exception:
        pass


class BrowserTool(Tool):
    """Fetch web pages and perform real browser interactions."""

    name = "browser"
    description = (
        "Control a web browser. Static actions (fetch, extract_links, "
        "extract_text) work without extra dependencies. Interactive actions "
        "(navigate, screenshot, click, fill, get_text, get_links, evaluate) "
        "require Playwright to be installed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "fetch",
                    "extract_links",
                    "extract_text",
                    "navigate",
                    "screenshot",
                    "click",
                    "fill",
                    "get_text",
                    "get_links",
                    "evaluate",
                ],
                "description": (
                    "The browser action to perform. 'fetch' returns raw HTML, "
                    "'extract_links' returns all links as JSON, 'extract_text' "
                    "strips HTML tags and returns visible text. Interactive "
                    "actions require Playwright: 'navigate' loads a URL, "
                    "'screenshot' captures the page or an element, 'click' "
                    "clicks an element, 'fill' types into an input, "
                    "'get_text' returns rendered visible text, 'get_links' "
                    "returns rendered links, 'evaluate' runs JavaScript."
                ),
            },
            "url": {
                "type": "string",
                "description": "The URL to fetch or navigate to.",
            },
            "selector": {
                "type": "string",
                "description": (
                    "CSS selector for interactive actions (click, fill, "
                    "screenshot element, or to scope text extraction)."
                ),
            },
            "text": {
                "type": "string",
                "description": "Text to type into an input for the 'fill' action.",
            },
            "script": {
                "type": "string",
                "description": "JavaScript code to run for the 'evaluate' action.",
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional file path to save a screenshot. If omitted, the "
                    "screenshot is returned as a base64 data URL."
                ),
            },
            "full_page": {
                "type": "boolean",
                "description": "Whether to capture the full page for 'screenshot'.",
                "default": False,
            },
            "selector_type": {
                "type": "string",
                "enum": ["css", "xpath", "text"],
                "description": "How to interpret 'selector' for interactive actions.",
                "default": "css",
            },
        },
        "required": ["action"],
    }

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    TIMEOUT = 15.0
    MAX_HTML_CHARS = 5000
    DEFAULT_VIEWPORT = {"width": 1280, "height": 720}

    # ------------------------------------------------------------------
    # Static httpx helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_blocked_ip(ip: Any) -> bool:
        """Return True if the IP address is in a disallowed range.

        Blocks private, loopback, link-local, multicast and unspecified
        addresses. This covers the cloud metadata endpoint
        ``169.254.169.254`` (link-local) and all common SSRF targets.
        """
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        )

    @staticmethod
    async def _validate_url(url: str) -> str | None:
        """Validate ``url`` for SSRF safety.

        Only ``http``/``https`` URLs are allowed. The host is resolved and
        every resolved IP is checked against disallowed ranges. IP literals
        are checked directly without DNS. Returns ``None`` when the URL is
        allowed, or an error-message string when it is blocked.

        DNS resolution failures are treated as allowed (fail-open): an
        unresolvable host cannot be reached anyway, and this keeps the tool
        usable in offline/test environments. IP-literal checks always apply.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: URL scheme '{parsed.scheme}' is not allowed (only http/https)."
        host = parsed.hostname
        if not host:
            return "Error: URL has no host."

        # IP literal: check directly (no DNS needed).
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None:
            if BrowserTool._is_blocked_ip(ip):
                return (
                    f"Error: URL host '{host}' is a blocked address "
                    "(private/loopback/link-local/multicast/unspecified)."
                )
            return None

        # Domain: resolve off the event loop and check every resolved IP.
        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
        except socket.gaierror:
            # Cannot resolve — cannot confirm a private IP; allow.
            return None
        for info in infos:
            sockaddr = info[4]
            ip_str = sockaddr[0]
            try:
                resolved = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if BrowserTool._is_blocked_ip(resolved):
                return (
                    f"Error: URL host '{host}' resolves to blocked address "
                    f"'{ip_str}'."
                )
        return None

    async def _fetch(self, url: str) -> str:
        headers = {"User-Agent": self.USER_AGENT}
        async with httpx.AsyncClient(
            timeout=self.TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text

    @staticmethod
    def _strip_tags(html: str) -> str:
        # Drop script/style blocks first so their contents aren't rendered.
        html = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_text(self, html: str, selector: str | None) -> str:
        if selector:
            # Support a simple tag-name selector (e.g. "p", "h1").
            tag = re.sub(r"[^a-zA-Z0-9]", "", selector)
            if tag:
                blocks = re.findall(
                    rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE
                )
                html = "\n".join(blocks)
        return self._strip_tags(html)

    @staticmethod
    def _extract_links(html: str) -> str:
        links: list[dict[str, str]] = []
        for match in re.finditer(
            r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            html,
            re.DOTALL | re.IGNORECASE,
        ):
            href = match.group(1).strip()
            label = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if href:
                links.append({"href": href, "text": label})
        return json.dumps(links, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Playwright helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _selector_for(selector: str | None, selector_type: str) -> str:
        """Build a Playwright locator string from selector and type."""
        if not selector:
            return ""
        selector_type = (selector_type or "css").lower()
        if selector_type == "css":
            return f"css={selector}"
        if selector_type == "xpath":
            return f"xpath={selector}"
        if selector_type == "text":
            return f"text={selector}"
        return selector

    @asynccontextmanager
    async def _with_page(self, url: str) -> AsyncIterator[Any]:
        """Async context manager yielding a Playwright page for ``url``.

        Resources (playwright, browser, context) are registered with an
        :class:`contextlib.AsyncExitStack` so they are always cleaned up even
        if setup fails partway (e.g. ``chromium.launch`` raises) or one of the
        close calls raises. Each close is wrapped in its own try/except so a
        single failure cannot skip the remaining cleanups.
        """
        async with contextlib.AsyncExitStack() as stack:
            playwright = await async_playwright().start()
            stack.push_async_callback(_safe_stop, playwright)
            browser = await playwright.chromium.launch(headless=True)
            stack.push_async_callback(_safe_close, browser)
            context = await browser.new_context(
                viewport=self.DEFAULT_VIEWPORT,  # type: ignore[arg-type]
                user_agent=self.USER_AGENT,
            )
            stack.push_async_callback(_safe_close, context)
            page = await context.new_page()
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(self.TIMEOUT * 1000),
            )
            yield page

    async def _navigate(self, url: str) -> str:
        async with self._with_page(url) as page:
            title = await page.title()
            return f"Navigated to {url}. Title: {title}"

    async def _screenshot(self, url: str, kwargs: dict[str, Any]) -> str:
        selector = str(kwargs.get("selector", "")) or None
        selector_type = str(kwargs.get("selector_type", "css"))
        full_page = bool(kwargs.get("full_page", False))
        output_path = str(kwargs.get("output_path", "")) or None

        if output_path:
            blocked = check_path(output_path)
            if blocked:
                return blocked

        async with self._with_page(url) as page:
            target = page
            if selector:
                locator = page.locator(self._selector_for(selector, selector_type))
                await locator.wait_for(state="visible", timeout=5000)
                target = locator

            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                await target.screenshot(path=str(path), full_page=full_page)
                return f"Screenshot saved to {path.resolve()}"

            screenshot_bytes = await target.screenshot(full_page=full_page)
            data_url = "data:image/png;base64," + base64.b64encode(screenshot_bytes).decode("ascii")
            return f"Screenshot (base64 PNG): {data_url}"

    async def _click(self, url: str, kwargs: dict[str, Any]) -> str:
        selector = str(kwargs.get("selector", ""))
        selector_type = str(kwargs.get("selector_type", "css"))
        if not selector:
            return "Error: 'selector' is required for click action."

        async with self._with_page(url) as page:
            locator = page.locator(self._selector_for(selector, selector_type))
            await locator.wait_for(state="visible", timeout=5000)
            await locator.click()
            await page.wait_for_load_state("networkidle", timeout=5000)
            title = await page.title()
            return f"Clicked '{selector}'. Current title: {title}"

    async def _fill(self, url: str, kwargs: dict[str, Any]) -> str:
        selector = str(kwargs.get("selector", ""))
        text = str(kwargs.get("text", ""))
        selector_type = str(kwargs.get("selector_type", "css"))
        if not selector:
            return "Error: 'selector' is required for fill action."

        async with self._with_page(url) as page:
            locator = page.locator(self._selector_for(selector, selector_type))
            await locator.wait_for(state="visible", timeout=5000)
            await locator.fill(text)
            return f"Filled '{selector}' with provided text."

    async def _get_text(self, url: str, kwargs: dict[str, Any]) -> str:
        selector = str(kwargs.get("selector", "")) or None
        selector_type = str(kwargs.get("selector_type", "css"))

        async with self._with_page(url) as page:
            if selector:
                locator = page.locator(self._selector_for(selector, selector_type))
                await locator.wait_for(state="visible", timeout=5000)
                text = await locator.inner_text()
            else:
                text = await page.inner_text("body")
            return str(text).strip()

    async def _get_links(self, url: str) -> str:
        async with self._with_page(url) as page:
            links = await page.eval_on_selector_all(
                "a",
                """elements => elements
                    .map(a => ({href: a.href, text: a.innerText.trim()}))
                    .filter(item => item.href)""",
            )
            return json.dumps(links, ensure_ascii=False)

    async def _evaluate(self, url: str, kwargs: dict[str, Any]) -> str:
        script = str(kwargs.get("script", ""))
        if not script:
            return "Error: 'script' is required for evaluate action."

        async with self._with_page(url) as page:
            result = await page.evaluate(script)
            return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def execute(self, **kwargs: object) -> str:
        action = str(kwargs.get("action", "")).lower()
        url = str(kwargs.get("url", ""))

        if not action:
            return (
                "Error: 'action' is required (fetch|extract_links|extract_text|"
                "navigate|screenshot|click|fill|get_text|get_links|evaluate)."
            )

        static_actions = {"fetch", "extract_links", "extract_text"}
        interactive_actions = {
            "navigate",
            "screenshot",
            "click",
            "fill",
            "get_text",
            "get_links",
            "evaluate",
        }
        if action not in static_actions | interactive_actions:
            return f"Error: unknown action '{action}'."

        if not url:
            return "Error: 'url' is required."

        # SSRF guard: always validate the URL (http/https only, no
        # private/loopback/link-local/multicast/unspecified targets) before
        # any network action. Applied to both static and interactive actions.
        url_error = await self._validate_url(url)
        if url_error:
            return url_error

        if action in static_actions:
            try:
                html = await self._fetch(url)
            except httpx.HTTPError as exc:
                return f"Error fetching URL: {exc}"
            if action == "fetch":
                return html[: self.MAX_HTML_CHARS]
            if action == "extract_links":
                return self._extract_links(html)
            selector = kwargs.get("selector")
            selector_str = str(selector) if selector else None
            return self._extract_text(html, selector_str)

        # Interactive actions require Playwright.
        if not _PLAYWRIGHT_AVAILABLE:
            return (
                "Error: Playwright is required for interactive browser actions. "
                "Install it with: pip install playwright && playwright install chromium"
            )

        try:
            if action == "navigate":
                return await self._navigate(url)
            if action == "screenshot":
                return await self._screenshot(url, kwargs)
            if action == "click":
                return await self._click(url, kwargs)
            if action == "fill":
                return await self._fill(url, kwargs)
            if action == "get_text":
                return await self._get_text(url, kwargs)
            if action == "get_links":
                return await self._get_links(url)
            return await self._evaluate(url, kwargs)
        except Exception as exc:  # noqa: BLE001
            return f"Error performing browser action: {type(exc).__name__}: {exc}"
