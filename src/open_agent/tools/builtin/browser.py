"""Builtin browser tool: lightweight web page fetching via httpx.

Provides a ``browser`` tool with actions to fetch raw HTML, extract links, and
extract text content from a URL. It intentionally depends only on
:mod:`httpx` (no Playwright/Selenium); the ``screenshot_stub`` action is a
placeholder signalling that headless rendering is not available in this
lightweight implementation.
"""
from __future__ import annotations

import json
import re

import httpx

from open_agent.tools.base import Tool


class BrowserTool(Tool):
    """Fetch web pages and extract content using httpx."""

    name = "browser"
    description = (
        "Fetch a web page and either return its HTML, extract the links it "
        "contains, or extract its visible text. Use 'action' to select "
        "'fetch', 'extract_links', 'extract_text', or 'screenshot_stub'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["fetch", "extract_links", "extract_text", "screenshot_stub"],
                "description": (
                    "The browser action to perform: 'fetch' returns raw HTML, "
                    "'extract_links' returns all links as JSON, 'extract_text' "
                    "strips HTML tags and returns visible text, 'screenshot_stub' "
                    "is a placeholder (screenshots unsupported)."
                ),
            },
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "selector": {
                "type": "string",
                "description": (
                    "Optional simple tag selector (e.g. 'p', 'h1') used to "
                    "scope text extraction for the 'extract_text' action."
                ),
            },
        },
        "required": ["action", "url"],
    }

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    TIMEOUT = 15.0
    MAX_HTML_CHARS = 5000

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

    async def execute(self, **kwargs: object) -> str:
        action = str(kwargs.get("action", "")).lower()
        url = str(kwargs.get("url", ""))
        if not action:
            return (
                "Error: 'action' is required "
                "(fetch|extract_links|extract_text|screenshot_stub)."
            )
        if action not in {"fetch", "extract_links", "extract_text", "screenshot_stub"}:
            return f"Error: unknown action '{action}'."
        if not url:
            return "Error: 'url' is required."
        if action == "screenshot_stub":
            return (
                "Error: screenshot action is not supported by the lightweight "
                "BrowserTool (requires a headless browser such as Playwright)."
            )
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
