"""Builtin web search tool using DuckDuckGo's Lite and HTML endpoints.

Tries the lightweight ``lite.duckduckgo.com/lite/`` endpoint first (simpler,
more stable markup) and falls back to ``html.duckduckgo.com/html/`` when the
Lite endpoint is unavailable or returns no usable results. Results are parsed
into titles, URLs and snippets with regular expressions and rendered as
numbered text. The tool is dependency-free beyond :mod:`httpx` and includes
timeout handling plus a small inter-request delay for rate limiting.
"""
from __future__ import annotations

import asyncio
import re
import time
from html import unescape
from urllib.parse import unquote

import httpx

from open_agent.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo and return results as text."""

    name = "web_search"
    description = (
        "Search the web for a query and return a text summary of the top "
        "results (titles, URLs and snippets)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    LITE_URL = "https://lite.duckduckgo.com/lite/"
    HTML_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = "open-agent/0.1 (+https://github.com/your-org/open-agent)"
    TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    # Minimum seconds between consecutive outbound search requests. DuckDuckGo
    # rate-limits aggressive clients; a short delay keeps the tool reliable.
    RATE_LIMIT_SECONDS = 1.0

    # Shared across instances so rapid repeated calls are throttled globally.
    _last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove HTML tags, decode entities and collapse whitespace."""
        text = re.sub(r"<[^>]+>", "", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_url(url: str) -> str:
        """Unwrap DuckDuckGo redirect URLs and normalize protocol-relative ones.

        DuckDuckGo wraps result links in a redirect such as
        ``//duckduckgo.com/l/?uddg=<encoded>``; the real target lives in the
        ``uddg`` query parameter.
        """
        if not url:
            return url
        match = re.search(r"[?&]uddg=([^&]+)", url)
        if match:
            return unquote(match.group(1))
        if url.startswith("//"):
            return "https:" + url
        return url

    async def _rate_limit(self) -> None:
        """Sleep just long enough to honor ``RATE_LIMIT_SECONDS`` between calls."""
        if self.RATE_LIMIT_SECONDS <= 0:
            return
        now = time.monotonic()
        elapsed = now - WebSearchTool._last_request_time
        if 0.0 <= elapsed < self.RATE_LIMIT_SECONDS:
            await asyncio.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        WebSearchTool._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_lite(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse ``lite.duckduckgo.com`` markup into result dicts.

        Result links are ``<a rel="nofollow" class="result-link" href="URL">``
        and snippets are ``<td class="result-snippet">`` cells. Attribute order
        is matched flexibly so layout tweaks don't break parsing.
        """
        snippets = [
            self._strip_tags(inner)
            for _, inner in re.findall(
                r'<(?P<tag>td|a|div)\b[^>]*\bclass="result-snippet"[^>]*>(.*?)</(?P=tag)>',
                html,
                re.DOTALL,
            )
        ]
        results: list[dict[str, str]] = []
        for attrs, title in re.findall(r'<a\b([^>]*)>(.*?)</a>', html, re.DOTALL):
            if "result-link" not in attrs:
                continue
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs)
            url = self._clean_url(href_match.group(1)) if href_match else ""
            results.append(
                {
                    "title": self._strip_tags(title),
                    "url": url,
                    "snippet": "",
                }
            )
            if len(results) >= max_results:
                break
        self._pair_snippets(results, snippets)
        return results

    def _parse_html(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse ``html.duckduckgo.com`` markup into result dicts.

        Result links carry ``class="result__a"`` (URLs wrapped in a redirect)
        and snippets carry ``class="result__snippet"``.
        """
        snippets = [
            self._strip_tags(inner)
            for _, inner in re.findall(
                r'<(?P<tag>a|td|div)\b[^>]*\bclass="result__snippet"[^>]*>(.*?)</(?P=tag)>',
                html,
                re.DOTALL,
            )
        ]
        results: list[dict[str, str]] = []
        for attrs, title in re.findall(r'<a\b([^>]*)>(.*?)</a>', html, re.DOTALL):
            if "result__a" not in attrs:
                continue
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs)
            url = self._clean_url(href_match.group(1)) if href_match else ""
            results.append(
                {
                    "title": self._strip_tags(title),
                    "url": url,
                    "snippet": "",
                }
            )
            if len(results) >= max_results:
                break
        self._pair_snippets(results, snippets)
        return results

    @staticmethod
    def _pair_snippets(
        results: list[dict[str, str]], snippets: list[str]
    ) -> None:
        """Attach snippets to results by document order (1:1)."""
        for i, result in enumerate(results):
            if i < len(snippets):
                result["snippet"] = snippets[i]

    def _format(self, results: list[dict[str, str]]) -> str:
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result['title']}")
            if result["url"]:
                lines.append(f"   URL: {result['url']}")
            if result["snippet"]:
                lines.append(f"   {result['snippet']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    async def _search_lite(self, query: str, max_results: int) -> str:
        await self._rate_limit()
        headers = {"User-Agent": self.USER_AGENT}
        data = {"q": query, "kl": "us-en"}
        async with httpx.AsyncClient(
            timeout=self.TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.post(self.LITE_URL, data=data, headers=headers)
            response.raise_for_status()
            results = self._parse_lite(response.text, max_results)
        return self._format(results)

    async def _search_html(self, query: str, max_results: int) -> str:
        await self._rate_limit()
        headers = {"User-Agent": self.USER_AGENT}
        data = {"q": query}
        async with httpx.AsyncClient(
            timeout=self.TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.post(self.HTML_URL, data=data, headers=headers)
            response.raise_for_status()
            results = self._parse_html(response.text, max_results)
        return self._format(results)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def execute(self, **kwargs: object) -> str:
        query = str(kwargs.get("query", ""))
        if not query:
            return "Error: no query provided."
        max_results = int(kwargs.get("max_results", 5))

        # Primary: lite.duckduckgo.com (more stable, simpler markup).
        try:
            results = await self._search_lite(query, max_results)
            if results and results != "No results found.":
                return results
        except Exception:
            pass

        # Fallback: html.duckduckgo.com.
        try:
            return await self._search_html(query, max_results)
        except Exception as exc:
            return f"Error performing search: {exc}"
