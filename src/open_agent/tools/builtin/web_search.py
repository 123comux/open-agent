"""Builtin web search tool using Bing and DuckDuckGo as fallback.

Primary search uses Bing (works in China), with DuckDuckGo's Lite and HTML
endpoints as fallback for users outside China or with VPN. Results are parsed
into titles, URLs and snippets with regular expressions.
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
    """Search the web via Bing (primary) or DuckDuckGo (fallback)."""

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

    BING_URL = "https://www.bing.com/search"
    LITE_URL = "https://lite.duckduckgo.com/lite/"
    HTML_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    RATE_LIMIT_SECONDS = 1.0
    _last_request_time: float = 0.0
    _rate_lock: asyncio.Lock | None = None

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove HTML tags, decode entities and collapse whitespace."""
        text = re.sub(r"<[^>]+>", "", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_url(url: str) -> str:
        """Unwrap redirect URLs and normalize protocol-relative ones."""
        if not url:
            return url
        match = re.search(r"[?&]uddg=([^&]+)", url)
        if match:
            return unquote(match.group(1))
        if url.startswith("//"):
            return "https:" + url
        return url

    async def _rate_limit(self) -> None:
        if self.RATE_LIMIT_SECONDS <= 0:
            return
        # Lazily create the lock on first use. asyncio.Lock must be created
        # inside a running event loop; doing it here (rather than at class
        # definition time) avoids "no running event loop" errors and makes
        # the lock safe across loop restarts. The check-then-assign is atomic
        # in asyncio's single-threaded model (no await between them).
        if WebSearchTool._rate_lock is None:
            WebSearchTool._rate_lock = asyncio.Lock()
        async with WebSearchTool._rate_lock:
            now = time.monotonic()
            elapsed = now - WebSearchTool._last_request_time
            if 0.0 <= elapsed < self.RATE_LIMIT_SECONDS:
                wait = self.RATE_LIMIT_SECONDS - elapsed
                await asyncio.sleep(wait)
            WebSearchTool._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Bing parsing (primary — works in China)
    # ------------------------------------------------------------------
    def _parse_bing(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse Bing search results page.

        Bing wraps results in <li class="b_algo"> blocks containing
        <h2><a href="URL">Title</a></h2> and <p class="b_lineclamp...">snippet</p>.
        """
        results: list[dict[str, str]] = []

        # Extract result blocks
        blocks = re.findall(
            r'<li\b[^>]*\bclass="b_algo"[^>]*>(.*?)</li>',
            html,
            re.DOTALL,
        )

        for block in blocks:
            # Title + URL — Bing wraps result links in <h2 class="..."><a href="...">
            # The h2 tag may carry attributes (class, id, etc.), so allow any attrs.
            title_match = re.search(
                r'<h2[^>]*>\s*<a\b([^>]*)>(.*?)</a>', block, re.DOTALL
            )
            if not title_match:
                continue
            attrs, title_html = title_match.groups()
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs)
            url = href_match.group(1) if href_match else ""
            title = self._strip_tags(title_html)

            # Snippet — Bing uses <p class="b_lineclampN"> inside <div class="b_caption">.
            # Fall back to any <p> with class containing "b_" or any <p> at all.
            snippet = ""
            for pattern in [
                r'<p\b[^>]*\bclass="b_lineclamp[^"]*"[^>]*>(.*?)</p>',
                r'<div\b[^>]*\bclass="b_caption"[^>]*>.*?<p\b[^>]*>(.*?)</p>',
                r'<p\b[^>]*\bclass="b_[^"]*"[^>]*>(.*?)</p>',
                r'<p\b[^>]*>(.*?)</p>',
            ]:
                snippet_match = re.search(pattern, block, re.DOTALL)
                if snippet_match:
                    snippet = self._strip_tags(snippet_match.group(1))
                    if snippet:
                        break

            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break

        return results

    async def _search_bing(self, query: str, max_results: int) -> str:
        """Search via Bing."""
        await self._rate_limit()
        headers = {"User-Agent": self.USER_AGENT, "Accept-Language": "zh-CN,en;q=0.9"}
        params = {"q": query, "count": str(max_results * 2)}
        client = _get_client()
        response = await client.get(self.BING_URL, params=params, headers=headers)
        response.raise_for_status()
        results = self._parse_bing(response.text, max_results)
        return self._format(results)

    # ------------------------------------------------------------------
    # DuckDuckGo parsing (fallback)
    # ------------------------------------------------------------------
    def _parse_lite(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo Lite results, pairing each link with its snippet.

        Snippets are extracted from the region between a result link and the
        next result link (rather than globally collecting all snippets and
        zipping by index) so a missing or extra snippet cannot shift the
        pairing — mirroring the block-scoped approach used by ``_parse_bing``.
        """
        link_pattern = re.compile(r'<a\b([^>]*)>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(
            r'<(?P<tag>td|a|div)\b[^>]*\bclass="result-snippet"[^>]*>(.*?)</(?P=tag)>',
            re.DOTALL,
        )
        links = [
            (m.start(), m)
            for m in link_pattern.finditer(html)
            if "result-link" in m.group(1)
        ]
        results: list[dict[str, str]] = []
        for i, (start, m) in enumerate(links):
            attrs, title_html = m.group(1), m.group(2)
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs)
            url = self._clean_url(href_match.group(1)) if href_match else ""
            end = links[i + 1][0] if i + 1 < len(links) else len(html)
            snippet = ""
            snip_match = snippet_pattern.search(html, start, end)
            if snip_match:
                snippet = self._strip_tags(snip_match.group(2))
            results.append(
                {"title": self._strip_tags(title_html), "url": url, "snippet": snippet}
            )
            if len(results) >= max_results:
                break
        return results

    def _parse_html(self, html: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML results, pairing each link with its snippet.

        Snippets are extracted from the region between a result link and the
        next result link so a missing/extra snippet cannot desync the pairing.
        """
        link_pattern = re.compile(r'<a\b([^>]*)>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(
            r'<(?P<tag>a|td|div)\b[^>]*\bclass="result__snippet"[^>]*>(.*?)</(?P=tag)>',
            re.DOTALL,
        )
        links = [
            (m.start(), m)
            for m in link_pattern.finditer(html)
            if "result__a" in m.group(1)
        ]
        results: list[dict[str, str]] = []
        for i, (start, m) in enumerate(links):
            attrs, title_html = m.group(1), m.group(2)
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs)
            url = self._clean_url(href_match.group(1)) if href_match else ""
            end = links[i + 1][0] if i + 1 < len(links) else len(html)
            snippet = ""
            snip_match = snippet_pattern.search(html, start, end)
            if snip_match:
                snippet = self._strip_tags(snip_match.group(2))
            results.append(
                {"title": self._strip_tags(title_html), "url": url, "snippet": snippet}
            )
            if len(results) >= max_results:
                break
        return results

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
    # DuckDuckGo endpoints (fallback)
    # ------------------------------------------------------------------
    async def _search_lite(self, query: str, max_results: int) -> str:
        await self._rate_limit()
        headers = {"User-Agent": self.USER_AGENT}
        data = {"q": query, "kl": "us-en"}
        client = _get_client()
        response = await client.post(self.LITE_URL, data=data, headers=headers)
        response.raise_for_status()
        results = self._parse_lite(response.text, max_results)
        return self._format(results)

    async def _search_html(self, query: str, max_results: int) -> str:
        await self._rate_limit()
        headers = {"User-Agent": self.USER_AGENT}
        data = {"q": query}
        client = _get_client()
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
        max_results_raw = kwargs.get("max_results", 5)
        max_results = int(max_results_raw) if isinstance(max_results_raw, int) else 5

        errors: list[str] = []

        # Primary: Bing (works in China).
        try:
            results = await self._search_bing(query, max_results)
            if results and results != "No results found.":
                return results
            errors.append("bing: no results parsed")
        except Exception as exc:
            errors.append(f"bing: {type(exc).__name__}: {exc}")

        # Fallback 1: lite.duckduckgo.com.
        try:
            results = await self._search_lite(query, max_results)
            if results and results != "No results found.":
                return results
            errors.append("ddg-lite: no results")
        except Exception as exc:
            errors.append(f"ddg-lite: {type(exc).__name__}: {exc}")

        # Fallback 2: html.duckduckgo.com.
        try:
            results = await self._search_html(query, max_results)
            if results and results != "No results found.":
                return results
            errors.append("ddg-html: no results")
        except Exception as exc:
            errors.append(f"ddg-html: {type(exc).__name__}: {exc}")

        return "Error performing search: " + " | ".join(errors)

# ---------------------------------------------------------------------------
# Shared HTTP client with connection pooling (module-level singleton)
# ---------------------------------------------------------------------------
# A single client is reused across requests so keep-alive connections are
# pooled, consistent with how the model adapters share an httpx.AsyncClient.
# Callers must NOT wrap this client in ``async with`` (that would close it
# after each request); use :func:`aclose` on shutdown instead.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared :class:`httpx.AsyncClient`, creating it lazily.

    The client is configured with the tool's granular timeout and connection
    limits that allow pooling of keep-alive connections.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=WebSearchTool.TIMEOUT,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
    return _client


async def aclose() -> None:
    """Close the shared HTTP client, if open. Idempotent."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
