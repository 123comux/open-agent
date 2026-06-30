"""Builtin web search tool using DuckDuckGo's HTML endpoint.

Performs a POST to ``https://html.duckduckgo.com/html/`` and scrapes the result
titles and snippets with simple regular expressions. This is intentionally
lightweight and dependency-free; for production use a dedicated search API
would be preferable.
"""
from __future__ import annotations

import re

import httpx

from open_agent.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo and return results as text."""

    name = "web_search"
    description = (
        "Search the web for a query and return a text summary of the top "
        "results (titles and snippets)."
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

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = "open-agent/0.1 (+https://github.com/your-org/open-agent)"

    @staticmethod
    def _strip_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    def _parse(self, html: str, max_results: int) -> str:
        results: list[str] = []
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', html, re.DOTALL
        )
        for i in range(min(max_results, len(titles))):
            title = self._strip_tags(titles[i])
            snippet = self._strip_tags(snippets[i]) if i < len(snippets) else ""
            results.append(f"{i + 1}. {title}\n   {snippet}")
        if not results:
            return "No results found."
        return "\n".join(results)

    async def execute(self, **kwargs: object) -> str:
        query = str(kwargs.get("query", ""))
        if not query:
            return "Error: no query provided."
        max_results = int(kwargs.get("max_results", 5))
        params = {"q": query}
        headers = {"User-Agent": self.USER_AGENT}
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.post(self.SEARCH_URL, data=params, headers=headers)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPError as exc:
            return f"Error performing search: {exc}"
        return self._parse(html, max_results)
