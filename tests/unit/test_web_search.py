"""Unit tests for :class:`open_agent.tools.builtin.web_search.WebSearchTool`.

Covers the rate-limit lock (M8) and the block-scoped DuckDuckGo snippet
parsing (M10).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from open_agent.tools.builtin.web_search import WebSearchTool

# ---------------------------------------------------------------------------
# Rate-limit lock (M8)
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_rate_limit_state():
    """Reset the class-level rate-limit state before and after a test."""
    WebSearchTool._last_request_time = 0.0
    WebSearchTool._rate_lock = None
    yield
    WebSearchTool._last_request_time = 0.0
    WebSearchTool._rate_lock = None


async def test_web_search_rate_limit_serializes_concurrent(_reset_rate_limit_state):
    """Two concurrent _rate_limit calls must serialize: the second waits for
    the rate-limit window to elapse while the first holds the lock.

    Without the lock, both would read the same ``_last_request_time`` and
    neither would wait (TOCTOU). With the lock, the total elapsed time is at
    least RATE_LIMIT_SECONDS (the second call waits it out).
    """
    tool = WebSearchTool()
    tool.RATE_LIMIT_SECONDS = 0.3  # shorten to keep the test fast

    start = time.monotonic()
    await asyncio.gather(tool._rate_limit(), tool._rate_limit())
    elapsed = time.monotonic() - start

    # The first call returns immediately (elapsed from 0.0 is large). The
    # second acquires the lock, observes ~0 elapsed since the first set the
    # timestamp, and waits RATE_LIMIT_SECONDS before returning.
    assert elapsed >= 0.28
    # Serialised, not doubled: total should be well under 2 * RATE_LIMIT_SECONDS.
    assert elapsed < 0.9
    # The lock was lazily created.
    assert WebSearchTool._rate_lock is not None


async def test_web_search_rate_limit_creates_lock_lazily(_reset_rate_limit_state):
    tool = WebSearchTool()
    assert WebSearchTool._rate_lock is None
    await tool._rate_limit()
    assert WebSearchTool._rate_lock is not None


async def test_web_search_rate_limit_noop_when_zero(_reset_rate_limit_state):
    tool = WebSearchTool()
    tool.RATE_LIMIT_SECONDS = 0.0
    # Should return immediately without creating a lock.
    await tool._rate_limit()
    assert WebSearchTool._rate_lock is None


# ---------------------------------------------------------------------------
# DuckDuckGo snippet pairing (M10)
# ---------------------------------------------------------------------------


# A Lite page with a stray snippet BEFORE the first result link, plus two
# results each followed by their own snippet. The old global-zip parser would
# pair result1 with the stray snippet; the block-scoped parser pairs each link
# with the snippet that follows it.
LITE_HTML_WITH_STRAY_SNIPPET = """
<table>
<tr><td class="result-snippet">stray snippet before any link</td></tr>
<tr><td><a class="result-link" href="//example.com/r1">Result One</a></td></tr>
<tr><td class="result-snippet">snippet for result one</td></tr>
<tr><td><a class="result-link" href="//example.com/r2">Result Two</a></td></tr>
<tr><td class="result-snippet">snippet for result two</td></tr>
</table>
"""


def test_parse_lite_pairs_snippet_with_correct_link():
    tool = WebSearchTool()
    results = tool._parse_lite(LITE_HTML_WITH_STRAY_SNIPPET, max_results=5)
    assert len(results) == 2
    assert results[0]["title"] == "Result One"
    assert results[0]["url"] == "https://example.com/r1"
    assert results[0]["snippet"] == "snippet for result one"
    assert results[1]["title"] == "Result Two"
    assert results[1]["url"] == "https://example.com/r2"
    assert results[1]["snippet"] == "snippet for result two"


def test_parse_lite_missing_snippet_leaves_empty():
    """A result with no following snippet must not steal the next result's."""
    html = """
    <a class="result-link" href="//example.com/a">Alpha</a>
    <a class="result-link" href="//example.com/b">Beta</a>
    <td class="result-snippet">only betas snippet</td>
    """
    tool = WebSearchTool()
    results = tool._parse_lite(html, max_results=5)
    assert len(results) == 2
    assert results[0]["snippet"] == ""  # no snippet between alpha and beta
    assert results[1]["snippet"] == "only betas snippet"


# Same structure for the HTML endpoint (result__a / result__snippet).
HTML_WITH_STRAY_SNIPPET = """
<div class="result__snippet">stray snippet before any link</div>
<a class="result__a" href="//example.com/r1">Result One</a>
<a class="result__snippet" href="//x">snippet for result one</a>
<a class="result__a" href="//example.com/r2">Result Two</a>
<td class="result__snippet">snippet for result two</td>
"""


def test_parse_html_pairs_snippet_with_correct_link():
    tool = WebSearchTool()
    results = tool._parse_html(HTML_WITH_STRAY_SNIPPET, max_results=5)
    assert len(results) == 2
    assert results[0]["title"] == "Result One"
    assert results[0]["snippet"] == "snippet for result one"
    assert results[1]["title"] == "Result Two"
    assert results[1]["snippet"] == "snippet for result two"


def test_parse_lite_respects_max_results():
    html = """
    <a class="result-link" href="//e.com/1">One</a>
    <td class="result-snippet">s1</td>
    <a class="result-link" href="//e.com/2">Two</a>
    <td class="result-snippet">s2</td>
    <a class="result-link" href="//e.com/3">Three</a>
    <td class="result-snippet">s3</td>
    """
    tool = WebSearchTool()
    results = tool._parse_lite(html, max_results=2)
    assert len(results) == 2
    assert results[0]["title"] == "One"
    assert results[1]["title"] == "Two"


def test_web_search_tool_schema():
    tool = WebSearchTool()
    schema = tool.to_schema()
    assert schema["name"] == "web_search"
    assert schema["parameters"]["required"] == ["query"]
