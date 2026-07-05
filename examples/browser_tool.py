"""Browser tool: fetch static pages or drive a headless browser.

This example demonstrates both static fetching (no extra dependencies) and the
interactive actions available when Playwright is installed:
  * ``fetch`` / ``extract_text`` / ``extract_links`` — lightweight httpx.
  * ``navigate`` / ``screenshot`` / ``click`` / ``fill`` / ``get_text`` /
    ``get_links`` / ``evaluate`` — Playwright headless browser.

Run with:  python examples/browser_tool.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.tools.builtin.browser import BrowserTool


async def main() -> None:
    tool = BrowserTool()
    url = "https://example.com"

    # Static fetch: always available.
    print("=== Static extract_text ===")
    result = await tool.execute(action="extract_text", url=url)
    print(result[:500])

    # Interactive actions require Playwright.
    print("\n=== Headless get_text ===")
    result = await tool.execute(action="get_text", url=url)
    print(result[:500])

    print("\n=== Headless screenshot (base64 preview) ===")
    result = await tool.execute(action="screenshot", url=url, full_page=True)
    print(result[:120] + "...")


if __name__ == "__main__":
    asyncio.run(main())
