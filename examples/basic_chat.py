"""Basic chat: run a simple conversation with a fake (mock) model.

This example shows how to:
  * Build a ``ToolRegistry`` (empty here -- the agent needs no tools).
  * Create an :class:`Agent` backed by a programmable fake model.
  * Run the agent and inspect the :class:`AgentOutput` it returns
    (``response``, ``steps``, ``tool_calls_made``).

Run with:  python examples/basic_chat.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the local ``open_agent`` package importable when running this script
# directly from a source checkout (without installing the package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import Message, ModelInterface, ModelResponse, ToolSchema
from open_agent.tools.registry import ToolRegistry


class FakeModel(ModelInterface):
    """A programmable stand-in for a real LLM.

    It replays a list of canned textual replies in order so the example runs
    with no network access and no API key. In a real application you would pass
    an actual provider (e.g. ``OpenAIModel``) to the agent instead.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        if self._index < len(self._replies):
            text = self._replies[self._index]
        else:
            text = "I have nothing more to add."
        self._index += 1
        # A plain content-only response is treated by the agent as a direct
        # answer (no tool call).
        return ModelResponse(content=text)


async def main() -> None:
    # 1. A tool registry with no tools registered. The agent still works and
    #    will answer directly from the model.
    registry = ToolRegistry()

    # 2. A fake model pre-loaded with the answer we want it to produce.
    model = FakeModel(
        [
            "Hello! I'm Open Agent, a general-purpose autonomous work "
            "assistant. I can reason step by step and call tools to help you.",
        ]
    )

    # 3. Create the agent. ``max_steps`` bounds the ReAct reasoning loop.
    agent = Agent(model=model, tool_registry=registry, max_steps=5)

    # 4. Run a single user turn. ``Agent.run`` is async, so we await it.
    output: AgentOutput = await agent.run(
        "Hi! Can you tell me what you can do?"
    )

    # 5. Inspect the structured result.
    print("=== Agent Response ===")
    print(output.response)
    print(f"\nSteps taken: {output.steps}")
    print(f"Tool calls made: {len(output.tool_calls_made)}")
    # ``tool_calls_made`` is a list of dicts; empty here because we used no tools.
    for call in output.tool_calls_made:
        print(
            f"  - {call['name']}({call['arguments']}) -> {call['observation']!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
