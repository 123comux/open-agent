"""Custom tool: define a new Tool subclass and let the agent use it.

This example shows how to:
  * Subclass :class:`open_agent.tools.base.Tool` to create a custom capability.
  * Register the custom tool in a :class:`ToolRegistry` alongside a builtin one.
  * Have the agent invoke the custom tool via a fake model.

A fake model is used so the example runs with no API key.

Run with:  python examples/custom_tool.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolSchema,
)
from open_agent.tools.base import Tool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.registry import ToolRegistry


class WordCountTool(Tool):
    """Count the words in a piece of text -- a custom tool defined inline.

    A :class:`Tool` subclass must set the ``name``, ``description`` and
    ``parameters`` class attributes and implement :meth:`execute`, which returns
    a textual result.
    """

    name = "word_count"
    description = (
        "Count the number of words in a given text and return the count."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text whose words should be counted.",
            }
        },
        "required": ["text"],
    }

    async def execute(self, **kwargs: object) -> str:
        text = str(kwargs.get("text", ""))
        if not text:
            return "0 words (empty input)."
        count = len(text.split())
        return f"{count} words"


class FakeModel(ModelInterface):
    """Programmable fake model that replays a list of canned ModelResponses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        if self._index < len(self._responses):
            resp = self._responses[self._index]
        else:
            resp = ModelResponse(content="Done.")
        self._index += 1
        return resp


async def main() -> None:
    # 1. Register the custom tool together with a builtin one.
    registry = ToolRegistry()
    registry.register(WordCountTool())
    registry.register(PythonTool())
    print("Registered tools:", registry.list_tools())

    # 2. Program the fake model to call our custom word_count tool, then answer.
    sample_text = "Open Agent makes it easy to build autonomous assistants."
    responses = [
        ModelResponse(
            content="I'll count the words with the word_count tool.",
            tool_calls=[
                ToolCall(name="word_count", arguments={"text": sample_text})
            ],
        ),
        ModelResponse(content="That sentence has 9 words."),
    ]
    model = FakeModel(responses)

    # 3. Run the agent.
    agent = Agent(model=model, tool_registry=registry, max_steps=5)
    output: AgentOutput = await agent.run(
        f"How many words are in this sentence? {sample_text!r}"
    )

    print("\n=== Agent Response ===")
    print(output.response)
    print(f"\nSteps taken: {output.steps}")
    print("Tool calls made:")
    for call in output.tool_calls_made:
        print(
            f"  - {call['name']}({call['arguments']}) -> {call['observation']!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
