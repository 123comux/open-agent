"""Using builtin tools (shell, python, file) with an agent.

This example shows how to:
  * Register the builtin ``ShellTool``, ``PythonTool`` and ``FileTool`` in a
    :class:`ToolRegistry`.
  * Wire that registry into an :class:`Agent`.
  * Have the agent invoke a tool (``PythonTool``) to answer a question.

A fake model is used so the example runs with no API key. The model is
programmed to (a) request the ``python`` tool, then (b) answer directly.

Run with:  python examples/with_tools.py
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
from open_agent.tools.builtin.file import FileTool
from open_agent.tools.builtin.python import PythonTool
from open_agent.tools.builtin.shell import ShellTool
from open_agent.tools.registry import ToolRegistry


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
            resp = ModelResponse(content="I'm done.")
        self._index += 1
        return resp


async def main() -> None:
    # 1. Build the tool registry and register the builtin tools.
    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(PythonTool())
    registry.register(FileTool())
    print("Registered tools:", registry.list_tools())

    # 2. Program the fake model to:
    #    (a) first call the 'python' tool to compute 23 * 17, then
    #    (b) answer the user directly using the observation.
    responses = [
        ModelResponse(
            content="Let me compute that with the python tool.",
            tool_calls=[
                ToolCall(
                    name="python",
                    arguments={"code": "print(23 * 17)"},
                )
            ],
        ),
        ModelResponse(content="23 multiplied by 17 is 391."),
    ]
    model = FakeModel(responses)

    # 3. Create and run the agent.
    agent = Agent(model=model, tool_registry=registry, max_steps=5)
    output: AgentOutput = await agent.run("What is 23 multiplied by 17?")

    print("\n=== Agent Response ===")
    print(output.response)
    print(f"\nSteps taken: {output.steps}")
    print("Tool calls made:")
    for call in output.tool_calls_made:
        print(f"  - step {call['step']}: {call['name']}({call['arguments']})")
        print(f"      observation: {call['observation']!r}")
        print(f"      is_error:    {call['is_error']}")


if __name__ == "__main__":
    asyncio.run(main())
