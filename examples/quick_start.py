"""Quick start: minimal example of using Open Agent."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from open_agent.agent.core import Agent
from open_agent.models.base import ModelInterface, ModelResponse, ToolCall
from open_agent.tools.builtin import PythonTool
from open_agent.tools.registry import ToolRegistry

class FakeModel(ModelInterface):
    """Demo model: calls the python tool, then answers. No API key needed."""
    _n = 0
    async def chat(self, messages, tools=None):
        self._n += 1
        if self._n == 1:
            return ModelResponse(content="Let me compute that.", tool_calls=[ToolCall(name="python", arguments={"code": "print(6*7)"})])
        return ModelResponse(content="6 × 7 = 42.")

async def main():
    registry = ToolRegistry()
    registry.register(PythonTool())
    agent = Agent(model=FakeModel(), tool_registry=registry, max_steps=5)
    output = await agent.run("What is 6 times 7?")
    print(f"Answer: {output.response}\nSteps: {output.steps} | Tool calls: {len(output.tool_calls_made)}")

if __name__ == "__main__":
    asyncio.run(main())
