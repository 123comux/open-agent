"""Multi-model: switch between OpenAI, Anthropic, and Ollama providers.

This example shows how to:
  * Configure each of the three builtin providers.
  * Select a provider based on environment variables / local availability.
  * Run the same prompt against one or more providers and compare outputs.

Set the following environment variables to use the real providers:
  * OPENAI_API_KEY    -> uses ``OpenAIModel``    (model: gpt-4o-mini by default)
  * ANTHROPIC_API_KEY -> uses ``AnthropicModel`` (model: claude-3-5-sonnet-...)
  * OLLAMA_ENABLED=1  -> uses ``OllamaModel`` pointing at a local Ollama server
                         (OLLAMA_BASE_URL and OLLAMA_MODEL override defaults).

When a key/endpoint is missing, the example prints a helpful message and falls
back to a ``MockModel`` so it always runs end-to-end.

Run with:  python examples/multi_model.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolSchema,
)
from open_agent.models.anthropic_provider import AnthropicModel
from open_agent.models.ollama_provider import OllamaModel
from open_agent.models.openai_provider import OpenAIModel
from open_agent.tools.registry import ToolRegistry


class MockModel(ModelInterface):
    """Deterministic stand-in used when no real provider is configured."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        # A short deterministic reply keeps the example self-contained.
        return ModelResponse(
            content=f"[mock {self.name}] The ReAct pattern alternates "
            "reasoning and acting until an answer is reached."
        )


def build_models() -> list[tuple[str, ModelInterface]]:
    """Build a list of (label, model) pairs from the environment.

    Real providers are used when their env var is set; otherwise a MockModel
    fallback is used so the example still runs.
    """
    models: list[tuple[str, ModelInterface]] = []

    # --- OpenAI (OpenAI-compatible API) ---
    # OpenAIModel takes an api_key, an optional base_url (any OpenAI-compatible
    # endpoint works: OpenAI itself, vLLM, LM Studio, ...), and a model name.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        models.append(
            ("openai", OpenAIModel(api_key=openai_key, model="gpt-4o-mini"))
        )
    else:
        print("OPENAI_API_KEY not set; using a mock for the OpenAI provider.")
        models.append(("openai (mock)", MockModel("openai")))

    # --- Anthropic (Claude) ---
    # AnthropicModel takes an api_key and a model name.
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        models.append(
            ("anthropic", AnthropicModel(api_key=anthropic_key))
        )
    else:
        print("ANTHROPIC_API_KEY not set; using a mock for the Anthropic provider.")
        models.append(("anthropic (mock)", MockModel("anthropic")))

    # --- Ollama (local server, no API key) ---
    # OllamaModel points at a local Ollama server; it needs no api_key. Because
    # contacting a server that isn't running would block until timeout, the real
    # provider is only enabled when OLLAMA_ENABLED is set.
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    if os.environ.get("OLLAMA_ENABLED", "").lower() in {"1", "true", "yes"}:
        models.append(
            ("ollama", OllamaModel(base_url=ollama_url, model=ollama_model))
        )
    else:
        print("OLLAMA_ENABLED not set; using a mock for the Ollama provider.")
        models.append(("ollama (mock)", MockModel("ollama")))

    return models


async def run_one(label: str, model: ModelInterface) -> tuple[str, str | None, str | None]:
    """Run a single prompt through one provider; return (label, response, error)."""
    registry = ToolRegistry()
    agent = Agent(model=model, tool_registry=registry, max_steps=3)
    prompt = "In one sentence, what is the ReAct pattern for agents?"
    try:
        output: AgentOutput = await agent.run(prompt)
        return label, output.response, None
    except Exception as exc:  # network/auth errors from a real provider
        return label, None, f"{type(exc).__name__}: {exc}"


async def main() -> None:
    models = build_models()
    print(f"\nRunning the same prompt against {len(models)} provider(s)...\n")

    # Run all providers concurrently and collect the results.
    results = await asyncio.gather(
        *(run_one(label, model) for label, model in models)
    )

    print("=== Comparison ===")
    for label, response, error in results:
        print(f"\n[{label}]")
        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  {response}")


if __name__ == "__main__":
    asyncio.run(main())
