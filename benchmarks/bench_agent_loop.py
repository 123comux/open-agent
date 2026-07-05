"""Benchmark Agent.run loop overhead (no real LLM call).

Usage: py -m benchmarks.bench_agent_loop

Measures the ReAct loop in ``open_agent.agent.core.Agent.run`` excluding any
real LLM latency: a ``MockModel`` returns preset responses instantly and a
mock ``Tool`` returns a string immediately. Uses the real ``ToolRegistry``,
``Planner`` and ``ToolExecutor``.

Measured:
- single-step overhead (planner parse + tool execute + reflection),
- N-step loop total cost and per-step average,
- effect of context-window truncation on loop time (varying history size and
  ``max_context_tokens``),
- token estimate of the assembled context.

Does NOT call any LLM API.
"""
from __future__ import annotations

import asyncio
import statistics
import string
import time
from collections.abc import AsyncIterator
from random import Random
from typing import Any

from rich.console import Console
from rich.table import Table

from open_agent.agent.context_window import estimate_messages_tokens
from open_agent.agent.core import Agent
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall as ModelToolCall,
    ToolSchema,
)
from open_agent.tools.base import Tool
from open_agent.tools.registry import ToolRegistry


class _MockModel(ModelInterface):
    """Returns queued responses in order; records every call.

    Mirrors the MockModel in ``tests/conftest.py``. When the queue is empty a
    default direct textual response is returned.
    """

    model = "mock-bench-model"

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses: list[ModelResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return ModelResponse(content="mock default response")
        return self.responses.pop(0)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        yield "mock default response"

    def queue(self, response: ModelResponse) -> None:
        self.responses.append(response)


class _EchoTool(Tool):
    """A trivial mock tool that echoes its arguments as text."""

    name = "echo"
    description = "Echo back the provided text. Used only for benchmarking."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."},
        },
        "required": ["text"],
    }

    async def execute(self, **kwargs: object) -> str:
        text = str(kwargs.get("text", ""))
        return f"echo: {text}"


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean": 0.0, "stdev": 0.0, "min": 0.0,
            "max": 0.0, "p50": 0.0, "p99": 0.0,
        }
    s = sorted(samples)
    n = len(s)

    def _pct(p: float) -> float:
        if n == 1:
            return s[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p50": _pct(0.50),
        "p99": _pct(0.99),
    }


def _build_agent(max_steps: int, max_context_tokens: int) -> tuple[Agent, _MockModel]:
    """Build an Agent with a fresh mock model and the echo tool registered."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    model = _MockModel()
    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
    )
    return agent, model


def _tool_call_response(text: str = "hello") -> ModelResponse:
    return ModelResponse(
        content=f"Calling echo with {text!r}.",
        tool_calls=[ModelToolCall(name="echo", arguments={"text": text})],
    )


def _direct_response() -> ModelResponse:
    return ModelResponse(content="done")


async def _time_run_async(coro_factory: Any, runs: int) -> list[float]:
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await coro_factory()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _make_run_factory(agent: Agent, model: _MockModel, responses: list[ModelResponse],
                      max_steps: int, user_input: str = "do the thing") -> Any:
    """Return a zero-arg async factory that re-arms the model queue and runs.

    A fresh Agent/registry is NOT rebuilt per run (we want to measure the loop,
    not construction), but the response queue is re-armed and the agent's
    short-term memory is reset so each run is independent.
    """

    async def _run() -> Any:
        model.responses = list(responses)
        agent.memory.clear()
        return await agent.run(user_input)

    return _run


async def _run_async(quick: bool) -> dict[str, Any]:
    runs = 5 if quick else 10
    n_steps = 3 if quick else 10

    headers = ["scenario", "total (ms)", "per-step (ms)", "steps", "tokens"]
    rows: list[list[str]] = []
    notes: list[str] = [
        "model: instant MockModel (no LLM API call)",
        "tool: _EchoTool (real ToolRegistry + ToolExecutor)",
        "per-step = total / steps (each step = planner parse + tool exec + reflect)",
    ]

    # 1. Single tool-call step followed by a direct answer (2 iterations).
    agent, model = _build_agent(max_steps=5, max_context_tokens=8000)
    responses = [_tool_call_response(), _direct_response()]
    factory = _make_run_factory(agent, model, responses, max_steps=5)
    samples = await _time_run_async(factory, runs)
    st = _stats(samples)
    steps = 2
    rows.append([
        "1-step (tool+direct)",
        f"{st['mean']:.3f}",
        f"{st['mean'] / steps:.3f}",
        str(steps),
        "-",
    ])

    # 2. N-step loop: queue N tool calls + 1 direct answer.
    agent2, model2 = _build_agent(max_steps=n_steps + 1, max_context_tokens=8000)
    responses_n = [_tool_call_response(f"step-{i}") for i in range(n_steps)]
    responses_n.append(_direct_response())
    factory_n = _make_run_factory(agent2, model2, responses_n, max_steps=n_steps + 1)
    samples_n = await _time_run_async(factory_n, runs)
    st_n = _stats(samples_n)
    total_steps = n_steps + 1
    rows.append([
        f"{n_steps}-step loop",
        f"{st_n['mean']:.3f}",
        f"{st_n['mean'] / total_steps:.3f}",
        str(total_steps),
        "-",
    ])

    # 3. Context-window truncation effect: pre-fill memory with K large
    # messages and run a 1-step direct response with a tight budget so
    # _apply_context_window has to truncate on every step.
    rng = Random(0)
    big_history: list[Message] = []
    for i in range(200):
        length = rng.randint(200, 400)
        body = "".join(rng.choice(string.ascii_letters + " ") for _ in range(length))
        big_history.append(
            Message(role="user" if i % 2 == 0 else "assistant", content=body)
        )

    # 3a. small context (no truncation needed): history empty, big budget.
    agent3, model3 = _build_agent(max_steps=1, max_context_tokens=8000)
    factory_small = _make_run_factory(
        agent3, model3, [_direct_response()], max_steps=1
    )
    samples_small = await _time_run_async(factory_small, runs)
    st_small = _stats(samples_small)

    # 3b. large context with truncation: pre-fill memory, tight budget.
    async def _run_large() -> Any:
        model4.responses = [_direct_response()]
        agent4.memory.clear()
        for m in big_history:
            agent4.memory.add(m)
        return await agent4.run("answer now")

    agent4, model4 = _build_agent(max_steps=1, max_context_tokens=2000)
    samples_large = await _time_run_async(_run_large, runs)
    st_large = _stats(samples_large)

    # Token estimate of the full (pre-truncation) assembled context.
    sys_msg = Message(role="system", content=agent4._system_prompt())
    full_msgs = [sys_msg] + big_history + [Message(role="user", content="answer now")]
    tokens_full = estimate_messages_tokens([m.model_dump() for m in full_msgs])

    rows.append([
        "ctx: small (no trunc)",
        f"{st_small['mean']:.3f}",
        f"{st_small['mean']:.3f}",
        "1",
        "-",
    ])
    rows.append([
        "ctx: 200 msgs, budget=2000",
        f"{st_large['mean']:.3f}",
        f"{st_large['mean']:.3f}",
        "1",
        str(tokens_full),
    ])
    notes.append(
        f"ctx-trunc: full context ~{tokens_full} tokens truncated to <=2000 "
        f"every step; overhead vs small = "
        f"{st_large['mean'] - st_small['mean']:.3f} ms"
    )

    return {
        "name": "agent_loop",
        "title": "Agent.run loop overhead (MockModel)",
        "skipped": False,
        "skip_reason": None,
        "headers": headers,
        "rows": rows,
        "notes": notes,
        "duration_s": 0.0,
    }


def run(quick: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Run the agent-loop benchmark and return a result dict."""
    console = Console(quiet=quiet)
    t_start = time.perf_counter()
    result = asyncio.run(_run_async(quick))
    duration = time.perf_counter() - t_start
    result["duration_s"] = round(duration, 4)

    table = Table(title=result["title"])
    for h in result["headers"]:
        table.add_column(h, overflow="fold")
    for r in result["rows"]:
        table.add_row(*r)
    console.print(table)
    for note in result["notes"]:
        console.print(f"  note: {note}")

    return result


if __name__ == "__main__":
    import sys as _sys
    run(quick="--quick" in _sys.argv)
