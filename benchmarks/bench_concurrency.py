"""Benchmark concurrency throughput under the project's asyncio.Lock points.

Usage: py -m benchmarks.bench_concurrency

Measures how the project's serialization points affect concurrent throughput:

- ``Agent.run`` concurrency under the per-instance ``_run_lock`` (calls on the
  same Agent are serialized) vs separate agents (truly parallel).
- ``FAISSStore.query`` concurrency under the store's ``_lock``.
- ``WebSearchTool._rate_limit`` concurrent serialization effect (measured with
  a reduced rate-limit interval so the benchmark finishes fast; the default
  1.0s interval is noted).

Uses an instant MockModel, a mock tool and a pure-Python FAISS stub. No
network, no real LLM API.
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import statistics
import sys
import time
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

from rich.console import Console
from rich.table import Table

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None  # type: ignore[assignment]

from open_agent.agent.core import Agent
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolSchema,
)
from open_agent.tools.base import Tool
from open_agent.tools.builtin.web_search import WebSearchTool
from open_agent.tools.registry import ToolRegistry


# --- faiss stub (compact, self-contained) ----------------------------------
def _install_faiss_stub() -> None:
    if np is None or "faiss" in sys.modules:
        return

    class _FakeIndex:
        def __init__(self, dim: int) -> None:
            self.d = dim
            self._vectors: Any = None

        def add(self, vectors: Any) -> None:
            arr = np.asarray(vectors, dtype=np.float32)
            if self._vectors is None:
                self._vectors = np.array(arr, dtype=np.float32)
            else:
                self._vectors = np.vstack([self._vectors, arr])

        def search(self, query: Any, k: int) -> tuple[Any, Any]:
            if self._vectors is None or len(self._vectors) == 0:
                return (
                    np.zeros((1, 0), dtype=np.float32),
                    np.full((1, k), -1, dtype=np.int64),
                )
            scores = np.dot(self._vectors, query[0])
            k2 = min(k, len(scores))
            top_idx = np.argsort(-scores)[:k2]
            return (
                np.array([scores[top_idx]], dtype=np.float32),
                np.array([top_idx], dtype=np.int64),
            )

    def _norm(arr: Any) -> None:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            nrm = float(np.linalg.norm(a))
            if nrm > 0:
                arr /= nrm
        else:
            nrms = np.linalg.norm(a, axis=1, keepdims=True)
            nrms[nrms == 0] = 1.0
            a /= nrms

    stub = types.ModuleType("faiss")
    stub.IndexFlatIP = _FakeIndex  # type: ignore[attr-defined]
    stub.normalize_L2 = _norm  # type: ignore[attr-defined]
    stub.write_index = lambda idx, p: pickle.dump(idx, open(p, "wb"))  # type: ignore[attr-defined]
    stub.read_index = lambda p: pickle.load(open(p, "rb"))  # type: ignore[attr-defined]
    sys.modules["faiss"] = stub


_install_faiss_stub()
if np is not None:
    from open_agent.rag.stores import faiss_store as faiss_store_mod  # noqa: E402
    from open_agent.rag.stores.faiss_store import FAISSStore  # noqa: E402


# --- mocks -----------------------------------------------------------------
class _InstantModel(ModelInterface):
    """Returns a direct response instantly. Tracks peak concurrency."""

    model = "mock-concurrency"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        # Tiny yield so concurrent coroutines can interleave.
        await asyncio.sleep(0)
        self.in_flight -= 1
        return ModelResponse(content="ok")

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[str]:
        yield "ok"


class _NoopTool(Tool):
    name = "noop"
    description = "Does nothing. Benchmark only."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: object) -> str:
        return "ok"


def _make_embedding_model(dim: int = 32) -> Any:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def encode(texts: list[str], **_kw: Any) -> Any:
        arr = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            d = hashlib.md5(t.encode("utf-8")).digest()
            for j in range(dim):
                arr[i, j] = (d[j % len(d)] / 255.0) - 0.5
        return arr

    model.encode = encode
    return model


# --- timing helpers --------------------------------------------------------
def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p99": 0.0}
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


async def _throughput(coro_factory: Any, n: int, rounds: int) -> dict[str, float]:
    """Run n concurrent coroutines ``rounds`` times; return wall/throughput."""
    walls: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        await asyncio.gather(*[coro_factory(i) for i in range(n)])
        walls.append(time.perf_counter() - t0)
    st = _stats([w * 1000.0 for w in walls])
    total = n * rounds
    mean_wall = sum(walls) / len(walls) if walls else 1.0
    return {
        "wall_ms": st["mean"],
        "qps": total / mean_wall if mean_wall > 0 else 0.0,
        "p50_ms": st["p50"],
        "p99_ms": st["p99"],
    }


# --- benchmarks ------------------------------------------------------------
async def _bench_agent_concurrency(level: int) -> dict[str, float]:
    """Same-agent (serialized by _run_lock) vs separate agents (parallel)."""

    async def shared(i: int) -> Any:
        await shared_agent.run(f"task {i}")

    async def separate(i: int) -> Any:
        ag = Agent(model=_InstantModel(), tool_registry=registry, max_steps=1)
        await ag.run(f"task {i}")

    registry = ToolRegistry()
    registry.register(_NoopTool())

    shared_agent = Agent(model=_InstantModel(), tool_registry=registry, max_steps=1)
    shared = await _throughput(shared, level, rounds=3)
    separate_res = await _throughput(separate, level, rounds=3)
    return {"shared_qps": shared["qps"], "separate_qps": separate_res["qps"],
            "shared_wall_ms": shared["wall_ms"]}


async def _bench_faiss_concurrency(level: int) -> dict[str, float]:
    """Concurrent FAISSStore.query under the store's _lock."""
    model = _make_embedding_model()
    orig = faiss_store_mod.get_embedding_model
    faiss_store_mod.get_embedding_model = lambda _name: model
    try:
        store = FAISSStore()
        ids = [f"d-{i}" for i in range(1000)]
        docs = [f"document content {i}" for i in range(1000)]
        batch = 256
        for i in range(0, 1000, batch):
            await store.add(ids[i:i + batch], docs[i:i + batch])

        async def one(i: int) -> Any:
            await store.query(f"document content {i % 1000}", n_results=5)

        return await _throughput(one, level, rounds=3)
    finally:
        faiss_store_mod.get_embedding_model = orig


async def _bench_rate_limiter(level: int, interval: float = 0.05) -> dict[str, float]:
    """Concurrent WebSearchTool._rate_limit calls (serialized, ~interval apart).

    Uses a reduced interval so the benchmark finishes quickly; the production
    default is 1.0s.
    """
    WebSearchTool.RATE_LIMIT_SECONDS = interval
    WebSearchTool._last_request_time = 0.0
    WebSearchTool._rate_lock = None  # force lazy re-create
    tool = WebSearchTool()

    async def one(i: int) -> None:
        await tool._rate_limit()

    res = await _throughput(one, level, rounds=1)
    # Restore defaults.
    WebSearchTool.RATE_LIMIT_SECONDS = 1.0
    WebSearchTool._last_request_time = 0.0
    WebSearchTool._rate_lock = None
    return res


async def _run_async(quick: bool) -> dict[str, Any]:
    levels = [1, 5, 10] if quick else [1, 5, 10, 50, 100]

    headers = [
        "concurrency", "agent shared qps", "agent separate qps",
        "faiss qps", "rate-limit wall (ms)",
    ]
    rows: list[list[str]] = []
    notes: list[str] = [
        "agent shared = same Agent instance (serialized by _run_lock)",
        "agent separate = N Agent instances (truly parallel, baseline)",
        "rate-limit interval reduced to 0.05s for the benchmark (default 1.0s)",
    ]

    for level in levels:
        ag = await _bench_agent_concurrency(level)
        faiss_res: dict[str, float] = {"qps": 0.0, "wall_ms": 0.0}
        if np is not None:
            faiss_res = await _bench_faiss_concurrency(level)
        rl = await _bench_rate_limiter(level, interval=0.05)
        rows.append([
            str(level),
            f"{ag['shared_qps']:.1f}",
            f"{ag['separate_qps']:.1f}",
            f"{faiss_res['qps']:.1f}" if np is not None else "-",
            f"{rl['wall_ms']:.1f}",
        ])

    return {
        "name": "concurrency",
        "title": "Concurrency throughput (asyncio.Lock serialization points)",
        "skipped": False,
        "skip_reason": None,
        "headers": headers,
        "rows": rows,
        "notes": notes,
        "duration_s": 0.0,
    }


def run(quick: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Run the concurrency benchmark and return a result dict."""
    console = Console(quiet=quiet)
    t_start = time.perf_counter()
    result = asyncio.run(_run_async(quick))
    duration = time.perf_counter() - t_start
    result["duration_s"] = round(duration, 4)

    if np is None:
        result["notes"].append(
            "numpy not installed -> FAISSStore query column skipped (-)"
        )

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
