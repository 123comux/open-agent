# Performance Benchmarks

This directory contains performance benchmarks for the `open-agent` project.
The goal is to **establish a performance baseline** so that subsequent
optimizations have a quantified reference point, and regressions can be
detected early.

The benchmarks are self-contained: they do **not** call any external LLM API
and do **not** load real SentenceTransformer models. Mocks and local stubs
(FAISS stub, hash-based fake embeddings, mock model) are used throughout, so
the numbers reflect the framework's own overhead — not network or model
inference latency.

## Running

```bash
cd d:\github1

# Full suite (all sizes, 10 runs each)
py -m benchmarks.run_all

# Quick version (fewer samples, smaller indexes)
py -m benchmarks.run_all --quick

# JSON output (machine-readable)
py -m benchmarks.run_all --json
```

Each benchmark is also independently runnable:

```bash
py -m benchmarks.bench_context_window
py -m benchmarks.bench_rag_retrieval
py -m benchmarks.bench_agent_loop
py -m benchmarks.bench_concurrency
```

Results are written to `benchmarks/results/YYYY-MM-DD-HHMMSS.md` (and printed
to stdout as a markdown table).

## What each benchmark measures

### 1. `bench_context_window.py` — `truncate_messages` performance

Measures the cost of `open_agent.agent.context_window.truncate_messages` over
mock conversation histories of 10 / 50 / 100 / 500 messages (content 100–500
chars each).

- Single-call latency (mean ± std over 10 runs; also min / max / p50 / p99).
- Two scenarios: **fit** (budget large enough, no truncation, cache unused)
  vs **truncate** (budget tight, the internal while-loops run and the
  per-message token cache is exercised). The difference shows the cache's
  value.
- Encoding-call count per truncate run (verifies caching keeps it linear in N,
  not quadratic).
- Uses real `tiktoken` when available; otherwise falls back to the same
  length-based heuristic as the production code (and the benchmark prints a
  notice).

### 2. `bench_rag_retrieval.py` — RAG retrieval performance

Measures `FAISSStore.query` and `HybridRetriever.retrieve` over indexes of
100 / 1000 / 10000 documents.

- Single-query latency.
- Concurrent query throughput (10 / 50 concurrent queries).
- Hybrid retriever end-to-end latency (vector + BM25 + RRF fusion).
- Reranker on (NoOp) vs a CPU-bound stub reranker — to show the reranker's
  marginal cost.

Uses a pure-Python FAISS stub (inner-product `IndexFlatIP`) and hash-based
fake embeddings, so it runs without `faiss-cpu` or `sentence-transformers`
installed. If `numpy` is missing the whole benchmark is skipped with a clear
reason.

### 3. `bench_agent_loop.py` — `Agent.run` loop overhead

Measures the ReAct loop in `open_agent.agent.core.Agent.run` excluding any
real LLM call (a `MockModel` returns preset responses instantly).

- Single-step overhead (planner parse + tool execute + reflection).
- N-step loop total cost and per-step average.
- Effect of context-window truncation on loop time (varying history size and
  `max_context_tokens`).
- Token estimate of the assembled context.

Uses the real `ToolRegistry` and a mock `Tool`.

### 4. `bench_concurrency.py` — Concurrency throughput

Measures how the project's `asyncio.Lock` serialization points affect
concurrent throughput.

- `Agent.run` concurrency under the per-instance `_run_lock` (calls on the
  same Agent are serialized).
- `FAISSStore.query` concurrency under the store's `_lock`.
- `WebSearchTool` rate limiter (`_rate_limit`) concurrent serialization
  effect — measured with a reduced rate-limit interval so the benchmark
  finishes quickly; the default 1.0s interval is noted.

Output: concurrency level vs throughput curve.

## Interpreting results

| Metric | What matters | Typical range (mock environment) |
| --- | --- | --- |
| `truncate_messages` mean | Per-call cost before every LLM call | sub-ms for ≤500 msgs |
| encoding calls / run | Should be ~linear in N (cache works) | ≤ 2·N |
| `FAISSStore.query` p50 | Single retrieval latency | grows with index size |
| concurrent throughput | queries/sec under lock contention | flattens under lock |
| `Agent.run` per-step | framework overhead per ReAct iteration | sub-ms with mock model |
| rate-limiter wall time | ~N · interval (serialized) | linear in N |

The absolute numbers are environment-specific (CPU, Python version, OS). What
matters for regression detection is the **relative** change between runs on
the same machine.

## Adding a new benchmark

1. Create `benchmarks/bench_<thing>.py` with:
   - `from __future__ import annotations`
   - a `run(quick: bool = False) -> dict` function returning a result dict
     with keys: `name`, `title`, `skipped`, `skip_reason`, `headers`,
     `rows`, `notes`, `duration_s`.
   - a `if __name__ == "__main__":` block that calls `run()` and prints.
2. Register it in `benchmarks/run_all.py`'s `_BENCH_MODULES` list.
3. Follow the conventions: `time.perf_counter()` for timing, graceful skip on
   missing optional dependencies (`try/except ImportError` + a clear reason),
   no network, no real LLM.

## Known limitations

- **Mock model ≠ real LLM latency.** The agent-loop and concurrency
  benchmarks use an instant mock model, so they measure framework overhead
  only. Real end-to-end latency is dominated by the LLM provider and is not
  captured here.
- **Stub FAISS ≠ real FAISS.** The pure-Python/NumPy stub does an O(N·d) dot
  product per query; real `faiss-cpu` (with BLAS) is faster and scales
  differently. Use the stub numbers for relative comparison only.
- **Hash embeddings ≠ real embeddings.** Retrieval quality is not measured;
  only latency.
- **Platform differences.** `time.perf_counter()` resolution, GIL behavior
  and thread-pool scheduling differ between Windows and Linux. Compare runs
  on the same OS.
- **tiktoken optional.** When `tiktoken` is not installed, the context-window
  benchmark falls back to the heuristic token estimator; absolute latencies
  will be lower but still comparable across runs.

## Historical results

Each `run_all.py` invocation writes a timestamped markdown report to
[`results/`](results/). Compare the latest report against an earlier one to
spot regressions. A baseline template lives at
`results/baseline-2026-07-06.md`.
