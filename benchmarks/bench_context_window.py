"""Benchmark context_window.truncate_messages performance.

Usage: py -m benchmarks.bench_context_window

Measures the cost of ``truncate_messages`` over mock conversation histories
of 10 / 50 / 100 / 500 messages (content 100-500 chars each). Reports
mean / stdev / min / max / p50 / p99 over 10 runs, plus the number of
``_estimate_single_message_tokens`` calls per run (to verify the per-message
token cache keeps encoding work linear in N rather than quadratic).

Uses real ``tiktoken`` when installed; otherwise falls back to the same
length-based heuristic the production code uses (a notice is printed).
Does not depend on any LLM API.
"""
from __future__ import annotations

import random
import statistics
import string
import time
from typing import Any

from rich.console import Console
from rich.table import Table

import open_agent.agent.context_window as cw
from open_agent.agent.context_window import (
    _estimate_single_message_tokens,
    estimate_messages_tokens,
    truncate_messages,
)

try:
    import tiktoken  # type: ignore[import-not-found]  # noqa: F401

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


def _stats(samples: list[float]) -> dict[str, float]:
    """Return mean/stdev/min/max/p50/p99 for a list of float samples."""
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


def _make_messages(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Build a mock conversation: 1 system message + n user/assistant turns.

    Each message content is 100-500 random ASCII characters.
    """
    rng = random.Random(seed)
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    for i in range(n):
        length = rng.randint(100, 500)
        body = "".join(rng.choice(string.ascii_letters + " ") for _ in range(length))
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": body})
    return msgs


def _time_call(fn: Any, runs: int) -> list[float]:
    """Run ``fn`` ``runs`` times, returning per-call latencies in ms."""
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _count_encodings(messages: list[dict[str, Any]], max_tokens: int) -> int:
    """Count ``_estimate_single_message_tokens`` invocations during one truncate.

    With the per-message cache active this stays ~linear in N (each message
    encoded at most once when the cache is warm). Without caching the
    while-loops would re-encode every remaining message per iteration (O(N^2)).
    """
    count = 0
    real = _estimate_single_message_tokens

    def counting(msg: dict[str, Any], model: str) -> int:
        nonlocal count
        count += 1
        return real(msg, model)

    orig = cw._estimate_single_message_tokens
    cw._estimate_single_message_tokens = counting
    try:
        truncate_messages(
            messages,
            max_tokens=max_tokens,
            preserve_system=True,
            preserve_recent=2,
        )
    finally:
        cw._estimate_single_message_tokens = orig
    return count


def _maybe_plot(sizes: list[int], fit_means: list[float],
                trunc_means: list[float]) -> str | None:
    """Save a bar chart to results/ if matplotlib is available."""
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return None
    import os

    x = list(range(len(sizes)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in x], fit_means, w, label="fit (no truncation)")
    ax.bar([i + w / 2 for i in x], trunc_means, w, label="truncate (cache hits)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("message count")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title("truncate_messages latency")
    ax.legend()
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "context_window.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run(quick: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Run the context-window benchmark and return a result dict.

    Args:
        quick: When True, use fewer sizes and runs for a fast smoke test.
        quiet: When True, suppress console table output (used by run_all).
    """
    t_start = time.perf_counter()
    console = Console(quiet=quiet)

    sizes = [10, 50] if quick else [10, 50, 100, 500]
    runs = 5 if quick else 10

    notes: list[str] = []
    if _HAS_TIKTOKEN:
        notes.append("tiktoken: real (cl100k_base encoding)")
    else:
        notes.append(
            "tiktoken: NOT installed -> using length-based heuristic estimator "
            "(same fallback as production code); absolute latencies are lower "
            "but still comparable across runs."
        )

    headers = [
        "messages", "fit mean (ms)", "trunc mean (ms)", "stdev",
        "min", "p50", "p99", "max", "encodes/call",
    ]
    rows: list[list[str]] = []
    fit_means: list[float] = []
    trunc_means: list[float] = []

    for n in sizes:
        msgs = _make_messages(n, seed=n)
        total_tokens = estimate_messages_tokens(msgs)
        # "fit" budget: large enough that no truncation is needed (cache built
        # but the while-loops never run -> no cache hits).
        fit_budget = total_tokens * 2 + 1000
        # "truncate" budget: forces the middle-drop loop to run and exercise
        # the per-message token cache.
        trunc_budget = max(64, total_tokens // 2)

        fit_samples = _time_call(
            lambda m=msgs, b=fit_budget: truncate_messages(
                m, max_tokens=b, preserve_system=True, preserve_recent=2
            ),
            runs,
        )
        trunc_samples = _time_call(
            lambda m=msgs, b=trunc_budget: truncate_messages(
                m, max_tokens=b, preserve_system=True, preserve_recent=2
            ),
            runs,
        )
        st = _stats(trunc_samples)
        encodes = _count_encodings(msgs, trunc_budget)

        fit_means.append(statistics.mean(fit_samples))
        trunc_means.append(statistics.mean(trunc_samples))
        rows.append([
            str(n),
            f"{statistics.mean(fit_samples):.4f}",
            f"{st['mean']:.4f}",
            f"{st['stdev']:.4f}",
            f"{st['min']:.4f}",
            f"{st['p50']:.4f}",
            f"{st['p99']:.4f}",
            f"{st['max']:.4f}",
            f"{encodes} (<=2·{len(msgs)}={2 * len(msgs)})",
        ])

    duration = time.perf_counter() - t_start

    plot_path = _maybe_plot(sizes, fit_means, trunc_means)
    if plot_path:
        notes.append(f"chart saved to: {plot_path}")

    # Print to console when run directly.
    table = Table(title="context_window.truncate_messages (ms over %d runs)" % runs)
    for h in headers:
        table.add_column(h, overflow="fold")
    for r in rows:
        table.add_row(*r)
    console.print(table)
    for note in notes:
        console.print(f"  note: {note}")

    return {
        "name": "context_window",
        "title": "context_window.truncate_messages",
        "skipped": False,
        "skip_reason": None,
        "headers": headers,
        "rows": rows,
        "notes": notes,
        "duration_s": round(duration, 4),
    }


if __name__ == "__main__":
    import sys as _sys
    run(quick="--quick" in _sys.argv)
