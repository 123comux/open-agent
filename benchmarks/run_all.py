"""Run all benchmarks and generate a report.

Usage:
  py -m benchmarks.run_all           # full suite
  py -m benchmarks.run_all --quick   # quick version (fewer samples)
  py -m benchmarks.run_all --json    # JSON output to stdout

Aggregates the four benchmark modules (context_window, rag_retrieval,
agent_loop, concurrency) into a single markdown report written to
``benchmarks/results/YYYY-MM-DD-HHMMSS.md`` and printed to stdout. Each
benchmark is imported lazily; if a module fails to import (e.g. a hard
dependency is missing) it is reported as skipped rather than aborting the
whole suite.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Callable

from rich.console import Console
from rich.markdown import Markdown


# (module_path, function_name) pairs. Imported lazily so a missing optional
# dependency in one module doesn't prevent the others from running.
_BENCH_MODULES: list[tuple[str, str]] = [
    ("benchmarks.bench_context_window", "run"),
    ("benchmarks.bench_rag_retrieval", "run"),
    ("benchmarks.bench_agent_loop", "run"),
    ("benchmarks.bench_concurrency", "run"),
]


def _import_bench(module_path: str, fn_name: str) -> tuple[Callable[..., Any] | None, str | None]:
    """Import a benchmark's run() function; return (fn, None) or (None, reason)."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - want to capture any import error
        return None, f"import {module_path} failed: {type(exc).__name__}: {exc}"
    fn = getattr(mod, fn_name, None)
    if fn is None:
        return None, f"{module_path} has no attribute {fn_name!r}"
    return fn, None


def _env_info() -> list[str]:
    lines = [
        f"- Python: {sys.version.split()[0]}",
        f"- OS: {platform.system()} {platform.release()}",
    ]
    try:
        import tiktoken  # type: ignore[import-not-found]  # noqa: F401
        lines.append("- tiktoken: installed")
    except ImportError:
        lines.append("- tiktoken: NOT installed (heuristic token estimator)")
    faiss_mod = sys.modules.get("faiss")
    if faiss_mod is not None and getattr(faiss_mod, "__file__", None):
        lines.append("- faiss: real")
    elif faiss_mod is not None:
        lines.append("- faiss: pure-Python stub (installed by a benchmark module)")
    else:
        try:
            import faiss  # type: ignore[import-not-found]  # noqa: F401
            lines.append("- faiss: real")
        except ImportError:
            lines.append("- faiss: not installed")
    try:
        import sentence_transformers  # type: ignore[import-not-found]  # noqa: F401
        lines.append("- sentence-transformers: installed")
    except ImportError:
        lines.append("- sentence-transformers: NOT installed (hash-embedding stub)")
    return lines


def _render_markdown(result: dict[str, Any]) -> str:
    """Render a single benchmark result as a markdown section."""
    lines: list[str] = [f"## {result['title']}", ""]
    if result.get("skipped"):
        lines.append(f"_Skipped: {result.get('skip_reason', 'unknown reason')}_")
        lines.append("")
        return "\n".join(lines)
    headers = result.get("headers", [])
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in result.get("rows", []):
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
    for note in result.get("notes", []):
        lines.append(f"- {note}")
    lines.append(f"- duration: {result.get('duration_s', 0)} s")
    return "\n".join(lines)


def _build_report(results: list[dict[str, Any]], quick: bool, total_s: float) -> str:
    """Assemble the full markdown report string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "quick" if quick else "full"
    lines: list[str] = [
        f"# Benchmark Report — {now}",
        "",
        f"- mode: **{mode}**",
        f"- total duration: {total_s:.2f} s",
        "",
        "## Environment",
        "",
        *_env_info(),
        "",
    ]
    for r in results:
        lines.append(_render_markdown(r))
        lines.append("")
    return "\n".join(lines)


def _save_report(text: str) -> str:
    """Write the report to benchmarks/results/YYYY-MM-DD-HHMMSS.md; return path."""
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all open-agent benchmarks.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer samples and smaller indexes.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON to stdout instead of a markdown report.",
    )
    args = parser.parse_args()

    console = Console()
    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for module_path, fn_name in _BENCH_MODULES:
        name = module_path.rsplit(".", 1)[-1]
        if not args.json:
            console.print(f"[cyan]running[/cyan] {name} ...")
        fn, err = _import_bench(module_path, fn_name)
        if fn is None:
            skipped = {
                "name": name,
                "title": module_path,
                "skipped": True,
                "skip_reason": err,
                "headers": [],
                "rows": [],
                "notes": [],
                "duration_s": 0.0,
            }
            results.append(skipped)
            console.print(f"[yellow]SKIP[/yellow] {module_path}: {err}")
            continue
        try:
            res = fn(quick=args.quick, quiet=True)
        except Exception as exc:  # noqa: BLE001 - don't let one bench abort others
            res = {
                "name": name,
                "title": module_path,
                "skipped": True,
                "skip_reason": f"runtime error: {type(exc).__name__}: {exc}",
                "headers": [],
                "rows": [],
                "notes": [],
                "duration_s": 0.0,
            }
            console.print(f"[red]ERROR[/red] {module_path}: {exc}")
        results.append(res)

    total_s = time.perf_counter() - t_start

    if args.json:
        payload = {
            "mode": "quick" if args.quick else "full",
            "total_duration_s": round(total_s, 4),
            "environment": _env_info(),
            "results": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    report = _build_report(results, args.quick, total_s)
    path = _save_report(report)
    console.print()
    console.print(Markdown(f"**Report saved to:** `{path}`"))
    console.print(Markdown(f"**Total duration:** {total_s:.2f} s ({'quick' if args.quick else 'full'} mode)"))
    # Print the markdown body to stdout (spec: "输出 markdown 表格到 stdout").
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
