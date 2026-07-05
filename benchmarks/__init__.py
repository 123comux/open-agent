"""Performance benchmarks for the open-agent project.

Each submodule under ``benchmarks`` is independently runnable via
``py -m benchmarks.bench_xxx`` and exposes a ``run(quick=False)`` function
returning a result dict consumed by :mod:`benchmarks.run_all`.

These benchmarks establish a performance baseline so the effect of future
optimizations can be quantified. They do NOT depend on any external LLM API
or remote embedding service: mocks and local stubs are used throughout.
"""
from __future__ import annotations
