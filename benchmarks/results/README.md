# Benchmark Results

Every run of `py -m benchmarks.run_all` writes a timestamped markdown report
to this directory named `YYYY-MM-DD-HHMMSS.md`. Each report contains:

- the aggregated markdown tables for all four benchmarks,
- any per-benchmark skip reasons (missing optional dependencies),
- the environment notes (Python version, tiktoken/faiss availability),
- the total wall-clock duration.

To compare two runs, diff the relevant tables (mean / p50 / p99 latency and
throughput). Keep older reports in place — they are the historical baseline.

## Baseline template (placeholder)

The following is the reference baseline template (`baseline-2026-07-06`).
Numbers are intentionally left blank — fill them in from the first full
`py -m benchmarks.run_all` run on a representative machine and commit the
filled-in copy as `baseline-2026-07-06.md` in this directory.

````markdown
# Baseline — 2026-07-06

> Fill in after the first representative run. Compare future runs against this.

## Environment
- Python:
- OS:
- tiktoken: installed / missing (heuristic)
- faiss: stub / real
- sentence-transformers: stub / real
- CPU:

## context_window.truncate_messages
| messages | fit mean (ms) | trunc mean (ms) | stdev | p50 | p99 | encodes/call |
| --- | --- | --- | --- | --- | --- | --- |
| 10  |  |  |  |  |  |  |
| 50  |  |  |  |  |  |  |
| 100 |  |  |  |  |  |  |
| 500 |  |  |  |  |  |  |

## rag retrieval
| docs | query p50 (ms) | qps @10 | qps @50 | hybrid (ms) | rerank on (ms) |
| --- | --- | --- | --- | --- | --- |
| 100   |  |  |  |  |  |
| 1000  |  |  |  |  |  |
| 10000 |  |  |  |  |  |

## agent loop
| scenario | total (ms) | per-step (ms) | tokens |
| --- | --- | --- | --- |
| 1-step  |  |  |  |
| N-step  |  |  |  |
| ctx-trunc (small) |  |  |  |
| ctx-trunc (large) |  |  |  |

## concurrency
| level | agent qps | faiss qps | rate-limit wall (s) |
| --- | --- | --- | --- |
| 1   |  |  |  |
| 10  |  |  |  |
| 50  |  |  |  |
| 100 |  |  |  |
````
