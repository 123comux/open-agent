"""Benchmark RAG retrieval performance (FAISSStore.query + HybridRetriever).

Usage: py -m benchmarks.bench_rag_retrieval

Measures retrieval latency and concurrent throughput over indexes of
100 / 1000 / 10000 documents. Uses a pure-Python FAISS stub (inner-product
``IndexFlatIP``) and hash-based fake embeddings so it runs without
``faiss-cpu`` or ``sentence-transformers`` installed. If ``numpy`` is missing
the whole benchmark is skipped with a clear reason.

Measured:
- single-query latency (mean / p50 / p99),
- concurrent query throughput (10 / 50 concurrent queries),
- HybridRetriever end-to-end latency (vector + BM25 + RRF fusion),
- reranker on (NoOp) vs a CPU-bound stub reranker.

Does NOT call any real embedding model or external service.
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import statistics
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

from rich.console import Console
from rich.table import Table

# numpy is required for the stub. If missing, the whole benchmark skips.
try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None  # type: ignore[assignment]


# --- Stub the faiss module so FAISSStore imports without faiss-cpu --------
def _install_faiss_stub() -> None:
    """Install a minimal pure-Python/NumPy faiss module into sys.modules.

    Mirrors the stub pattern in ``tests/unit/test_faiss_store.py``:
    ``IndexFlatIP`` stores vectors and ranks them by inner product (which
    equals cosine similarity for L2-normalized vectors).
    """
    if "faiss" in sys.modules:
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

        def search(
            self, query: Any, k: int
        ) -> tuple[Any, Any]:
            if self._vectors is None or len(self._vectors) == 0:
                return (
                    np.zeros((1, 0), dtype=np.float32),
                    np.full((1, k), -1, dtype=np.int64),
                )
            scores = np.dot(self._vectors, query[0])
            k2 = min(k, len(scores))
            top_idx = np.argsort(-scores)[:k2]
            top_scores = scores[top_idx]
            return (
                np.array([top_scores], dtype=np.float32),
                np.array([top_idx], dtype=np.int64),
            )

    def _fake_normalize_l2(arr: Any) -> None:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            norm = float(np.linalg.norm(a))
            if norm > 0:
                arr /= norm
        else:
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            a /= norms

    def _fake_write_index(index: Any, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(index, f)

    def _fake_read_index(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    stub = types.ModuleType("faiss")
    stub.IndexFlatIP = _FakeIndex  # type: ignore[attr-defined]
    stub.normalize_L2 = _fake_normalize_l2  # type: ignore[attr-defined]
    stub.write_index = _fake_write_index  # type: ignore[attr-defined]
    stub.read_index = _fake_read_index  # type: ignore[attr-defined]
    sys.modules["faiss"] = stub


if np is not None:
    _install_faiss_stub()
    # Import here so it picks up the stub. Patch the embedding-model factory
    # with a hash-based stub before constructing any store.
    from open_agent.rag.hybrid_retriever import HybridRetriever  # noqa: E402
    from open_agent.rag.reranker import NoOpReranker, Reranker  # noqa: E402
    from open_agent.rag.stores import faiss_store as faiss_store_mod  # noqa: E402
    from open_agent.rag.stores.faiss_store import FAISSStore  # noqa: E402


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


_EMBED_DIM = 32


def _make_mock_embedding_model(dim: int = _EMBED_DIM) -> Any:
    """Build a fake SentenceTransformer returning deterministic hash embeddings.

    The same text always maps to the same vector (MD5-derived), so a query
    equal to a stored document's text returns it as the top hit. No neural
    model is loaded.
    """
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def encode(texts: list[str], **_kwargs: Any) -> Any:
        arr = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.md5(text.encode("utf-8")).digest()
            for j in range(dim):
                arr[i, j] = (digest[j % len(digest)] / 255.0) - 0.5
        return arr

    model.encode = encode
    return model


def _make_docs(n: int) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Build n mock documents with distinct, searchable text."""
    ids = [f"doc-{i}" for i in range(n)]
    docs = [f"document content number {i} about topic {(i * 7) % 23}" for i in range(n)]
    metas = [{"index": i, "topic": (i * 7) % 23} for i in range(n)]
    return ids, docs, metas


async def _build_store(n_docs: int) -> Any:
    """Construct a FAISSStore (stub-backed) and index n_docs documents."""
    model = _make_mock_embedding_model()
    orig_get = faiss_store_mod.get_embedding_model
    faiss_store_mod.get_embedding_model = lambda _name: model
    try:
        store = FAISSStore()
        ids, docs, metas = _make_docs(n_docs)
        # Add in batches to avoid repeated vstack of single rows.
        batch = 256
        for i in range(0, n_docs, batch):
            await store.add(ids[i:i + batch], docs[i:i + batch], metas[i:i + batch])
        return store
    finally:
        faiss_store_mod.get_embedding_model = orig_get


async def _bench_single_query(store: Any, runs: int) -> list[float]:
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await store.query("document content number 5", n_results=5)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


async def _bench_concurrent_query(store: Any, concurrency: int, rounds: int) -> float:
    """Return throughput (queries/sec) for ``concurrency`` parallel queries."""
    total = concurrency * rounds
    t0 = time.perf_counter()
    await asyncio.gather(*[
        store.query(f"document content number {r}", n_results=5)
        for _ in range(rounds) for r in range(concurrency)
    ])
    elapsed = time.perf_counter() - t0
    return total / elapsed if elapsed > 0 else float("inf")


async def _bench_hybrid(store: Any, runs: int) -> list[float]:
    retriever = HybridRetriever(store, top_k=5, reranker=NoOpReranker())
    # Warm the BM25 index so the first measured call isn't penalized.
    await retriever.retrieve("document content number 5")
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await retriever.retrieve("document content number 5")
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


class _StubReranker(Reranker):
    """A reranker that does real CPU work (sort + trivial re-score).

    Stands in for a cross-encoder so the reranker-on path has non-zero cost
    without loading sentence-transformers.
    """

    def rank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for d in documents:
            doc = str(d.get("document", ""))
            # Trivial lexical-overlap score: fraction of query tokens in doc.
            qtoks = set(query.lower().split())
            dtoks = doc.lower().split()
            overlap = sum(1 for t in dtoks if t in qtoks)
            d["rerank_score"] = float(overlap) / max(1, len(dtoks))
        documents.sort(key=lambda d: d["rerank_score"], reverse=True)
        return documents


async def _bench_rerank_on(store: Any, runs: int) -> list[float]:
    retriever = HybridRetriever(store, top_k=5, reranker=_StubReranker())
    await retriever.retrieve("document content number 5")
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await retriever.retrieve("document content number 5")
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


async def _run_async(quick: bool) -> dict[str, Any]:
    sizes = [100, 1000] if quick else [100, 1000, 10000]
    concurrency_levels = [10] if quick else [10, 50]
    runs = 5 if quick else 10

    headers = [
        "docs", "q mean (ms)", "q p50", "q p99",
        "qps@10", "qps@50", "hybrid (ms)", "rerank on (ms)",
    ]
    rows: list[list[str]] = []
    notes: list[str] = [
        "faiss: pure-Python/NumPy stub (inner-product IndexFlatIP)",
        "embeddings: deterministic MD5 hash stub (no SentenceTransformer)",
        "BM25: project's built-in _BM25 fallback (rank_bm25 optional)",
    ]

    for n_docs in sizes:
        store = await _build_store(n_docs)
        single = await _bench_single_query(store, runs)
        st = _stats(single)

        qps: dict[int, float] = {}
        for c in concurrency_levels:
            qps[c] = await _bench_concurrent_query(store, c, rounds=3)

        hybrid = await _bench_hybrid(store, runs)
        rerank_on = await _bench_rerank_on(store, runs)

        rows.append([
            str(n_docs),
            f"{st['mean']:.3f}",
            f"{st['p50']:.3f}",
            f"{st['p99']:.3f}",
            f"{qps.get(10, 0):.1f}" if 10 in qps else "-",
            f"{qps.get(50, 0):.1f}" if 50 in qps else "-",
            f"{statistics.mean(hybrid):.3f}",
            f"{statistics.mean(rerank_on):.3f}",
        ])

    return {
        "name": "rag_retrieval",
        "title": "RAG retrieval (FAISSStore + HybridRetriever)",
        "skipped": False,
        "skip_reason": None,
        "headers": headers,
        "rows": rows,
        "notes": notes,
        "duration_s": 0.0,  # filled by caller
    }


def run(quick: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Run the RAG retrieval benchmark and return a result dict."""
    console = Console(quiet=quiet)
    if np is None:
        msg = "numpy not installed -> RAG retrieval benchmark skipped"
        console.print(f"[yellow]skip:[/yellow] {msg}")
        return {
            "name": "rag_retrieval",
            "title": "RAG retrieval (FAISSStore + HybridRetriever)",
            "skipped": True,
            "skip_reason": msg,
            "headers": [],
            "rows": [],
            "notes": [],
            "duration_s": 0.0,
        }

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
