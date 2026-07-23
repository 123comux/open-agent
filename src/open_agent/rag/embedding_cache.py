"""Process-level embedding model cache.

Avoids loading the same SentenceTransformer model multiple times.
Each unique model name maps to a single shared instance.
"""
from __future__ import annotations

from typing import Any

_embedding_cache: dict[str, Any] = {}


def get_embedding_model(model_name: str) -> Any:
    """Return a shared SentenceTransformer instance for the given model name.

    The model is loaded on first access and cached for the lifetime of the
    process. Subsequent calls with the same model name return the same
    instance, avoiding redundant multi-hundred-MB loads.
    """
    if model_name not in _embedding_cache:
        from sentence_transformers import SentenceTransformer

        _embedding_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_cache[model_name]


def clear_embedding_cache() -> None:
    """Clear the cache. Mainly useful for tests."""
    _embedding_cache.clear()
