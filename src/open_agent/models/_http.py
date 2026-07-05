"""Shared HTTP utilities for model providers.

Provides a reusable :class:`httpx.AsyncClient` with connection pooling and a
retry helper with exponential backoff for transient failures (429, 5xx,
network errors).
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Status codes that warrant a retry.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 30.0  # seconds

__all__ = ["request_with_retry"]


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: Any | None = None,
) -> httpx.Response:
    """Send an HTTP request with exponential-backoff retry.

    Retries on HTTP 429/5xx and :class:`httpx.TransportError`. Raises the last
    error if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        retry_after = 0.0
        try:
            response = await client.request(
                method, url, headers=headers, json=json
            )
            if response.status_code not in _RETRYABLE_STATUS:
                return response
            if response.status_code == 429:
                raw = response.headers.get("Retry-After")
                if raw is not None:
                    try:
                        retry_after = float(int(raw))
                    except (TypeError, ValueError):
                        try:
                            from datetime import datetime, timezone
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(raw)
                            if dt is not None:
                                retry_after = max(
                                    0.0,
                                    (dt - datetime.now(timezone.utc)).total_seconds(),
                                )
                            else:
                                retry_after = 0.0
                        except (TypeError, ValueError, OverflowError):
                            retry_after = 0.0
            last_exc = httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
        except httpx.TransportError as exc:
            last_exc = exc

        if attempt < _MAX_RETRIES:
            delay = min(
                _BASE_DELAY * (2**attempt) + random.uniform(0, _BASE_DELAY),
                _MAX_DELAY,
            )
            if retry_after:
                delay = max(delay, retry_after)
            logger.warning(
                "HTTP %s retry %d/%d after %.2fs: %s",
                method,
                attempt + 1,
                _MAX_RETRIES,
                delay,
                url,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
