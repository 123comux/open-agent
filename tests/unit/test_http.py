"""Tests for the shared HTTP retry helper in :mod:`open_agent.models._http`."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from open_agent.models import _http
from open_agent.models._http import request_with_retry


def _make_response(
    status_code: int, headers: dict[str, str] | None = None
) -> MagicMock:
    """Build a fake ``httpx.Response`` with the given status and headers."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers if headers is not None else {}
    response.request = MagicMock()
    return response


def _mock_client(side_effect: list) -> AsyncMock:
    """Build a fake ``httpx.AsyncClient`` whose ``request`` follows ``side_effect``."""
    client = AsyncMock()
    client.request = AsyncMock(side_effect=side_effect)
    return client


@pytest.fixture
def no_sleep() -> AsyncMock:
    """Patch ``asyncio.sleep`` (as seen by ``_http``) so retries don't really wait."""
    sleep_mock = AsyncMock()
    fake_asyncio = MagicMock()
    fake_asyncio.sleep = sleep_mock
    with patch.object(_http, "asyncio", fake_asyncio):
        yield sleep_mock


@pytest.fixture
def no_jitter() -> MagicMock:
    """Patch ``random.uniform`` (as seen by ``_http``) to return 0.0."""
    uniform_mock = MagicMock(return_value=0.0)
    fake_random = MagicMock()
    fake_random.uniform = uniform_mock
    with patch.object(_http, "random", fake_random):
        yield uniform_mock


async def test_success_no_retry(no_sleep, no_jitter):
    """A 200 response is returned immediately without sleeping."""
    response = _make_response(200)
    client = _mock_client([response])
    result = await request_with_retry(client, "GET", "https://example.com")
    assert result is response
    assert client.request.await_count == 1
    assert no_sleep.await_count == 0


async def test_retry_on_429_then_success(no_sleep, no_jitter):
    """A 429 followed by 200 triggers exactly one retry."""
    err_response = _make_response(429)
    ok_response = _make_response(200)
    client = _mock_client([err_response, ok_response])
    result = await request_with_retry(client, "POST", "https://example.com")
    assert result is ok_response
    assert client.request.await_count == 2
    assert no_sleep.await_count == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_retry_on_5xx(status, no_sleep, no_jitter):
    """Each 5xx retryable status triggers one retry before succeeding."""
    err_response = _make_response(status)
    ok_response = _make_response(200)
    client = _mock_client([err_response, ok_response])
    result = await request_with_retry(client, "GET", "https://example.com")
    assert result is ok_response
    assert client.request.await_count == 2
    assert no_sleep.await_count == 1


async def test_retry_on_408(no_sleep, no_jitter):
    """The newly-added 408 status is retryable."""
    err_response = _make_response(408)
    ok_response = _make_response(200)
    client = _mock_client([err_response, ok_response])
    result = await request_with_retry(client, "GET", "https://example.com")
    assert result is ok_response
    assert client.request.await_count == 2
    assert no_sleep.await_count == 1


@pytest.mark.parametrize("status", [400, 404])
async def test_no_retry_on_non_retryable(status, no_sleep, no_jitter):
    """Non-retryable statuses are returned immediately (caller raises later)."""
    response = _make_response(status)
    client = _mock_client([response])
    result = await request_with_retry(client, "GET", "https://example.com")
    assert result is response
    assert client.request.await_count == 1
    assert no_sleep.await_count == 0


async def test_max_retries_exhausted(no_sleep, no_jitter):
    """All retries exhausted raises the last ``HTTPStatusError``."""
    total = _http._MAX_RETRIES + 1
    responses = [_make_response(429) for _ in range(total)]
    client = _mock_client(responses)
    with pytest.raises(httpx.HTTPStatusError):
        await request_with_retry(client, "GET", "https://example.com")
    assert client.request.await_count == total
    assert no_sleep.await_count == _http._MAX_RETRIES


async def test_retry_after_header_honored(no_sleep, no_jitter):
    """A 429 with ``Retry-After: 5`` waits at least 5 seconds."""
    err_response = _make_response(429, headers={"Retry-After": "5"})
    ok_response = _make_response(200)
    client = _mock_client([err_response, ok_response])
    await request_with_retry(client, "GET", "https://example.com")
    assert no_sleep.await_count == 1
    delay = no_sleep.await_args.args[0]
    assert delay >= 5.0


async def test_jitter_added_to_delay(no_sleep):
    """The delay includes random jitter on top of the base backoff."""
    fake_random = MagicMock()
    fake_random.uniform = MagicMock(return_value=0.5)
    with patch.object(_http, "random", fake_random):
        err_response = _make_response(429)
        ok_response = _make_response(200)
        client = _mock_client([err_response, ok_response])
        await request_with_retry(client, "GET", "https://example.com")
    assert no_sleep.await_count == 1
    delay = no_sleep.await_args.args[0]
    assert delay == pytest.approx(1.5)


async def test_exponential_backoff(no_sleep, no_jitter):
    """Delays grow exponentially (1s, 2s, 4s) with jitter zeroed out."""
    total = _http._MAX_RETRIES + 1
    responses = [_make_response(429) for _ in range(total)]
    client = _mock_client(responses)
    with pytest.raises(httpx.HTTPStatusError):
        await request_with_retry(client, "GET", "https://example.com")
    delays = [call.args[0] for call in no_sleep.await_args_list]
    assert delays == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(4.0)]


async def test_max_delay_cap(no_sleep, no_jitter):
    """Delay never exceeds ``_MAX_DELAY`` even with a large base delay."""
    total = _http._MAX_RETRIES + 1
    responses = [_make_response(429) for _ in range(total)]
    client = _mock_client(responses)
    with patch.object(_http, "_BASE_DELAY", 20.0):
        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retry(client, "GET", "https://example.com")
    delays = [call.args[0] for call in no_sleep.await_args_list]
    assert all(d <= _http._MAX_DELAY for d in delays)
    assert delays[1] == pytest.approx(_http._MAX_DELAY)


async def test_retry_on_transport_error(no_sleep, no_jitter):
    """A :class:`httpx.TransportError` triggers a retry before succeeding."""
    ok_response = _make_response(200)
    client = _mock_client([httpx.ConnectError("boom"), ok_response])
    result = await request_with_retry(client, "GET", "https://example.com")
    assert result is ok_response
    assert client.request.await_count == 2
    assert no_sleep.await_count == 1


async def test_request_args_forwarded(no_sleep, no_jitter):
    """Headers and json body are forwarded to ``client.request``."""
    response = _make_response(200)
    client = _mock_client([response])
    headers = {"Authorization": "Bearer x"}
    payload = {"model": "gpt"}
    await request_with_retry(
        client, "POST", "https://example.com", headers=headers, json=payload
    )
    client.request.assert_awaited_once_with(
        "POST", "https://example.com", headers=headers, json=payload
    )
