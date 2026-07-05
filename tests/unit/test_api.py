"""Unit tests for the FastAPI API server (async/concurrent scenarios)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx


async def test_update_settings_endpoint_serializes_concurrent_rebuilds(monkeypatch):
    """Two concurrent update requests must not crash; the lock serializes them."""
    from open_agent.server import api as api_module

    async def fake_build():
        return ("mock_agent", "mock_registry", "mock_tracer")

    monkeypatch.setattr(api_module, "_build_agent", fake_build)
    monkeypatch.setattr(api_module, "_close_model", AsyncMock())
    monkeypatch.setattr(api_module, "_close_mcp_client", AsyncMock())
    # Use a fresh lock bound to this test's event loop.
    monkeypatch.setattr(api_module, "_rebuild_lock", asyncio.Lock())
    # Snapshot module-level state so monkeypatch restores after the endpoint
    # reassigns these globals.
    monkeypatch.setattr(api_module, "_settings", api_module._settings)
    monkeypatch.setattr(api_module, "_agent", api_module._agent)
    monkeypatch.setattr(api_module, "_registry", api_module._registry)
    monkeypatch.setattr("open_agent.config.set_settings", lambda s: None)

    transport = httpx.ASGITransport(app=api_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            client.post("/api/settings", json={"max_steps": 15}),
            client.post("/api/settings", json={"max_steps": 20}),
        )

    assert all(r.status_code == 200 for r in responses)
