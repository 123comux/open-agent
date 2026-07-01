"""Integration tests for the FastAPI API server."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from open_agent.server.api import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    """Mock settings to avoid requiring real API keys."""
    monkeypatch.setenv("OPEN_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_AGENT_MODEL_PROVIDER", "openai")


@pytest.fixture
def mock_agent(monkeypatch):
    """Mock the agent's run method to avoid real LLM calls."""
    from open_agent.agent.core import AgentOutput
    from open_agent.server import api as api_module

    async def fake_run(message, session_id="default"):
        return AgentOutput(
            response=f"Mock response to: {message}", steps=1, tool_calls_made=[]
        )

    monkeypatch.setattr(api_module._agent, "run", fake_run)
    return api_module._agent


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestToolsEndpoint:
    def test_list_tools(self, client):
        response = client.get("/api/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        # Should have at least the built-in tools
        tool_names = [t["name"] for t in data["tools"]]
        assert "shell" in tool_names
        assert "python" in tool_names


class TestChatEndpoint:
    def test_chat_returns_response(self, client, mock_agent):
        response = client.post(
            "/api/chat", json={"message": "hello", "session_id": "test-session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "steps" in data
        assert "tool_calls_made" in data


class TestSessionEndpoints:
    def test_list_sessions(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data

    def test_get_history_empty(self, client):
        response = client.get("/api/sessions/nonexistent-session/history")
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []

    def test_clear_session(self, client):
        response = client.delete("/api/sessions/test-session")
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}


class TestOpenAPISchema:
    def test_openapi_docs_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Open Agent API"
        assert "paths" in data
