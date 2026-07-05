"""Integration tests for the FastAPI API server."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """Create a test client for the FastAPI app (session-scoped to avoid rebuilding agent)."""
    from open_agent.server.api import app
    with TestClient(app) as c:
        yield c


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


# ---------------------------------------------------------------------------
# Session ID validation
# ---------------------------------------------------------------------------


class TestSessionIdValidation:
    """Verify _validate_session_id rejects path traversal and invalid chars."""

    def test_clear_session_rejects_double_dot(self, client):
        response = client.delete("/api/sessions/foo..bar")
        assert response.status_code == 400

    def test_get_history_rejects_double_dot(self, client):
        response = client.get("/api/sessions/foo..bar/history")
        assert response.status_code == 400

    def test_rename_rejects_invalid_old_id(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager, "rename_session", lambda *a, **kw: None
        )
        response = client.post(
            "/api/sessions/bad..id/rename",
            json={"new_session_id": "valid-id"},
        )
        assert response.status_code == 400

    def test_rename_rejects_invalid_new_id(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager, "rename_session", lambda *a, **kw: None
        )
        response = client.post(
            "/api/sessions/valid-id/rename",
            json={"new_session_id": "bad..id"},
        )
        assert response.status_code == 400

    def test_export_rejects_double_dot(self, client):
        response = client.get("/api/sessions/foo..bar/export")
        assert response.status_code == 400

    def test_valid_session_id_with_dashes_dots_underscores(self, client):
        """Common safe IDs should pass validation."""
        for sid in ("my-session", "session.1", "user_42", "a-b.c_d"):
            response = client.get(f"/api/sessions/{sid}/history")
            assert response.status_code == 200, f"sid={sid} rejected"


# ---------------------------------------------------------------------------
# Rename / Search / Export
# ---------------------------------------------------------------------------


class TestRenameSession:
    def test_rename_success(self, client, monkeypatch):
        from open_agent.server import api as api_module

        calls: list[tuple] = []

        def fake_rename(old_id, new_id):
            calls.append((old_id, new_id))

        monkeypatch.setattr(api_module._session_manager, "rename_session", fake_rename)
        response = client.post(
            "/api/sessions/old-id/rename",
            json={"new_session_id": "new-id"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "renamed"
        assert data["old_session_id"] == "old-id"
        assert data["new_session_id"] == "new-id"
        assert calls == [("old-id", "new-id")]

    def test_rename_same_id_is_noop(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager, "rename_session", lambda *a, **kw: None
        )
        response = client.post(
            "/api/sessions/same-id/rename",
            json={"new_session_id": "same-id"},
        )
        assert response.status_code == 200

    def test_rename_duplicate_returns_409(self, client, monkeypatch):
        from open_agent.server import api as api_module

        def raise_duplicate(*a, **kw):
            raise ValueError("Session already exists")

        monkeypatch.setattr(api_module._session_manager, "rename_session", raise_duplicate)
        response = client.post(
            "/api/sessions/old-id/rename",
            json={"new_session_id": "existing-id"},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestSearchSessions:
    def test_search_returns_results(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager,
            "search_sessions",
            lambda q: [{"session_id": "s1", "matches": 2}],
        )
        response = client.get("/api/sessions/search?q=hello")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello"
        assert len(data["results"]) == 1
        assert data["results"][0]["session_id"] == "s1"

    def test_search_empty_results(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager, "search_sessions", lambda q: []
        )
        response = client.get("/api/sessions/search?q=nonexistent")
        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_search_missing_query_param(self, client):
        """FastAPI should reject requests without the required ``q`` param."""
        response = client.get("/api/sessions/search")
        assert response.status_code == 422


class TestExportSession:
    def test_export_json(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager,
            "export_session",
            lambda sid, fmt="json": '{"session_id": "test", "messages": []}',
        )
        response = client.get("/api/sessions/test-session/export?fmt=json")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]

    def test_export_markdown(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager,
            "export_session",
            lambda sid, fmt="json": "# Session: test\n",
        )
        response = client.get("/api/sessions/test-session/export?fmt=md")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]

    def test_export_markdown_alias(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(
            api_module._session_manager,
            "export_session",
            lambda sid, fmt="json": "# Session: test\n",
        )
        response = client.get("/api/sessions/test-session/export?fmt=markdown")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]

    def test_export_unsupported_format(self, client):
        response = client.get("/api/sessions/test-session/export?fmt=xml")
        assert response.status_code == 400
        assert response.json()["error"] == "unsupported_format"


# ---------------------------------------------------------------------------
# Knowledge base endpoints
# ---------------------------------------------------------------------------


class TestKnowledgeBaseEndpoints:
    def test_list_knowledge_bases(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.list_kbs.return_value = ["default", "docs"]
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.get("/api/knowledge-bases")
        assert response.status_code == 200
        assert response.json()["knowledge_bases"] == ["default", "docs"]

    def test_list_knowledge_bases_empty_on_error(self, client, monkeypatch):
        from open_agent.server import api as api_module

        def raise_error():
            raise RuntimeError("not ready")

        monkeypatch.setattr(api_module, "_get_kb_manager", raise_error)
        response = client.get("/api/knowledge-bases")
        assert response.status_code == 200
        assert response.json()["knowledge_bases"] == []

    def test_list_kb_documents(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.list_documents.return_value = [
            {"source": "a.txt", "chunks": 3},
            {"source": "b.md", "chunks": 5},
        ]
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.get("/api/knowledge-bases/test-kb/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["kb_name"] == "test-kb"
        assert len(data["documents"]) == 2

    def test_list_kb_documents_not_found(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.list_documents.side_effect = KeyError("not found")
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.get("/api/knowledge-bases/nonexistent/documents")
        assert response.status_code == 404
        assert response.json()["error"] == "kb_not_found"

    def test_delete_kb_document(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.delete_document = AsyncMock(return_value=3)
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.delete(
            "/api/knowledge-bases/test-kb/documents?source=a.txt"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["removed"] == 3
        assert data["source"] == "a.txt"

    def test_delete_kb_document_not_found(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.delete_document = AsyncMock(side_effect=KeyError("not found"))
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.delete(
            "/api/knowledge-bases/nonexistent/documents?source=a.txt"
        )
        assert response.status_code == 404

    def test_upload_document(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.index_file = AsyncMock(return_value=5)
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            params={"kb_name": "test-kb"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["chunks"] == 5
        assert data["kb_name"] == "test-kb"

    def test_upload_unsupported_file_type(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
            params={"kb_name": "test-kb"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unsupported_file_type"

    def test_upload_indexing_failure(self, client, monkeypatch):
        from open_agent.server import api as api_module

        mock_manager = MagicMock()
        mock_manager.index_file = AsyncMock(side_effect=RuntimeError("embed failed"))
        monkeypatch.setattr(api_module, "_get_kb_manager", lambda: mock_manager)

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
            params={"kb_name": "test-kb"},
        )
        assert response.status_code == 500
        assert response.json()["error"] == "indexing_failed"


# ---------------------------------------------------------------------------
# Trace endpoints
# ---------------------------------------------------------------------------


class TestTraceEndpoints:
    def test_list_traces_default(self, client):
        response = client.get("/api/traces")
        assert response.status_code == 200
        data = response.json()
        assert "traces" in data
        assert isinstance(data["traces"], list)

    def test_list_traces_with_data(self, client, monkeypatch):
        from open_agent.observability.tracer import LocalJsonlTracer
        from open_agent.server import api as api_module

        mock_tracer = MagicMock(spec=LocalJsonlTracer)
        mock_tracer.list_traces.return_value = [
            {"id": "t1", "name": "run", "status": "ok"}
        ]
        monkeypatch.setattr(api_module, "_tracer", mock_tracer)

        response = client.get("/api/traces?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["traces"]) == 1
        assert data["traces"][0]["id"] == "t1"
        mock_tracer.list_traces.assert_called_once_with(limit=10)

    def test_list_traces_with_non_local_tracer(self, client, monkeypatch):
        """When tracer is not LocalJsonlTracer, traces list is empty."""
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module, "_tracer", MagicMock())
        response = client.get("/api/traces")
        assert response.status_code == 200
        assert response.json()["traces"] == []

    def test_get_trace_found(self, client, monkeypatch):
        from open_agent.observability.tracer import LocalJsonlTracer
        from open_agent.server import api as api_module

        mock_tracer = MagicMock(spec=LocalJsonlTracer)
        mock_tracer.get_trace.return_value = {"id": "t1", "name": "run"}
        monkeypatch.setattr(api_module, "_tracer", mock_tracer)

        response = client.get("/api/traces/t1")
        assert response.status_code == 200
        assert response.json()["id"] == "t1"

    def test_get_trace_not_found(self, client, monkeypatch):
        from open_agent.observability.tracer import LocalJsonlTracer
        from open_agent.server import api as api_module

        mock_tracer = MagicMock(spec=LocalJsonlTracer)
        mock_tracer.get_trace.return_value = None
        monkeypatch.setattr(api_module, "_tracer", mock_tracer)

        response = client.get("/api/traces/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "trace_not_found"


# ---------------------------------------------------------------------------
# Settings endpoint
# ---------------------------------------------------------------------------


class TestSettingsEndpoint:
    def test_get_settings(self, client):
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "model_provider" in data
        assert "max_steps" in data
        assert "embedding_model" in data
        # Secrets should not be exposed
        assert "api_key" not in data

    def test_update_settings(self, client, monkeypatch):
        from open_agent.server import api as api_module

        async def fake_build():
            return ("mock_agent", "mock_registry", "mock_tracer")

        monkeypatch.setattr(api_module, "_build_agent", fake_build)
        # Snapshot module-level state so monkeypatch restores after the
        # endpoint reassigns these globals.
        monkeypatch.setattr(api_module, "_settings", api_module._settings)
        monkeypatch.setattr(api_module, "_agent", api_module._agent)
        monkeypatch.setattr(api_module, "_registry", api_module._registry)
        # Avoid polluting the config singleton.
        monkeypatch.setattr("open_agent.config.set_settings", lambda s: None)

        response = client.post("/api/settings", json={"max_steps": 15})
        assert response.status_code == 200
        assert response.json() == {"status": "updated"}
        assert api_module._settings.max_steps == 15

    def test_update_settings_rebuild_failure(self, client, monkeypatch):
        from open_agent.server import api as api_module

        async def failing_build():
            raise RuntimeError("rebuild failed")

        monkeypatch.setattr(api_module, "_build_agent", failing_build)
        monkeypatch.setattr(api_module, "_settings", api_module._settings)
        monkeypatch.setattr(api_module, "_agent", api_module._agent)
        monkeypatch.setattr(api_module, "_registry", api_module._registry)
        monkeypatch.setattr("open_agent.config.set_settings", lambda s: None)

        response = client.post("/api/settings", json={"max_steps": 99})
        assert response.status_code == 500
        assert response.json()["error"] == "rebuild_failed"


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    """Verify Bearer token auth protects sensitive endpoints."""

    def test_unprotected_endpoints_work_without_token(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        # Only health remains unprotected
        assert client.get("/api/health").status_code == 200
        assert client.get(
            "/api/settings",
            headers={"Authorization": "Bearer secret-token"},
        ).status_code == 200

    def test_protected_endpoint_without_token_returns_401(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        response = client.post(
            "/api/sessions/test/rename",
            json={"new_session_id": "new"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing Bearer token."
        # Newly-protected endpoints also require auth
        assert client.get("/api/tools").status_code == 401
        assert client.get("/api/sessions").status_code == 401
        assert client.get("/api/traces").status_code == 401
        assert client.get("/api/settings").status_code == 401

    def test_protected_endpoint_with_invalid_token_returns_401(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        response = client.post(
            "/api/sessions/test/rename",
            json={"new_session_id": "new"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API token."

    def test_protected_endpoint_with_valid_token_succeeds(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        monkeypatch.setattr(
            api_module._session_manager, "rename_session", lambda *a, **kw: None
        )
        response = client.post(
            "/api/sessions/old-id/rename",
            json={"new_session_id": "new-id"},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200

    def test_upload_requires_auth_when_enabled(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hi", "text/plain")},
        )
        assert response.status_code == 401

    def test_settings_update_requires_auth_when_enabled(self, client, monkeypatch):
        from open_agent.server import api as api_module

        monkeypatch.setattr(api_module._settings, "api_auth_token", "secret-token")
        response = client.post("/api/settings", json={"max_steps": 20})
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------


class TestWebSocketChat:
    def test_ws_chat_streams_events(self, client, monkeypatch):
        from open_agent.server import api as api_module

        async def fake_run_stream(message, session_id="default"):
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " world"}
            yield {
                "type": "done",
                "response": "Hello world",
                "steps": 1,
                "tool_calls_made": [],
            }

        monkeypatch.setattr(api_module._agent, "run_stream", fake_run_stream)

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hi", "session_id": "ws-test"})
            events = []
            for _ in range(10):
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "done":
                    break

        assert len(events) == 3
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "Hello"
        assert events[1]["type"] == "token"
        assert events[2]["type"] == "done"
        assert events[2]["response"] == "Hello world"

    def test_ws_chat_with_tool_events(self, client, monkeypatch):
        from open_agent.server import api as api_module

        async def fake_run_stream(message, session_id="default"):
            yield {"type": "tool_start", "name": "python", "arguments": {"code": "1+1"}}
            yield {"type": "tool_end", "name": "python", "observation": "2", "is_error": False}
            yield {"type": "done", "response": "2", "steps": 1, "tool_calls_made": []}

        monkeypatch.setattr(api_module._agent, "run_stream", fake_run_stream)

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "calc", "session_id": "ws-test"})
            events = []
            for _ in range(10):
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "done":
                    break

        types = [e["type"] for e in events]
        assert types == ["tool_start", "tool_end", "done"]
        assert events[0]["name"] == "python"
        assert events[1]["observation"] == "2"


class TestOpenAPISchema:
    def test_openapi_docs_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Open Agent API"
        assert "paths" in data

    def test_openapi_includes_new_endpoints(self, client):
        """Verify all new endpoints appear in the OpenAPI schema."""
        response = client.get("/openapi.json")
        paths = response.json()["paths"]
        expected = [
            "/api/sessions/{session_id}/rename",
            "/api/sessions/search",
            "/api/sessions/{session_id}/export",
            "/api/upload",
            "/api/knowledge-bases",
            "/api/knowledge-bases/{kb_name}/documents",
            "/api/traces",
            "/api/traces/{trace_id}",
            "/api/settings",
        ]
        for path in expected:
            assert path in paths, f"Missing endpoint {path} in OpenAPI schema"
