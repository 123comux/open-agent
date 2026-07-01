"""FastAPI server exposing the open-agent core over HTTP and WebSocket.

This module is a thin adapter layer: it builds an :class:`Agent` from settings
and exposes it via REST and WebSocket endpoints. It depends on the optional
``server`` extra (``fastapi`` and ``uvicorn``); importing it without those
packages raises a clear :class:`ImportError`.
"""
from __future__ import annotations

from typing import Any

from open_agent.config import get_settings
from open_agent.memory.session_manager import SessionManager

# FastAPI is an optional dependency; provide a clear error if missing.
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The server extra is required to use open_agent.server.api. "
        "Install it with: pip install 'open-agent[server]'"
    ) from exc


def _build_agent():
    """Construct an Agent and its registry from settings (lazy heavy imports)."""
    from open_agent.agent.core import Agent
    from open_agent.models.base import ModelInterface
    from open_agent.tools.builtin import FileTool, PythonTool, ShellTool, WebSearchTool
    from open_agent.tools.registry import ToolRegistry

    settings = get_settings()
    provider = settings.model_provider
    model: ModelInterface
    if provider == "anthropic":
        from open_agent.models.anthropic_provider import AnthropicModel

        model = AnthropicModel(
            api_key=settings.api_key, model=settings.model_name, timeout=settings.request_timeout
        )
    elif provider == "ollama":
        from open_agent.models.ollama_provider import OllamaModel

        model = OllamaModel(
            base_url=settings.base_url, model=settings.model_name, timeout=settings.request_timeout
        )
    else:
        from open_agent.models.openai_provider import OpenAIModel

        model = OpenAIModel(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model_name,
            timeout=settings.request_timeout,
        )

    registry = ToolRegistry()
    for tool in (ShellTool(), PythonTool(), FileTool(), WebSearchTool()):
        registry.register(tool)
    return (
        Agent(
            model=model,
            tool_registry=registry,
            max_steps=settings.max_steps,
            session_manager=_session_manager,
        ),
        registry,
    )


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    response: str
    steps: int
    tool_calls_made: list[dict] = Field(default_factory=list)


_settings = get_settings()
_session_manager = SessionManager(
    max_messages=_settings.short_term_memory_size,
    storage_dir=_settings.session_storage_dir,
)

app = FastAPI(title="Open Agent API", version="0.1.0")
_agent, _registry = _build_agent()


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/tools")
async def list_tools() -> dict[str, Any]:
    """List the names and schemas of available tools."""
    return {"tools": _registry.schemas()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a single message and return the agent's response."""
    output = await _agent.run(request.message, session_id=request.session_id)
    return ChatResponse(
        response=output.response,
        steps=output.steps,
        tool_calls_made=output.tool_calls_made,
    )


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Stream a conversation over WebSocket with real-time token streaming.

    Receives JSON messages of shape ``{"message": "...", "session_id": "..."}``
    and streams back events as they occur:

    - ``{"type": "token", "content": "..."}`` -- incremental response text
    - ``{"type": "tool_start", "name": "...", "arguments": {...}}``
    - ``{"type": "tool_end", "name": "...", "observation": "...", "is_error": bool}``
    - ``{"type": "done", "response": "...", "steps": N, "tool_calls_made": [...]}``

    The connection stays open for multiple round trips until the client disconnects.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", "default")

            async for event in _agent.run_stream(message, session_id=session_id):
                await websocket.send_json(event)

    except WebSocketDisconnect:
        return


@app.get("/api/sessions")
async def list_sessions() -> dict[str, list[str]]:
    """List all known session IDs."""
    return {"sessions": _session_manager.list_sessions()}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str) -> dict[str, str]:
    """Clear a session's conversation history."""
    _session_manager.clear_session(session_id)
    return {"status": "cleared"}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str) -> dict[str, list[dict[str, str]]]:
    """Return the stored conversation history for a session."""
    history = _session_manager.get_history(session_id)
    return {"messages": [{"role": m.role, "content": m.content} for m in history]}


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server with uvicorn (entry point for the console script)."""
    import uvicorn

    uvicorn.run("open_agent.server.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
