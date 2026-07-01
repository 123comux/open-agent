"""FastAPI server exposing the open-agent core over HTTP and WebSocket.

This module is a thin adapter layer: it builds an :class:`Agent` from settings
and exposes it via REST and WebSocket endpoints. It depends on the optional
``server`` extra (``fastapi`` and ``uvicorn``); importing it without those
packages raises a clear :class:`ImportError`.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from open_agent.config import get_settings
from open_agent.logging_config import get_logger, setup_logging
from open_agent.memory.session_manager import SessionManager

# FastAPI is an optional dependency; provide a clear error if missing.
try:
    from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The server extra is required to use open_agent.server.api. "
        "Install it with: pip install 'open-agent[server]'"
    ) from exc

# Configure logging early so all subsequent module-level code is logged.
setup_logging()
logger = get_logger("server")


async def _build_agent():
    """Construct an Agent and its registry from settings (lazy heavy imports)."""
    from open_agent.agent.core import Agent
    from open_agent.models.base import ModelInterface
    from open_agent.tools.builtin import FileTool, KnowledgeBaseTool, PythonTool, ShellTool, WebSearchTool
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
    try:
        from open_agent.rag.kb_manager import KBManager

        kb_manager = KBManager(
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            split_unit=settings.split_unit,
            top_k=settings.rag_top_k,
        )
        kb_tool = KnowledgeBaseTool(kb_manager=kb_manager, top_k=settings.rag_top_k)
    except ImportError:
        kb_tool = KnowledgeBaseTool()
    for tool in (ShellTool(), PythonTool(), FileTool(), WebSearchTool(), kb_tool):
        registry.register(tool)

    # Load MCP servers if configured.
    if settings.mcp_servers_file:
        try:
            from open_agent.mcp import MCPClient, adapt_mcp_tools, load_mcp_servers

            servers = load_mcp_servers(settings.mcp_servers_file)
            if servers:
                mcp_client = MCPClient(servers)
                await mcp_client.connect()
                for mcp_tool in adapt_mcp_tools(mcp_client):
                    registry.register(mcp_tool)
        except Exception as exc:  # pragma: no cover - optional integration
            logger.warning("Failed to load MCP servers: %s", exc)

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

app = FastAPI(
    title="Open Agent API",
    version="0.2.0",
    description="""## Open Agent — Agentic RAG Autonomous Work Assistant

A general-purpose AI agent with:
- **Autonomous tool decision** — ReAct loop with intent classification
- **Multi-tool orchestration** — Shell, Python, File, Web Search, Knowledge Base
- **RAG** — FAISS vector search + BM25 keyword search + knowledge base routing
- **Streaming** — Real-time token streaming via WebSocket
- **Session memory** — Per-session conversation history with persistence

### Quick Start
```bash
open-agent serve
```
""",
    openapi_tags=[
        {"name": "chat", "description": "Chat endpoints for agent interaction"},
        {"name": "tools", "description": "Tool management and listing"},
        {"name": "sessions", "description": "Session management"},
        {"name": "health", "description": "Health checks"},
    ],
)
_agent, _registry = asyncio.run(_build_agent())

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configurable in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every HTTP request with method, path, status, and duration."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={"duration_ms": round(duration_ms, 1)},
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a structured JSON error response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "type": type(exc).__name__,
        },
    )


@app.get("/api/health", tags=["health"])
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/tools", tags=["tools"])
async def list_tools() -> dict[str, Any]:
    """List the names and schemas of available tools."""
    return {"tools": _registry.schemas()}


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a single message and return the agent's response."""
    logger.info(
        f"Chat request from session={request.session_id}",
        extra={"session_id": request.session_id},
    )
    try:
        output = await _agent.run(request.message, session_id=request.session_id)
        logger.info(
            f"Chat completed: {output.steps} steps, "
            f"{len(output.tool_calls_made)} tool calls",
            extra={"session_id": request.session_id},
        )
        return ChatResponse(
            response=output.response,
            steps=output.steps,
            tool_calls_made=output.tool_calls_made,
        )
    except Exception as exc:
        logger.error(
            f"Chat failed: {exc}",
            exc_info=True,
            extra={"session_id": request.session_id},
        )
        raise


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
            logger.info(
                f"WS chat request from session={session_id}",
                extra={"session_id": session_id},
            )

            async for event in _agent.run_stream(message, session_id=session_id):
                await websocket.send_json(event)

            logger.info(
                f"WS chat completed for session={session_id}",
                extra={"session_id": session_id},
            )

    except WebSocketDisconnect:
        return


@app.get("/api/sessions", tags=["sessions"])
async def list_sessions() -> dict[str, list[str]]:
    """List all known session IDs."""
    return {"sessions": _session_manager.list_sessions()}


@app.delete("/api/sessions/{session_id}", tags=["sessions"])
async def clear_session(session_id: str) -> dict[str, str]:
    """Clear a session's conversation history."""
    _session_manager.clear_session(session_id)
    return {"status": "cleared"}


@app.get("/api/sessions/{session_id}/history", tags=["sessions"])
async def get_history(session_id: str) -> dict[str, list[dict[str, str]]]:
    """Return the stored conversation history for a session."""
    history = _session_manager.get_history(session_id)
    return {"messages": [{"role": m.role, "content": m.content} for m in history]}


@app.post("/api/upload", tags=["rag"])
async def upload_document(
    file: UploadFile,
    kb_name: str = "default",
) -> dict[str, Any]:
    """Upload a document file and index it into a knowledge base.

    Supports: .txt, .md, .rst, .pdf, .docx, .csv, .json, .html
    """
    import os
    import tempfile

    from open_agent.rag.document_loaders import load_file, SUPPORTED_EXTENSIONS

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_file_type",
                "message": f"File type '{ext}' not supported. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            },
        )

    # Save to temp file, load, index
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from open_agent.rag.kb_manager import KBManager

        manager = KBManager(
            embedding_model=_settings.embedding_model,
            chunk_size=_settings.chunk_size,
            chunk_overlap=_settings.chunk_overlap,
            split_unit=_settings.split_unit,
            top_k=_settings.rag_top_k,
        )
        chunk_count = await manager.index_file(tmp_path, kb_name)
        logger.info(
            f"Uploaded file '{file.filename}' indexed into KB '{kb_name}': {chunk_count} chunks"
        )
        return {
            "status": "indexed",
            "filename": file.filename,
            "kb_name": kb_name,
            "chunks": chunk_count,
        }
    except Exception as exc:
        logger.error(f"Upload indexing failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "indexing_failed", "message": str(exc)},
        )
    finally:
        os.unlink(tmp_path)


@app.get("/api/knowledge-bases", tags=["rag"])
async def list_knowledge_bases() -> dict[str, Any]:
    """List all registered knowledge bases."""
    try:
        from open_agent.rag.kb_manager import KBManager

        manager = KBManager()
        return {"knowledge_bases": manager.list_kbs()}
    except Exception:
        return {"knowledge_bases": []}


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server with uvicorn (entry point for the console script)."""
    import uvicorn

    uvicorn.run("open_agent.server.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
