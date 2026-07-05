"""FastAPI server exposing the open-agent core over HTTP and WebSocket.

This module is a thin adapter layer: it builds an :class:`Agent` from settings
and exposes it via REST and WebSocket endpoints. It depends on the optional
``server`` extra (``fastapi`` and ``uvicorn``); importing it without those
packages raises a clear :class:`ImportError`.
"""
from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi.responses import Response

from open_agent.config import Settings, get_settings
from open_agent.logging_config import get_logger, setup_logging
from open_agent.memory.session_manager import SessionManager
from open_agent.observability.tracer import LocalJsonlTracer, NoOpTracer, Tracer

# FastAPI is an optional dependency; provide a clear error if missing.
try:
    from fastapi import (
        Depends,
        FastAPI,
        HTTPException,
        Request,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
    )
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


async def _build_agent() -> tuple[Any, Any, Tracer]:
    """Construct an Agent and its registry from settings (lazy heavy imports)."""
    from open_agent.agent.core import Agent
    from open_agent.models.base import ModelInterface
    from open_agent.tools.builtin import (
        BrowserTool,
        FileTool,
        KnowledgeBaseTool,
        PythonTool,
        ShellTool,
        WebSearchTool,
    )
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
        kb_manager = _get_kb_manager()
        kb_tool = KnowledgeBaseTool(kb_manager=kb_manager, top_k=settings.rag_top_k)
    except ImportError:
        kb_tool = KnowledgeBaseTool()
    builtin_tools = [
        ShellTool(),
        PythonTool(),
        FileTool(),
        WebSearchTool(),
        BrowserTool(),
        kb_tool,
    ]
    enabled = set(settings.enabled_tools)
    for tool in builtin_tools:
        if not enabled or tool.name in enabled:
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
                global _mcp_client
                _mcp_client = mcp_client
        except Exception as exc:  # pragma: no cover - optional integration
            logger.warning("Failed to load MCP servers: %s", exc)

    tracer: Tracer = NoOpTracer()
    if settings.enable_observability:
        try:
            if settings.observability_provider == "langsmith":
                from open_agent.observability.tracer import LangSmithTracer

                tracer = LangSmithTracer(
                    api_key=settings.langsmith_api_key or None,
                    api_url=settings.langsmith_api_url or None,
                    project_name=settings.langsmith_project or None,
                )
            elif settings.observability_provider == "langfuse":
                from open_agent.observability.tracer import LangfuseTracer

                tracer = LangfuseTracer(
                    public_key=settings.langfuse_public_key or None,
                    secret_key=settings.langfuse_secret_key or None,
                    host=settings.langfuse_host or None,
                )
            else:
                from open_agent.observability.tracer import LocalJsonlTracer

                tracer = LocalJsonlTracer(settings.observability_output_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize observability tracer: %s", exc)

    long_term_memory = None
    if settings.enable_long_term_memory:
        try:
            from open_agent.memory.long_term import LongTermMemory

            long_term_memory = LongTermMemory(
                embedding_model=settings.embedding_model,
                storage_dir=settings.long_term_memory_dir,
                top_k=settings.long_term_memory_top_k,
            )
        except Exception as exc:  # pragma: no cover - optional integration
            logger.warning("Failed to initialize long-term memory: %s", exc)

    return (
        Agent(
            model=model,
            tool_registry=registry,
            max_steps=settings.max_steps,
            session_manager=_session_manager,
            long_term_memory=long_term_memory,
            tracer=tracer,
        ),
        registry,
        tracer,
    )


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    response: str
    steps: int
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)


class SettingsUpdateRequest(BaseModel):
    """Editable subset of :class:`Settings` exposed through the API."""

    model_provider: Literal["openai", "anthropic", "ollama"] | None = None
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=200)
    request_timeout: float | None = Field(default=None, ge=1.0, le=3600.0)
    embedding_model: str | None = None
    chunk_size: int | None = Field(default=None, ge=1, le=100000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=100000)
    split_unit: str | None = None
    rag_top_k: int | None = Field(default=None, ge=1, le=100)
    reranker_model: str | None = None
    rerank_k: int | None = Field(default=None, ge=1, le=200)
    enabled_tools: list[str] | None = None
    enable_long_term_memory: bool | None = None
    long_term_memory_top_k: int | None = Field(default=None, ge=1, le=50)


_settings = get_settings()
_session_manager = SessionManager(
    max_messages=_settings.short_term_memory_size,
    storage_dir=_settings.session_storage_dir,
)


_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-.]+$")

# Maximum allowed size for a single document upload (50 MB).
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _validate_session_id(session_id: str) -> str:
    """Validate session_id to prevent path traversal."""
    if not session_id or not _SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id: only alphanumeric, dash, dot, underscore allowed.",
        )
    if ".." in session_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id: '..' is not allowed.",
        )
    return session_id


def _validate_kb_name(kb_name: str) -> str:
    """Validate kb_name to prevent path traversal (same rules as session_id)."""
    if not kb_name or not _SESSION_ID_PATTERN.match(kb_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid kb_name: only alphanumeric, dash, dot, underscore allowed.",
        )
    if ".." in kb_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid kb_name: '..' is not allowed.",
        )
    return kb_name


async def _require_auth(request: Request) -> None:
    """Validate Bearer token when OPEN_AGENT_API_AUTH_TOKEN is configured.

    When the token is empty (default), authentication is disabled for local
    development. Set a token in production to protect sensitive endpoints.
    """
    token = _settings.api_auth_token
    if not token:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    if not secrets.compare_digest(auth[7:], token):
        raise HTTPException(status_code=401, detail="Invalid API token.")


_kb_manager: Any = None
_mcp_client: Any = None
_rebuild_lock = asyncio.Lock()


def _get_kb_manager() -> Any:
    """Return a shared KBManager for document upload and RAG management."""
    global _kb_manager
    if _kb_manager is not None:
        return _kb_manager
    from open_agent.rag.kb_manager import KBManager

    _kb_manager = KBManager(
        storage_dir=".open_agent_kb_indexes",
        embedding_model=_settings.embedding_model,
        chunk_size=_settings.chunk_size,
        chunk_overlap=_settings.chunk_overlap,
        split_unit=_settings.split_unit,
        top_k=_settings.rag_top_k,
        reranker_model=_settings.reranker_model,
        rerank_k=_settings.rerank_k,
    )
    return _kb_manager


async def _close_mcp_client() -> None:
    """Close the module-level MCP client if one is active."""
    global _mcp_client
    if _mcp_client is not None:
        try:
            await _mcp_client.close()
        except Exception:
            logger.debug("MCP client close failed", exc_info=True)
        _mcp_client = None


async def _close_model() -> None:
    """Close the current agent's model httpx client if it exposes ``aclose``."""
    if _agent is not None:
        model = getattr(_agent, "model", None)
        if model is not None:
            aclose = getattr(model, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.debug("model aclose failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the agent on startup; clean up on shutdown."""
    await _initialize_agent()
    yield
    await _close_mcp_client()
    await _close_model()
    try:
        from open_agent.tools.builtin.web_search import aclose as _close_web_search

        await _close_web_search()
    except Exception:
        pass
    if isinstance(_tracer, LocalJsonlTracer):
        try:
            await _tracer.aclose()
        except Exception:
            pass


app = FastAPI(
    lifespan=lifespan,
    title="Open Agent API",
    version="0.2.0",
    description="""## Open Agent — Agentic RAG Autonomous Work Assistant

A general-purpose AI agent with:
- **Autonomous tool decision** — ReAct loop with intent classification
- **Multi-tool orchestration** — Shell, Python, File, Web Search, Knowledge Base
- **RAG** — FAISS vector search + BM25 keyword search + knowledge base routing
- **Streaming** — Real-time token streaming via WebSocket
- **Session memory** — Per-session conversation history with persistence
- **Runtime settings** — Update model, API, tools, and RAG parameters via `/api/settings`

### Quick Start
```bash
open-agent serve
```
""",
    openapi_tags=[
        {"name": "chat", "description": "Chat endpoints for agent interaction"},
        {"name": "tools", "description": "Tool management and listing"},
        {"name": "sessions", "description": "Session management"},
        {"name": "rag", "description": "Knowledge base and document indexing"},
        {"name": "health", "description": "Health checks"},
        {"name": "settings", "description": "Runtime configuration"},
        {"name": "observability", "description": "Trace inspection and observability"},
    ],
)
_agent: Any = None
_registry: Any = None
_tracer: Tracer = NoOpTracer()


async def _initialize_agent() -> None:
    """Build the global agent and registry on startup."""
    global _agent, _registry, _tracer
    _agent, _registry, _tracer = await _build_agent()


# CORS middleware
_cors_origins = _settings.cors_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
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
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred.",
        },
    )


@app.get("/api/health", tags=["health"])
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/ready", tags=["health"], response_model=None)
async def readiness_check() -> dict[str, object] | JSONResponse:
    """Readiness check — verifies the agent is initialized."""
    if _agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "agent_not_initialized"},
        )
    return {"status": "ready"}


@app.get("/api/tools", tags=["tools"], dependencies=[Depends(_require_auth)])
async def list_tools() -> dict[str, Any]:
    """List the names and schemas of available tools."""
    return {"tools": _registry.schemas()}


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(_require_auth)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a single message and return the agent's response."""
    _validate_session_id(request.session_id)
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
    token = _settings.api_auth_token
    if token:
        auth = websocket.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not secrets.compare_digest(auth[7:], token):
            await websocket.close(code=4001, reason="Unauthorized")
            return
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", "default")
            try:
                _validate_session_id(session_id)
            except HTTPException:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid session_id"}
                )
                continue
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


@app.get("/api/sessions", tags=["sessions"], dependencies=[Depends(_require_auth)])
async def list_sessions() -> dict[str, list[str]]:
    """List all known session IDs."""
    return {"sessions": _session_manager.list_sessions()}


@app.delete("/api/sessions/{session_id}", tags=["sessions"], dependencies=[Depends(_require_auth)])
async def clear_session(session_id: str) -> dict[str, str]:
    """Clear a session's conversation history."""
    _validate_session_id(session_id)
    _session_manager.clear_session(session_id)
    return {"status": "cleared"}


@app.get(
    "/api/sessions/{session_id}/history",
    tags=["sessions"],
    dependencies=[Depends(_require_auth)],
)
async def get_history(session_id: str) -> dict[str, list[dict[str, str]]]:
    """Return the stored conversation history for a session."""
    _validate_session_id(session_id)
    history = _session_manager.get_history(session_id)
    return {"messages": [{"role": m.role, "content": m.content} for m in history]}


class RenameSessionRequest(BaseModel):
    """Request body for renaming a session."""

    new_session_id: str


@app.post(
    "/api/sessions/{session_id}/rename",
    tags=["sessions"],
    dependencies=[Depends(_require_auth)],
)
async def rename_session(session_id: str, request: RenameSessionRequest) -> dict[str, str]:
    """Rename a session."""
    _validate_session_id(session_id)
    _validate_session_id(request.new_session_id)
    try:
        _session_manager.rename_session(session_id, request.new_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "status": "renamed",
        "old_session_id": session_id,
        "new_session_id": request.new_session_id,
    }


@app.get("/api/sessions/search", tags=["sessions"], dependencies=[Depends(_require_auth)])
async def search_sessions(q: str) -> dict[str, Any]:
    """Search sessions by id and message content."""
    results = _session_manager.search_sessions(q)
    return {"query": q, "results": results}


@app.get(
    "/api/sessions/{session_id}/export",
    tags=["sessions"],
    dependencies=[Depends(_require_auth)],
)
async def export_session(session_id: str, fmt: str = "json") -> Any:
    """Export a session as JSON or Markdown."""
    _validate_session_id(session_id)
    supported = {"json", "md", "markdown"}
    if fmt not in supported:
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_format",
                "message": f"Format '{fmt}' not supported. Use: {', '.join(sorted(supported))}",
            },
        )
    content = _session_manager.export_session(session_id, fmt="md" if fmt == "markdown" else fmt)
    media_type = "application/json" if fmt == "json" else "text/markdown"
    ext = "json" if fmt == "json" else "md"
    from fastapi.responses import Response

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={session_id}.{ext}"},
    )


@app.post("/api/upload", tags=["rag"], dependencies=[Depends(_require_auth)], response_model=None)
async def upload_document(
    file: UploadFile,
    kb_name: str = "default",
) -> dict[str, Any] | JSONResponse:
    """Upload a document file and index it into a knowledge base.

    Supports: .txt, .md, .rst, .pdf, .docx, .csv, .json, .html
    """
    _validate_kb_name(kb_name)
    import os
    import tempfile

    from open_agent.rag.document_loaders import SUPPORTED_EXTENSIONS

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_file_type",
                "message": f"File type '{ext}' not supported. Supported: {supported}",
            },
        )

    # Stream upload to a temp file with a size cap to prevent abuse.
    chunk_size = 64 * 1024  # 64 KB
    total = 0
    too_large = False
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as tmp:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    too_large = True
                    break
                tmp.write(chunk)
        if too_large:
            return JSONResponse(
                status_code=413,
                content={"error": "file_too_large", "max_bytes": MAX_UPLOAD_BYTES},
            )
        manager = _get_kb_manager()
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
        logger.error("Upload indexing failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "indexing_failed", "message": "indexing failed"},
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/knowledge-bases", tags=["rag"], dependencies=[Depends(_require_auth)])
async def list_knowledge_bases() -> dict[str, Any]:
    """List all registered knowledge bases."""
    try:
        manager = _get_kb_manager()
        return {"knowledge_bases": manager.list_kbs()}
    except Exception:
        return {"knowledge_bases": []}


@app.get(
    "/api/knowledge-bases/{kb_name}/documents",
    tags=["rag"],
    dependencies=[Depends(_require_auth)],
    response_model=None,
)
async def list_kb_documents(kb_name: str) -> dict[str, Any] | JSONResponse:
    """List documents indexed in a knowledge base."""
    try:
        manager = _get_kb_manager()
        documents = manager.list_documents(kb_name)
        return {"kb_name": kb_name, "documents": documents}
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": "kb_not_found", "message": f"KB '{kb_name}' not found"},
        )
    except Exception as exc:
        logger.error("Failed to list KB documents: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "list_failed", "message": "failed to list documents"},
        )


@app.delete(
    "/api/knowledge-bases/{kb_name}/documents",
    tags=["rag"],
    dependencies=[Depends(_require_auth)],
    response_model=None,
)
async def delete_kb_document(kb_name: str, source: str) -> dict[str, Any] | JSONResponse:
    """Delete all chunks from ``source`` in ``kb_name``."""
    try:
        manager = _get_kb_manager()
        removed = await manager.delete_document(kb_name, source)
        logger.info("Deleted %d chunks from KB '%s' source '%s'", removed, kb_name, source)
        return {"kb_name": kb_name, "source": source, "removed": removed}
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": "kb_not_found", "message": f"KB '{kb_name}' not found"},
        )
    except Exception as exc:
        logger.error("Failed to delete KB document: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "delete_failed", "message": "failed to delete document"},
        )


def _list_local_traces(limit: int = 100) -> list[dict[str, Any]]:
    """Return traces if the configured tracer supports local listing."""
    if isinstance(_tracer, LocalJsonlTracer):
        return _tracer.list_traces(limit=limit)
    return []


@app.get("/api/traces", tags=["observability"], dependencies=[Depends(_require_auth)])
async def list_traces(limit: int = 100) -> dict[str, Any]:
    """List recent traces from the local tracer."""
    limit = max(1, min(limit, 1000))
    traces = await asyncio.to_thread(_list_local_traces, limit=limit)
    return {"traces": traces}


@app.get("/api/traces/{trace_id}", tags=["observability"], dependencies=[Depends(_require_auth)])
async def get_trace(trace_id: str) -> Any:
    """Get a single trace by id."""

    def _get() -> dict[str, Any] | None:
        if isinstance(_tracer, LocalJsonlTracer):
            return _tracer.get_trace(trace_id)
        return None

    trace = await asyncio.to_thread(_get)
    if trace is not None:
        return trace
    return JSONResponse(
        status_code=404,
        content={"error": "trace_not_found", "message": f"Trace '{trace_id}' not found"},
    )


@app.get("/api/settings", tags=["settings"], dependencies=[Depends(_require_auth)])
async def get_settings_endpoint() -> dict[str, Any]:
    """Return the current runtime settings (excluding secrets)."""
    return _settings.to_safe_dict()


@app.post(
    "/api/settings",
    tags=["settings"],
    dependencies=[Depends(_require_auth)],
    response_model=None,
)
async def update_settings_endpoint(request: SettingsUpdateRequest) -> dict[str, str] | JSONResponse:
    """Apply new runtime settings and rebuild the agent/registry."""
    global _settings, _agent, _registry, _tracer, _kb_manager, _session_manager

    async with _rebuild_lock:
        updates = request.model_dump(exclude_unset=True)
        old_settings = _settings.model_copy(deep=True)
        old_kb_manager = _kb_manager
        old_session_manager = _session_manager
        new_settings = _settings.model_copy(update=updates)
        # Re-validate to run model_validator (model_copy bypasses it).
        new_settings = Settings.model_validate(new_settings.model_dump())
        from open_agent.config import set_settings

        set_settings(new_settings)
        _settings = new_settings

        kb_fields = {
            "chunk_size",
            "chunk_overlap",
            "split_unit",
            "embedding_model",
            "rag_top_k",
            "reranker_model",
            "rerank_k",
        }
        if updates.keys() & kb_fields:
            _kb_manager = None

        session_fields = {"short_term_memory_size", "session_storage_dir"}
        if updates.keys() & session_fields:
            _session_manager = SessionManager(
                max_messages=_settings.short_term_memory_size,
                storage_dir=_settings.session_storage_dir,
            )

        try:
            # Close resources held by the old agent before rebuilding.
            await _close_mcp_client()
            await _close_model()
            _agent, _registry, _tracer = await _build_agent()
        except Exception as exc:
            logger.error("Failed to rebuild agent after settings update: %s", exc, exc_info=True)
            set_settings(old_settings)
            _settings = old_settings
            _kb_manager = old_kb_manager
            _session_manager = old_session_manager
            # Old agent's MCP client was closed above; mark as not ready so
            # /api/ready reports 503 until a successful rebuild or restart.
            _agent = None
            _registry = None
            return JSONResponse(
                status_code=500,
                content={"error": "rebuild_failed", "message": "settings update failed"},
            )

    logger.info("Runtime settings updated: %s", ", ".join(updates.keys()))
    return {"status": "updated"}


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API server with uvicorn (entry point for the console script)."""
    import uvicorn

    uvicorn.run("open_agent.server.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
