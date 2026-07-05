# Changelog

All notable changes to Open Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-01

### Added
- Streaming LLM output: `stream_chat` on all model providers, `run_stream` on Agent
- Session-based conversation memory with JSON persistence (`SessionManager`)
- Multi-format document parsing: PDF, DOCX, CSV, JSON, HTML, Markdown
- RAG evaluation module: Faithfulness, Answer Relevance, Context Recall/Precision
- Docker containerization: Dockerfile, docker-compose.yml
- Web UI WebSocket streaming with real-time token display
- CLI streaming output for `chat` and `ask` commands
- CORS middleware and global exception handling
- Structured logging system
- OpenAPI documentation with tagged endpoints
- Session management REST API endpoints
- Thinking chain events in streaming
- Integration tests for API server and agent flow

### Changed
- KBManager now uses `load_file()` supporting multiple document formats
- WebSocket `/ws/chat` streams events instead of full responses
- CLI `chat` and `ask` now stream tokens in real-time

## [0.1.0] - 2026-07-06

### Added
- Agentic RAG with ReAct planning loop (intent classification → planning → execution → reflection → termination)
- Multi-tool support: Shell, Python, File, Web Search, Browser, Knowledge Base
- Multi-model providers: OpenAI, Anthropic, Ollama, Zhipu AI (GLM-4-Flash, GLM-4.7-Flash)
- MCP (Model Context Protocol) integration with stdio and SSE transports
- Hybrid RAG: FAISS vector + BM25 keyword search with RRF fusion + cross-encoder reranking (BAAI/bge-reranker-v2-m3)
- Chinese-optimized embedding default (BAAI/bge-small-zh-v1.5)
- Multi-frontend: CLI (Rich/Typer), FastAPI server (REST + WebSocket), React+Vite web UI, VS Code extension
- Short-term sliding window memory + optional long-term vector memory
- Observability: local JSONL tracer, LangSmith, Langfuse
- Tool sandbox with shell/python/path guards
- Session persistence with atomic writes
- Context window management with tiktoken + CJK heuristic truncation
- Configurable runtime settings via /api/settings endpoint
- Docker and docker-compose deployment support
- mkdocs documentation site

### Security
- Bearer token authentication (secrets.compare_digest, constant-time)
- Path traversal protection for session_id and kb_name
- SSRF guard for browser tool (blocks private/loopback/link-local IPs)
- Upload size limit (50 MB)
- CORS configuration with credentials safety
- Python sandbox with safe_builtins (getattr/setattr/delattr/super removed)

### Changed
- Adopted Pydantic v2 Settings with model_validator and to_safe_dict masking
- Refactored to "one core library, many frontends" architecture
- Async-first I/O with httpx.AsyncClient and asyncio.Lock serialization

### Fixed
- 50+ bugs across 3 audit rounds (see git log for details)
