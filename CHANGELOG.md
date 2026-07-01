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

## [0.1.0] - 2026-07-01

### Added
- Initial release of Open Agent
- ReAct agent core with multi-step reasoning loop
- Multi-model support: OpenAI, Anthropic, Ollama
- Built-in tools: Shell, Python, File, Web Search, Knowledge Base
- LangGraph agent with intent classification, planning, and reflection
- FAISS vector store with hybrid retrieval (BM25 + vector + RRF fusion)
- Knowledge base routing and management
- MCP protocol adapter for external tools
- CLI interface (Typer + Rich)
- FastAPI server with WebSocket support
- React Web UI with dark theme
- VS Code extension
- 84 unit tests
