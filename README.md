# Open Agent

[![CI](https://github.com/your-org/open-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/open-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

> A general-purpose Agentic RAG autonomous work assistant with multi-tool support.

Open Agent is an open-source autonomous work assistant that combines a **ReAct
planning loop** with **RAG retrieval augmentation** and a **pluggable tool
ecosystem**. It can reason step-by-step, call built-in tools (shell, Python,
file I/O, web search) or any [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
server, and ground its answers in your own documents via a vector store.

The project follows a **"one core library, many frontends"** principle: a
framework-agnostic Python core that powers a Rich/Typer CLI, a FastAPI server
(with a React + Vite web UI), and a VS Code extension.

## Key Features

- **Agentic RAG** — ReAct reasoning loop (`Thought → Action → Observation`)
  grounded by vector retrieval over your own documents (ChromaDB).
- **Multi-tool** — Built-in tools (shell, Python interpreter, file I/O, web
  search) plus a unified `ToolRegistry` for runtime registration.
- **Multi-model** — A single `ModelInterface` supports OpenAI-compatible
  endpoints, Anthropic Claude, and local Ollama. Switch providers with one env
  var.
- **MCP support** — Connect any MCP server (stdio or SSE) and its tools become
  first-class citizens alongside the built-ins.
- **Multi-frontend delivery** — Use it as a CLI, a library, an HTTP/WebSocket
  API server, a web UI, or a VS Code extension.
- **Memory** — Short-term sliding-window context plus optional long-term vector
  memory for cross-session recall.
- **Safety** — Per-tool exception isolation, LLM API retry with exponential
  backoff, and a configurable ReAct step cap to prevent runaway loops.

## Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CLI Frontend  │  │   Web Frontend  │  │  VS Code Plugin │
│   (Python/Rich) │  │  (React + Vite) │  │   (TypeScript)  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │ HTTP / stdio
         ┌────────────────────┴────────────────────┐
         │          API Gateway (FastAPI)          │
         │         (适配层，不含业务逻辑)           │
         └────────────────────┬────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │           Agent Core (Python)           │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
         │  │ Planner │ │  RAG    │ │ Tool    │  │
         │  │ (ReAct) │ │ (Vector)│ │ Registry│  │
         │  └─────────┘ └─────────┘ └─────────┘  │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
         │  │ Memory  │ │ Model   │ │ MCP     │  │
         │  │ (Short/ │ │ (Multi- │ │ Adapter │  │
         │  │  Long)  │ │provider)│ │         │  │
         │  └─────────┘ └─────────┘ └─────────┘  │
         └─────────────────────────────────────────┘
```

The **API Gateway** is a thin adapter with no business logic. All intelligence
lives in the **Agent Core**, which is a pure Python library that can be embedded
without FastAPI, React, or VS Code APIs.

## Quick Start

### Prerequisites

- Python **3.10**, **3.11**, or **3.12**
- An API key for your model provider (OpenAI, Anthropic) **or** a local
  [Ollama](https://ollama.ai/) instance.

### Install

```bash
git clone https://github.com/your-org/open-agent.git
cd open-agent
pip install -e ".[all]"
```

> Use `[all]` to pull in RAG, server, and MCP extras at once. For a minimal
> install, use `pip install -e .` and add extras as needed (`[rag]`, `[server]`,
> `[mcp]`).

### Configure

Open Agent reads all configuration from environment variables prefixed with
`OPEN_AGENT_`. The quickest way is a `.env` file (make sure to source it or use
a tool like [direnv](https://direnv.net/)):

```bash
# .env
OPEN_AGENT_MODEL_PROVIDER=openai        # openai | anthropic | ollama
OPEN_AGENT_API_KEY=sk-...               # your API key (omit for ollama)
OPEN_AGENT_MODEL_NAME=gpt-4o-mini       # model identifier
```

For a fully local setup with Ollama:

```bash
OPEN_AGENT_MODEL_PROVIDER=ollama
OPEN_AGENT_BASE_URL=http://localhost:11434
OPEN_AGENT_MODEL_NAME=llama3
```

### Run the CLI

```bash
open-agent chat
```

## Usage Examples

### Interactive chat (REPL)

Start a multi-turn session. The agent maintains short-term memory across turns
within the session.

```bash
open-agent chat
```

```
╭─ open-agent ────────────────────────────────────────────╮
│ Open Agent ready. Provider: openai, Model: gpt-4o-mini. │
│ Type 'exit' to quit.                                    │
╰─────────────────────────────────────────────────────────╯
you> Summarize the largest files in this repo
...
```

### Single ask (one-shot)

Ask one question and print the answer plus step/tool statistics.

```bash
open-agent ask "What does the planner module do in this codebase?"
```

### Serve mode (HTTP + WebSocket API)

Launch the FastAPI server so the web UI or VS Code extension can connect:

```bash
open-agent serve --host 0.0.0.0 --port 8000
# or directly:
make serve
```

Endpoints:

| Method | Path          | Description                                   |
|--------|---------------|-----------------------------------------------|
| GET    | `/api/health` | Health check                                  |
| GET    | `/api/tools`  | List available tools and their schemas        |
| POST   | `/api/chat`   | Single message → `ChatResponse`               |
| WS     | `/ws/chat`    | Streaming conversation over WebSocket         |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "List the Python files in src", "session_id": "cli-1"}'
```

### As a library

The core is a plain Python package, so you can embed it directly:

```python
import asyncio
from open_agent.agent.core import Agent
from open_agent.models.ollama_provider import OllamaModel
from open_agent.tools.builtin import ShellTool, WebSearchTool
from open_agent.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(ShellTool())
registry.register(WebSearchTool())

agent = Agent(
    model=OllamaModel(base_url="http://localhost:11434", model="llama3"),
    tool_registry=registry,
    max_steps=10,
)

output = asyncio.run(agent.run("Find the top 5 largest .py files here"))
print(output.response)
```

## Configuration

All settings are environment variables prefixed with `OPEN_AGENT_` and loaded
by `open_agent.config.Settings`:

| Variable                            | Default                     | Description                                              |
|-------------------------------------|-----------------------------|----------------------------------------------------------|
| `OPEN_AGENT_MODEL_PROVIDER`         | `openai`                    | One of `openai`, `anthropic`, `ollama`.                  |
| `OPEN_AGENT_API_KEY`                | `""`                        | API key for the provider (omit for local Ollama).        |
| `OPEN_AGENT_BASE_URL`               | `https://api.openai.com`    | Base URL of the model endpoint (OpenAI-compatible).      |
| `OPEN_AGENT_MODEL_NAME`             | `gpt-4o-mini`               | Model identifier passed to the provider.                 |
| `OPEN_AGENT_MAX_STEPS`              | `10`                        | Maximum ReAct iterations per turn (runaway protection).  |
| `OPEN_AGENT_REQUEST_TIMEOUT`        | `60`                        | Per-request timeout in seconds for model calls.          |
| `OPEN_AGENT_SERVER_HOST`            | `127.0.0.1`                 | Bind host for the FastAPI server.                        |
| `OPEN_AGENT_SERVER_PORT`            | `8000`                      | Bind port for the FastAPI server.                        |
| `OPEN_AGENT_SHORT_TERM_MEMORY_SIZE` | `20`                        | Number of recent messages kept in short-term memory.     |

## Development

Set up a full development environment with lint, type-check, and test tooling:

```bash
make dev          # pip install -e ".[dev,all]"
```

Common tasks (see `Makefile`):

```bash
make test         # run pytest
make lint         # ruff check src tests
make typecheck    # mypy src/open_agent
make serve        # uvicorn open_agent.server.api:app --reload --port 8000
make clean        # remove caches and build artifacts
```

CI (`.github/workflows/ci.yml`) runs `ruff`, `mypy`, and `pytest` across Python
3.10 / 3.11 / 3.12 on every push to `main` and on pull requests.

## Project Structure

```
open-agent/
├── pyproject.toml              # Package config, deps, ruff/mypy/pytest settings
├── Makefile                    # install / dev / test / lint / serve / clean
├── README.md
├── LICENSE
├── .github/
│   ├── workflows/ci.yml        # Matrix CI: ruff + mypy + pytest
│   └── CONTRIBUTING.md
├── docs/
│   └── superpowers/specs/      # Design documents
├── src/open_agent/             # Core library (no frontend deps)
│   ├── __init__.py
│   ├── cli.py                  # Typer/Rich CLI (chat / ask / serve)
│   ├── config.py               # Settings (OPEN_AGENT_ env vars)
│   ├── agent/
│   │   ├── core.py             # ReAct main loop
│   │   ├── planner.py          # Parse LLM output → DirectResponse | ToolCall
│   │   └── executor.py         # Route ToolCalls through the registry
│   ├── models/
│   │   ├── base.py             # ModelInterface abstraction
│   │   ├── openai_provider.py  # OpenAI / compatible endpoints
│   │   ├── anthropic_provider.py
│   │   └── ollama_provider.py
│   ├── rag/
│   │   ├── indexer.py          # Document chunking + embedding
│   │   └── retriever.py        # Top-K vector retrieval
│   ├── tools/
│   │   ├── registry.py         # ToolRegistry
│   │   ├── base.py             # Tool abstraction
│   │   ├── builtin/            # shell, python, file, web_search
│   │   └── mcp/                # MCP client + adapter → internal Tool
│   ├── memory/
│   │   ├── short_term.py       # Sliding-window conversation history
│   │   └── long_term.py        # Vector-backed long-term memory
│   └── server/
│       └── api.py              # FastAPI adapter (REST + WebSocket)
└── tests/                      # unit / integration / tools / e2e
```

## Contributing

Contributions are welcome! Please read [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md)
for guidelines on setting up a dev environment, running tests, code style, and
submitting pull requests.

In short:

1. Fork the repo and create a feature branch.
2. Run `make dev` to install all dependencies.
3. Ensure `make lint`, `make typecheck`, and `make test` pass.
4. Open a pull request describing your change.

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Open Agent
Contributors.
