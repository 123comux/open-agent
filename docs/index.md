# Open Agent

[![CI](https://github.com/123comux/open-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/123comux/open-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

> 一个通用型 **Agentic RAG 自主工作助手**，将 ReAct 推理循环、RAG 检索增强与可插拔的工具生态结合在一起。

Open Agent 是一个开源的自主工作助手。它通过 **ReAct 规划循环**（`Thought → Action → Observation`）逐步推理，调用内置工具（shell、Python、文件 I/O、网页搜索、无头浏览器）或任意 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 服务器，并基于你自己的文档向量库对答案进行落地。项目遵循 **"one core library, many frontends"** 原则：一个与框架无关的纯 Python 核心，同时驱动 Rich/Typer CLI、FastAPI 服务（含 React + Vite Web UI）与 VS Code 扩展三种前端。

## 核心特性

- :material-robot-outline: **Agentic RAG** — ReAct 推理循环（`Thought → Action → Observation`），基于自有文档的向量检索（FAISS，可选 ChromaDB）做答案落地。
- :material-tools: **多工具** — 内置工具（shell、Python 解释器、文件 I/O、网页搜索、无头浏览器）+ 统一的 `ToolRegistry` 支持运行时注册。
- :material-brain: **多模型** — 单一 `ModelInterface` 支持 OpenAI 兼容端点、Anthropic Claude、智谱 AI 与本地 Ollama；一个环境变量或 Web UI 即可切换。
- :material-link-variant: **MCP 支持** — 接入任意 MCP 服务器（stdio / SSE），其工具与内置工具一视同仁。
- :material-monitor-multiple: **多前端交付** — CLI、库、HTTP/WebSocket API 服务、Web UI、VS Code 扩展任你选。
- :material-database: **记忆** — 短期滑动窗口上下文 + 可选的向量长期记忆，实现跨会话召回。
- :material-shield-check: **安全** — 工具级异常隔离、LLM API 指数退避重试、可配置的 ReAct 步数上限防止失控循环。

## 核心架构

```text
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

**API Gateway** 是一层薄适配，不含业务逻辑。所有智能都集中在 **Agent Core**——一个纯 Python 库，可在不引入 FastAPI、React 或 VS Code API 的情况下被嵌入使用。详见 [架构设计](architecture.md)。

## 快速链接

<div class="grid cards" markdown>

- :material-rocket-launch: **快速开始**

    ---

    5 分钟内跑起来：CLI、API 服务、Web UI，以及第一个 RAG 知识库。

    [:octicons-arrow-right-24: 开始](quickstart.md)

- :material-sitemap: **架构设计**

    ---

    Agent Core、Planner、RAG、Tool Registry、Memory、Model Interface、MCP Adapter 与请求生命周期。

    [:octicons-arrow-right-24: 查看](architecture.md)

- :material-api: **API 参考**

    ---

    REST 端点、WebSocket 流式事件、认证、错误响应与 Python 客户端示例。

    [:octicons-arrow-right-24: 查看](api.md)

- :material-tools: **工具系统**

    ---

    Tool 基类、ToolRegistry、内置工具、沙箱机制与自定义工具开发指南。

    [:octicons-arrow-right-24: 查看](tools.md)

- :material-database-search: **RAG 知识库**

    ---

    文档索引、混合检索（向量 + BM25 + RRF）、重排序与知识库路由。

    [:octicons-arrow-right-24: 查看](rag.md)

- :material-cog: **配置参考**

    ---

    完整环境变量表、配置优先级、运行时更新与脱敏机制。

    [:octicons-arrow-right-24: 查看](configuration.md)

</div>

## 下一步

- 想立刻体验？前往 [快速开始](quickstart.md)。
- 想了解内部机制？阅读 [架构设计](architecture.md)。
- 准备部署？查看 [部署指南](deployment.md)。
