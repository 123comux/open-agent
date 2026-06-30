# Open Agent 设计文档

## 1. 项目概述

Open Agent 是一个开源的 **通用多工具 Agentic RAG 自主智能工作助手**。它通过 ReAct 规划循环结合 RAG 检索增强，能够自主调用多种工具（内置工具 + MCP 生态）完成复杂任务。

项目以 **"一套核心库 + 多端交付"** 为架构原则：
- **核心库**：纯 Python，包含 Agent 循环、RAG、记忆、工具注册表、多模型支持
- **CLI**：基于 Rich/Typer 的命令行交互界面
- **Web UI**：React + Vite 的浏览器界面，通过 FastAPI 与核心通信
- **VS Code 插件**：TypeScript 编写，通过 stdio/HTTP 与 Python 后端通信

## 2. 架构设计

### 2.1 整体架构

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

### 2.2 目录结构

```
open-agent/
├── pyproject.toml              # Python 包配置
├── README.md
├── Makefile
├── .github/workflows/ci.yml    # 测试 + 类型检查
├── src/open_agent/             # 核心库
│   ├── __init__.py
│   ├── cli.py                  # CLI 入口 (Rich/Typer)
│   ├── agent/
│   │   ├── core.py             # ReAct 主循环
│   │   ├── planner.py          # 任务规划/分解
│   │   └── executor.py         # 工具执行器
│   ├── models/
│   │   ├── base.py             # 模型抽象接口
│   │   ├── openai.py           # OpenAI/兼容端点
│   │   ├── anthropic.py        # Claude
│   │   └── ollama.py           # 本地 Ollama
│   ├── rag/
│   │   ├── indexer.py          # 文档索引
│   │   ├── retriever.py        # 向量检索
│   │   └── stores/chroma.py    # ChromaDB 向量存储
│   ├── tools/
│   │   ├── registry.py         # 工具注册表
│   │   ├── base.py             # 工具抽象
│   │   ├── builtin/            # 内置工具
│   │   │   ├── shell.py
│   │   │   ├── python.py       # 代码解释器
│   │   │   ├── file.py
│   │   │   ├── web_search.py
│   │   │   └── browser.py
│   │   └── mcp/
│   │       ├── client.py       # MCP 客户端
│   │       └── adapter.py      # 适配为内部 Tool
│   ├── memory/
│   │   ├── short_term.py       # 滑动窗口对话记忆
│   │   └── long_term.py        # 向量长期记忆
│   └── server/
│       └── api.py              # FastAPI 服务
├── web/                        # Web UI
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── Chat.tsx
│       │   ├── ToolCall.tsx
│       │   └── Settings.tsx
│       └── api/client.ts
└── vscode-extension/           # VS Code 插件
    ├── package.json
    └── src/
        ├── extension.ts
        └── panel/chatPanel.ts
```

### 2.3 关键设计决策

1. **核心库完全独立**：`src/open_agent/` 不依赖 FastAPI、React 或 VS Code API，可作为纯 Python 库安装使用。
2. **统一模型接口**：所有 LLM 调用通过 `models/base.py` 的抽象接口，支持无缝切换供应商。
3. **工具即插即用**：内置工具 + MCP 工具统一通过 `ToolRegistry` 注册，Agent 不区分来源。
4. **RAG 与记忆分离**：RAG 用于检索外部知识，Memory 用于维护对话上下文，两者独立可配置。

## 3. 数据流设计

### 3.1 单次用户请求生命周期

```
用户输入
   │
   ▼
┌─────────────────────┐
│ 1. 记忆加载          │  ← 从 ShortTermMemory 获取近期对话历史
│    (MemoryLoader)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. RAG 检索          │  ← 如用户问题需要外部知识，调用 Retriever
│    (Retriever)      │     从向量库检索 Top-K 相关文档片段
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. 上下文组装        │  ← 拼接：System Prompt + 检索结果 + 历史 + 用户输入
│  (ContextBuilder)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. LLM 规划/响应     │  ← 调用模型，支持两种模式：
│    (ModelInterface) │     • 直接回答（无需工具）
│                     │     • ReAct 循环：Thought → Action → Observation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. 工具执行（如需）   │  ← Planner 解析工具调用，Executor 通过 ToolRegistry
│  (ToolExecutor)     │     路由到具体工具（内置或 MCP），返回 Observation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. 循环或输出        │  ← 若工具返回，回到步骤 4 继续 ReAct 循环；
│    (AgentCore)      │     若模型给出最终答案，进入步骤 7
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 7. 记忆更新与返回    │  ← 将本轮对话写入 ShortTermMemory，返回给用户
│  (MemoryUpdater)    │
└─────────────────────┘
```

## 4. 组件设计

### 4.1 Agent Core (`agent/core.py`)

- **职责**：ReAct 主循环的编排器，协调 Planner、Executor、Memory、RAG 的交互。
- **接口**：
  ```python
  class Agent:
      async def run(self, user_input: str, session_id: str) -> AgentOutput: ...
  ```
- **循环限制**：单轮对话最多 10 步 ReAct，超过则强制返回总结，防止无限循环。

### 4.2 Planner (`agent/planner.py`)

- **职责**：解析 LLM 输出，判断是直接回答还是工具调用，解析具体工具名和参数。
- **输出类型**：
  - `DirectResponse(text: str)`
  - `ToolCall(tool_name: str, arguments: dict)`

### 4.3 ToolExecutor (`agent/executor.py`)

- **职责**：接收 `ToolCall`，通过 `ToolRegistry` 路由到实际工具，执行并返回 `Observation`。
- **隔离性**：每个工具在独立异常域中执行，单个工具失败不影响其他工具。

### 4.4 ModelInterface (`models/base.py`)

- **职责**：屏蔽不同 LLM 供应商的差异，提供统一的 `chat(messages, tools) -> response` 接口。
- **支持供应商**：OpenAI API 兼容端点、Anthropic Claude、本地 Ollama。

### 4.5 RAG 模块 (`rag/`)

- **Indexer**：将文档切分后存入向量数据库（默认 ChromaDB）。
- **Retriever**：基于用户查询做向量相似度检索，返回 Top-K 文档片段。

### 4.6 Memory 模块 (`memory/`)

- **ShortTermMemory**：滑动窗口对话历史，按 token 数或消息数截断。
- **LongTermMemory**：基于向量存储的持久记忆，用于跨会话回忆。

### 4.7 ToolRegistry (`tools/registry.py`)

- **职责**：管理所有可用工具的注册与发现。
- **注册来源**：内置工具（启动时自动注册）、运行时动态注册的 MCP 工具。

### 4.8 MCP Adapter (`tools/mcp/`)

- **职责**：将外部 MCP 服务器暴露的工具转换为内部 `Tool` 接口。
- **客户端**：`mcp/client.py` 负责与 MCP 服务器的 stdio/SSE 通信。

### 4.9 API Server (`server/api.py`)

- **职责**：FastAPI 服务，将核心库的 `Agent.run()` 暴露为 WebSocket/HTTP 端点。
- **设计原则**：纯适配层，不含业务逻辑，便于 Web UI 和 VS Code 插件消费。

## 5. 错误处理策略

| 错误类型 | 处理方式 | 说明 |
|---|---|---|
| **工具执行失败** | 捕获异常，将错误信息作为 Observation 返回给 LLM | 让模型自己决定重试、换工具或告知用户 |
| **LLM API 错误** | 指数退避重试 3 次，仍失败则返回友好错误信息 | 包含 rate limit、超时、鉴权失败等 |
| **RAG 检索为空** | 继续执行，但不在上下文中附加检索结果 | 避免阻塞用户请求 |
| **MCP 连接失败** | 标记该工具为不可用，记录日志，不影响其他工具 | 启动时健康检查，运行时隔离 |
| **循环/超时保护** | 单轮对话最多 10 步 ReAct，超限时强制返回总结 | 防止无限循环和 runaway 开销 |

## 6. 测试策略

1. **单元测试**（`tests/unit/`）：覆盖 `Planner`、`ToolRegistry`、`ModelInterface` 等纯逻辑组件，使用 `pytest` + `unittest.mock`。
2. **集成测试**（`tests/integration/`）：测试完整 ReAct 循环，LLM 调用使用 fake/mock 模型。
3. **工具测试**（`tests/tools/`）：每个内置工具独立测试，MCP 工具使用 mock 服务器。
4. **E2E 测试**（`tests/e2e/`）：CLI 层面用 `subprocess` 跑完整对话流程。
5. **CI**：GitHub Actions 运行全部测试 + `ruff` 代码检查 + `mypy` 类型检查。

## 7. 技术栈

| 层级 | 技术 |
|---|---|
| 核心库 | Python 3.10+, `pydantic`, `httpx`, `typer`, `rich` |
| RAG/向量 | `chromadb`, `sentence-transformers` |
| Web 服务 | `fastapi`, `uvicorn` |
| Web UI | React 18, TypeScript, Vite, Tailwind CSS |
| VS Code 插件 | TypeScript, VS Code Extension API |
| 测试 | `pytest`, `pytest-asyncio`, `pytest-cov` |
| 代码质量 | `ruff`, `mypy` |

## 8. 开源与交付

- 代码托管于 GitHub，采用 MIT 许可证。
- `pyproject.toml` 支持 `pip install -e .` 开发安装。
- CLI 命令为 `open-agent`。
- FastAPI 服务启动命令为 `open-agent-server`。
- 提供 `Makefile` 封装常用命令（安装、测试、lint、启动）。
