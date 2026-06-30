# Open Agent 实施计划

> 设计文档：`docs/superpowers/specs/2026-07-01-open-agent-design.md`

## 阶段一：项目骨架与核心抽象

### Task 1.1 — 项目初始化
- 创建 `pyproject.toml`（包名 `open-agent`，Python 3.10+）
- 配置依赖：`pydantic`, `httpx`, `typer`, `rich`, `fastapi`, `uvicorn`
- 配置开发依赖：`pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`
- 创建 `src/open_agent/__init__.py`
- 创建 `Makefile`（install / test / lint / dev 命令）
- 创建 `.github/workflows/ci.yml`
- 创建 `README.md`（基础说明）

### Task 1.2 — 模型抽象接口 (`models/base.py`)
- 定义 `Message`、`ToolSchema`、`ModelResponse` 等 Pydantic 模型
- 定义 `ModelInterface` 抽象基类：
  ```python
  async def chat(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> ModelResponse: ...
  ```

### Task 1.3 — 工具抽象与注册表 (`tools/base.py`, `tools/registry.py`)
- 定义 `Tool` 抽象基类：`name`, `description`, `parameters_schema`, `async execute(**kwargs) -> str`
- 实现 `ToolRegistry`：注册、查找、列出工具

### Task 1.4 — Planner 与 Executor (`agent/planner.py`, `agent/executor.py`)
- 定义 `PlannerOutput` 联合类型：`DirectResponse | ToolCall`
- 实现 `Planner.parse(llm_output) -> PlannerOutput`
- 实现 `ToolExecutor.execute(tool_call, registry) -> Observation`

### Task 1.5 — Agent Core ReAct 循环 (`agent/core.py`)
- 实现 `Agent` 类，编排 Memory → RAG → Context → LLM → Tool → 循环
- 循环上限 10 步，超限强制总结
- 定义 `AgentOutput` 返回结构

**验证标准**：使用 mock 模型和 mock 工具，能完成完整 ReAct 循环的单元测试通过。

---

## 阶段二：模型适配器实现

### Task 2.1 — OpenAI 适配器 (`models/openai.py`)
- 使用 `httpx` 调用 OpenAI Chat Completions API
- 支持自定义 `base_url`（兼容所有 OpenAI API 端点）
- 支持 function calling / tool use
- 实现指数退避重试（3 次）

### Task 2.2 — Anthropic Claude 适配器 (`models/anthropic.py`)
- 调用 Anthropic Messages API
- 转换内部 Message 格式 ↔ Claude 格式
- 支持 tool use

### Task 2.3 — Ollama 适配器 (`models/ollama.py`)
- 调用本地 Ollama REST API
- 支持本地模型列表获取

**验证标准**：每个适配器有独立单元测试（mock HTTP），集成测试使用 fake 模型跑通完整循环。

---

## 阶段三：内置工具实现

### Task 3.1 — Shell 工具 (`tools/builtin/shell.py`)
- 执行 shell 命令，返回 stdout/stderr/exit_code
- 超时保护（默认 30s）

### Task 3.2 — Python 代码解释器 (`tools/builtin/python.py`)
- 在子进程中执行 Python 代码，返回结果
- 沙箱隔离，超时保护

### Task 3.3 — 文件操作工具 (`tools/builtin/file.py`)
- 读文件、写文件、列目录、搜索文件内容

### Task 3.4 — 网络搜索工具 (`tools/builtin/web_search.py`)
- 集成搜索 API（默认 DuckDuckGo，可配置）
- 返回结构化搜索结果

### Task 3.5 — 浏览器工具 (`tools/builtin/browser.py`)
- 基于 Playwright 的页面抓取与交互
- 支持 navigate / click / extract_text / screenshot

**验证标准**：每个工具有独立测试，Shell 和 Python 工具有超时测试。

---

## 阶段四：RAG 与记忆系统

### Task 4.1 — 短期记忆 (`memory/short_term.py`)
- 滑动窗口实现，按 token 数截断
- 会话隔离（session_id）

### Task 4.2 — 长期记忆 (`memory/long_term.py`)
- 基于向量存储的持久记忆
- 支持写入、检索、删除

### Task 4.3 — RAG Indexer (`rag/indexer.py`)
- 文档切分（递归字符切分）
- 向量化并存入 ChromaDB
- 支持批量索引目录

### Task 4.4 — RAG Retriever (`rag/retriever.py`)
- 向量相似度检索 Top-K
- 结果重排序（可选）
- 空结果优雅降级

### Task 4.5 — ChromaDB 存储 (`rag/stores/chroma.py`)
- 封装 ChromaDB 客户端
- 支持持久化路径配置

**验证标准**：索引 → 检索 → 注入上下文的完整流程测试通过。

---

## 阶段五：MCP 集成

### Task 5.1 — MCP 客户端 (`tools/mcp/client.py`)
- 实现 MCP 协议客户端（stdio + SSE 传输）
- 连接管理、健康检查、自动重连

### Task 5.2 — MCP 适配器 (`tools/mcp/adapter.py`)
- 将 MCP 工具转换为内部 `Tool` 接口
- 启动时自动发现并注册到 `ToolRegistry`
- 工具不可用时隔离降级

**验证标准**：使用 mock MCP 服务器测试工具发现和调用。

---

## 阶段六：CLI 界面

### Task 6.1 — CLI 入口 (`cli.py`)
- 使用 Typer + Rich 构建交互式 REPL
- 支持 `open-agent chat`、`open-agent index <dir>`、`open-agent config` 子命令
- 配置文件 `~/.open-agent/config.yaml`（模型选择、API key、工具开关）

### Task 6.2 — 流式输出与工具调用展示
- 实时显示 Thought / Action / Observation
- 工具调用结果折叠展示

**验证标准**：E2E 测试通过 `subprocess` 跑完整 CLI 对话。

---

## 阶段七：FastAPI 服务

### Task 7.1 — API 服务 (`server/api.py`)
- `POST /api/chat` — 同步对话
- `WS /api/chat/stream` — 流式对话（WebSocket）
- `GET /api/tools` — 列出可用工具
- `POST /api/index` — 触发文档索引
- CORS 配置，允许 Web UI 跨域访问

### Task 7.2 — 服务启动入口
- `open-agent-server` 命令启动 FastAPI 服务
- 支持配置端口和 host

**验证标准**：API 接口测试通过，WebSocket 流式输出正常。

---

## 阶段八：Web UI

### Task 8.1 — 项目初始化
- Vite + React 18 + TypeScript + Tailwind CSS
- API client 封装（HTTP + WebSocket）

### Task 8.2 — 聊天界面 (`components/Chat.tsx`)
- 消息列表、输入框、流式渲染
- Markdown 渲染支持

### Task 8.3 — 工具调用展示 (`components/ToolCall.tsx`)
- 折叠展示 Thought / Action / Observation
- 工具执行状态指示

### Task 8.4 — 设置页面 (`components/Settings.tsx`)
- 模型选择、API Key 配置
- 工具开关、RAG 配置

**验证标准**：本地启动 Web UI 能完成对话和工具调用展示。

---

## 阶段九：VS Code 插件

### Task 9.1 — 插件骨架
- `package.json` 配置（命令、视图容器）
- 侧边栏 Webview 面板

### Task 9.2 — 与后端通信
- 通过 stdio 启动 Python 后端进程
- 或通过 HTTP 连接已运行的 `open-agent-server`

### Task 9.3 — 聊天面板 (`panel/chatPanel.ts`)
- 内嵌聊天 UI
- 工具调用展示

**验证标准**：VS Code 中能打开面板并完成对话。

---

## 阶段十：开源打磨

### Task 10.1 — README 完善
- 项目介绍、截图、快速开始、配置说明
- 架构图、贡献指南

### Task 10.2 — LICENSE 与 CONTRIBUTING
- MIT LICENSE
- CONTRIBUTING.md（开发环境搭建、代码规范、PR 流程）

### Task 10.3 — GitHub Actions CI 完善
- Python 测试 + ruff + mypy
- Web UI 构建 + 类型检查
- VS Code 插件编译

### Task 10.4 — 示例与文档
- `examples/` 目录：基础对话、RAG 索引、自定义工具、MCP 集成示例
- 文档站配置（可选 mkdocs 或 docusaurus）

---

## 依赖关系

```
阶段一（核心骨架）─┬─→ 阶段二（模型适配器）
                  ├─→ 阶段三（内置工具）
                  ├─→ 阶段四（RAG与记忆）
                  └─→ 阶段五（MCP集成）
                          │
                          ▼
                   阶段六（CLI）─┬─→ 阶段七（FastAPI）
                                │           │
                                │           ▼
                                │    阶段八（Web UI）
                                │
                                └─→ 阶段九（VS Code 插件）
                                          │
                                          ▼
                                   阶段十（开源打磨）
```

- 阶段一是所有后续阶段的前置依赖
- 阶段二、三、四、五可并行开发
- 阶段六、七依赖前置阶段完成
- 阶段八、九依赖阶段七
- 阶段十最后执行
