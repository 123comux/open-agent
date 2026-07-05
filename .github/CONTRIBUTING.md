# Contributing to Open Agent

感谢你对 Open Agent 的兴趣！无论是 bug 修复、新功能、文档改进还是问题反馈，社区贡献都是项目成长的关键。

## 开发环境

1. Fork 仓库并克隆到本地。
2. 安装 Python 3.10 / 3.11 / 3.12。
3. 安装完整开发依赖：

```bash
pip install -e ".[dev,all]"
```

如果只需要最小核心（不含 RAG、server、MCP）：

```bash
pip install -e ".[dev]"
```

4. （可选）安装 Playwright 浏览器依赖：

```bash
pip install -e ".[browser]"
playwright install chromium
```

## 项目结构

- `src/open_agent/` —— 纯 Python 核心库，不依赖前端。
- `src/open_agent/agent/` —— ReAct / LangGraph Agent 实现。
- `src/open_agent/models/` —— 多 LLM 提供商适配器。
- `src/open_agent/tools/` —— 工具基类、内置工具、MCP 适配器。
- `src/open_agent/rag/` —— RAG 索引、检索、知识库管理。
- `src/open_agent/memory/` —— 短期与长期记忆。
- `src/open_agent/server/` —— FastAPI HTTP/WebSocket 适配层。
- `web/` —— React + Vite Web UI。
- `vscode-extension/` —— VS Code 插件。
- `tests/` —— 单元测试与集成测试。

## 代码规范

- 使用 `ruff` 进行代码格式与 lint：
  ```bash
  ruff check src tests
  ```
- 使用 `mypy` 进行类型检查（当前处于严格模式，历史债务逐步清理中）：
  ```bash
  mypy src/open_agent
  ```
- 所有新增功能必须附带单元测试。
- 保持核心库独立于前端/插件代码。

## 提交前检查

```bash
make lint
make typecheck
make test
```

Windows 用户如果没有 `make`，可以直接运行：

```bash
ruff check src tests
mypy src/open_agent
pytest
```

前端项目请在对应目录下运行：

```bash
cd web && npm ci && npm run build
cd vscode-extension && npm ci && npm run compile
```

## Pull Request 流程

1. 从 `main` 切出功能分支：`git checkout -b feature/your-feature`。
2. 提交清晰、独立的 commit。
3. 确保 CI 通过（Python 测试、ruff、Web UI 构建、VS Code 编译）。
4. 在 PR 描述中说明变更动机、主要改动和测试方式。
5. 等待维护者 review。

## 报告问题

提交 Issue 时请尽量包含：

- 运行环境（OS、Python 版本、依赖版本）。
- 复现步骤。
- 期望行为与实际行为。
- 相关日志或报错堆栈。

## 行为准则

请保持友善、尊重与建设性。所有参与者都应遵守本项目的行为准则：互相尊重、欢迎新人、对事不对事。
