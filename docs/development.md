# 开发指南

本页面向贡献者：项目结构、开发环境搭建、代码规范、测试、提交规范、发布流程与调试技巧。

## 项目结构

```text
open-agent/
├── pyproject.toml              # 包配置、依赖、ruff/mypy/pytest 设置
├── Makefile                    # install / dev / test / lint / serve / clean
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Dockerfile                  # 多阶段构建：deps → runtime
├── docker-compose.yml          # 开发：api + web (Vite dev)
├── docker-compose.prod.yml     # 生产：api + web (nginx)
├── mkdocs.yml                  # 本文档站配置
├── .env.example                # 环境变量模板
├── mcp_servers.example.json    # MCP 配置示例
├── .github/
│   ├── workflows/ci.yml        # 矩阵 CI：ruff + mypy + pytest
│   ├── workflows/release.yml
│   ├── CONTRIBUTING.md
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/                       # mkdocs 文档源
├── examples/                   # 可运行示例脚本
├── shared/types.ts             # 前后端共享类型
├── src/open_agent/             # 核心库（无前端依赖）
│   ├── __init__.py
│   ├── cli.py                  # Typer/Rich CLI: chat / ask / serve / index / evaluate
│   ├── config.py               # Settings（OPEN_AGENT_ 环境变量）
│   ├── logging_config.py       # 结构化日志
│   ├── agent/
│   │   ├── core.py             # ReAct 主循环 + run_stream
│   │   ├── planner.py          # 解析 LLM 输出 → DirectResponse | ToolCall
│   │   ├── executor.py         # 路由 ToolCall 到 registry
│   │   ├── context_window.py   # 上下文截断
│   │   ├── langchain_tools.py  # LangChain 工具适配
│   │   └── langgraph_agent.py  # 可选 LangGraph agent（带思考链）
│   ├── models/
│   │   ├── base.py             # ModelInterface 抽象
│   │   ├── _http.py            # 共享 httpx 客户端与重试
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── ollama_provider.py
│   │   └── langchain_adapter.py
│   ├── rag/
│   │   ├── document_loaders.py # 多格式解析：txt/md/pdf/docx/csv/json/html
│   │   ├── indexer.py          # 切块
│   │   ├── embedding_cache.py  # 嵌入磁盘缓存
│   │   ├── stores/faiss_store.py / chroma.py
│   │   ├── hybrid_retriever.py # 向量 + BM25 + RRF
│   │   ├── reranker.py         # 交叉编码器重排序
│   │   ├── kb_router.py        # 多知识库路由
│   │   ├── kb_manager.py       # 高层管理器
│   │   └── evaluation.py       # RAG 质量评估
│   ├── tools/
│   │   ├── base.py             # Tool 抽象基类
│   │   ├── registry.py         # ToolRegistry
│   │   ├── sandbox.py          # check_shell / check_python / check_path
│   │   ├── builtin/            # shell / python / file / web_search / browser / knowledge_base
│   │   └── mcp/                # MCP 客户端与适配器
│   ├── memory/
│   │   ├── short_term.py       # 滑动窗口
│   │   ├── session_manager.py  # 会话持久化
│   │   └── long_term.py        # 向量长期记忆
│   ├── mcp/                    # 顶层 MCP（loader / client / adapter）
│   ├── observability/
│   │   └── tracer.py           # NoOp / LocalJsonl / LangSmith / Langfuse
│   └── server/
│       └── api.py              # FastAPI 适配层（REST + WebSocket）
├── tests/
│   ├── conftest.py             # 共享 fixture（MockModel 等）
│   ├── unit/                   # 单元测试（不依赖外部服务）
│   └── integration/            # 集成测试（API、agent flow）
├── web/                        # React + Vite + Tailwind 前端
└── vscode-extension/           # VS Code 扩展（TypeScript）
```

## 开发环境搭建

```bash
# 1. 克隆你的 fork
git clone https://github.com/<your-username>/open-agent.git
cd open-agent

# 2. 安装开发依赖（含 dev + all extras）
make dev          # 等同 pip install -e ".[dev,all]"

# 3.（可选）浏览器工具运行时
pip install -e ".[browser]"
playwright install chromium

# 4. 验证
pytest tests/
```

!!! note "Python 版本"
    项目要求 Python ≥ 3.10，CI 矩阵覆盖 3.10 / 3.11 / 3.12。建议本地用 3.11 或 3.12 以匹配生产。

### pre-commit（可选）

仓库未强制 pre-commit，但建议配置以避免 CI 失败：

```bash
pip install pre-commit
pre-commit run --all-files
```

可参考 `.pre-commit-config.yaml`（若仓库未提供，自行创建一个跑 `ruff check` 与 `ruff format` 的 hook 即可）。

## 代码规范

### ruff

- 行宽 **100**（`pyproject.toml` 的 `[tool.ruff] line-length`）。
- Lint 规则：`E`（pycodestyle errors）、`F`（pyflakes）、`I`（isort）、`N`（命名）、`W`（warnings）、`UP`（pyupgrade）。
- 运行：

```bash
make lint               # ruff check src tests
ruff format src tests   # 自动格式化（可选）
```

### mypy

- 严格模式：`[tool.mypy] strict = true`、`warn_return_any = true`。
- 目标版本：Python 3.10。
- 运行：

```bash
make typecheck           # mypy src/open_agent
```

### 通用约定

- 每个 Python 模块开头 `from __future__ import annotations`，启用 PEP 563 延迟注解求值。
- Google 风格 docstring，公共 API 必须有 docstring。
- 复用 `src/open_agent/models/_http.py` 的共享 `httpx.AsyncClient` 与重试逻辑，不要在 provider 里自建客户端。
- 工具异常不要抛出——`ToolRegistry.execute` 会兜底，但返回明确错误字符串对模型更友好。

## 测试

### 测试组织

- `tests/unit/` — 单元测试，不依赖外部服务（LLM、网络、文件系统都 mock）。运行快。
- `tests/integration/` — 集成测试，包含 FastAPI `TestClient` 端到端测试与 agent flow。
- `tests/conftest.py` — 共享 fixture，包括 `MockModel`（模拟 `ModelInterface` 的工具调用序列）。

### 运行

```bash
make test                # pytest（asyncio_mode=auto）
pytest tests/unit/       # 仅单元
pytest tests/integration/# 仅集成
pytest -k "test_planner" # 按名过滤
pytest --cov=open_agent  # 覆盖率
```

### 异步测试

`pyproject.toml` 配置 `asyncio_mode = "auto"`，所以 `async def test_...` 函数自动跑在事件循环里，**不需要** `@pytest.mark.asyncio` 装饰器。

### 测试约定

- 不打真实 LLM 或网络请求——用 `unittest.mock.AsyncMock` 或 `conftest.py` 的 `MockModel`。
- 集成测试需要 `server` extra（`pip install -e ".[server]"`）。
- 新增功能或修 bug 时，附带一个聚焦的测试。
- 覆盖率配置：`[tool.coverage.run] source = ["src/open_agent"]`。

## 提交规范

### Commit message

使用**祈使句**作为标题，像下命令一样：

- `Add shell sandbox pattern matching`
- `Fix retry backoff for Anthropic rate limits`
- `Refactor retriever to share the embedding cache`
- `Update README quick-start with the new CLI flags`

标题 ≤ 72 字符，首字母大写。可选 body 说明**为什么**（动机、权衡、背景）。

避免模糊的 `fix`、`update stuff`、`wip`。

### Conventional Commits（推荐）

虽然仓库未强制，但推荐遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```text
feat(rag): add hybrid retrieval with RRF fusion
fix(sandbox): block path traversal in file tool
docs(api): document /api/sessions/search endpoint
refactor(agent): extract _llm_step for testability
test(tools): add browser SSRF guard tests
chore(deps): bump faiss-cpu to 1.9
```

### Pull Request 流程

1. Fork 仓库，从 `main` 切特性分支（`feature/rag-reranker`、`fix/shell-sandbox-glob`）。
2. 改动保持聚焦，代码风格干净。
3. 提 PR 前本地跑：

```bash
ruff check src tests
mypy src/open_agent
pytest tests/
```

4. 推分支开 PR 到 `main`，填 PR 模板说明改了什么、为什么。
5. 应对 review 反馈，往同一分支追加提交；维护者可能在合并时 squash。

详见 [`CONTRIBUTING.md`](https://github.com/123comux/open-agent/blob/main/CONTRIBUTING.md)。

## 发布流程

### 版本号

项目遵循 [Semantic Versioning](https://semver.org/)：`MAJOR.MINOR.PATCH`。

- `0.x.y` — Alpha 阶段（当前 `0.2.0`），API 可能变动。
- `1.0.0` 起 — 公共 API 稳定，破坏性变更才升 MAJOR。

版本号定义在 `pyproject.toml` 的 `[project] version`。

### CHANGELOG

每次发布前更新 `CHANGELOG.md`，遵循 [Keep a Changelog](https://keepachangelog.com/) 格式：

```markdown
## [0.3.0] - 2026-08-01

### Added
- New feature X

### Changed
- Behavior of Y

### Fixed
- Bug Z
```

### Tag 与发布

```bash
# 1. 更新版本号与 CHANGELOG
# 编辑 pyproject.toml 的 version，更新 CHANGELOG.md

# 2. 提交
git commit -am "Release 0.3.0"

# 3. 打 tag
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin main --tags

# 4. GitHub Actions (.github/workflows/release.yml) 自动构建并发布
```

## 调试技巧

### 用 --demo 模式

无需 API Key，用内置 `DemoModel` 模拟工具调用，验证 ReAct 循环与工具集成：

```bash
open-agent chat --demo
open-agent ask "calculate 2 to the power of 10" --demo
```

`DemoModel` 会根据问题关键词触发 `python`、`file`、`shell` 工具调用，第二轮返回固定总结。

### 用 LangGraph 思考链

`--langgraph` 切换到 `LangGraphAgent`，输出意图分类、子任务分解、思考链与反思：

```bash
open-agent chat --langgraph
```

### 日志

调到 DEBUG 看完整调用链：

```python
import logging
from open_agent.logging_config import setup_logging
setup_logging(level="DEBUG")
# 或运行时
logging.getLogger("open_agent").setLevel(logging.DEBUG)
```

关键 logger：

- `open_agent.server` — HTTP 请求/响应、session_id。
- `open_agent.agent.core` — ReAct 每步、步数耗尽。
- `open_agent.tools.registry` — 工具执行与异常。
- `open_agent.rag.kb_manager` — 索引、路由、查询。

### Trace

开 local provider，跑一次对话后查看 trace：

```bash
# 直接看文件
cat .open_agent_traces/traces.jsonl | python -m json.tool

# 或通过 API
curl http://127.0.0.1:8000/api/traces?limit=1 | python -m json.tool
```

trace 里每个 span 有 `metrics.latency_ms`，能快速定位慢调用。详见 [可观测性](observability.md)。

### 单元测试调试

```bash
# 跑单个测试并打印输出
pytest tests/unit/test_planner.py -v -s

# 失败时进入 pdb
pytest tests/unit/test_planner.py --pdb

# 用 -k 过滤
pytest -k "test_planner and not test_parse_text"
```

### 在 Notebook 里实验

核心是纯 Python 库，可在 Jupyter 直接嵌入：

```python
import asyncio
from open_agent.agent.core import Agent
from open_agent.models.openai_provider import OpenAIModel
from open_agent.tools.builtin import ShellTool
from open_agent.tools.registry import ToolRegistry

reg = ToolRegistry()
reg.register(ShellTool())
agent = Agent(
    model=OpenAIModel(api_key="sk-...", model="gpt-4o-mini"),
    tool_registry=reg,
    max_steps=5,
)
out = asyncio.run(agent.run("列出当前目录"))
print(out.response, out.tool_calls_made)
```

## 下一步

- [架构设计](architecture.md) — 理解核心模块与数据流。
- [工具系统](tools.md) — 自定义工具开发。
- [CONTRIBUTING.md](https://github.com/123comux/open-agent/blob/main/CONTRIBUTING.md) — 完整贡献指南。
