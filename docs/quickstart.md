# 快速开始

本页带你 5 分钟内跑通 Open Agent：安装、配置、运行 CLI/API/Web，并完成第一个 RAG 知识库与自定义工具。

## 前置条件

- **Python 3.10 / 3.11 / 3.12**（CI 矩阵覆盖这三个版本）。
- 一个模型 API Key（OpenAI、Anthropic 等）**或** 一个本地 [Ollama](https://ollama.ai/) 实例。
- （可选）[git](https://git-scm.com/) 用于克隆仓库；[Docker](https://www.docker.com/) 用于容器化部署。

!!! tip "不想配 API Key？"
    CLI 提供 `--demo` 模式，使用内置 `DemoModel` 模拟工具调用，可零配置体验 ReAct 循环：
    ```bash
    open-agent chat --demo
    ```

## 1. 安装

```bash
git clone https://github.com/123comux/open-agent.git
cd open-agent
pip install -e ".[all]"
```

`[all]` 会一次性拉取 RAG、server、MCP、browser、observability 全部可选依赖。如需最小安装：

```bash
pip install -e .                  # 仅核心
pip install -e ".[rag]"           # + RAG 后端
pip install -e ".[server]"        # + FastAPI 服务
pip install -e ".[mcp]"           # + MCP 客户端
```

!!! note "Playwright 浏览器工具"
    若要使用 `browser` 工具的交互式动作（click、fill、screenshot），还需安装 Chromium 运行时：
    ```bash
    pip install -e ".[browser]"
    playwright install chromium
    ```

## 2. 配置

Open Agent 通过 `OPEN_AGENT_` 前缀的环境变量读取所有配置。最简便的方式是放一个 `.env` 文件（配合 [direnv](https://direnv.net/) 或手动 `source`）：

```bash
# .env
OPEN_AGENT_MODEL_PROVIDER=openai        # openai | anthropic | ollama
OPEN_AGENT_API_KEY=sk-...               # 模型 API Key（ollama 可省略）
OPEN_AGENT_MODEL_NAME=gpt-4o-mini       # 模型标识符
OPEN_AGENT_BASE_URL=https://api.openai.com/v1   # OpenAI 兼容端点
OPEN_AGENT_MAX_STEPS=10                 # 单轮最大 ReAct 迭代数
```

完全本地化的 Ollama 配置：

```bash
OPEN_AGENT_MODEL_PROVIDER=ollama
OPEN_AGENT_BASE_URL=http://localhost:11434
OPEN_AGENT_MODEL_NAME=llama3
```

!!! warning "完整变量表"
    以上仅是最关键的 5 个变量。完整配置（RAG、沙箱、可观测、MCP、Server 等）见 [配置参考](configuration.md)。

## 3. 五分钟跑起来

### 3.1 CLI 模式

启动多轮交互式 REPL，会话内自动维护短期记忆：

```bash
open-agent chat
```

预期输出：

```text
╭─ open-agent ────────────────────────────────────────────╮
│ Open Agent ready. Provider: openai, Model: gpt-4o-mini. │
│ Type 'exit' to quit.                                    │
╰─────────────────────────────────────────────────────────╯
you> 总结这个仓库里最大的几个文件
...
```

单次提问（一次性问答，附带步数与工具调用统计）：

```bash
open-agent ask "planner 模块在这个代码库里做什么？"
```

CLI 流式输出 token，并以 `[dim]` 颜色标注工具调用与观察结果。

### 3.2 API 模式

启动 FastAPI 服务，供 Web UI 或 VS Code 扩展连接：

```bash
open-agent serve --host 0.0.0.0 --port 8000
# 或直接：
make serve
```

快速测试：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "列出 src 下的 Python 文件", "session_id": "cli-1"}'
```

!!! info "OpenAPI 文档"
    服务启动后访问 `http://127.0.0.1:8000/docs`（Swagger UI）或 `/redoc`（ReDoc）浏览完整 API。

### 3.3 Web UI

在另一个终端启动前端开发服务器：

```bash
cd web
npm install
npm run dev
```

打开 `http://localhost:5173`，在侧边栏选择会话即可与 Agent 对话，并实时查看工具调用、token 流与 Trace。

## 4. 第一个 RAG 知识库

Open Agent 的 RAG 支持 `.txt`、`.md`、`.rst`、`.pdf`、`.docx`、`.csv`、`.json`、`.html` 八种格式。

### 4.1 通过 CLI 索引文档

```bash
# 索引单个文件
open-agent index ./docs/handbook.md --kb company

# 索引整个目录（仅扫描顶层支持的文件）
open-agent index ./docs/policies --kb company --desc "公司规章制度"
```

### 4.2 通过 API 上传文档

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@./docs/handbook.md" \
  -F "kb_name=company"
```

响应示例：

```json
{"status": "indexed", "filename": "handbook.md", "kb_name": "company", "chunks": 42}
```

!!! warning "上传大小限制"
    单个文件最大 **50 MB**（`MAX_UPLOAD_BYTES = 50 * 1024 * 1024`）。超出会返回 `413 file_too_large`。

### 4.3 查询知识库

索引完成后，Agent 会自动通过 `knowledge_base` 工具检索相关段落：

```bash
open-agent ask "公司年假政策是怎样的？"
```

或直接用 Python 库调用：

```python
import asyncio
from open_agent.rag.kb_manager import KBManager

async def main():
    manager = KBManager(
        embedding_model="BAAI/bge-small-zh-v1.5",
        chunk_size=500,
        chunk_overlap=50,
        reranker_model="BAAI/bge-reranker-v2-m3",
        rerank_k=20,
    )
    await manager.index_file("./docs/handbook.md", "company")
    result = await manager.query("年假政策", top_k=5)
    for chunk in result["chunks"]:
        print(chunk["document"][:80], "| score:", chunk["score"])

asyncio.run(main())
```

检索流程为 **路由 → 向量 + BM25 混合检索 → RRF 融合 → 重排序**，详见 [RAG 知识库](rag.md)。

## 5. 第一个自定义工具

自定义工具只需继承 `Tool` 基类，实现 `execute` 方法，再注册到 `ToolRegistry`：

```python
import asyncio
from open_agent.agent.core import Agent
from open_agent.models.openai_provider import OpenAIModel
from open_agent.tools.base import Tool
from open_agent.tools.registry import ToolRegistry


class ExchangeRateTool(Tool):
    name = "exchange_rate"
    description = "查询货币对的实时汇率（演示用：返回固定值）。"
    parameters = {
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "基础货币，如 USD"},
            "quote": {"type": "string", "description": "报价货币，如 CNY"},
        },
        "required": ["base", "quote"],
    }

    async def execute(self, **kwargs) -> str:
        base = str(kwargs.get("base", "USD"))
        quote = str(kwargs.get("quote", "CNY"))
        # 这里替换为真实 API 调用
        return f"1 {base} = 7.18 {quote} (演示值)"


registry = ToolRegistry()
registry.register(ExchangeRateTool())

agent = Agent(
    model=OpenAIModel(api_key="sk-...", model="gpt-4o-mini"),
    tool_registry=registry,
    max_steps=10,
)

output = asyncio.run(agent.run("100 美元等于多少人民币？"))
print(output.response)
```

!!! tip "工具参数 schema"
    `parameters` 字段是标准 JSON Schema，会原样透传给模型的 function-calling 接口。务必为每个字段写清晰的 `description`，这直接影响模型调用工具的准确率。

更完整的工具开发指南（沙箱、命名空间、注册方式）见 [工具系统](tools.md)。

## 下一步

- [架构设计](architecture.md) — 理解 ReAct 循环、Planner 与请求生命周期。
- [配置参考](configuration.md) — 完整环境变量表与运行时更新。
- [API 参考](api.md) — REST/WebSocket 端点、认证与错误处理。
- [工具系统](tools.md) — 内置工具详解与自定义工具开发。
