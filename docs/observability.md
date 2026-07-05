# 可观测性

Open Agent 内置轻量 trace 体系，记录每次 Agent 运行的完整 span 树（LLM 调用、工具执行、延迟、错误）。本页介绍三种 trace 后端、配置切换、本地 JSONL 格式、性能影响与日志配置。

## 三种 Provider

`OPEN_AGENT_OBSERVABILITY_PROVIDER` 选择 trace 后端：

| Provider | 类 | 适用场景 |
|---|---|---|
| `local` | `LocalJsonlTracer` | 默认。本地开发、调试、无外部依赖。把 trace 写入 JSONL 文件。 |
| `langsmith` | `LangSmithTracer` | 生产。转发到 [LangSmith](https://smith.langchain.com/) 平台，支持可视化与团队协作。 |
| `langfuse` | `LangfuseTracer` | 生产。转发到 [Langfuse](https://langfuse.com/)（自托管或云）。 |

当 `OPEN_AGENT_ENABLE_OBSERVABILITY=false` 时，使用 `NoOpTracer`——仅维护内存结构，不做任何持久化，零开销。

## 配置切换

切换 provider 只需改环境变量并重启：

```bash
# local（默认）
OPEN_AGENT_ENABLE_OBSERVABILITY=true
OPEN_AGENT_OBSERVABILITY_PROVIDER=local
OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR=/var/open-agent/traces

# LangSmith
OPEN_AGENT_OBSERVABILITY_PROVIDER=langsmith
OPEN_AGENT_LANGSMITH_API_KEY=ls-...
OPEN_AGENT_LANGSMITH_PROJECT=open-agent-prod

# Langfuse
OPEN_AGENT_OBSERVABILITY_PROVIDER=langfuse
OPEN_AGENT_LANGFUSE_PUBLIC_KEY=pk-lf-...
OPEN_AGENT_LANGFUSE_SECRET_KEY=sk-lf-...
OPEN_AGENT_LANGFUSE_HOST=https://cloud.langfuse.com
```

!!! warning "provider 不在运行时可更新集合内"
    `observability_provider` 不在 `SettingsUpdateRequest` 的字段里，运行时通过 `/api/settings` 改它**不会**生效。切换 provider 必须重启进程。其余可观测字段（`enable_observability`、`observability_output_dir`、`langsmith_*`、`langfuse_*`）同理。

## LocalJsonlTracer

`LocalJsonlTracer` 把每个完成的 trace 以一行 JSON 追加到 `<output_dir>/traces.jsonl`。

### 输出位置

- 默认目录：`.open_agent_traces/`（即 `OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR`）。
- Docker 部署映射到 `/app/.open_agent_traces`（见 `docker-compose.yml`）。
- 目录不存在会自动创建（`mkdir(parents=True, exist_ok=True)`）。

### 输出格式

每行一个 JSON 对象，结构等同于 `Trace.to_dict()`：

```json
{
  "id": "uuid4",
  "name": "agent.run",
  "start_time": "2026-07-01T08:00:00+00:00",
  "end_time": "2026-07-01T08:00:03.5+00:00",
  "input": {"user_input": "...", "session_id": "cli-1"},
  "metadata": {},
  "status": "ok",
  "root_span": {
    "id": "uuid4",
    "parent_id": null,
    "trace_id": "uuid4",
    "type": "agent",
    "name": "react_loop",
    "start_time": "...",
    "end_time": "...",
    "input": {"user_input": "..."},
    "output": {"response": "...", "steps": 2},
    "metadata": {},
    "metrics": {},
    "status": "ok",
    "children": [
      {
        "id": "...", "type": "llm", "name": "step_1.llm",
        "metrics": {"latency_ms": 850},
        "output": {"content": "...", "tool_calls": []}
      },
      {
        "id": "...", "type": "tool", "name": "shell",
        "metrics": {"latency_ms": 120, "is_error": false},
        "input": {"command": "ls"}, "output": {"observation": "..."}
      }
    ]
  }
}
```

文件以行缓冲模式打开（`buffering=1`），每行写完立即 flush，进程崩溃也不会丢失已完成的 trace。

## Trace 结构

`Trace` 与 `TraceSpan` 是两个核心数据类：

### Trace（顶层）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` (uuid4) | trace 唯一 ID，对应 `AgentOutput.trace_id`。 |
| `name` | `str` | `agent.run` 或 `agent.run_stream`。 |
| `start_time` / `end_time` | `str` (ISO8601 UTC) | 起止时间。 |
| `input` | `Any` | `{"user_input": ..., "session_id": ...}`。 |
| `metadata` | `dict` | 保留扩展。 |
| `status` | `str` | `ok` / `error` / `incomplete`。 |
| `root_span` | `TraceSpan \| None` | 根 span，类型为 `agent`。 |

### TraceSpan（span）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | span 唯一 ID。 |
| `parent_id` | `str \| None` | 父 span ID（根 span 为 `None`）。 |
| `trace_id` | `str` | 所属 trace。 |
| `type` | `str` | span 类型，见下表。 |
| `name` | `str` | 具体名字，如 `step_1.llm`、`shell`。 |
| `start_time` / `end_time` | `str` | 起止时间。 |
| `input` / `output` | `Any` | 输入输出快照。 |
| `metrics` | `dict` | 如 `{"latency_ms": 850, "is_error": false}`。 |
| `status` | `str` | `ok` / `error`。 |
| `children` | `list[TraceSpan]` | 子 span（工具/LLM 都挂在 root 下）。 |

### SpanType

| `type` | 含义 | 由谁创建 |
|---|---|---|
| `agent` | 一次完整 ReAct 循环 | `Agent._prepare_run` 创建 root span |
| `llm` | 一次模型 `chat` 调用 | `Agent._llm_step` |
| `llm_stream` | 一次模型 `stream_chat` 调用 | `run_stream` 的最终回答流 |
| `tool` | 一次工具执行 | `Agent._execute_tool_call` |

LangSmith/Langfuse 各自有自己的类型映射：

| Open Agent | LangSmith `run_type` | Langfuse `observation_type` |
|---|---|---|
| `agent` | `chain` | `chain` |
| `llm` | `llm` | `generation` |
| `tool` | `tool` | `tool` |
| `retrieval` | `retriever` | `retriever` |

## 性能影响

!!! warning "LocalJsonlTracer 的同步写阻塞"
    `LocalJsonlTracer.end_trace` 是**同步**方法（受共享 `Tracer` 接口约束），每次完成 trace 时会同步写一行 JSON 到磁盘并 `flush()`。这会短暂阻塞事件循环。

    - 单次写入开销通常 < 1ms（一行 JSON + 一次 `flush`），对交互式对话几乎无感。
    - 高并发场景下，多个 trace 同时结束可能累积阻塞，事件循环延迟会上升。
    - **生产环境若 QPS 较高，建议切换到 `langsmith` 或 `langfuse`**——它们的 `end_trace` 内部做网络上报但会捕获所有异常，不会阻塞 Agent 主循环（failure-fail-open，trace 丢失不影响业务）。

LangSmith/Langfuse 的设计原则是「失败静默」：所有后端 API 调用都包在 `try/except` 里，异常仅 `logger.debug`，不抛出。这保证可观测后端故障永远不会让 Agent 运行失败。

## Trace 端点

### GET /api/traces?limit=

列出最近的本地 trace（仅 `LocalJsonlTracer` 模式下有数据）。`limit` 范围 1~1000，默认 100，返回最新在前。读取操作在 `asyncio.to_thread` 中执行，避免阻塞事件循环。

```bash
curl http://127.0.0.1:8000/api/traces?limit=5
```

### GET /api/traces/{trace_id}

取单个 trace 详情（含完整 span 树）。`LocalJsonlTracer.get_trace` 会扫描最多 10000 条 trace 匹配 id。未找到返回 `404 trace_not_found`。

!!! note "仅 local 模式可用"
    这两个端点只读 `LocalJsonlTracer` 写的 JSONL 文件。LangSmith/Langfuse 模式下返回空列表 / 404——请到对应平台的 UI 查看 trace。

## 日志配置

`src/open_agent/logging_config.py` 提供结构化日志：

### 日志格式

`AgentFormatter` 输出形如：

```text
INFO    open_agent.server    | Chat request from session=cli-1 [session_id=cli-1]
INFO    open_agent.server    | GET /api/chat -> 200 [duration_ms=1234.5]
ERROR   open_agent.server    | Chat failed: ConnectionError [session_id=cli-1]
```

支持的结构化字段（通过 `extra={...}` 传入）：`session_id`、`tool_name`、`step`、`model`、`duration_ms`。

### 日志级别

- 根 logger：`open_agent`，默认 `INFO`。
- 噪声库降级：`httpx` / `httpcore` / `uvicorn.access` 设为 `WARNING`。
- 调整级别：`setup_logging(level="DEBUG")`，或在代码里 `logging.getLogger("open_agent").setLevel(logging.DEBUG)`。

### 关键日志点

- `server` — 每次 HTTP 请求的方法、路径、状态码、耗时。
- `Agent` — ReAct 循环的每步 LLM 调用、工具调用、步数耗尽。
- `KBManager` — 文档索引、删除、查询路由。
- `MCPClient` — 子进程启动、工具发现、调用。

## 调试技巧

- **想看完整 trace**：开 `local` provider，跑一次对话后查看 `.open_agent_traces/traces.jsonl`，或调 `/api/traces/{id}`。
- **想看 LLM 原始输入输出**：trace 的 `llm` span 的 `input.messages` 与 `output` 字段包含完整消息列表与模型响应。
- **想看工具调用细节**：`tool` span 的 `input` 是参数，`output.observation` 是返回字符串，`metrics.is_error` 标记是否出错。
- **想定位性能瓶颈**：看每个 span 的 `metrics.latency_ms`，找出最慢的 LLM 调用或工具执行。

## 下一步

- [配置参考 § 可观测性](configuration.md#可观测性) — 完整可观测性环境变量。
- [API 参考 § observability](api.md#observability) — trace 端点。
- [部署指南](deployment.md) — 生产环境 trace 持久化卷映射。
