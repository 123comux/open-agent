# API 参考

Open Agent 的 FastAPI 服务器以 REST + WebSocket 暴露全部能力。本页列出所有端点、Schema、认证、错误处理与客户端示例。

服务由 `open-agent serve` 启动，默认监听 `127.0.0.1:8000`。所有业务端点都在 `/api/` 前缀下。

## 端点总览

| 方法 | 路径 | 标签 | 说明 |
|---|---|---|---|
| GET | `/api/health` | health | 存活检查，恒返回 `{"status":"ok"}`。 |
| GET | `/api/ready` | health | 就绪检查，Agent 未初始化时返回 503。 |
| GET | `/api/tools` | tools | 列出已注册工具的 JSON Schema。 |
| GET | `/api/settings` | settings | 返回当前配置（凭据脱敏）。 |
| POST | `/api/settings` | settings | 运行时更新配置并重建 Agent。 |
| POST | `/api/chat` | chat | 单轮同步对话。 |
| WS | `/ws/chat` | chat | 流式对话（token / 工具事件）。 |
| GET | `/api/sessions` | sessions | 列出所有 session_id。 |
| GET | `/api/sessions/{id}/history` | sessions | 取某会话的历史消息。 |
| DELETE | `/api/sessions/{id}` | sessions | 清空某会话。 |
| POST | `/api/sessions/{id}/rename` | sessions | 重命名会话。 |
| GET | `/api/sessions/search?q=` | sessions | 按内容搜索会话。 |
| GET | `/api/sessions/{id}/export?fmt=` | sessions | 导出会话为 JSON 或 Markdown。 |
| POST | `/api/upload` | rag | 上传文档并索引到 KB。 |
| GET | `/api/knowledge-bases` | rag | 列出所有知识库。 |
| GET | `/api/knowledge-bases/{kb}/documents` | rag | 列出 KB 内文档。 |
| DELETE | `/api/knowledge-bases/{kb}/documents?source=` | rag | 删除某 source 的所有 chunk。 |
| GET | `/api/traces?limit=` | observability | 列出最近的本地 trace。 |
| GET | `/api/traces/{id}` | observability | 取单个 trace 详情。 |

!!! info "OpenAPI 文档"
    - Swagger UI：`http://127.0.0.1:8000/docs`
    - ReDoc：`http://127.0.0.1:8000/redoc`
    - OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`

## health

### GET /api/health

存活检查，不依赖 Agent 是否初始化完成。

```http
GET /api/health
```

```json
{"status": "ok"}
```

### GET /api/ready

就绪检查。Agent 未初始化时返回 503，适合用作容器编排的 readiness probe。

```http
GET /api/ready
```

```json
{"status": "ready"}
```

未就绪时：

```json
{"status": "not_ready", "reason": "agent_not_initialized"}
```

## chat

### POST /api/chat

同步单轮对话。Agent 完整跑完 ReAct 循环后返回最终回答与工具调用统计。

**请求体**（`ChatRequest`）：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `message` | `str` | — | 用户消息，必填。 |
| `session_id` | `str` | `"default"` | 会话 ID，用于历史持久化。仅允许 `[a-zA-Z0-9_\-.]`，禁止 `..`。 |

**响应体**（`ChatResponse`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `response` | `str` | 最终回答文本。 |
| `steps` | `int` | ReAct 循环步数。 |
| `tool_calls_made` | `list[dict]` | 每步的工具调用记录（name/arguments/observation/is_error/step）。 |

**示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message": "列出 src 下的 Python 文件", "session_id": "cli-1"}'
```

```json
{
  "response": "src 下共有 47 个 Python 文件，包括 agent/core.py、tools/registry.py ...",
  "steps": 2,
  "tool_calls_made": [
    {"step": 1, "name": "shell", "arguments": {"command": "find src -name '*.py'"}, "observation": "...", "is_error": false}
  ]
}
```

### WebSocket /ws/chat

流式对话，连接保持期间可多次发送消息。每次消息流式返回 token 与工具事件，最后推送一个 `done` 事件。

**鉴权**：若 `OPEN_AGENT_API_AUTH_TOKEN` 非空，连接时需带 `Authorization: Bearer <token>` 头；否则服务端会以 `4001 Unauthorized` 关闭连接。

**客户端→服务端**：

```json
{"message": "总结这个仓库的架构", "session_id": "cli-1"}
```

**服务端→客户端事件**：

| `type` | 字段 | 说明 |
|---|---|---|
| `token` | `content: str` | 流式回答的一个 token 片段。 |
| `tool_start` | `name: str`, `arguments: dict` | 工具调用开始。 |
| `tool_end` | `name: str`, `observation: str`, `is_error: bool` | 工具调用结束。 |
| `thought` | `content: str`, `step: int` | 推理步说明（CLI 流式用）。 |
| `done` | `response: str`, `steps: int`, `tool_calls_made: list`, `trace_id: str` | 整轮结束。 |
| `error` | `message: str` | 错误（如 session_id 非法）。 |

**Python 客户端示例**：

```python
import asyncio
import json
import websockets

async def main():
    url = "ws://127.0.0.1:8000/ws/chat"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}  # 无 token 时省略
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"message": "你好", "session_id": "cli-1"}))
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_start":
                print(f"\n[tool] {event['name']} {event['arguments']}")
            elif event["type"] == "tool_end":
                print(f"[obs] {event['observation'][:80]}")
            elif event["type"] == "done":
                print(f"\n[done] steps={event['steps']} trace={event['trace_id']}")
                break

asyncio.run(main())
```

## tools

### GET /api/tools

返回所有已注册工具的 JSON Schema 列表，供前端动态渲染工具面板。

```http
GET /api/tools
```

```json
{
  "tools": [
    {
      "name": "shell",
      "description": "Execute a shell command and return its output.",
      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }
  ]
}
```

## sessions

### GET /api/sessions

列出所有已知的 session_id。

```json
{"sessions": ["cli-1", "web-abc", "default"]}
```

### GET /api/sessions/{id}/history

返回某会话的历史消息列表（role + content）。

```json
{
  "messages": [
    {"role": "user", "content": "列出 src 下的文件"},
    {"role": "assistant", "content": "src 下有 ..."}
  ]
}
```

### DELETE /api/sessions/{id}

清空某会话的历史。

```json
{"status": "cleared"}
```

### POST /api/sessions/{id}/rename

重命名会话。请求体 `{"new_session_id": "..."}`。目标已存在时返回 409。

```json
{"status": "renamed", "old_session_id": "cli-1", "new_session_id": "web-1"}
```

### GET /api/sessions/search?q=

按 `q` 在 session id 与消息内容中搜索。

```json
{"query": "年假", "results": [{"session_id": "cli-1", "matches": [...]}]}
```

### GET /api/sessions/{id}/export?fmt=

导出会话。`fmt` 支持 `json` / `md` / `markdown`，返回对应文件下载。

## rag

### POST /api/upload

上传文档并索引到指定知识库。`multipart/form-data`。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `file` | 文件 | — | 必填。扩展名必须在 `SUPPORTED_EXTENSIONS` 内。 |
| `kb_name` | `str` | `"default"` | 目标知识库名。 |

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@./docs/handbook.md" \
  -F "kb_name=company"
```

成功响应：

```json
{"status": "indexed", "filename": "handbook.md", "kb_name": "company", "chunks": 42}
```

!!! warning "上传限制"
    - 单文件最大 **50 MB**（`MAX_UPLOAD_BYTES = 50 * 1024 * 1024`），超出返回 `413 file_too_large`。
    - 不支持的扩展名返回 `400 unsupported_file_type`。
    - 上传以流式写入临时文件，超限时立即中断写入。

### GET /api/knowledge-bases

列出所有知识库名。

```json
{"knowledge_bases": ["company", "policies"]}
```

### GET /api/knowledge-bases/{kb}/documents

列出 KB 内的文档（按 source 分组，含 chunk 数）。

### DELETE /api/knowledge-bases/{kb}/documents?source=

删除某 source 的所有 chunk。返回 `{"kb_name": "...", "source": "...", "removed": N}`。

## settings

### GET /api/settings

返回当前运行时配置，凭据字段已脱敏（见 [to_safe_dict 脱敏机制](configuration.md#to_safe_dict-脱敏机制)）。

```json
{
  "model_provider": "openai",
  "api_key": "<redacted>",
  "model_name": "gpt-4o-mini",
  "max_steps": 10,
  "enable_tool_sandbox": false,
  "observability_provider": "local",
  ...
}
```

### POST /api/settings

运行时更新配置并重建 Agent。请求体是 `SettingsUpdateRequest`，所有字段都是可选的——只传需要改的字段。

```bash
curl -X POST http://127.0.0.1:8000/api/settings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model_name": "gpt-4o", "max_steps": 15, "rag_top_k": 8}'
```

```json
{"status": "updated"}
```

可更新字段与重建流程详见 [配置参考 § 运行时更新](configuration.md#运行时更新)。重建失败时返回 `500 rebuild_failed` 并回滚到旧配置。

## observability

### GET /api/traces?limit=

列出最近的本地 trace（`LocalJsonlTracer` 模式下可用）。`limit` 范围 1~1000，默认 100。

```json
{"traces": [{"id": "...", "name": "agent.run", "start_time": "...", "status": "ok", ...}]}
```

### GET /api/traces/{id}

取单个 trace 详情，包含完整的 span 树。未找到返回 `404 trace_not_found`。

## 认证

通过 `OPEN_AGENT_API_AUTH_TOKEN` 配置 Bearer token 认证。

- **token 为空**（默认）：认证禁用，所有端点免鉴权（仅本地开发用）。
- **token 非空**：所有 `/api/*` 端点（除 `/api/health` 与 `/api/ready`）与 `/ws/chat` 都要求 `Authorization: Bearer <token>` 头。

```bash
# 设置 token
export OPEN_AGENT_API_AUTH_TOKEN=$(openssl rand -hex 32)

# 请求时带上
curl -H "Authorization: Bearer $OPEN_AGENT_API_AUTH_TOKEN" http://127.0.0.1:8000/api/tools
```

!!! danger "生产必须设置 token"
    空 token 时任何人都能调用 `/api/chat`（消耗你的 LLM 配额）、`/api/upload`（向磁盘写文件）、`/api/settings`（修改配置）。生产部署**必须**设置强随机 token。

WebSocket 的鉴权在 `accept()` 之后立即检查 `Authorization` 头，失败时以 `4001` 关闭连接。

## 错误响应

所有未捕获异常由全局 `exception_handler` 转为统一 JSON 结构：

```json
{"error": "internal_server_error", "message": "An internal error occurred."}
```

业务错误（4xx）的格式：

```json
{"error": "<error_code>", "message": "<人类可读说明>"}
```

常见错误码：

| HTTP | `error` | 触发场景 |
|---|---|---|
| 400 | `unsupported_file_type` | 上传扩展名不在支持列表。 |
| 400 | `unsupported_format` | 会话导出格式不支持。 |
| 401 | — | 缺少或错误的 Bearer token。 |
| 404 | `kb_not_found` | 知识库不存在。 |
| 404 | `trace_not_found` | trace id 不存在。 |
| 409 | — | 会话重命名时目标已存在。 |
| 413 | `file_too_large` | 上传超过 50 MB。 |
| 500 | `indexing_failed` | 上传后索引失败。 |
| 500 | `rebuild_failed` | `/api/settings` 重建 Agent 失败。 |
| 500 | `internal_server_error` | 未捕获异常。 |

## 限流与上传大小

- **上传大小**：单文件 50 MB（`MAX_UPLOAD_BYTES`）。流式写入临时文件，超限即中断。
- **session_id 校验**：仅允许 `[a-zA-Z0-9_\-.]`，禁止 `..`，防止路径穿越。
- **kb_name 校验**：同 session_id 规则。
- **限流**：当前版本未内置 HTTP 限流，建议在反向代理（nginx、Cloudflare）层做。

## Python 客户端示例

用 `httpx` 同步调用 `/api/chat`：

```python
import httpx

BASE = "http://127.0.0.1:8000"
TOKEN = "your-token"

def chat(message: str, session_id: str = "default") -> dict:
    resp = httpx.post(
        f"{BASE}/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"message": message, "session_id": session_id},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()

result = chat("用一句话介绍 Open Agent")
print(result["response"])
print(f"steps={result['steps']}, tools={len(result['tool_calls_made'])}")
```

异步流式（WebSocket）客户端见上文 [WebSocket](#WebSocket-wschat) 章节。

## 下一步

- [配置参考](configuration.md) — `/api/settings` 可更新字段详解。
- [可观测性](observability.md) — `/api/traces` 与 trace 结构。
- [部署指南](deployment.md) — 反向代理与生产部署。
