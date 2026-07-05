# 配置参考

Open Agent 的所有运行时配置都通过 `OPEN_AGENT_` 前缀的环境变量加载，由 `open_agent.config.Settings` 类统一管理。本页给出完整变量表、加载机制、优先级与运行时更新规则。

## 配置加载机制

### 三层来源

1. **环境变量** — `OPEN_AGENT_<NAME>`，最优先。
2. **`.env` 文件** — 通过 [direnv](https://direnv.net/) 或手动 `source` 加载到环境，本质上仍是环境变量。
3. **代码默认值** — `Settings` 类的字段默认值。

加载入口是 `Settings.load()`，它逐个读取 `os.environ` 并做类型转换（`_parse_bool` / `_parse_int` / `_parse_float` / `_parse_list`），对非法值回退到默认值而非抛异常，保证配置错误不会让进程启动失败。

### 单例缓存

```python
from open_agent.config import get_settings, set_settings, reload_settings

settings = get_settings()        # 首次调用时从环境加载，之后返回缓存单例
set_settings(new_settings)       # 替换缓存单例（服务器运行时更新用）
reload_settings()                # 清空缓存并重新从环境加载
```

`get_settings()` 是全局单例：CLI、Server、KBManager、Tracer 等所有模块共享同一份 `Settings`。这样运行时通过 `/api/settings` 更新配置后，新构建的组件会读到最新值。

!!! warning "已构造的组件不会自动更新"
    `set_settings` 只替换缓存。已经基于旧 settings 构造的 `Agent` / `KBManager` / `SessionManager` 不会自动重建——服务器层在 `/api/settings` 端点里显式调用 `_build_agent()` 重建 Agent。详见 [运行时更新](#运行时更新)。

### 校验

`Settings` 用 pydantic `model_validator` 做跨字段校验。当前唯一的约束：

- `chunk_overlap` **必须小于** `chunk_size`，否则启动时报 `ValueError`。

`Literal` 类型字段（`model_provider`、`observability_provider`、`split_unit`）在 `load()` 中做枚举校验，非法值回退到默认值。

## 完整环境变量表

下表按类别分组。所有变量都以 `OPEN_AGENT_` 为前缀；「类型」列指 `Settings` 字段的 Python 类型；「默认」列为代码默认值；「示例」列给出典型的字符串值。

### 模型

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_MODEL_PROVIDER` | `Literal["openai","anthropic","ollama"]` | `openai` | 模型 provider。非法值回退到 `openai`。 | `anthropic` |
| `OPEN_AGENT_API_KEY` | `str` | `""` | Provider 的 API Key。Ollama 可留空。 | `sk-...` |
| `OPEN_AGENT_BASE_URL` | `str` | `https://api.openai.com/v1` | OpenAI 兼容端点。Anthropic/Ollama 各自的 provider 会忽略此值或按需使用。 | `https://open.bigmodel.cn/api/paas/v4` |
| `OPEN_AGENT_MODEL_NAME` | `str` | `gpt-4o-mini` | 模型标识符，透传给 provider。 | `claude-3-5-sonnet-20241022` |
| `OPEN_AGENT_MAX_STEPS` | `int` | `10` | 单轮 ReAct 最大迭代数（防失控）。 | `15` |
| `OPEN_AGENT_MAX_CONTEXT_TOKENS` | `int` | `8000` | 单次 LLM 请求的最大上下文 token 数；超出会按 `truncate_messages` 截断。 | `16000` |
| `OPEN_AGENT_REQUEST_TIMEOUT` | `float` | `60.0` | 单次模型请求超时（秒）。 | `120.0` |

### RAG / 向量化

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_EMBEDDING_MODEL` | `str` | `BAAI/bge-small-zh-v1.5` | sentence-transformers 嵌入模型名。 | `BAAI/bge-large-zh-v1.5` |
| `OPEN_AGENT_CHUNK_SIZE` | `int` | `500` | 文档切块大小（字符或段落数）。 | `800` |
| `OPEN_AGENT_CHUNK_OVERLAP` | `int` | `50` | 块重叠。**必须小于** `chunk_size`。 | `100` |
| `OPEN_AGENT_SPLIT_UNIT` | `Literal["char","paragraph"]` | `char` | 切块单位。 | `paragraph` |
| `OPEN_AGENT_RAG_TOP_K` | `int` | `5` | 每次查询返回的 chunk 数。 | `8` |
| `OPEN_AGENT_RERANKER_MODEL` | `str` | `BAAI/bge-reranker-v2-m3` | 交叉编码器重排序模型。设为空可禁用重排序。 | `BAAI/bge-reranker-base` |
| `OPEN_AGENT_RERANK_K` | `int` | `20` | 送入重排序的候选数。 | `50` |

### 工具

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_ENABLED_TOOLS` | `list[str]` | `[]` | 启用的工具名（逗号分隔）。空表示启用全部。 | `shell,python,knowledge_base` |
| `OPEN_AGENT_MCP_SERVERS_FILE` | `str` | `""` | MCP 服务器 JSON 配置文件路径。 | `mcp_servers.json` |

### 沙箱

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_ENABLE_TOOL_SANDBOX` | `bool` | `false` | 启用文件系统与命令沙箱。生产建议开启。 | `true` |
| `OPEN_AGENT_SANDBOX_ALLOWED_PATHS` | `list[str]` | `[]` | 工具可访问的白名单路径（逗号分隔）。 | `./data,./docs` |
| `OPEN_AGENT_SANDBOX_BLOCKED_PATHS` | `list[str]` | `[]` | 工具被拒绝的黑名单路径。 | `/etc,/root` |

!!! warning "Python 沙箱非硬隔离"
    `PythonTool` 的沙箱基于 `exec` + 受限 builtins + 正则黑名单，**不是**真正的进程级隔离。详见 [工具系统 § 沙箱机制](tools.md#沙箱机制)。

### 记忆

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_SHORT_TERM_MEMORY_SIZE` | `int` | `20` | 短期记忆保留的最近消息数。 | `40` |
| `OPEN_AGENT_SESSION_STORAGE_DIR` | `str` | `.open_agent_sessions` | 会话历史持久化目录。 | `/var/open-agent/sessions` |
| `OPEN_AGENT_ENABLE_LONG_TERM_MEMORY` | `bool` | `false` | 启用向量长期记忆。 | `true` |
| `OPEN_AGENT_LONG_TERM_MEMORY_DIR` | `str` | `.open_agent_long_term` | 长期记忆存储目录。 | `/var/open-agent/ltm` |
| `OPEN_AGENT_LONG_TERM_MEMORY_TOP_K` | `int` | `3` | 每次召回的长期记忆条数。 | `5` |

### API / Server

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_SERVER_HOST` | `str` | `127.0.0.1` | FastAPI 绑定地址。生产用 `0.0.0.0`。 | `0.0.0.0` |
| `OPEN_AGENT_SERVER_PORT` | `int` | `8000` | FastAPI 绑定端口。 | `8080` |
| `OPEN_AGENT_API_AUTH_TOKEN` | `str` | `""` | Bearer token；空表示禁用认证（仅本地开发）。 | `s3cret-token` |
| `OPEN_AGENT_CORS_ORIGINS` | `list[str]` | `[]` | 允许的 CORS 源（逗号分隔）。空表示 `["*"]`。 | `https://app.example.com` |

### 可观测性

| 变量 | 类型 | 默认 | 说明 | 示例 |
|---|---|---|---|---|
| `OPEN_AGENT_ENABLE_OBSERVABILITY` | `bool` | `true` | 启用 trace 收集。 | `true` |
| `OPEN_AGENT_OBSERVABILITY_PROVIDER` | `Literal["local","langsmith","langfuse"]` | `local` | trace 后端。 | `langsmith` |
| `OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR` | `str` | `.open_agent_traces` | local 模式的 trace 输出目录。 | `/var/open-agent/traces` |
| `OPEN_AGENT_LANGSMITH_API_KEY` | `str` | `""` | LangSmith 个人 API Key。 | `ls-...` |
| `OPEN_AGENT_LANGSMITH_API_URL` | `str` | `https://api.smith.langchain.com` | LangSmith API 端点。 | — |
| `OPEN_AGENT_LANGSMITH_PROJECT` | `str` | `open-agent` | LangSmith 项目名。 | `prod-agent` |
| `OPEN_AGENT_LANGFUSE_PUBLIC_KEY` | `str` | `""` | Langfuse 公钥。 | `pk-lf-...` |
| `OPEN_AGENT_LANGFUSE_SECRET_KEY` | `str` | `""` | Langfuse 密钥。 | `sk-lf-...` |
| `OPEN_AGENT_LANGFUSE_HOST` | `str` | `https://cloud.langfuse.com` | Langfuse 主机 URL。 | `https://lf.internal.com` |

### MCP

MCP 的连接配置不在环境变量里，而在 `OPEN_AGENT_MCP_SERVERS_FILE` 指向的 JSON 文件中。详见 [工具系统 § MCP 集成](tools.md#MCP-集成)。

## 配置优先级

```text
环境变量  >  .env 文件  >  代码默认值
```

`.env` 文件本身只是把值注入环境变量，所以前两者其实是同一种来源；区别仅在于是否需要手动 `source .env`。

!!! tip "Shell 已导出的变量优先"
    如果一个变量同时存在于 `.env` 和 shell 的 `export` 中，shell 中的值会覆盖 `.env`（因为 `source .env` 只在变量未设置时才赋值，取决于你的 shell 语义）。建议生产环境用 secrets manager 注入，避免 `.env` 提交进 git。

## 运行时更新

服务运行时可以通过 `POST /api/settings` 修改一部分配置，无需重启：

```bash
curl -X POST http://127.0.0.1:8000/api/settings \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt-4o", "max_steps": 15}'
```

可更新字段（来自 `SettingsUpdateRequest`）：

- 模型：`model_provider`、`api_key`、`base_url`、`model_name`、`max_steps`、`request_timeout`
- RAG：`embedding_model`、`chunk_size`、`chunk_overlap`、`split_unit`、`rag_top_k`、`reranker_model`、`rerank_k`
- 工具：`enabled_tools`
- 记忆：`enable_long_term_memory`、`long_term_memory_top_k`

更新流程：

1. 服务器在 `_rebuild_lock` 内合并新旧 settings，重新跑 `model_validator` 校验。
2. 调用 `set_settings(new_settings)` 替换缓存单例。
3. 若改动涉及 KB 字段，清空 `_kb_manager` 缓存（下次请求时重建）。
4. 若改动涉及 session 字段（`short_term_memory_size`、`session_storage_dir`），重建 `SessionManager`。
5. 关闭旧 Agent 持有的 MCP client 与 model httpx client，调用 `_build_agent()` 重建 Agent / Registry / Tracer。
6. 重建失败时回滚到旧 settings，并把 `_agent` 置为 `None`（`/api/ready` 会返回 503）。

!!! warning "运行时更新的边界"
    - `server_host` / `server_port` / `cors_origins` / `api_auth_token` / `observability_provider` 等启动期参数**不在**可更新集合内，修改它们需要重启进程。
    - 更新 `api_key` 后会重建 model provider，但**不会**撤销已建立的连接池，下一次请求时才使用新 key。
    - 并发的 `POST /api/settings` 会被 `_rebuild_lock` 串行化。

## to_safe_dict 脱敏机制

`GET /api/settings` 返回 `Settings.to_safe_dict()` 的结果。任何字段名以 `_key`、`_token`、`_secret` 结尾的值都会被脱敏：

- 已设置的非空值 → `"<redacted>"`
- 空值 → `""`（保留为空，便于前端判断「未配置」）

被脱敏的字段包括：`api_key`、`api_auth_token`、`langsmith_api_key`、`langfuse_secret_key`、`langfuse_public_key`。这保证 `GET /api/settings` 永远不会泄露凭据，可安全返回给前端展示。

!!! note "脱敏仅作用于读取"
    `to_safe_dict` 只影响 `GET` 的响应；`POST /api/settings` 接收的是明文新值，会正常写入 `Settings`。

## 最佳实践

### 开发环境

```bash
# .env (开发)
OPEN_AGENT_MODEL_PROVIDER=openai
OPEN_AGENT_API_KEY=sk-...
OPEN_AGENT_MODEL_NAME=gpt-4o-mini
OPEN_AGENT_ENABLE_OBSERVABILITY=true
OPEN_AGENT_OBSERVABILITY_PROVIDER=local
# 认证关闭，方便本地调试
# OPEN_AGENT_API_AUTH_TOKEN=
OPEN_AGENT_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 生产环境

```bash
# /etc/open-agent/env (生产)
OPEN_AGENT_MODEL_PROVIDER=openai
OPEN_AGENT_API_KEY=<from secrets manager>
OPEN_AGENT_MODEL_NAME=gpt-4o
OPEN_AGENT_SERVER_HOST=0.0.0.0
OPEN_AGENT_SERVER_PORT=8000
OPEN_AGENT_API_AUTH_TOKEN=<strong random token>
OPEN_AGENT_CORS_ORIGINS=https://app.example.com
OPEN_AGENT_ENABLE_TOOL_SANDBOX=true
OPEN_AGENT_SANDBOX_ALLOWED_PATHS=/app/data,/app/docs
OPEN_AGENT_ENABLE_OBSERVABILITY=true
OPEN_AGENT_OBSERVABILITY_PROVIDER=langsmith
OPEN_AGENT_LANGSMITH_API_KEY=<from secrets manager>
OPEN_AGENT_SESSION_STORAGE_DIR=/var/open-agent/sessions
OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR=/var/open-agent/traces
```

!!! tip "生产 vs 开发的关键差异"
    - **必须**设置 `OPEN_AGENT_API_AUTH_TOKEN`（开发可空）。
    - **建议**开启 `OPEN_AGENT_ENABLE_TOOL_SANDBOX=true`（开发可关）。
    - CORS **严格列举**域名，不要用 `*`（`allow_credentials` 在 `*` 下会被禁用）。
    - 可观测性建议切换到 `langsmith` 或 `langfuse`，避免 local JSONL 的同步写阻塞事件循环。

更多部署清单见 [部署指南](deployment.md)。

## 下一步

- [运行时更新](api.md#settings) — `/api/settings` 端点详解。
- [可观测性](observability.md) — trace 后端切换与性能影响。
- [部署指南](deployment.md) — 生产环境配置清单。
