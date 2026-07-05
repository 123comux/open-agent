# 工具系统

工具是 Agent 感知与操作世界的接口。本页涵盖 Tool 基类、ToolRegistry、内置工具、沙箱机制、MCP 集成与自定义工具开发。

## Tool 基类与 execute 接口

`src/open_agent/tools/base.py` 定义了所有工具的抽象基类：

```python
class Tool(ABC):
    name: str                       # 工具名（registry 中的 key，全局唯一）
    description: str                # 给模型看的描述（影响调用准确率）
    parameters: dict[str, Any]      # JSON Schema，描述入参

    @abstractmethod
    async def execute(self, **kwargs: object) -> str:
        """执行工具，返回文本结果。"""
        ...

    def to_schema(self) -> dict[str, Any]:
        """返回 {name, description, parameters} 给模型的 function-calling 接口。"""
```

关键约定：

- `execute` 是 **异步** 方法，返回 **字符串**。字符串会被原样回灌给模型作为 `Observation`。
- `parameters` 是标准 JSON Schema，会原样透传给模型 provider 的 tool-calling 接口。
- `name` 在同一个 `ToolRegistry` 中必须唯一；同名注册会覆盖旧工具。

`ToolResult` 是 registry 层的返回结构（`success` / `output` / `error`），用于区分「工具执行失败」与「工具执行成功但返回了错误信息」。

## ToolRegistry 注册与查询

`src/open_agent/tools/registry.py` 的 `ToolRegistry` 是工具的中央目录：

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None: ...        # 注册（同名覆盖）
    def get(self, name: str) -> Tool: ...               # 按名查找；未找到抛 KeyError
    def list_tools(self) -> list[str]: ...              # 列出所有工具名
    def schemas(self) -> list[dict[str, Any]]: ...      # 输出 JSON Schema 列表
    async def execute(self, name: str, **kwargs) -> ToolResult: ...
```

!!! note "异常隔离"
    `ToolRegistry.execute` 会捕获两类异常：查找失败（`KeyError`）与执行异常（任意 `Exception`），都转换为 `ToolResult(success=False, error=...)`。这保证单个工具崩溃不会让 Agent 循环退出——模型会看到错误信息并自行决定重试、换工具或直接回答。

服务器启动时（`_build_agent`）会按 `OPEN_AGENT_ENABLED_TOOLS` 过滤注册内置工具；空表示全部注册。CLI 则默认注册除 `BrowserTool` 外的全部内置工具（browser 仅在 server 路径注册，因其依赖较重）。

## 内置工具详解

Open Agent 自带六个内置工具，全部位于 `src/open_agent/tools/builtin/`。

### ShellTool

执行 shell 命令，返回 stdout（或失败时的 stdout + stderr）。

| 字段 | 值 |
|---|---|
| `name` | `shell` |
| 必需参数 | `command: str` |
| 可选参数 | `timeout: float`（默认 30 秒） |

```python
# 模型可能的调用
{"command": "git log --oneline -5", "timeout": 10}
```

实现细节：

- 通过 `asyncio.create_subprocess_exec` 启动子进程，**不经过 shell**，用 `shlex.split` 做 POSIX 风格分词。
- 超时后 `proc.kill()` 并 `await proc.wait()`，返回 `Error: command timed out after {timeout}s.`。
- 命令不存在时返回 `Error: command not found: {tokens[0]}`。

!!! warning "安全"
    沙箱开启时，`check_shell_safety` 会用正则匹配危险命令（`rm -rf`、`mkfs`、`shutdown`、`curl ... | sh` 等）并抛 `PermissionError`，被工具捕获后返回阻塞信息给模型。完整黑名单见 `src/open_agent/tools/sandbox.py` 的 `DEFAULT_SHELL_BLOCKED_PATTERNS`。

### PythonTool

执行 Python 源码，捕获 stdout 返回。

| 字段 | 值 |
|---|---|
| `name` | `python` |
| 必需参数 | `code: str` |
| 可选参数 | `timeout: float`（默认 10 秒） |

```python
# 模型可能的调用
{"code": "import statistics\nprint(statistics.mean([1,2,3,4,5]))"}
```

实现细节：

- 在 worker thread 中 `exec`，用 `contextlib.redirect_stdout` 捕获输出。
- 先尝试 `compile(code, ..., "eval")`（像 Jupyter 一样求值最后一行），失败则回退到 `exec` 整段。
- 沙箱开启时，`_safe_builtins()` 仅保留类型转换、数值、迭代、异常等安全 builtin，剔除 `__import__`、`open`、`eval`、`exec`、`compile`、`globals`、`locals`、`getattr` 等；同时 `check_python` 用正则拦截 `__import__(`、`subprocess`、`os.system` 等。

!!! danger "非硬隔离"
    Python 沙箱基于 `exec` + 受限命名空间 + 正则黑名单，**不是**进程级隔离。有经验的攻击者可能通过反射链逃逸。**生产环境若运行不可信模型，请用容器或沙箱进程隔离，而非依赖此沙箱。** 详见 [沙箱机制](#沙箱机制)。

### FileTool

读写、列出本地文件。

| 字段 | 值 |
|---|---|
| `name` | `file` |
| 必需参数 | `action: "read" \| "write" \| "list"`, `path: str` |
| 条件必需 | `content: str`（仅 write） |

```python
{"action": "read", "path": "./README.md"}
{"action": "write", "path": "./out.txt", "content": "hello"}
{"action": "list", "path": "./src"}
```

实现细节：

- `read` 用 UTF-8 读取整文件；`write` 用「写临时文件 + `os.replace`」做原子写入，避免崩溃留下半写文件。
- `list` 返回排序后的目录条目，每行一个。
- 沙箱开启时，`check_path` 拦截路径穿越（`..`）、敏感目录（`/etc/`、`/root/`、`C:\Windows\System32`）、白名单外的路径。

### WebSearchTool

通过 Bing（主）+ DuckDuckGo（兜底）搜索网页，返回格式化的标题/URL/摘要。

| 字段 | 值 |
|---|---|
| `name` | `web_search` |
| 必需参数 | `query: str` |
| 可选参数 | `max_results: int`（默认 5） |

```python
{"query": "Open Agent RAG 架构", "max_results": 5}
```

实现细节：

- 主用 Bing（中国大陆可达）；解析失败或无结果时依次回退到 `lite.duckduckgo.com` 与 `html.duckduckgo.com`。
- 共享模块级 `httpx.AsyncClient` 做连接池（`max_connections=20`），在服务关闭时由 `aclose()` 释放。
- 内置 1 秒最小请求间隔（`RATE_LIMIT_SECONDS`），用 `asyncio.Lock` 串行化，避免触发搜索引擎限流。
- 解析逻辑用正则按结果块提取，每条结果单独配对标题与摘要，避免错位。

!!! tip "查询语言建议"
    系统提示要求 `web_search` 的查询与用户问题**同语言**，不要混用中英文（如 `today news 2026`）。中文新闻用 `今日新闻 最新消息` 这类自然表达。

### BrowserTool

轻量抓取（httpx）+ 真实无头浏览器交互（Playwright，可选）。

| 字段 | 值 |
|---|---|
| `name` | `browser` |
| 必需参数 | `action: str`, `url: str` |
| 可选参数 | `selector`, `text`, `script`, `output_path`, `full_page`, `selector_type` |

`action` 取值：

| 动作 | 依赖 | 说明 |
|---|---|---|
| `fetch` | httpx | 返回原始 HTML（截断到 5000 字符）。 |
| `extract_links` | httpx | 返回所有 `<a>` 链接的 JSON。 |
| `extract_text` | httpx | 剥离 HTML 标签返回可见文本。 |
| `navigate` | Playwright | 加载 URL，返回页面标题。 |
| `screenshot` | Playwright | 截图，返回 base64 data URL 或保存到 `output_path`。 |
| `click` / `fill` | Playwright | 点击 / 填充元素。 |
| `get_text` / `get_links` | Playwright | 返回渲染后的文本 / 链接。 |
| `evaluate` | Playwright | 执行页面内 JavaScript。 |

```python
{"action": "screenshot", "url": "https://example.com", "full_page": true}
```

!!! warning "SSRF 防护"
    所有动作执行前都会调用 `_validate_url`：仅允许 `http`/`https`；解析主机所有 IP，拦截私有、loopback、link-local（含云元数据 `169.254.169.254`）、multicast、unspecified 地址。IP 字面量直接检查，域名做 DNS 解析后逐 IP 检查。

### KnowledgeBaseTool

通过 RAG 检索已索引文档，返回相关文本段落。

| 字段 | 值 |
|---|---|
| `name` | `knowledge_base` |
| 必需参数 | `query: str` |
| 可选参数 | `top_k: int`（默认取 settings 里的 `rag_top_k`） |

```python
{"query": "公司年假政策", "top_k": 5}
```

实现细节：

- 包装 `KBManager.query`，返回的每个 chunk 包含 `kb_name`、`score`、`document` 三段。
- 无知识库或无结果时返回友好提示，引导先 `open-agent index`。
- 检索流程（路由 → 混合检索 → 重排序）详见 [RAG 知识库](rag.md)。

## 沙箱机制

沙箱由 `src/open_agent/tools/sandbox.py` 提供，受 `OPEN_AGENT_ENABLE_TOOL_SANDBOX` 开关控制。三类检查：

### check_shell / check_shell_safety

`DEFAULT_SHELL_BLOCKED_PATTERNS` 是一组正则，匹配即抛 `PermissionError`。覆盖：

- 破坏性删除：`rm -rf`、`rmdir /s`、`del /f`、`format C:`、`mkfs`、`dd if=`
- 系统控制：`shutdown`、`reboot`、`halt`、`poweroff`
- fork 炸弹：` :(){ :|:& };: `
- 提权与持久化：`chmod 777`、`crontab -r`、`reg add ... /f`、`takeown /f`、`icacls ... /grant`
- 管道执行：`curl ... | sh`、`wget ... | bash`
- 包卸载：`pip uninstall`、`npm uninstall`

### check_python

`DEFAULT_PYTHON_BLOCKED_PATTERNS` 拦截：

- 导入系统：`__import__(`、`importlib`、`import os`、`subprocess`
- 危险调用：`os.system`、`os.popen`、`shutil.rmtree`、`eval(`、`exec(`、`compile(`
- 反射逃逸：`getattr(__builtins__`、`__builtins__`、`globals()`、`locals()`、`vars()`
- 文件访问：`open('...`、`breakpoint(`

同时 `_safe_builtins()` 在开启沙箱时替换 `__builtins__`，剔除 `__import__`、`open`、`eval`、`exec`、`compile`、`getattr`、`setattr`、`delattr`、`super` 等。

### check_path

按以下顺序拦截：

1. **黑名单** `OPEN_AGENT_SANDBOX_BLOCKED_PATHS` — 命中即拒。
2. **白名单** `OPEN_AGENT_SANDBOX_ALLOWED_PATHS` — 非空时，路径必须在白名单内。
3. **路径穿越** — `..` 出现在路径分段中即拒。
4. **敏感目录** — `/etc/`、`/root/`、`/var/log/`、`C:\Windows\System32` 一律拒。

### 已知限制

!!! danger "沙箱不是安全边界"
    - **Python 沙箱非硬隔离**：基于 `exec` + 正则 + 受限 builtins，无法防止有经验的攻击者通过反射链（如 `().__class__.__mro__` 系列）逃逸。不可信代码请用容器或独立子进程隔离。
    - **Shell 沙箱是黑名单**：只能拦截已知危险模式，新型攻击命令可能漏网。
    - **check_path 是路径级**：不限制命令的网络副作用（如 `curl` 上传文件）。
    - **沙箱状态在启动时确定**：`sandbox_enabled()` 读 `get_settings()` 缓存，运行时通过 `/api/settings` 改 `enable_tool_sandbox` 后需要重建 Agent 才生效。

## MCP 集成

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 让你接入任意外部工具服务器。Open Agent 通过 `src/open_agent/mcp/` 把 MCP 工具包装成普通 `Tool`。

### 传输方式

当前实现支持 **stdio** 传输：每个 MCP 服务器作为子进程启动，通过 stdin/stdout 通信。SSE 传输在路线图上。

### 配置文件格式

通过 `OPEN_AGENT_MCP_SERVERS_FILE` 指定一个 JSON 文件，schema 见 `mcp_servers.example.json`：

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {"optional": "value"},
      "cwd": "/optional/workdir"
    },
    {
      "name": "fetch",
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  ]
}
```

| 字段 | 必需 | 说明 |
|---|---|---|
| `name` | 是 | 逻辑名，用于工具命名空间。 |
| `command` | 是 | 可执行文件（`npx`、`uvx`、`python` 等）。 |
| `args` | 否 | 命令行参数列表。 |
| `env` | 否 | 子进程环境变量。 |
| `cwd` | 否 | 子进程工作目录。 |

### 命名空间前缀

MCP 工具在 registry 中注册时，名字会被改成 `server_name/tool_name`，例如 `filesystem/read_file`、`fetch/fetch`。这避免与内置工具（`shell`、`python` 等）或其他 MCP 服务器的同名工具冲突。模型在 function-calling 时使用带前缀的全名。

### 加载流程

`Agent` 构建时（CLI 的 `_build_agent` 或 server 的 `_build_agent`）：

1. 若 `settings.mcp_servers_file` 非空，调用 `load_mcp_servers(path)` 解析 JSON。
2. 用 `MCPClient(servers)` 创建客户端，`await client.connect()` 启动所有子进程并 `list_tools`。
3. `adapt_mcp_tools(client)` 把每个 MCP 工具包装成 `MCPToolAdapter`，注册进 `ToolRegistry`。
4. 客户端句柄存到 `agent._mcp_client`，在 Agent 退出时 `close()` 释放子进程。

!!! warning "MCP 加载失败不致命"
    MCP 子进程启动失败只记录 warning（CLI）或 logger.warning（server），不会让 Agent 启动失败。其余内置工具仍可用。

## 自定义工具开发

### 完整示例

下面是一个完整的自定义工具：调用外部天气 API 并格式化结果。

```python
import asyncio
import json
from typing import Any

import httpx

from open_agent.agent.core import Agent
from open_agent.models.openai_provider import OpenAIModel
from open_agent.tools.base import Tool
from open_agent.tools.registry import ToolRegistry


class WeatherTool(Tool):
    name = "weather"
    description = (
        "查询指定城市的当前天气。返回温度、天气状况与湿度。"
        "适用于用户询问天气、温度、是否需要带伞等问题。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名，可以是中文或英文，如 '北京' 或 'Tokyo'。",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "温度单位：metric=摄氏，imperial=华氏。",
                "default": "metric",
            },
        },
        "required": ["city"],
    }

    async def execute(self, **kwargs: object) -> str:
        city = str(kwargs.get("city", "")).strip()
        if not city:
            return "Error: 'city' is required."
        units = str(kwargs.get("units", "metric"))
        # 替换为你的真实 API key 与端点
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={
                        "q": city,
                        "appid": "YOUR_API_KEY",
                        "units": units,
                        "lang": "zh_cn",
                    },
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            except httpx.HTTPError as exc:
                return f"Error fetching weather: {exc}"

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return f"{city}：{desc}，温度 {temp}°，湿度 {humidity}%"


# 注册并使用
registry = ToolRegistry()
registry.register(WeatherTool())
# 也可以同时注册内置工具
from open_agent.tools.builtin import ShellTool, PythonTool
registry.register(ShellTool())
registry.register(PythonTool())

agent = Agent(
    model=OpenAIModel(api_key="sk-...", model="gpt-4o-mini"),
    tool_registry=registry,
    max_steps=10,
)

output = asyncio.run(agent.run("北京今天天气怎么样？需要带伞吗？"))
print(output.response)
```

### 注册方式

- **库内使用** — 直接 `registry.register(MyTool())`，传给 `Agent(tool_registry=...)`。
- **CLI / Server** — 在 `_build_agent`（`cli.py` 与 `server/api.py`）的 `builtin_tools` 列表里追加你的工具实例。两者都会按 `OPEN_AGENT_ENABLED_TOOLS` 过滤，所以若你启用了过滤，记得把工具名加进白名单。

### 最佳实践

- **`description` 要写清使用场景**：模型靠它决定何时调用。与其写「查询天气」，不如写「查询指定城市的当前天气，适用于用户询问天气、温度、是否需要带伞等问题」。
- **`parameters` 的每个字段都写 `description`**：模型用它构造参数。模糊的字段名会导致参数错乱。
- **`execute` 返回结构化文本**：模型更易解析「城市：北京，温度：25°，湿度：60%」这样的格式。
- **捕获异常并返回错误字符串**：不要让 `execute` 抛异常（虽然 registry 会兜底，但返回明确的错误信息能让模型更好地决定下一步）。
- **保持工具幂等**：模型可能重试同一个工具调用，副作用型工具（写入、删除）应做幂等设计或返回明确的「已完成」标识。

## 下一步

- [RAG 知识库](rag.md) — `KnowledgeBaseTool` 背后的检索链路。
- [API 参考](api.md) — 通过 `/api/tools` 端点查看运行时工具列表。
- [配置参考](configuration.md#工具) — 工具与沙箱相关环境变量。
