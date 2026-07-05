# Docker Quick Start（5 分钟体验）

本目录提供一个独立、自包含的 Open Agent 体验环境，让你在 5 分钟内跑通
“构建镜像 → 启动服务 → 索引文档 → 提问”的完整流程。

> 这个 compose 文件**独立于项目根目录的 `docker-compose.yml`**（那个是生产部署用的）。
> 它只关注“快速体验”，所有持久化数据放在命名卷里，删掉即净。

---

## 前置条件

- **Docker**（建议 20.10+）
- **Docker Compose**（建议 v2，即 `docker compose` 子命令）
- 一个可用的 LLM 后端：
  - OpenAI API key，或
  - 本地运行的 [Ollama](https://ollama.com)（可选，见下方“使用本地 Ollama”）

> Windows 用户使用 PowerShell 即可；macOS/Linux 用户使用 bash。
> 验证安装：`docker --version` 和 `docker compose version`。

---

## 步骤

### 1. 准备配置

把 `.env.example` 复制为 `.env`，并填入你的 API key：

```bash
# Linux / macOS
cp .env.example .env
# 然后编辑 .env，把 OPEN_AGENT_API_KEY 改成你的真实 key
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
# 然后用记事本编辑 .env
```

`.env` 中**只有 `OPEN_AGENT_API_KEY` 是必填项**（使用 OpenAI 时），其余都有合理默认值。

### 2. 构建并启动

```bash
docker compose up -d --build
```

首次构建需要拉取基础镜像并安装 Python 依赖，大约 3–5 分钟；之后启动只需几秒。

### 3. 查看启动日志

```bash
docker compose logs -f
```

看到类似下面的日志就说明启动完成：

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

按 `Ctrl+C` 退出日志查看（不会停止服务）。

### 4. 运行初始化脚本

脚本会自动：等待服务就绪 → 上传 `sample-docs/` → 提问 “What is Acme Corp?” → 打印响应。

```bash
# Linux / macOS
bash init-and-query.sh
```

```powershell
# Windows PowerShell
.\init-and-query.ps1
```

> 脚本默认使用 `demo-token-please-change` 作为鉴权 token（与 `docker-compose.yml` 中一致）。
> 如果你改了 compose 里的 `OPEN_AGENT_API_AUTH_TOKEN`，请通过环境变量或参数同步覆盖：
> `bash init-and-query.sh` 前 `export OPEN_AGENT_API_AUTH_TOKEN=你的token`，
> 或 `.\init-and-query.ps1 -AuthToken 你的token`。

### 5. 查看 API 文档

浏览器打开：<http://localhost:8000/docs>

这是 FastAPI 自动生成的交互式 OpenAPI 文档，可以直接在页面上试调每个接口
（点击右上角 **Authorize**，输入 `demo-token-please-change` 即可）。

### 6. 使用 Web UI（如果启用了）

本 quick-start compose 默认**只启动 API 服务**，不包含前端。
如果你需要 Web UI，请使用项目根目录的完整 `docker-compose.yml`：

```bash
# 回到项目根目录
cd ../..
docker compose up -d --build
# 前端: http://localhost:5173
```

---

## 清理

停止并删除容器与持久化卷（**会清空知识库索引、会话、trace**）：

```bash
docker compose down -v
```

只停止容器、保留数据：

```bash
docker compose down
```

---

## 故障排查

| 现象 | 排查 |
| --- | --- |
| `docker compose up` 报端口占用 | 修改 `docker-compose.yml` 中 `ports: - "8000:8000"` 的左侧端口（例如 `"18000:8000"`），并同步修改脚本里的 `BASE_URL`/`-BaseUrl`。 |
| 服务一直不 ready / `503` | `docker compose logs` 查看错误。常见原因：API key 无效、模型名错误、或镜像构建失败。 |
| `401 Unauthorized` | `.env` 里的 token 与脚本不一致。脚本默认 token 是 `demo-token-please-change`，与 `docker-compose.yml` 一致；如改动请两边同步。 |
| 上传文档 `400 unsupported_file_type` | `sample-docs/` 只放了 `.txt` / `.md`，是支持的。如果你自己加了别的文件，参考支持列表：`.txt .md .rst .pdf .docx .csv .json .html`。 |
| 嵌入模型下载慢 / 卡住 | 首次启动会下载 `BAAI/bge-small-zh-v1.5` 等模型，耐心等待，或配置代理后重试。 |
| 脚本报 `curl: command not found` | `init-and-query.sh` 依赖 curl；Windows 请改用 `init-and-query.ps1`。 |

查看实时日志：

```bash
docker compose logs -f open-agent-demo
```

进入容器排查：

```bash
docker compose exec open-agent-demo bash
```

---

## 使用本地 Ollama（可选，免 API 费用）

如果你本地已经跑着 [Ollama](https://ollama.com)，可以完全脱离云端 API 体验。

1. 先用 Ollama 拉一个模型：`ollama pull llama3.2`
2. 编辑 `.env`，**注释掉 OpenAI 的 `OPEN_AGENT_API_KEY`**，改成：

   ```dotenv
   OPEN_AGENT_MODEL_PROVIDER=ollama
   OPEN_AGENT_BASE_URL=http://host.docker.internal:11434/v1
   OPEN_AGENT_API_KEY=ollama
   OPEN_AGENT_MODEL_NAME=llama3.2
   ```

   > `host.docker.internal` 在 Docker Desktop（Mac/Windows）上会自动指向宿主机。
   > Linux 用户需要加 `--add-host=host.docker.internal:host-gateway`，或在
   > `docker-compose.yml` 的 `extra_hosts` 中添加。

3. `docker compose up -d --build` 重建即可。

---

## 下一步

- 阅读 [主 README](../../README.md) 了解项目全貌与架构。
- 跟着 [docs/quickstart.md](../../docs/quickstart.md) 走一遍 pip 安装 + 本地开发流程。
- 查看 [API 文档](http://localhost:8000/docs) 试调更多接口（会话、工具、知识库管理等）。
- 阅读 [examples/](../) 目录下的示例代码，了解如何用 Python SDK 编程调用。
- 想做生产部署？回到项目根目录的 `docker-compose.yml` / `docker-compose.prod.yml`。
