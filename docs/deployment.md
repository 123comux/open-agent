# 部署指南

本页覆盖开发部署、Docker 部署与生产部署注意事项，包含认证、CORS、持久化卷、反向代理、资源限制与升级回滚。

## 开发部署

最简方式：本地装包 + 跑服务。

```bash
git clone https://github.com/123comux/open-agent.git
cd open-agent
pip install -e ".[all]"

# 配置最小 .env
cat > .env <<'EOF'
OPEN_AGENT_MODEL_PROVIDER=openai
OPEN_AGENT_API_KEY=sk-...
OPEN_AGENT_MODEL_NAME=gpt-4o-mini
EOF

# 启动服务（默认 127.0.0.1:8000）
open-agent serve
# 或带 reload 的开发模式
make serve
```

开发模式下可同时跑前端：

```bash
cd web && npm install && npm run dev   # http://localhost:5173
```

!!! tip "开发模式特点"
    - 认证关闭（`OPEN_AGENT_API_AUTH_TOKEN` 留空）。
    - CORS 允许 `*`，方便前端联调。
    - 可观测性用 `local`，trace 写到 `.open_agent_traces/`。
    - 沙箱关闭，方便工具读写本地文件。

## Docker 部署

仓库自带 `docker-compose.yml`（开发，含前端 Vite dev server）与 `docker-compose.prod.yml`（生产，前端用 nginx 静态服务）。

### 开发 compose

```bash
# 准备 .env（至少 OPEN_AGENT_API_KEY）
cp .env.example .env
# 编辑 .env 填入 API Key

docker compose up --build
```

启动后：

- API 服务：`http://localhost:8000`
- Web UI（Vite dev server）：`http://localhost:5173`

`docker-compose.yml` 关键配置：

- `api` 服务构建 `Dockerfile` 的 `runtime` target，监听 `0.0.0.0:8000`。
- 健康检查：每 30 秒 `GET /api/health`，启动宽限 60 秒。
- `web` 服务依赖 `api` 健康，否则不启动。
- 持久化卷：`.open_agent_kb_indexes`、`.open_agent_sessions`、`.open_agent_long_term`、`.open_agent_traces`、`./data`。

### 生产 compose

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

差异：

- `api` 服务用 `expose` 而非 `ports`，仅对内网开放。
- `web` 服务用 `Dockerfile.prod` 构建优化镜像，nginx 监听 `80` 端口对外。
- 适合放在反向代理（如云负载均衡器或外层 nginx）后面。

## 生产部署注意事项

### 1. 必须设置认证 token

```bash
OPEN_AGENT_API_AUTH_TOKEN=$(openssl rand -hex 32)
```

空 token 时 `/api/chat`、`/api/upload`、`/api/settings` 全部免鉴权，等同于把你的 LLM 配额与磁盘开放给公网。WebSocket 同样受保护。

### 2. 建议开启工具沙箱

```bash
OPEN_AGENT_ENABLE_TOOL_SANDBOX=true
OPEN_AGENT_SANDBOX_ALLOWED_PATHS=/app/data,/app/docs
OPEN_AGENT_SANDBOX_BLOCKED_PATHS=/etc,/root,/var/log
```

!!! warning "沙箱非硬隔离"
    Python 沙箱基于 `exec` + 正则黑名单，**不能**防止有经验的攻击者逃逸。若 Agent 可能执行不可信模型生成的代码，请额外用容器、gVisor 或独立子进程隔离。详见 [工具系统 § 沙箱机制](tools.md#沙箱机制)。

### 3. CORS 严格列举

```bash
OPEN_AGENT_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

不要用 `*`：当 `allow_origins=["*"]` 时，FastAPI 会自动把 `allow_credentials` 设为 `False`，带 cookie 的跨域请求会失败。严格列举域名才能保留 credentials。

### 4. 持久化卷

下列目录保存运行时状态，必须映射到宿主机或持久卷，否则容器重建后数据丢失：

| 容器路径 | 内容 | 对应环境变量 |
|---|---|---|
| `/app/.open_agent_kb_indexes` | FAISS 知识库索引 | `KBManager` 默认 |
| `/app/.open_agent_sessions` | 会话历史 JSON | `OPEN_AGENT_SESSION_STORAGE_DIR` |
| `/app/.open_agent_long_term` | 长期向量记忆 | `OPEN_AGENT_LONG_TERM_MEMORY_DIR` |
| `/app/.open_agent_traces` | 本地 trace JSONL | `OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR` |
| `/app/data` | 上传文档等业务数据 | 自定义 |

`docker-compose.yml` 已把这五个目录映射到宿主机的同名隐藏目录。生产建议改用命名卷（named volume）或挂载到分布式文件系统。

### 5. 反向代理（nginx 示例）

生产建议在最外层放 nginx，处理 TLS、限流、静态资源：

```nginx
upstream open_agent_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name agent.example.com;

    ssl_certificate     /etc/ssl/agent.example.com.pem;
    ssl_certificate_key /etc/ssl/agent.example.com.key;

    # 上传大小限制（与 MAX_UPLOAD_BYTES 对齐）
    client_max_body_size 60m;

    # 限流：每个 IP 每秒 10 个请求
    limit_req_zone $binary_remote_addr zone=agent:10m rate=10r/s;

    # API 与 WebSocket
    location /api/ {
        limit_req zone=agent burst=20 nodelay;
        proxy_pass http://open_agent_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /ws/ {
        proxy_pass http://open_agent_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;   # WebSocket 长连接
    }

    # 前端静态资源（生产 compose 的 web 服务）
    location / {
        proxy_pass http://127.0.0.1:80;
        proxy_set_header Host $host;
    }
}
```

关键点：

- `client_max_body_size` 至少 50 MB（`MAX_UPLOAD_BYTES`），留余量建议 60 MB。
- WebSocket 路径必须设置 `Upgrade` / `Connection` 头，并加长 `proxy_read_timeout`。
- `/api/` 限流防止滥用；`/ws/` 不限流但靠 token 鉴权。
- `X-Forwarded-Proto` 让 FastAPI 知道原始协议（影响 OpenAPI 文档里的 URL 生成）。

### 6. 资源限制

在 `docker-compose.yml` 的 `api` 服务里加 `deploy.resources`（Swarm）或 `mem_limit` / `cpus`（Compose v2）：

```yaml
services:
  api:
    # ...
    deploy:
      resources:
        limits:
          memory: 2g
          cpus: "2.0"
        reservations:
          memory: 512m
          cpus: "0.5"
```

资源建议：

- **内存**：嵌入模型（`bge-small-zh-v1.5`）常驻约 500 MB；`bge-large` 约 1.5 GB。生产建议 ≥ 2 GB。
- **CPU**：FAISS 检索与 BM25 是 CPU 密集，建议 ≥ 2 核。
- **GPU**：当前版本不依赖 GPU；若用本地嵌入服务（如 TEI）可单独部署。

## 环境变量检查清单

部署前逐项确认：

| 项目 | 检查 |
|---|---|
| 认证 | `OPEN_AGENT_API_AUTH_TOKEN` 已设置且为强随机值。 |
| CORS | `OPEN_AGENT_CORS_ORIGINS` 严格列举域名，未用 `*`。 |
| 沙箱 | `OPEN_AGENT_ENABLE_TOOL_SANDBOX=true`，白名单/黑名单已配。 |
| 模型 | `OPEN_AGENT_MODEL_PROVIDER`、`OPEN_AGENT_API_KEY`、`OPEN_AGENT_MODEL_NAME` 正确。 |
| 监听 | `OPEN_AGENT_SERVER_HOST=0.0.0.0`（容器内），`OPEN_AGENT_SERVER_PORT=8000`。 |
| 持久化 | 五个数据目录已映射到宿主或持久卷。 |
| 可观测 | `OPEN_AGENT_OBSERVABILITY_PROVIDER` 已选（生产建议 `langsmith`/`langfuse`），对应密钥已配。 |
| 长期记忆 | 若启用 `OPEN_AGENT_ENABLE_LONG_TERM_MEMORY=true`，目录已映射。 |
| MCP | 若用 MCP，`OPEN_AGENT_MCP_SERVERS_FILE` 指向的 JSON 已挂载。 |
| 超时 | `OPEN_AGENT_REQUEST_TIMEOUT` 与 `OPEN_AGENT_MAX_STEPS` 符合业务预期。 |
| 上游限制 | nginx `client_max_body_size` ≥ 50 MB；`proxy_read_timeout` ≥ 120s。 |
| 健康检查 | 编排器配置 `/api/ready` 作为 readiness probe、`/api/health` 作为 liveness probe。 |

## 升级与回滚

### 升级

1. **备份数据**：先停服务，备份五个持久化目录（至少 sessions 与 kb_indexes）。
2. **拉取新版本**：`git pull` 或下载指定 tag。
3. **重建镜像**：`docker compose build`（依赖变了才需要）。
4. **滚动重启**：`docker compose up -d`。容器会按 `healthcheck` 优雅替换。
5. **验证**：`curl /api/ready` 返回 `{"status":"ready"}`；跑一次 smoke test 对话。

```bash
# 备份
tar czf backup-$(date +%F).tar.gz .open_agent_sessions .open_agent_kb_indexes .open_agent_long_term .open_agent_traces data

# 升级
git pull
docker compose build
docker compose up -d
```

### 回滚

1. **停服务**：`docker compose down`。
2. **切回旧版本**：`git checkout <previous-tag>`。
3. **恢复数据**（仅当数据格式不兼容时）：`tar xzf backup-YYYY-MM-DD.tar.gz`。
4. **重启**：`docker compose up -d`。

!!! tip "兼容性"
    - 会话历史 JSON 格式向后兼容；新版本通常能读旧版本数据。
    - FAISS 索引格式由 faiss-cpu 版本决定，跨大版本可能不兼容——升级 `faiss-cpu` 前先备份索引。
    - 嵌入模型变更（`OPEN_AGENT_EMBEDDING_MODEL`）会导致旧索引失效，需重新索引。

### 蓝绿部署

生产建议蓝绿：

- 蓝环境（当前版本）继续服务，绿环境（新版本）独立部署。
- 绿环境 `/api/ready` 通过后，把负载均衡流量切到绿环境。
- 观察一段时间无异常后，停掉蓝环境；有问题立即切回。

注意蓝绿共享持久化卷时需要只读挂载蓝、读写挂载绿，避免并发写损坏数据；或各自独立卷 + 数据同步。

## 下一步

- [配置参考](configuration.md) — 完整环境变量表。
- [可观测性](observability.md) — 生产 trace 后端选择。
- [API 参考](api.md) — 健康检查与就绪检查端点。
