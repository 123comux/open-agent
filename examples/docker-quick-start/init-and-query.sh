#!/usr/bin/env bash
# Open Agent — Docker Quick Start 演示脚本 (Linux/macOS)
#
# 流程：
#   1. 等待服务 ready（轮询 /api/ready 直到 200）
#   2. 上传 sample-docs 到知识库（POST /api/upload）
#   3. 发送一个 chat 请求查询 "What is Acme Corp?"（POST /api/chat）
#   4. 打印响应
#
# 用法：
#   bash init-and-query.sh
#
# 可通过环境变量覆盖默认值：
#   OPEN_AGENT_BASE_URL=http://localhost:8000
#   OPEN_AGENT_API_AUTH_TOKEN=demo-token-please-change
#   SAMPLE_DIR=./sample-docs
set -euo pipefail

# ---- 配置 ----
BASE_URL="${OPEN_AGENT_BASE_URL:-http://localhost:8000}"
AUTH_TOKEN="${OPEN_AGENT_API_AUTH_TOKEN:-demo-token-please-change}"
SAMPLE_DIR="${SAMPLE_DIR:-$(dirname "$(readlink -f "$0")")/sample-docs}"
SESSION_ID="demo-$(date +%s)"
READY_TIMEOUT="${READY_TIMEOUT:-120}"   # 等待就绪的总秒数
READY_INTERVAL="${READY_INTERVAL:-3}"   # 轮询间隔
UPLOAD_TIMEOUT="${UPLOAD_TIMEOUT:-60}"
CHAT_TIMEOUT="${CHAT_TIMEOUT:-120}"

# ---- 颜色 ----
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_CYAN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_CYAN=""
fi

log()  { printf '%s[%s]%s %s\n' "$C_BOLD" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
info() { printf '%s%s%s\n' "$C_CYAN" "$*" "$C_RESET"; }
ok()   { printf '%s%s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s%s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }
err()  { printf '%s%s%s\n' "$C_RED" "$*" "$C_RESET" >&2; }
die()  { err "ERROR: $*"; exit 1; }

# ---- 前置检查 ----
command -v curl >/dev/null 2>&1 || die "curl 未安装，请先安装 curl。"
[[ -d "$SAMPLE_DIR" ]] || die "示例文档目录不存在: $SAMPLE_DIR"

# ---- HTTP 辅助函数 ----
# 输出: 第一行 HTTP 状态码，其余为响应体。
http_code_body() {
  local tmp
  tmp="$(mktemp)"
  local code
  code=$(curl --silent --show-error --max-time "$1" \
            -o "$tmp" \
            -w '%{http_code}' \
            "${@:2}" || true)
  echo "$code"
  cat "$tmp"
  rm -f "$tmp"
}

# ---- Step 1: 等待服务就绪 ----
log "等待 Open Agent 就绪: ${BASE_URL}/api/ready （最多 ${READY_TIMEOUT}s）..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready=0
while [[ $(date +%s) -lt $deadline ]]; do
  code="$(curl --silent --output /dev/null --write-out '%{http_code}' \
              --max-time 5 "${BASE_URL}/api/ready" || true)"
  if [[ "$code" == "200" ]]; then
    ready=1
    break
  fi
  printf '%s.%s' "$C_YELLOW" "$C_RESET"
  sleep "$READY_INTERVAL"
done
echo
[[ $ready -eq 1 ]] || die "服务在 ${READY_TIMEOUT}s 内未就绪。请运行 'docker compose logs' 查看日志。"
ok "服务已就绪。"

# ---- Step 2: 上传示例文档 ----
info "上传示例文档: ${SAMPLE_DIR}"
uploaded=0
failed=0
shopt -s nullglob
for f in "$SAMPLE_DIR"/*; do
  [[ -f "$f" ]] || continue
  fname="$(basename "$f")"
  log "  - 上传 ${fname} ..."
  out="$(http_code_body "$UPLOAD_TIMEOUT" \
            -X POST "${BASE_URL}/api/upload" \
            -H "Authorization: Bearer ${AUTH_TOKEN}" \
            -F "kb_name=default" \
            -F "file=@${f}")"
  code="$(printf '%s' "$out" | head -n1)"
  body="$(printf '%s' "$out" | tail -n +2)"
  if [[ "$code" == "200" ]]; then
    ok "  已上传 ${fname} (HTTP ${code}): ${body}"
    uploaded=$((uploaded + 1))
  else
    warn "  上传失败 ${fname} (HTTP ${code}): ${body}"
    failed=$((failed + 1))
  fi
done
shopt -u nullglob

[[ $uploaded -gt 0 ]] || die "没有文档被成功上传，无法继续 RAG 查询。"
warn "已上传 ${uploaded} 个文件；失败 ${failed} 个。"

# ---- Step 3: 发送 chat 查询 ----
question="What is Acme Corp?"
log "提问: ${C_BOLD}${question}${C_RESET}"

# 用 python 构造 JSON 以正确转义（“-” 表示从 stdin 读取程序体）。
payload="$(python3 - "$question" "$SESSION_ID" <<'PY'
import json, sys
print(json.dumps({"message": sys.argv[1], "session_id": sys.argv[2]}))
PY
)" || die "构造 JSON 失败，请确认已安装 python3。"

out="$(http_code_body "$CHAT_TIMEOUT" \
          -X POST "${BASE_URL}/api/chat" \
          -H "Authorization: Bearer ${AUTH_TOKEN}" \
          -H "Content-Type: application/json" \
          -d "$payload")"
code="$(printf '%s' "$out" | head -n1)"
body="$(printf '%s' "$out" | tail -n +2)"

if [[ "$code" != "200" ]]; then
  die "Chat 请求失败 (HTTP ${code}): ${body}"
fi

# ---- Step 4: 显示响应 ----
echo
ok "===== 响应 ====="
# 尝试美化 JSON；失败则原样输出。
if command -v python3 >/dev/null 2>&1; then
  printf '%s' "$body" | python3 -m json.tool 2>/dev/null || printf '%s\n' "$body"
else
  printf '%s\n' "$body"
fi
echo
log "Session ID: ${SESSION_ID}"
info "提示: 打开 ${BASE_URL}/docs 查看完整交互式 API 文档。"
info "提示: 如启用了前端，可访问 ${BASE_URL}/ 使用 Web UI。"
