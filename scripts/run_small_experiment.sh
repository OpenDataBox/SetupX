#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "错误: 找不到虚拟环境 Python: $ROOT_DIR/.venv/bin/python" >&2
  echo "请先创建并安装虚拟环境依赖，例如: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

TOOLS_CONFIG="experiment/configs/tools.json"
if [[ ! -f "$TOOLS_CONFIG" ]]; then
  TOOLS_CONFIG="experiment/configs/tools.example.json"
  echo "提示: 未找到 experiment/configs/tools.json，回退使用 $TOOLS_CONFIG" >&2
fi

export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENCODE_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENCODE_BASE_URL:-}}"
export OPENAI_MODEL="${OPENAI_MODEL:-${OPENCODE_MODEL:-gpt-4o}}"

RUN_CMD=(
  "$ROOT_DIR/.venv/bin/python"
  experiment/run_cli_benchmark.py
  --tools-config "$TOOLS_CONFIG"
  --repo-list data/python179.jsonl
  --limit 20
)

if [[ $# -gt 0 ]]; then
  RUN_CMD+=("$@")
fi

echo "运行命令: ${RUN_CMD[*]}" >&2
exec "${RUN_CMD[@]}"
