#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

if [[ -f .env.local ]]; then
  set -a
  . ./.env.local
  set +a
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "错误: 找不到虚拟环境 Python: $ROOT_DIR/.venv/bin/python" >&2
  echo "请先创建并安装虚拟环境依赖，例如: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

TOOLS_CONFIG="experiment/configs/tools.claude_qwen.json"
REPO_LIST="data/benchmark100.jsonl"

RUN_CMD=(
  "$ROOT_DIR/.venv/bin/python"
  experiment/run_cli_benchmark.py
  --tools-config "$TOOLS_CONFIG"
  --repo-list "$REPO_LIST"
  --limit 50
  --tool-parallelism 2
)

if [[ $# -gt 0 ]]; then
  RUN_CMD+=("$@")
fi

echo "运行命令: ${RUN_CMD[*]}" >&2
exec "${RUN_CMD[@]}"
