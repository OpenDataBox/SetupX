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
SOURCE_REPO_LIST="data/benchmark100.jsonl"
START_INDEX=21
END_INDEX=50

TEMP_REPO_LIST="$(mktemp /tmp/benchmark100_21_50.XXXXXX.jsonl)"
cleanup() {
  rm -f "$TEMP_REPO_LIST"
}
trap cleanup EXIT

python - <<'PY' "$SOURCE_REPO_LIST" "$TEMP_REPO_LIST" "$START_INDEX" "$END_INDEX"
from pathlib import Path
import sys

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
start_index = int(sys.argv[3])
end_index = int(sys.argv[4])

if start_index < 1 or end_index < start_index:
    raise SystemExit("无效区间")

with source_path.open("r", encoding="utf-8") as src, target_path.open("w", encoding="utf-8") as dst:
    for idx, line in enumerate(src, start=1):
        if idx < start_index:
            continue
        if idx > end_index:
            break
        dst.write(line)
PY

RUN_CMD=(
  "$ROOT_DIR/.venv/bin/python"
  experiment/run_cli_benchmark.py
  --tools-config "$TOOLS_CONFIG"
  --repo-list "$TEMP_REPO_LIST"
  --limit $((END_INDEX - START_INDEX + 1))
  --tool-parallelism 2
)

if [[ $# -gt 0 ]]; then
  RUN_CMD+=("$@")
fi

echo "运行命令: ${RUN_CMD[*]}" >&2
exec "${RUN_CMD[@]}"
