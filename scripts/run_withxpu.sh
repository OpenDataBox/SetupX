#
# 带 XPU 跑 benchmark 指定范围的仓库（从 python329.jsonl 中按行号切片）
#
# 前置条件:
#   - .env 中配置好 OPENAI_API_KEY、OPENAI_BASE_URL、dns 等（参考 .env.example）
#   - pgvector 数据库中已导入 xpu_final.jsonl（600 条，共用同一台服务器的数据库）
#
# 用法:
#   bash scripts/run_withxpu.sh <start> <end> [parallelism]
#
# 示例:
#   bash scripts/run_withxpu.sh 1 81 5                # 跑第 1-81 个仓库，5 并发
#   bash scripts/run_withxpu.sh 82 164 5              # 跑第 82-164 个仓库，5 并发
#   bash scripts/run_withxpu.sh 165 246 5             # 跑第 165-246 个仓库，5 并发
#   bash scripts/run_withxpu.sh 247 329 5             # 跑第 247-329 个仓库，5 并发
#
# 说明:
#   - XPU 数据已在 pgvector 数据库中（postgresql://...localhost:5433/xpu_db）
#   - 跑实验时 FREEZE_TELEMETRY=1，不修改数据库中的 telemetry 计数
#   - 结果输出到 experiment/results_withxpu_<start>-<end>/ 目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- 检查 .env ----
if [ ! -f .env ]; then
    echo "错误: 项目根目录下没有 .env 文件"
    echo "请确保 .env 中包含以下配置:"
    echo "  OPENAI_API_KEY=..."
    echo "  OPENAI_BASE_URL=..."
    echo "  dns=postgresql://zihang:123456@localhost:5433/xpu_db"
    exit 1
fi

# ---- 参数解析 ----
if [ $# -lt 2 ]; then
    echo "用法: bash scripts/run_withxpu.sh <start> <end> [parallelism]"
    echo ""
    echo "  start: 起始行号（从 1 开始）"
    echo "  end:   结束行号（包含）"
    echo "  parallelism: 并发数（默认 5）"
    echo ""
    echo "示例:"
    echo "  bash scripts/run_withxpu.sh 1 81 5      # 跑第 1-81 个"
    echo "  bash scripts/run_withxpu.sh 82 164 5     # 跑第 82-164 个"
    echo "  bash scripts/run_withxpu.sh 165 246 5    # 跑第 165-246 个"
    echo "  bash scripts/run_withxpu.sh 247 329 5    # 跑第 247-329 个"
    exit 1
fi

START=$1
END=$2
PARALLELISM=${3:-5}
TOTAL_LINES=$(wc -l < data/python329.jsonl)

if [ "$START" -lt 1 ] || [ "$END" -gt "$TOTAL_LINES" ] || [ "$START" -gt "$END" ]; then
    echo "错误: 行号范围无效 (1-$TOTAL_LINES)，输入了 $START-$END"
    exit 1
fi

SLICE_FILE="data/_slice_${START}-${END}.jsonl"
OUTPUT_DIR="experiment/results_withxpu_${START}-${END}"
COUNT=$((END - START + 1))

echo "=================================================="
echo " With-XPU Benchmark Runner"
echo "=================================================="
echo " 仓库范围: 第 $START - $END 个（共 $COUNT 个）"
echo " 并发数:   $PARALLELISM"
echo " 输出目录: $OUTPUT_DIR"
echo "=================================================="

# ---- 步骤 1: 切片仓库列表 ----
echo ""
echo "[1/2] 切片仓库列表..."
sed -n "${START},${END}p" data/python329.jsonl > "$SLICE_FILE"
ACTUAL=$(wc -l < "$SLICE_FILE")
echo "  已生成 $SLICE_FILE ($ACTUAL 个仓库)"

# ---- 步骤 2: 跑实验（冻结 telemetry）----
echo ""
echo "[2/2] 启动实验（FREEZE_TELEMETRY=1）..."
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

FREEZE_TELEMETRY=1 .venv/bin/python experiment/ours/run_benchmark_ours.py \
    --repo-list "$SLICE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --parallelism "$PARALLELISM" \
    --phase1-timeout 3600

echo ""
echo "=================================================="
echo " 实验完成！"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果目录: $OUTPUT_DIR"
echo "=================================================="
