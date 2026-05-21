#!/usr/bin/env python3
"""
汇总 CLI benchmark 结果，支持 tool 级与 run 级 summary。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 CLI benchmark 结果")
    parser.add_argument("--result-dir", required=True, help="单次实验结果目录")
    parser.add_argument(
        "--output",
        default="",
        help="run 级汇总输出路径；默认写入 result-dir 下新的 run_summary_*.json",
    )
    return parser.parse_args()


def load_rows(raw_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success = sum(1 for row in rows if row.get("success"))
    failed = total - success
    timeout = sum(1 for row in rows if row.get("timeout"))
    avg_duration = round(sum(float(row.get("duration_sec", 0.0)) for row in rows) / total, 2) if total else 0.0
    success_rate = round(success / total, 4) if total else 0.0
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "timeout": timeout,
        "avg_duration_sec": avg_duration,
        "success_rate": success_rate,
    }


def summarize_by_tool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tool"], []).append(row)

    tools: list[dict[str, Any]] = []
    for tool_name, tool_rows in sorted(grouped.items()):
        item = {"tool": tool_name, **summarize_rows(tool_rows)}
        tools.append(item)
    return tools


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir)
    raw_path = result_dir / "raw_results.jsonl"
    if not raw_path.exists():
        print(f"错误: 找不到 {raw_path}", file=sys.stderr)
        return 1

    rows = load_rows(raw_path)
    tool_summaries = summarize_by_tool(rows)
    for item in tool_summaries:
        tool_dir = result_dir / sanitize_name(item["tool"])
        if tool_dir.exists():
            summary_path = tool_dir / f"summary_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

    run_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": result_dir.name,
        "overall": summarize_rows(rows),
        "tools": tool_summaries,
    }
    if args.output:
        run_summary_path = Path(args.output)
    else:
        run_summary_path = result_dir / f"run_summary_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    for item in tool_summaries:
        print(
            f"{item['tool']}: total={item['total']} "
            f"success={item['success']} failed={item['failed']} "
            f"timeout={item['timeout']} success_rate={item['success_rate']:.2%} "
            f"avg_duration={item['avg_duration_sec']}s"
        )
    print(f"run_summary 已写入: {run_summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
