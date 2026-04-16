#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    label: str
    run_dir: Path
    start_idx: int
    end_idx: int


def load_benchmark_order(path: Path) -> dict[str, int]:
    order: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            repo = obj.get("repository")
            repo_url = obj.get("repo_url", "")
            keys: list[str] = []
            if repo:
                keys.append(repo)
            if repo_url:
                normalized = repo_url.rstrip("/")
                keys.append(normalized)
                if normalized.endswith(".git"):
                    normalized = normalized[:-4]
                    keys.append(normalized)
                if "/" in normalized:
                    keys.append(normalized.rsplit("/", 1)[-1])
            for key in keys:
                if key and key not in order:
                    order[key] = idx
    return order


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bool_to_text(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="合并指定区间的 Claude/Qwen benchmark 结果")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("data/benchmark100.jsonl"),
        help="benchmark100 列表路径",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment/results_merged_claude_qwen"),
        help="输出目录根路径",
    )
    args = parser.parse_args()

    sources = [
        SourceSpec(
            label="first20_1_20",
            run_dir=Path("experiment/results_benchmark100_first20_claude_qwen/20260406-170005-379912"),
            start_idx=1,
            end_idx=20,
        ),
        SourceSpec(
            label="r21_50",
            run_dir=Path("experiment/results_benchmark100_21_50_claude_qwen/20260407-005903-945681"),
            start_idx=21,
            end_idx=50,
        ),
        SourceSpec(
            label="r51_59_from_51_100",
            run_dir=Path("experiment/results_benchmark100_51_100_claude_qwen/20260407-093446-521662"),
            start_idx=51,
            end_idx=59,
        ),
        SourceSpec(
            label="r60_100",
            run_dir=Path("experiment/results_benchmark100_60_100_claude_qwen/20260409-002416-053374"),
            start_idx=60,
            end_idx=100,
        ),
    ]

    benchmark_order = load_benchmark_order(args.benchmark_file)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_root / f"merged_selected_runs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    missing_repo_rows: list[dict[str, Any]] = []

    for source in sources:
        raw_path = source.run_dir / "raw_results.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"找不到 raw_results.jsonl: {raw_path}")
        rows = load_jsonl(raw_path)
        for row in rows:
            repo = row.get("repository", "")
            repo_url = row.get("repo_url", "")
            candidates = [repo, repo_url.rstrip("/")]
            if repo_url.endswith(".git"):
                candidates.append(repo_url[:-4].rstrip("/"))
            idx = None
            for c in candidates:
                if c in benchmark_order:
                    idx = benchmark_order[c]
                    break
            if idx is None:
                missing_repo_rows.append(
                    {
                        "source": source.label,
                        "repository": repo,
                        "tool": row.get("tool", ""),
                    }
                )
                continue
            if source.start_idx <= idx <= source.end_idx:
                obj = dict(row)
                obj["repo_index"] = idx
                obj["source_label"] = source.label
                obj["source_run_dir"] = str(source.run_dir)
                merged_rows.append(obj)

    merged_rows.sort(key=lambda x: (x["repo_index"], x.get("tool", ""), x.get("started_at", "")))

    merged_jsonl = out_dir / "merged_raw_results.jsonl"
    with merged_jsonl.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_repo: dict[int, dict[str, Any]] = defaultdict(dict)
    for row in merged_rows:
        idx = row["repo_index"]
        repo = row.get("repository", "")
        tool = row.get("tool", "")
        per_repo[idx]["repo_index"] = idx
        per_repo[idx]["repository"] = repo
        per_repo[idx]["repo_url"] = row.get("repo_url", "")
        per_repo[idx][f"{tool}_success"] = bool_to_text(row.get("success"))
        per_repo[idx][f"{tool}_timeout"] = bool_to_text(row.get("timeout"))
        per_repo[idx][f"{tool}_duration_sec"] = row.get("duration_sec", "")
        per_repo[idx][f"{tool}_verify_success"] = bool_to_text(row.get("verify_success"))
        per_repo[idx][f"{tool}_phase2_verdict"] = row.get("phase2_verdict", "")
        per_repo[idx][f"{tool}_source"] = row.get("source_label", "")

    matrix_csv = out_dir / "repo_tool_matrix.csv"
    fieldnames = [
        "repo_index",
        "repository",
        "repo_url",
        "claude_code_success",
        "claude_code_timeout",
        "claude_code_duration_sec",
        "claude_code_verify_success",
        "claude_code_phase2_verdict",
        "claude_code_source",
        "qwen_code_success",
        "qwen_code_timeout",
        "qwen_code_duration_sec",
        "qwen_code_verify_success",
        "qwen_code_phase2_verdict",
        "qwen_code_source",
    ]
    with matrix_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in sorted(per_repo):
            writer.writerow(per_repo[idx])

    tool_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in merged_rows:
        tool = row.get("tool", "unknown")
        if row.get("success") is True:
            tool_counter[tool]["success"] += 1
        else:
            tool_counter[tool]["failed"] += 1
        if row.get("timeout") is True:
            tool_counter[tool]["timeout"] += 1

    repo_indexes = sorted({row["repo_index"] for row in merged_rows})
    summary_md = out_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Claude/Qwen 实验数据整合汇总\n\n")
        f.write(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 合并记录数: {len(merged_rows)}\n")
        f.write(f"- 覆盖仓库数: {len(set(row.get('repository') for row in merged_rows))}\n")
        f.write(f"- 覆盖索引范围: {min(repo_indexes)}-{max(repo_indexes)}\n")
        f.write("- 目标区间: 1-20, 21-50, 51-59, 60-100\n\n")
        f.write("## 工具级统计\n\n")
        f.write("| tool | success | failed | timeout |\n")
        f.write("|---|---:|---:|---:|\n")
        for tool in sorted(tool_counter):
            c = tool_counter[tool]
            f.write(f"| {tool} | {c['success']} | {c['failed']} | {c['timeout']} |\n")
        f.write("\n## 输入源\n\n")
        for source in sources:
            f.write(
                f"- `{source.label}`: `{source.run_dir}` (索引 {source.start_idx}-{source.end_idx})\n"
            )
        if missing_repo_rows:
            f.write("\n## 警告\n\n")
            f.write(f"- 有 {len(missing_repo_rows)} 条记录在 benchmark100 中找不到 repository。\n")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_file": str(args.benchmark_file),
        "sources": [
            {
                "label": s.label,
                "run_dir": str(s.run_dir),
                "start_idx": s.start_idx,
                "end_idx": s.end_idx,
            }
            for s in sources
        ],
        "output": {
            "merged_raw_results_jsonl": str(merged_jsonl),
            "repo_tool_matrix_csv": str(matrix_csv),
            "summary_md": str(summary_md),
        },
        "merged_row_count": len(merged_rows),
        "missing_repo_row_count": len(missing_repo_rows),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"整合完成: {out_dir}")
    print(f"merged_raw_results.jsonl: {merged_jsonl}")
    print(f"repo_tool_matrix.csv: {matrix_csv}")
    print(f"summary.md: {summary_md}")


if __name__ == "__main__":
    main()
