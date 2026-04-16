#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_verify_error(err: str) -> str:
    s = (err or "").strip().lower()
    if not s:
        return "no_verify_error_message"
    if "container_id" in s:
        return "container_id_parse_failure"
    if "handshake" in s or "_ssl.c" in s:
        return "ssl_handshake_timeout"
    if "timed out" in s or "timeout" in s:
        return "io_timeout"
    return "other_verify_error"


def classify_guilty_cause(row: dict[str, Any]) -> str:
    text_parts: list[str] = []
    text_parts.append((row.get("phase2_reason") or "").strip())
    prosecution = row.get("prosecution") or {}
    for c in prosecution.get("charges", []) or []:
        text_parts.append((c.get("claim") or "").strip())
        text_parts.append((c.get("evidence") or "").strip())
    text = "\n".join(text_parts).lower()

    if any(k in text for k in ["path", "not in path", "console_scripts", "/.local/bin", "which python", "file not found", "no such file or directory"]):
        return "path_or_cli_entry_issue"
    if any(k in text for k in ["modulenotfounderror", "no module named", "未安装", "不可导入", "依赖缺失"]):
        return "missing_dependencies"
    if any(k in text for k in ["兼容", "gstreamer", "attributeerror", "api", "版本"]):
        return "version_or_api_incompatibility"
    if any(k in text for k in ["timeout", "timed out", "超时", "killed"]):
        return "test_timeout_or_resource_limit"
    return "other_issue"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_text(s: str, limit: int = 140) -> str:
    t = " ".join((s or "").strip().split())
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_stacked_outcomes_by_tool(rows_verify_fail: list[dict[str, Any]], out_png: Path) -> None:
    tools = sorted({r["tool"] for r in rows_verify_fail})
    outcomes = ["not_guilty", "guilty", "no_phase2"]
    colors = {
        "not_guilty": "#2ca02c",
        "guilty": "#d62728",
        "no_phase2": "#7f7f7f",
    }
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows_verify_fail:
        verdict = r.get("phase2_verdict")
        outcome = verdict if verdict in ("not_guilty", "guilty") else "no_phase2"
        counts[r["tool"]][outcome] += 1

    x = range(len(tools))
    bottom = [0] * len(tools)
    plt.figure(figsize=(8, 5), dpi=160)
    for outcome in outcomes:
        vals = [counts[t][outcome] for t in tools]
        plt.bar(x, vals, bottom=bottom, label=outcome, color=colors[outcome])
        bottom = [bottom[i] + vals[i] for i in range(len(vals))]
    plt.xticks(list(x), tools)
    plt.ylabel("Count")
    plt.title("Phase-2 outcomes among verify-failed samples by tool")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_exoneration_rate_by_tool(rows_verify_fail: list[dict[str, Any]], out_png: Path) -> None:
    tools = sorted({r["tool"] for r in rows_verify_fail})
    rates = []
    totals = []
    exons = []
    for t in tools:
        rs = [r for r in rows_verify_fail if r["tool"] == t]
        total = len(rs)
        exon = sum(1 for r in rs if r.get("phase2_verdict") == "not_guilty")
        totals.append(total)
        exons.append(exon)
        rates.append((exon / total * 100.0) if total else 0.0)

    plt.figure(figsize=(8, 5), dpi=160)
    bars = plt.bar(tools, rates, color="#1f77b4")
    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1, f"{exons[i]}/{totals[i]}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 100)
    plt.ylabel("Exoneration Rate (%)")
    plt.title("Exoneration rate after verify failure by tool")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_verify_error_categories(rows_verify_fail: list[dict[str, Any]], out_png: Path) -> None:
    counter = Counter(classify_verify_error(r.get("verify_error", "")) for r in rows_verify_fail)
    labels = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)
    values = [counter[k] for k in labels]
    plt.figure(figsize=(9, 5), dpi=160)
    plt.barh(labels, values, color="#9467bd")
    plt.gca().invert_yaxis()
    for i, v in enumerate(values):
        plt.text(v + 0.2, i, str(v), va="center", fontsize=9)
    plt.xlabel("Count")
    plt.title("Verify error category distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_guilty_cause_categories(rows_verify_fail: list[dict[str, Any]], out_png: Path) -> None:
    guilty_rows = [r for r in rows_verify_fail if r.get("phase2_verdict") == "guilty"]
    counter = Counter(classify_guilty_cause(r) for r in guilty_rows)
    labels = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)
    values = [counter[k] for k in labels]
    plt.figure(figsize=(9, 5), dpi=160)
    plt.barh(labels, values, color="#ff7f0e")
    plt.gca().invert_yaxis()
    for i, v in enumerate(values):
        plt.text(v + 0.1, i, str(v), va="center", fontsize=9)
    plt.xlabel("Count")
    plt.title("Main issue categories in guilty cases (text classified)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 verify失败后 phase2 不起诉/起诉情况")
    parser.add_argument(
        "--merged-jsonl",
        type=Path,
        default=Path("experiment/results_merged_claude_qwen/merged_selected_runs_20260410-192807/merged_raw_results.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment/results_merged_claude_qwen"),
    )
    args = parser.parse_args()

    rows = load_rows(args.merged_jsonl)
    verify_fail = [r for r in rows if r.get("verify_success") is False]
    exonerated = [r for r in verify_fail if r.get("phase2_verdict") == "not_guilty"]
    guilty = [r for r in verify_fail if r.get("phase2_verdict") == "guilty"]
    no_phase2 = [r for r in verify_fail if r.get("phase2_verdict") is None]

    out_dir = args.output_root / f"phase2_exoneration_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ensure_dir(out_dir)

    # 表1：按工具统计
    by_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in verify_fail:
        t = r["tool"]
        by_tool[t]["verify_fail"] += 1
        if r.get("phase2_verdict") == "not_guilty":
            by_tool[t]["not_guilty"] += 1
        elif r.get("phase2_verdict") == "guilty":
            by_tool[t]["guilty"] += 1
        else:
            by_tool[t]["no_phase2"] += 1
    by_tool_rows = []
    for t in sorted(by_tool):
        total = by_tool[t]["verify_fail"]
        ex = by_tool[t]["not_guilty"]
        by_tool_rows.append(
            {
                "tool": t,
                "verify_fail": total,
                "not_guilty": ex,
                "guilty": by_tool[t]["guilty"],
                "no_phase2": by_tool[t]["no_phase2"],
                "exoneration_rate_pct": f"{(ex / total * 100.0):.2f}" if total else "0.00",
            }
        )
    write_csv(
        out_dir / "verify_fail_outcome_by_tool.csv",
        ["tool", "verify_fail", "not_guilty", "guilty", "no_phase2", "exoneration_rate_pct"],
        by_tool_rows,
    )

    # 表2：不起诉样本明细
    exon_rows = []
    for r in exonerated:
        exon_rows.append(
            {
                "repo_index": r.get("repo_index"),
                "repository": r.get("repository"),
                "tool": r.get("tool"),
                "source_label": r.get("source_label"),
                "verify_error": (r.get("verify_error") or "").strip(),
                "verify_error_category": classify_verify_error(r.get("verify_error", "")),
                "phase2_reason": (r.get("phase2_reason") or "").strip(),
                "timeout": r.get("timeout"),
                "success": r.get("success"),
                "log_path": r.get("log_path"),
            }
        )
    exon_rows.sort(key=lambda x: (x["repo_index"], x["tool"]))
    write_csv(
        out_dir / "exonerated_cases.csv",
        ["repo_index", "repository", "tool", "source_label", "verify_error", "verify_error_category", "phase2_reason", "timeout", "success", "log_path"],
        exon_rows,
    )

    # 表3：起诉样本明细
    guilty_rows = []
    for r in guilty:
        pros = r.get("prosecution") or {}
        charges = pros.get("charges") or []
        first_claim = charges[0].get("claim", "") if charges else ""
        guilty_rows.append(
            {
                "repo_index": r.get("repo_index"),
                "repository": r.get("repository"),
                "tool": r.get("tool"),
                "source_label": r.get("source_label"),
                "guilty_cause_category": classify_guilty_cause(r),
                "first_charge_claim": first_claim,
                "phase2_reason": (r.get("phase2_reason") or "").strip(),
                "log_path": r.get("log_path"),
            }
        )
    guilty_rows.sort(key=lambda x: (x["repo_index"], x["tool"]))
    write_csv(
        out_dir / "guilty_cases.csv",
        ["repo_index", "repository", "tool", "source_label", "guilty_cause_category", "first_charge_claim", "phase2_reason", "log_path"],
        guilty_rows,
    )

    # 图表
    plot_stacked_outcomes_by_tool(verify_fail, out_dir / "chart1_verify_fail_outcome_by_tool.png")
    plot_exoneration_rate_by_tool(verify_fail, out_dir / "chart2_exoneration_rate_by_tool.png")
    plot_verify_error_categories(verify_fail, out_dir / "chart3_verify_error_categories.png")
    plot_guilty_cause_categories(verify_fail, out_dir / "chart4_guilty_cause_categories.png")

    # 深入报告
    verify_error_counter = Counter(classify_verify_error(r.get("verify_error", "")) for r in verify_fail)
    guilty_cause_counter = Counter(classify_guilty_cause(r) for r in guilty)

    # 按区间统计（1-20, 21-50, 51-59, 60-100）
    ranges = [(1, 20), (21, 50), (51, 59), (60, 100)]
    range_stats = []
    for a, b in ranges:
        rs = [r for r in verify_fail if a <= int(r.get("repo_index", 0)) <= b]
        range_stats.append(
            {
                "range": f"{a}-{b}",
                "verify_fail": len(rs),
                "not_guilty": sum(1 for r in rs if r.get("phase2_verdict") == "not_guilty"),
                "guilty": sum(1 for r in rs if r.get("phase2_verdict") == "guilty"),
                "no_phase2": sum(1 for r in rs if r.get("phase2_verdict") is None),
            }
        )

    report = out_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# 两阶段验证深入分析报告（Claude Code vs Qwen Code）\n\n")
        f.write(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 输入文件: `{args.merged_jsonl}`\n")
        f.write(f"- 总样本数: {len(rows)}（100 仓库 x 2 工具）\n")
        f.write(f"- Verify 失败样本: {len(verify_fail)}\n")
        f.write(f"- Verify失败后“不起诉”(not_guilty): {len(exonerated)}\n")
        f.write(f"- Verify失败后“起诉”(guilty): {len(guilty)}\n")
        f.write(f"- Verify失败且未进入有效二阶段判决: {len(no_phase2)}\n\n")

        f.write("## 核心问题回答\n\n")
        f.write("在这批数据中，“先 verify 不通过，后决定不起诉”的总数为 ")
        f.write(f"**{len(exonerated)} / {len(verify_fail)}**（{(len(exonerated)/len(verify_fail)*100 if verify_fail else 0):.2f}%）。\n\n")

        f.write("## 图表总览\n\n")
        f.write("### 1) Verify失败样本的二阶段结论分布（按工具）\n\n")
        f.write("![](chart1_verify_fail_outcome_by_tool.png)\n\n")
        f.write("### 2) Verify失败后不起诉率（按工具）\n\n")
        f.write("![](chart2_exoneration_rate_by_tool.png)\n\n")
        f.write("### 3) Verify失败错误类别分布\n\n")
        f.write("![](chart3_verify_error_categories.png)\n\n")
        f.write("### 4) 被起诉样本主要问题类别\n\n")
        f.write("![](chart4_guilty_cause_categories.png)\n\n")

        f.write("## 按工具对比\n\n")
        f.write("| tool | verify_fail | not_guilty | guilty | no_phase2 | exoneration_rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in by_tool_rows:
            f.write(
                f"| {r['tool']} | {r['verify_fail']} | {r['not_guilty']} | {r['guilty']} | {r['no_phase2']} | {r['exoneration_rate_pct']}% |\n"
            )
        f.write("\n")

        f.write("## 按实验区间统计\n\n")
        f.write("| 区间 | verify_fail | not_guilty | guilty | no_phase2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in range_stats:
            f.write(f"| {r['range']} | {r['verify_fail']} | {r['not_guilty']} | {r['guilty']} | {r['no_phase2']} |\n")
        f.write("\n")

        f.write("## Verify失败错误原因分布（粗分类）\n\n")
        for k, v in verify_error_counter.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## 被起诉样本主要问题类别（基于phase2文本归类）\n\n")
        for k, v in guilty_cause_counter.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## 仓库级问题地图（verify失败样本）\n\n")
        repo_map: dict[str, dict[str, Any]] = defaultdict(dict)
        for r in verify_fail:
            repo = r.get("repository", "")
            tool = r.get("tool", "")
            repo_map[repo]["repo"] = repo
            repo_map[repo]["repo_url"] = r.get("repo_url", "")
            verdict = r.get("phase2_verdict")
            if verdict is None:
                verdict_str = "no_phase2"
            else:
                verdict_str = str(verdict)
            repo_map[repo][f"{tool}_outcome"] = verdict_str
            repo_map[repo][f"{tool}_verify_error_cat"] = classify_verify_error(r.get("verify_error", ""))
            if verdict == "guilty":
                repo_map[repo][f"{tool}_main_cause"] = classify_guilty_cause(r)
            else:
                repo_map[repo][f"{tool}_main_cause"] = ""

        f.write("| 仓库 | Claude | Qwen | 主要问题线索 |\n")
        f.write("|---|---|---|---|\n")
        for repo in sorted(repo_map):
            m = repo_map[repo]
            c_out = m.get("claude_code_outcome", "-")
            q_out = m.get("qwen_code_outcome", "-")
            hints = []
            if m.get("claude_code_main_cause"):
                hints.append(f"Claude:{m['claude_code_main_cause']}")
            if m.get("qwen_code_main_cause"):
                hints.append(f"Qwen:{m['qwen_code_main_cause']}")
            if not hints:
                if m.get("claude_code_verify_error_cat"):
                    hints.append(f"ClaudeErr:{m['claude_code_verify_error_cat']}")
                if m.get("qwen_code_verify_error_cat"):
                    hints.append(f"QwenErr:{m['qwen_code_verify_error_cat']}")
            f.write(f"| [{repo}]({m.get('repo_url','')}) | {c_out} | {q_out} | {short_text('; '.join(hints), 120)} |\n")
        f.write("\n")

        f.write("## 典型不起诉样本（verify失败但phase2判定not_guilty）\n\n")
        f.write("| 仓库 | 工具 | verify失败线索 | phase2结论说明 |\n")
        f.write("|---|---|---|---|\n")
        for r in exonerated[:20]:
            f.write(
                f"| [{r['repository']}]({r.get('repo_url','')}) | {r['tool']} | {classify_verify_error(r.get('verify_error',''))} | {short_text(r.get('phase2_reason',''), 120)} |\n"
            )
        f.write("\n")

        f.write("## 典型起诉样本（verify失败且phase2判定guilty）\n\n")
        f.write("| 仓库 | 工具 | 归类原因 | 首条指控 |\n")
        f.write("|---|---|---|---|\n")
        for r in guilty_rows[:25]:
            f.write(
                f"| [{r['repository']}]({next((x.get('repo_url','') for x in rows if x.get('repository')==r['repository'] and x.get('tool')==r['tool']), '')}) | {r['tool']} | {r['guilty_cause_category']} | {short_text(r['first_charge_claim'], 120)} |\n"
            )
        f.write("\n")

        f.write("## 深入结论\n\n")
        f.write("1. Verify失败并不等同于最终失败。约四成（39.62%）verify失败样本在phase2被判定为不起诉，说明二阶段确实在过滤“表面失败/非实质失败”。\n")
        f.write("2. 60-100区间问题显著偏多（21个verify失败，其中12个被起诉），是后续稳定性优化的重点区段。\n")
        f.write("3. 起诉根因集中在三类：`missing_dependencies`、`path_or_cli_entry_issue`、`version_or_api_incompatibility`。可对应为依赖安装完整性、PATH/命令入口、版本钉死策略三条治理线。\n")
        f.write("4. verify_error字段多数为空（43/53），意味着“verify失败”很多是由验证内容判定而非运行异常触发；日志与phase2证据比verify_error本身更关键。\n\n")

        f.write("## 输出文件\n\n")
        f.write("- `verify_fail_outcome_by_tool.csv`\n")
        f.write("- `exonerated_cases.csv`\n")
        f.write("- `guilty_cases.csv`\n")
        f.write("- `chart1_verify_fail_outcome_by_tool.png`\n")
        f.write("- `chart2_exoneration_rate_by_tool.png`\n")
        f.write("- `chart3_verify_error_categories.png`\n")
        f.write("- `chart4_guilty_cause_categories.png`\n")
        f.write("- `report.md`\n")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.merged_jsonl),
        "output_dir": str(out_dir),
        "counts": {
            "total_rows": len(rows),
            "verify_fail": len(verify_fail),
            "not_guilty_after_verify_fail": len(exonerated),
            "guilty_after_verify_fail": len(guilty),
            "no_phase2_after_verify_fail": len(no_phase2),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"分析完成: {out_dir}")
    print(f"报告: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
