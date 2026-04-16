#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    idx = (len(vs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vs[lo]
    w = idx - lo
    return vs[lo] * (1 - w) + vs[hi] * w


def parse_tool_tokens(run_log: Path) -> dict[str, int] | None:
    if not run_log.exists():
        return None
    tokens_obj: dict[str, int] | None = None
    with run_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '"type":"result"' not in line and '"type": "result"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Pattern A: OpenCode style: message.info.tokens
            tokens_a = (((obj.get("message") or {}).get("info") or {}).get("tokens") or {})
            if tokens_a:
                tokens_obj = {
                    "total": int(tokens_a.get("total", 0) or 0),
                    "input": int(tokens_a.get("input", 0) or 0),
                    "output": int(tokens_a.get("output", 0) or 0),
                    "reasoning": int(tokens_a.get("reasoning", 0) or 0),
                    "cache_read": int((((tokens_a.get("cache") or {}).get("read", 0)) or 0)),
                    "cache_write": int((((tokens_a.get("cache") or {}).get("write", 0)) or 0)),
                }
                continue

            # Pattern B: Claude/Qwen style: usage.*
            usage = obj.get("usage") or {}
            if usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
                cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                total = input_tokens + output_tokens + cache_create + cache_read
                tokens_obj = {
                    "total": total,
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": 0,
                    "cache_read": cache_read,
                    "cache_write": cache_create,
                }
    return tokens_obj


def build_tool_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        log_path = r.get("log_path")
        if not log_path:
            continue
        tokens = parse_tool_tokens(Path(log_path))
        if not tokens:
            continue
        out.append(
            {
                "tool": r.get("tool", ""),
                "repository": r.get("repository", ""),
                **tokens,
            }
        )
    return out


def make_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out: Path,
    annotate_fmt: str = "{:.1f}",
) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    bars = plt.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(labels)])
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + (max(values) * 0.01 if values else 0), annotate_fmt.format(v), ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def make_stacked_phase2_chart(rows_by_tool: dict[str, list[dict[str, Any]]], out: Path) -> None:
    tools = list(rows_by_tool.keys())
    keys = ["not_guilty", "guilty", "none"]
    colors = {"not_guilty": "#2ca02c", "guilty": "#d62728", "none": "#7f7f7f"}
    counts = defaultdict(lambda: defaultdict(int))
    for t, rows in rows_by_tool.items():
        for r in rows:
            v = r.get("phase2_verdict")
            k = v if v in ("not_guilty", "guilty") else "none"
            counts[t][k] += 1

    x = range(len(tools))
    bottom = [0] * len(tools)
    plt.figure(figsize=(8, 5), dpi=160)
    for k in keys:
        vals = [counts[t][k] for t in tools]
        plt.bar(x, vals, bottom=bottom, label=k, color=colors[k])
        bottom = [bottom[i] + vals[i] for i in range(len(vals))]
    plt.xticks(list(x), tools)
    plt.title("Phase-2 verdict distribution by tool")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def make_overlap_outcome_chart(overlap_rows: dict[str, dict[str, dict[str, Any]]], out: Path) -> None:
    # overlap rows contains repo -> tool -> row, tools all three present
    counter = Counter()
    for repo, toolmap in overlap_rows.items():
        s = {t: bool(toolmap[t].get("success")) for t in ("claude_code", "qwen_code", "open_code")}
        if all(s.values()):
            counter["all_three_success"] += 1
        elif not any(s.values()):
            counter["all_three_failed"] += 1
        else:
            counter["mixed"] += 1
    labels = ["all_three_success", "mixed", "all_three_failed"]
    vals = [counter[l] for l in labels]
    plt.figure(figsize=(8, 5), dpi=160)
    bars = plt.bar(labels, vals, color=["#2ca02c", "#1f77b4", "#d62728"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)
    plt.title("Outcome overlap on repos with all 3 tools")
    plt.ylabel("Repo count")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def tool_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durs = [float(r.get("duration_sec", 0.0) or 0.0) for r in rows]
    succ = sum(1 for r in rows if r.get("success") is True)
    verify_succ = sum(1 for r in rows if r.get("verify_success") is True)
    timeout_n = sum(1 for r in rows if r.get("timeout") is True)
    p2 = Counter((r.get("phase2_verdict") if r.get("phase2_verdict") in ("guilty", "not_guilty") else "none") for r in rows)
    return {
        "n": len(rows),
        "success_n": succ,
        "success_rate": pct(succ, len(rows)),
        "verify_success_n": verify_succ,
        "verify_success_rate": pct(verify_succ, len(rows)),
        "timeout_n": timeout_n,
        "avg_duration_sec": mean(durs) if durs else float("nan"),
        "median_duration_sec": median(durs) if durs else float("nan"),
        "p90_duration_sec": quantile(durs, 0.9) if durs else float("nan"),
        "phase2_not_guilty_n": p2["not_guilty"],
        "phase2_guilty_n": p2["guilty"],
        "phase2_none_n": p2["none"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成三工具综合实验报告")
    parser.add_argument(
        "--merged-claude-qwen",
        type=Path,
        default=Path("experiment/results_merged_claude_qwen/merged_selected_runs_20260410-192807/merged_raw_results.jsonl"),
    )
    parser.add_argument(
        "--open-run",
        type=Path,
        default=Path("experiment/results_benchmark100_open_code/20260410-004552-102621/raw_results.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment/results_merged_claude_qwen"),
    )
    args = parser.parse_args()

    cq_rows = load_jsonl(args.merged_claude_qwen)
    open_rows = load_jsonl(args.open_run)

    # build unified rows
    rows = list(cq_rows) + list(open_rows)
    rows_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rows_by_tool[r.get("tool", "unknown")].append(r)

    out_dir = args.output_root / f"multi_cli_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # per tool stats
    stats = {tool: tool_stats(rs) for tool, rs in rows_by_tool.items()}

    # overlap analysis on repos with all 3 tools
    by_repo_tool: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        repo = r.get("repository", "")
        tool = r.get("tool", "")
        if repo and tool:
            by_repo_tool[repo][tool] = r
    overlap_repos = {repo: m for repo, m in by_repo_tool.items() if {"claude_code", "qwen_code", "open_code"}.issubset(set(m.keys()))}

    all_three_failed = []
    all_three_success = []
    mixed = []
    for repo, m in overlap_repos.items():
        s = {t: bool(m[t].get("success")) for t in ("claude_code", "qwen_code", "open_code")}
        if all(s.values()):
            all_three_success.append(repo)
        elif not any(s.values()):
            all_three_failed.append(repo)
        else:
            mixed.append(repo)

    # pairwise disagreement on overlap repos
    pair_disagree = {}
    pairs = [("claude_code", "qwen_code"), ("claude_code", "open_code"), ("qwen_code", "open_code")]
    for a, b in pairs:
        n = 0
        for repo, m in overlap_repos.items():
            if bool(m[a].get("success")) != bool(m[b].get("success")):
                n += 1
        pair_disagree[f"{a}__{b}"] = n

    # tool order
    tool_order = [t for t in ("claude_code", "qwen_code", "open_code") if t in stats]

    # token stats for all tools (from run.log)
    token_rows = build_tool_token_rows(rows)
    token_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in token_rows:
        token_by_tool[t["tool"]].append(t)

    token_stats_by_tool: dict[str, dict[str, dict[str, float | int]]] = {}
    for tool, trs in token_by_tool.items():
        token_stats_by_tool[tool] = {}
        for key in ("total", "input", "output", "reasoning", "cache_read", "cache_write"):
            vals = [int(x[key]) for x in trs]
            token_stats_by_tool[tool][key] = {
                "n": len(vals),
                "sum": int(sum(vals)),
                "avg": float(mean(vals)),
                "median": float(median(vals)),
                "p90": float(quantile(vals, 0.9)),
            }

    # pricing (CNY / 1M tokens)
    # claude_code:
    #   input=2, output=12, cache_write=2.5, cache_read=0.2
    # others (qwen_code/open_code):
    #   input=0.8, output=4.8
    pricing = {
        "claude_code": {"input": 2.0, "output": 12.0, "cache_write": 2.5, "cache_read": 0.2},
        "qwen_code": {"input": 0.8, "output": 4.8, "cache_write": 0.0, "cache_read": 0.0},
        "open_code": {"input": 0.8, "output": 4.8, "cache_write": 0.0, "cache_read": 0.0},
    }

    tool_costs: dict[str, dict[str, float | int]] = {}
    for t in tool_order:
        if t not in token_stats_by_tool:
            continue
        p = pricing.get(t, pricing["qwen_code"])
        input_sum = float(token_stats_by_tool[t]["input"]["sum"])
        output_sum = float(token_stats_by_tool[t]["output"]["sum"])
        cache_write_sum = float(token_stats_by_tool[t]["cache_write"]["sum"])
        cache_read_sum = float(token_stats_by_tool[t]["cache_read"]["sum"])
        total_cost = (
            input_sum / 1_000_000.0 * p["input"]
            + output_sum / 1_000_000.0 * p["output"]
            + cache_write_sum / 1_000_000.0 * p["cache_write"]
            + cache_read_sum / 1_000_000.0 * p["cache_read"]
        )
        n = int(token_stats_by_tool[t]["input"]["n"])
        tool_costs[t] = {
            "n_priced": n,
            "input_tokens": int(input_sum),
            "output_tokens": int(output_sum),
            "cache_write_tokens": int(cache_write_sum),
            "cache_read_tokens": int(cache_read_sum),
            "price_input_per_m": p["input"],
            "price_output_per_m": p["output"],
            "price_cache_write_per_m": p["cache_write"],
            "price_cache_read_per_m": p["cache_read"],
            "estimated_total_cny": total_cost,
            "estimated_avg_cny_per_run": (total_cost / n if n else 0.0),
        }

    # charts
    make_bar_chart(
        tool_order,
        [stats[t]["success_rate"] for t in tool_order],
        "Success rate by tool",
        "Success rate (%)",
        out_dir / "chart1_success_rate_by_tool.png",
        annotate_fmt="{:.2f}%",
    )
    make_bar_chart(
        tool_order,
        [stats[t]["avg_duration_sec"] / 60.0 for t in tool_order],
        "Average runtime by tool",
        "Minutes",
        out_dir / "chart2_avg_runtime_minutes.png",
        annotate_fmt="{:.1f}",
    )
    make_stacked_phase2_chart({t: rows_by_tool[t] for t in tool_order}, out_dir / "chart3_phase2_verdict_by_tool.png")
    make_overlap_outcome_chart(overlap_repos, out_dir / "chart4_overlap_outcome_all3.png")

    # csv outputs
    # per-tool stats csv
    tool_stats_csv = out_dir / "tool_stats.csv"
    with tool_stats_csv.open("w", encoding="utf-8") as f:
        f.write(
            "tool,n,success_n,success_rate,verify_success_n,verify_success_rate,timeout_n,avg_duration_sec,median_duration_sec,p90_duration_sec,phase2_not_guilty_n,phase2_guilty_n,phase2_none_n\n"
        )
        for t in tool_order:
            s = stats[t]
            f.write(
                f"{t},{s['n']},{s['success_n']},{s['success_rate']:.4f},{s['verify_success_n']},{s['verify_success_rate']:.4f},{s['timeout_n']},{s['avg_duration_sec']:.4f},{s['median_duration_sec']:.4f},{s['p90_duration_sec']:.4f},{s['phase2_not_guilty_n']},{s['phase2_guilty_n']},{s['phase2_none_n']}\n"
            )

    overlap_csv = out_dir / "repos_all_three_failed.csv"
    with overlap_csv.open("w", encoding="utf-8") as f:
        f.write("repository,repo_url,claude_success,qwen_success,open_success\n")
        for repo in sorted(all_three_failed):
            m = overlap_repos[repo]
            url = m["claude_code"].get("repo_url") or m["qwen_code"].get("repo_url") or m["open_code"].get("repo_url") or ""
            f.write(
                f"{repo},{url},{m['claude_code'].get('success')},{m['qwen_code'].get('success')},{m['open_code'].get('success')}\n"
            )

    cost_csv = out_dir / "tool_costs.csv"
    with cost_csv.open("w", encoding="utf-8") as f:
        f.write(
            "tool,n_priced,input_tokens,output_tokens,cache_write_tokens,cache_read_tokens,price_input_per_m,price_output_per_m,price_cache_write_per_m,price_cache_read_per_m,estimated_total_cny,estimated_avg_cny_per_run\n"
        )
        for t in tool_order:
            if t not in tool_costs:
                continue
            c = tool_costs[t]
            f.write(
                f"{t},{c['n_priced']},{c['input_tokens']},{c['output_tokens']},{c['cache_write_tokens']},{c['cache_read_tokens']},{c['price_input_per_m']},{c['price_output_per_m']},{c['price_cache_write_per_m']},{c['price_cache_read_per_m']},{c['estimated_total_cny']:.6f},{c['estimated_avg_cny_per_run']:.6f}\n"
            )

    # detailed markdown report
    report = out_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# 三工具实验综合报告（Claude Code / Qwen Code / OpenCode）\n\n")
        f.write(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Claude/Qwen 输入: `{args.merged_claude_qwen}`\n")
        f.write(f"- OpenCode 输入: `{args.open_run}`（未跑完，仅纳入当前已完成样本）\n\n")

        f.write("## 核心结论\n\n")
        f.write(
            f"1. 目前样本量：Claude={stats.get('claude_code', {}).get('n', 0)}，Qwen={stats.get('qwen_code', {}).get('n', 0)}，OpenCode={stats.get('open_code', {}).get('n', 0)}。\n"
        )
        f.write(
            f"2. 成功率：Claude={stats.get('claude_code', {}).get('success_rate', 0):.2f}%，Qwen={stats.get('qwen_code', {}).get('success_rate', 0):.2f}%，OpenCode={stats.get('open_code', {}).get('success_rate', 0):.2f}%。\n"
        )
        f.write(
            f"3. 平均耗时（分钟）：Claude={stats.get('claude_code', {}).get('avg_duration_sec', 0)/60.0:.2f}，Qwen={stats.get('qwen_code', {}).get('avg_duration_sec', 0)/60.0:.2f}，OpenCode={stats.get('open_code', {}).get('avg_duration_sec', 0)/60.0:.2f}。\n"
        )
        f.write(
            f"4. 三工具共同覆盖仓库数={len(overlap_repos)}；其中三工具共同失败={len(all_three_failed)}，共同成功={len(all_three_success)}，混合结果={len(mixed)}。\n\n"
        )

        f.write("## 图表\n\n")
        f.write("### 1) Success rate by tool\n\n![](chart1_success_rate_by_tool.png)\n\n")
        f.write("### 2) Average runtime by tool\n\n![](chart2_avg_runtime_minutes.png)\n\n")
        f.write("### 3) Phase-2 verdict distribution by tool\n\n![](chart3_phase2_verdict_by_tool.png)\n\n")
        f.write("### 4) Outcome overlap on repos with all 3 tools\n\n![](chart4_overlap_outcome_all3.png)\n\n")

        f.write("## 指标总表\n\n")
        f.write("| Tool | N | Success | SuccessRate | VerifySuccessRate | Timeout | AvgSec | MedianSec | P90Sec | Phase2 NotGuilty | Phase2 Guilty | Phase2 None |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for t in tool_order:
            s = stats[t]
            f.write(
                f"| {t} | {s['n']} | {s['success_n']} | {s['success_rate']:.2f}% | {s['verify_success_rate']:.2f}% | {s['timeout_n']} | {s['avg_duration_sec']:.1f} | {s['median_duration_sec']:.1f} | {s['p90_duration_sec']:.1f} | {s['phase2_not_guilty_n']} | {s['phase2_guilty_n']} | {s['phase2_none_n']} |\n"
            )
        f.write("\n")

        f.write("## Token消耗（从各工具 run.log 解析）\n\n")
        if token_stats_by_tool:
            f.write("| Tool | Metric | N | Total | Avg | Median | P90 |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|\n")
            for t in tool_order:
                if t not in token_stats_by_tool:
                    continue
                for k in ("total", "input", "output", "reasoning", "cache_read", "cache_write"):
                    st = token_stats_by_tool[t][k]
                    f.write(
                        f"| {t} | {k} | {st['n']} | {st['sum']} | {st['avg']:.1f} | {st['median']:.1f} | {st['p90']:.1f} |\n"
                    )
            f.write("\n")
            f.write("说明：`cache_write` 对应 cache_creation_input_tokens，`cache_read` 对应 cache_read_input_tokens。\n\n")
        else:
            f.write("当前未从运行日志解析到稳定 token 字段。\n\n")

        f.write("## 成本估算（人民币）\n\n")
        f.write("计价规则：\n\n")
        f.write("- Claude Code：input ¥2/M，output ¥12/M，cache_write ¥2.5/M，cache_read ¥0.2/M\n")
        f.write("- 其余工具（Qwen/OpenCode）：input ¥0.8/M，output ¥4.8/M\n\n")
        if tool_costs:
            f.write("| Tool | N(priced) | InputTokens | OutputTokens | CacheWrite | CacheRead | Estimated Total (¥) | Avg / Run (¥) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for t in tool_order:
                if t not in tool_costs:
                    continue
                c = tool_costs[t]
                f.write(
                    f"| {t} | {c['n_priced']} | {c['input_tokens']} | {c['output_tokens']} | {c['cache_write_tokens']} | {c['cache_read_tokens']} | {c['estimated_total_cny']:.4f} | {c['estimated_avg_cny_per_run']:.4f} |\n"
                )
            f.write("\n")
            f.write("注：OpenCode 日志当前未见 cache token 计数，按 0 处理。\n\n")
        else:
            f.write("当前无可用 token 样本，无法估算成本。\n\n")

        f.write("## 三工具共同失败仓库\n\n")
        if all_three_failed:
            f.write("| Repository | URL | Claude | Qwen | Open |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for repo in sorted(all_three_failed):
                m = overlap_repos[repo]
                url = m["claude_code"].get("repo_url") or m["qwen_code"].get("repo_url") or m["open_code"].get("repo_url") or ""
                f.write(
                    f"| {repo} | {url} | {int(bool(m['claude_code'].get('success')))} | {int(bool(m['qwen_code'].get('success')))} | {int(bool(m['open_code'].get('success')))} |\n"
                )
            f.write("\n")
        else:
            f.write("当前三工具共同覆盖的仓库里，没有出现“三者同时失败”。\n\n")

        f.write("## 两两分歧计数（基于三工具共同覆盖仓库）\n\n")
        f.write("| Pair | Disagree Count |\n")
        f.write("|---|---:|\n")
        for pair, n in pair_disagree.items():
            f.write(f"| {pair} | {n} |\n")
        f.write("\n")

        # top slow repos by tool
        f.write("## 各工具最慢仓库 Top10（按 duration_sec）\n\n")
        for t in tool_order:
            f.write(f"### {t}\n\n")
            top = sorted(rows_by_tool[t], key=lambda r: float(r.get("duration_sec", 0) or 0), reverse=True)[:10]
            f.write("| Repository | DurationSec | Success | VerifySuccess | Phase2 |\n")
            f.write("|---|---:|---:|---:|---|\n")
            for r in top:
                f.write(
                    f"| {r.get('repository','')} | {float(r.get('duration_sec',0) or 0):.1f} | {r.get('success')} | {r.get('verify_success')} | {r.get('phase2_verdict')} |\n"
                )
            f.write("\n")

        f.write("## 可执行建议\n\n")
        f.write("1. 优先排查三工具共同失败仓库（若存在），这些最可能是 benchmark 任务定义/依赖前置的系统性问题。\n")
        f.write("2. 对 `missing_dependencies` 与 `path_or_cli_entry_issue` 类问题，增加统一的安装后检查清单（import 检查 + PATH/entrypoint 检查）。\n")
        f.write("3. OpenCode 仍在运行，建议每批次重算本报告，观察其成功率和耗时是否向 Claude/Qwen 收敛。\n")
        f.write("4. 若要跨工具比较 token，请在 Claude/Qwen 的 result schema 中补充 token 字段（与 OpenCode 对齐 total/input/output/reasoning）。\n")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"merged_claude_qwen": str(args.merged_claude_qwen), "open_run": str(args.open_run)},
        "counts": {
            "claude_rows": stats.get("claude_code", {}).get("n", 0),
            "qwen_rows": stats.get("qwen_code", {}).get("n", 0),
            "open_rows": stats.get("open_code", {}).get("n", 0),
            "overlap_repos_all_three": len(overlap_repos),
            "all_three_failed": len(all_three_failed),
            "all_three_success": len(all_three_success),
            "mixed": len(mixed),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report_dir={out_dir}")
    print(f"report_md={out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
