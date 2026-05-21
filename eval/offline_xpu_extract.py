"""离线 XPU 经验提取脚本

从 runs_v6_phase2check/ 的日志文件中提取轨迹，结合 Phase 2 数据，
离线提取 XPU 经验。不修改核心提取逻辑，复用 extract_xpu_from_trajs_mvp 的
build_traj_prompt + LLM 提取。

用法:
  .venv/bin/python scripts/offline_xpu_extract.py \
    --log-dir runs_v6_phase2check \
    --results results/ours_92_final.jsonl \
    --output data/offline_xpu_extracted.jsonl \
    [--import-db]  # 可选：直接导入数据库
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)
sys.path.append(str(Path(__file__).parent.parent))

# 日志行前缀正则：2026-03-14 16:58:40 | LEVEL    | logger |
LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|")


def find_latest_log(log_dir: Path, repo: str) -> Path | None:
    """为仓库找到最新的日志文件。repo 格式: org/name"""
    safe = repo.replace("/", "__")
    pattern = f"{safe}@HEAD_*.log"
    matches = sorted(log_dir.glob(pattern))
    return matches[-1] if matches else None


def load_phase2_results(results_path: Path) -> dict[str, dict]:
    """从 ours_92_final.jsonl 加载 Phase 2 数据，key=repository"""
    data = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data[rec["repository"]] = rec
    return data


def parse_setup_traj_from_log(log_path: Path) -> list[dict]:
    """从日志文件解析 Setup Agent 的最后一轮 LLM 对话，返回 [{role, content}, ...]。

    策略：找到 Phase 2 之前最后一个 "LLM 输入 (Full Prompt)" 块，
    解析其中的 [N] role=X / content: ... 消息。
    """
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 找所有 setup_agent.llm 的 prompt 标记和 Phase 2 起点
    prompt_starts: list[int] = []
    phase2_idx = len(lines)

    for i, line in enumerate(lines):
        if "setup_agent.llm" in line and "LLM 输入 (Full Prompt)" in line:
            prompt_starts.append(i)
        if "阶段2: Phase 2" in line or "Phase 2 诉讼裁决" in line:
            phase2_idx = i
            break

    # 取 Phase 2 之前最后一个 prompt section
    valid = [s for s in prompt_starts if s < phase2_idx]
    if not valid:
        return []

    section_start = valid[-1]

    # 找 section 结束：从 section_start 往后，找第三个 === 分隔线（开头、标题、结尾各一行）
    # 实际上找 "LLM 输出" 或下一个 "=== Step" 即可
    section_end = phase2_idx
    eq_count = 0
    for i in range(section_start, phase2_idx):
        if "setup_agent.llm" in lines[i] and "====" in lines[i]:
            eq_count += 1
            if eq_count >= 3:
                # 第三个 === 行是 prompt section 的结束标记
                # 但后面紧跟的是 LLM 输出，不是我们要的
                pass
        # 找到下一个 LLM 输出标记就停
        if i > section_start + 2 and "LLM 输出 (Raw Response)" in lines[i]:
            section_end = i
            break

    # 从 section 中提取消息
    messages: list[dict] = []
    current_role: str | None = None
    content_lines: list[str] = []

    def flush():
        nonlocal current_role, content_lines
        if current_role and content_lines:
            text = "\n".join(content_lines)
            # 跳过 system prompt（index 0）：它是 agent 的指令，不是轨迹
            if current_role != "system" or len(messages) > 0:
                messages.append({"role": current_role, "content": text})
        current_role = None
        content_lines = []

    for i in range(section_start, section_end):
        line = lines[i]
        is_log = bool(LOG_LINE_RE.match(line))

        if is_log and "setup_agent.llm" in line:
            # 检查是否是新消息开始: [N] role=X
            role_m = re.search(r"\[(\d+)\] role=(\w+)", line)
            if role_m:
                flush()
                current_role = role_m.group(2)
                continue

            # 检查 content: 或 content (truncated):
            content_m = re.search(r"content(?:\s*\(truncated\))?:\s?(.*)", line)
            if content_m and current_role is not None:
                first_line = content_m.group(1)
                if first_line:
                    content_lines.append(first_line)
                continue

            # 截断标记: ... (N chars total) — 跳过
            if "chars total)" in line:
                continue

            # section 分隔线 === — 跳过
            if "====" in line:
                continue

        elif not is_log and current_role is not None:
            # 非日志行 = 上一条 content 的延续
            content_lines.append(line)

    flush()

    # 去掉 index 0 的 system prompt（agent 指令，不是轨迹数据）
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]

    return messages


def parse_prosecutor_from_log(log_path: Path) -> str:
    """从日志文件解析检察官调查轨迹，返回文本摘要。

    检察官 LLM 输出是多行 JSON，需要跨行收集。
    """
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    parts: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 只处理检察官相关日志
        if "setup_agent.prosecutor" not in line:
            i += 1
            continue

        # LLM 输出（多行 JSON）：收集到完整 JSON
        if "LLM 输出:" in line:
            m = re.search(r"LLM 输出: (.+)", line)
            if m:
                json_lines = [m.group(1)]
                j = i + 1
                while j < len(lines) and not LOG_LINE_RE.match(lines[j]):
                    json_lines.append(lines[j])
                    j += 1
                raw = "\n".join(json_lines)
                try:
                    parsed = json.loads(raw)
                    thought = parsed.get("thought", "")[:200]
                    action = parsed.get("action", "")
                    args = parsed.get("args", {})
                    if action == "exec_run":
                        parts.append(f"[检察官] {thought}\n  → 执行: {args.get('command', '')}")
                    elif action == "finish":
                        prosecute = args.get("prosecute", False)
                        charges = args.get("charges", [])
                        parts.append(f"[检察官结论] prosecute={prosecute}, 指控数={len(charges)}")
                        for c in charges:
                            parts.append(f"  指控: {c.get('claim', '')[:200]}")
                            parts.append(f"  证据: {c.get('evidence', '')[:200]}")
                except json.JSONDecodeError:
                    parts.append(f"[检察官输出] {raw[:200]}")
                i = j
                continue

        # 命令执行结果（含 exit_code）
        m = re.search(r"exec_run \[(.+?)\] → exit_code=(\d+)", line)
        if m:
            parts.append(f"  结果: exit_code={m.group(2)}")
            i += 1
            continue

        # 调查完成
        if "调查完成" in line:
            m = re.search(r"调查完成: (.+)", line)
            if m:
                parts.append(f"[调查完成] {m.group(1)}")

        i += 1

    return "\n".join(parts[-40:])  # 最后30行，覆盖关键步骤


def extract_one_repo(
    repo: str,
    traj: list[dict],
    phase2_data: dict[str, Any],
    prosecutor_investigation: str,
    cfg: dict,
    api_key: str,
    base_url: str,
) -> list[dict]:
    """对单个仓库运行 XPU 提取，返回提取出的 XPU 对象列表"""
    from src.xpu.extract_xpu_from_trajs_mvp import (
        build_traj_prompt,
        heuristic_stats_for_traj,
        heuristic_is_candidate,
        openai_compatible_chat_completions,
        parse_llm_json,
    )

    stats = heuristic_stats_for_traj(traj)
    is_candidate, score = heuristic_is_candidate(stats)
    if not is_candidate:
        return []

    # 构造 phase2_context
    phase2_context = None
    charges = phase2_data.get("charge_details", [])
    verdict = phase2_data.get("phase2_verdict")
    reason = phase2_data.get("phase2_reason", "")

    if verdict and verdict != "setup_failed":
        phase2_context = {
            "prosecution_charges": [{"claim": c} for c in charges] if charges else [],
            "verdict": verdict,
            "judge_reasoning": reason,
            "verifier_summary": phase2_data.get("final_message", ""),
            "prosecutor_investigation": prosecutor_investigation[:4000],
        }

    messages = build_traj_prompt(repo, "HEAD", traj, stats, cfg, phase2_context=phase2_context)

    raw = openai_compatible_chat_completions(
        model=cfg["llm_model"],
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=cfg["timeout_sec"],
        response_format_json=True,
    )
    content = raw["choices"][0]["message"]["content"]
    parsed = parse_llm_json(content)

    decision = str(parsed.get("decision", ""))
    if decision != "xpu":
        return []

    xpu_list = parsed.get("xpus") or []
    if not xpu_list:
        single = parsed.get("xpu")
        if single:
            xpu_list = [single]

    return xpu_list


def import_to_db(xpu_records: list[dict]) -> None:
    """将提取的 XPU 经验导入数据库（去重）"""
    from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
    from src.xpu.xpu_adapter import XpuEntry, XpuAtom
    from src.xpu.xpu_dedup import dedup_and_store

    store = XpuVectorStore()
    ok, dup, fail = 0, 0, 0

    for rec in tqdm(xpu_records, desc="导入数据库"):
        xpu_obj = rec["xpu"]
        repo = rec["repository"]
        try:
            auto_id = f"xpu_{int(time.time())}_{os.urandom(3).hex()}"
            atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                     for a in xpu_obj.get("atoms", [])]
            entry = XpuEntry(
                id=auto_id,
                context=xpu_obj.get("context", {}),
                signals=xpu_obj.get("signals", {}),
                advice_nl=xpu_obj.get("advice_nl", []),
                atoms=atoms,
            )
            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)
            result = dedup_and_store(store, entry, embedding, use_llm=True)
            action = result["action"]
            if action == "insert":
                ok += 1
            elif action in ("skip_duplicate", "merge"):
                dup += 1
            print(f"  [{repo}] {auto_id}: {action} - {result['reason'][:60]}")
        except Exception as e:
            fail += 1
            print(f"  [{repo}] 失败: {e}", file=sys.stderr)

    store.close()
    print(f"\n导入完成: 新增={ok}, 去重/合并={dup}, 失败={fail}")


def main():
    parser = argparse.ArgumentParser(description="离线 XPU 经验提取")
    parser.add_argument("--log-dir", type=Path, default=Path("runs_v6_phase2check"))
    parser.add_argument("--results", type=Path, default=Path("results/ours_92_final.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/offline_xpu_extracted.jsonl"))
    parser.add_argument("--import-db", action="store_true", help="提取后直接导入数据库")
    args = parser.parse_args()

    from src.xpu.extract_xpu_from_trajs_mvp import load_llm_config_from_env, get_env_or_raise

    cfg = load_llm_config_from_env()
    api_key = get_env_or_raise(cfg["api_key_env_var"])
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    phase2_data = load_phase2_results(args.results)
    print(f"加载 {len(phase2_data)} 个仓库的 Phase 2 结果")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    skip, error, extracted = 0, 0, 0

    repos = list(phase2_data.keys())
    for repo in tqdm(repos, desc="离线提取 XPU"):
        log_path = find_latest_log(args.log_dir, repo)
        if not log_path:
            print(f"  [{repo}] 未找到日志文件，跳过")
            skip += 1
            continue

        try:
            traj = parse_setup_traj_from_log(log_path)
            if not traj:
                print(f"  [{repo}] 日志中未解析出轨迹，跳过")
                skip += 1
                continue

            prosecutor_inv = parse_prosecutor_from_log(log_path)
            p2 = phase2_data[repo]

            xpu_list = extract_one_repo(repo, traj, p2, prosecutor_inv, cfg, api_key, base_url)

            if xpu_list:
                for xpu_obj in xpu_list:
                    rec = {
                        "repository": repo,
                        "llm_decision": "xpu",
                        "xpu": xpu_obj,
                        "phase2_verdict": p2.get("phase2_verdict"),
                        "source_log": str(log_path.name),
                    }
                    all_records.append(rec)
                extracted += len(xpu_list)
                print(f"  [{repo}] 提取 {len(xpu_list)} 条 XPU")
            else:
                print(f"  [{repo}] LLM 决定跳过（无可提炼经验）")

        except Exception as e:
            error += 1
            print(f"  [{repo}] 提取失败: {e}", file=sys.stderr)

    # 写入输出文件
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n提取完成: 共{len(repos)}个仓库, 提取{extracted}条XPU, 跳过{skip}, 失败{error}")
    print(f"输出文件: {args.output}")

    if args.import_db and all_records:
        print("\n开始导入数据库...")
        import_to_db(all_records)


if __name__ == "__main__":
    main()
