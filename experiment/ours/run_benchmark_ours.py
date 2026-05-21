#!/usr/bin/env python3
"""
用 src.main 并行跑 benchmark100，每次最多 N 个仓库并发。
用法:
    python experiment/ours/run_benchmark_ours.py \
        --repo-list data/benchmark100.jsonl \
        --output-dir experiment/results_benchmark100_ours_no_xpu \
        --parallelism 4 \
        [--no-xpu]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-list", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--parallelism", type=int, default=4)
    p.add_argument("--no-xpu", action="store_true")
    p.add_argument("--phase1-timeout", type=int, default=1800)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def run_one(repo: dict, output_dir: Path, no_xpu: bool, phase1_timeout: int) -> dict:
    repo_url = repo.get("repo_url")
    if not repo_url and repo.get("repository"):
        repo_url = f"https://github.com/{repo['repository']}"
    repository = repo.get("repository", repo_url.rstrip("/").split("/")[-1])
    if "/" in repository:
        repository = repository.split("/")[-1]
    started_at = time.time()

    # 每个仓库独立子文件夹
    repo_dir = output_dir / repository
    repo_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "-m", "src.main",
        repo_url,
        "--max-steps", "50",
        "--phase1-timeout", str(phase1_timeout),
        "--output-dir", str(repo_dir),
    ]
    if no_xpu:
        cmd.append("--no-xpu")

    env = os.environ.copy()

    log_path = repo_dir / f"{repository}.log"
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=None,      # 外层不限时，Phase 1 内部已有 signal.alarm
            )
        return_code = proc.returncode
        success = return_code == 0
    except Exception as e:
        return_code = -1
        success = False
        with open(log_path, "a") as f:
            f.write(f"\n[benchmark] 异常: {e}\n")

    duration = time.time() - started_at

    # 读 main.py 写入的 result json
    result_json_path = repo_dir / f"{repository}_result.json"
    phase2_verdict = None
    phase2_reason = None
    if result_json_path.exists():
        try:
            data = json.loads(result_json_path.read_text())
            phase2 = data.get("phase2", {})
            phase2_verdict = "not_guilty" if phase2.get("success") else ("guilty" if phase2.get("success") is False else None)
            phase2_reason = phase2.get("reason", "")
        except Exception:
            pass

    return {
        "repository": repository,
        "repo_url": repo_url,
        "started_at": started_at,
        "duration_sec": duration,
        "return_code": return_code,
        "success": success,
        "phase2_verdict": phase2_verdict,
        "phase2_reason": phase2_reason,
        "log_path": str(repo_dir / f"{repository}.log"),
    }


def main() -> None:
    args = parse_args()

    repos = [json.loads(l) for l in Path(args.repo_list).read_text().splitlines() if l.strip()]
    if args.limit:
        repos = repos[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "raw_results.jsonl"

    # 跳过已有结果的仓库（支持断点续跑）
    done_repos = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                try:
                    done_repos.add(json.loads(line)["repository"])
                except (json.JSONDecodeError, KeyError):
                    pass

    todo = []
    for repo in repos:
        repo_url = repo.get("repo_url")
        if not repo_url and repo.get("repository"):
            repo_url = f"https://github.com/{repo['repository']}"
        repository = repo.get("repository", repo_url.rstrip("/").split("/")[-1])
        if "/" in repository:
            repository = repository.split("/")[-1]
        if repository in done_repos:
            continue
        todo.append(repo)

    if not todo:
        print(f"[benchmark] 所有 {len(repos)} 个仓库已完成，无需重跑")
        return

    print(f"[benchmark] 仓库数={len(repos)}, 跳过={len(repos)-len(todo)}, 待跑={len(todo)}, 并发={args.parallelism}, phase1_timeout={args.phase1_timeout}s")
    print(f"[benchmark] 结果目录: {output_dir}")

    completed = 0
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {
            pool.submit(run_one, repo, output_dir, args.no_xpu, args.phase1_timeout): repo
            for repo in todo
        }
        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            if result["success"]:
                success_count += 1
            with open(results_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            status = "✓" if result["success"] else "✗"
            print(f"[{completed}/{len(todo)}] {status} {result['repository']} ({result['duration_sec']:.0f}s)")

    print(f"\n[benchmark] 完成: 成功={success_count}/{len(todo)}")


if __name__ == "__main__":
    main()
