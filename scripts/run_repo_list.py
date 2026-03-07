#!/usr/bin/env python3
"""
批量跑仓库清单（JSONL），支持多 worker 与简单进度条。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行仓库清单")
    parser.add_argument("--list", default="data/python329.jsonl", help="仓库清单 JSONL 路径")
    parser.add_argument("--limit", type=int, default=50, help="最多处理多少条")
    parser.add_argument("--workers", type=int, default=2, help="并发 worker 数")
    parser.add_argument("--max-steps", type=int, default=50, help="单仓库最大迭代步数")
    parser.add_argument("--output-dir", default="runs", help="日志输出目录")
    parser.add_argument("--disable-xpu", action="store_true", help="强制禁用 XPU")
    return parser.parse_args()


def load_repo_list(path: Path, limit: int) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "repository" not in obj:
                continue
            items.append(obj)
            if len(items) >= limit:
                break
    return items


def build_repo_url(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo
    return f"https://github.com/{repo}"


def run_one(
    repo_obj: Dict[str, str],
    max_steps: int,
    output_dir: Path,
    disable_xpu: bool,
) -> Tuple[str, bool, str]:
    repo = repo_obj.get("repository", "unknown/repo")
    revision = repo_obj.get("revision", "HEAD")
    repo_url = build_repo_url(repo)

    safe_name = repo.replace("/", "__")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"{safe_name}@{revision}_{timestamp}.log"

    env = os.environ.copy()
    if disable_xpu:
        env["XPU_DISABLED"] = "true"
    env["LOG_FILE"] = str(log_path)
    env["LOG_FILE_PREFIX"] = f"{safe_name}@{revision}"

    cmd = [sys.executable, "-m", "src.main", repo_url, str(max_steps)]
    with log_path.open("w", encoding="utf-8") as fp:
        result = subprocess.run(cmd, stdout=fp, stderr=fp, env=env)

    return repo, result.returncode == 0, str(log_path)


def format_progress(done: int, total: int, ok: int, fail: int) -> str:
    percent = (done / total * 100.0) if total else 100.0
    bar_len = 24
    filled = int(bar_len * done / total) if total else bar_len
    bar = "#" * filled + "-" * (bar_len - filled)
    return f"[{bar}] {done}/{total} ({percent:5.1f}%) 成功:{ok} 失败:{fail}"


def main() -> int:
    args = parse_args()
    list_path = Path(args.list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repos = load_repo_list(list_path, args.limit)
    total = len(repos)
    if total == 0:
        print("未读取到任何仓库", file=sys.stderr)
        return 1

    lock = threading.Lock()
    done = 0
    ok = 0
    fail = 0
    results: List[Tuple[str, bool, str]] = []

    print("开始执行批量任务...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one, repo, args.max_steps, output_dir, args.disable_xpu)
            for repo in repos
        ]

        for future in as_completed(futures):
            repo, success, log_path = future.result()
            with lock:
                done += 1
                if success:
                    ok += 1
                else:
                    fail += 1
                results.append((repo, success, log_path))
                progress = format_progress(done, total, ok, fail)
                print("\r" + progress, end="", flush=True)

    print()
    print("批量任务完成")
    print(f"成功: {ok}  失败: {fail}  输出目录: {output_dir}")

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for repo, success, log_path in results:
            f.write(json.dumps({
                "repository": repo,
                "success": success,
                "log_path": log_path,
            }, ensure_ascii=False) + "\n")

    print(f"汇总已写入: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
