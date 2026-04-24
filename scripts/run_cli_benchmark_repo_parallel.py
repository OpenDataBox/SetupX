#!/usr/bin/env python3
"""
多仓库并发运行 CLI benchmark。

复用 experiment/run_cli_benchmark.py 里的单工具执行、verify、phase2 与结果汇总逻辑，
仅增加“多个仓库可同时处理”的调度层。
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.run_cli_benchmark import (
    build_crash_row,
    build_repo_dir,
    build_repo_dir_path,
    build_run_dir,
    format_status,
    get_repository_name,
    persist_row_artifacts,
    read_json,
    read_jsonl,
    run_one_safe,
    sanitize_name,
    write_incremental_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多仓库并发运行 CLI 仓库配置基准测试")
    parser.add_argument(
        "--tools-config",
        default="experiment/configs/tools.json",
        help="工具配置 JSON 路径",
    )
    parser.add_argument(
        "--repo-list",
        default="data/python329.jsonl",
        help="仓库清单 JSONL 路径",
    )
    parser.add_argument(
        "--prompt-file",
        default="experiment/prompts/repo_setup_task.txt",
        help="任务提示词模板路径",
    )
    parser.add_argument("--limit", type=int, default=10, help="最多评测多少个仓库")
    parser.add_argument("--timeout", type=int, default=3600, help="单次运行超时秒数")
    parser.add_argument(
        "--output-root",
        default="experiment/results",
        help="结果根目录",
    )
    parser.add_argument(
        "--tool-parallelism",
        type=int,
        default=0,
        help="每个仓库同时运行多少个 CLI 工具，0 表示按启用工具数自动设置",
    )
    parser.add_argument(
        "--repo-parallelism",
        type=int,
        default=1,
        help="同时处理多少个仓库",
    )
    return parser.parse_args()


def run_repo_tools(
    repo_obj: dict[str, Any],
    tools: list[dict[str, Any]],
    prompt_template: str,
    prompt_path: Path,
    run_dir: Path,
    timeout: int,
    tool_parallelism: int,
) -> list[dict[str, Any]]:
    repository = get_repository_name(repo_obj)
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=tool_parallelism) as executor:
        future_map = {}
        for tool in tools:
            repo_dir = build_repo_dir(run_dir, tool["name"], repository)
            future = executor.submit(
                run_one_safe,
                tool,
                repo_obj,
                prompt_template,
                prompt_path,
                repo_dir,
                timeout,
            )
            future_map[future] = tool["name"]

        for future in as_completed(future_map):
            tool_name = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                repo_dir = build_repo_dir_path(run_dir, tool_name, repository)
                error_text = f"并发任务异常: {exc}\n{traceback.format_exc()}"
                row = persist_row_artifacts(
                    build_crash_row(
                        tool=next(item for item in tools if item["name"] == tool_name),
                        repo_obj=repo_obj,
                        repo_dir=repo_dir,
                        prompt_path=prompt_path,
                        started_at=datetime.now().isoformat(timespec="seconds"),
                        duration_sec=0.0,
                        error_text=error_text,
                    )
                )
            rows.append(row)

    return rows


def main() -> int:
    args = parse_args()
    tools_config_path = Path(args.tools_config)
    repo_list_path = Path(args.repo_list)
    prompt_path = Path(args.prompt_file)

    if not tools_config_path.exists():
        print(f"错误: 找不到工具配置文件 {tools_config_path}", file=sys.stderr)
        return 1
    if not repo_list_path.exists():
        print(f"错误: 找不到仓库清单 {repo_list_path}", file=sys.stderr)
        return 1
    if not prompt_path.exists():
        print(f"错误: 找不到提示词模板 {prompt_path}", file=sys.stderr)
        return 1
    if args.repo_parallelism < 1:
        print("错误: --repo-parallelism 必须 >= 1", file=sys.stderr)
        return 1

    tools = [tool for tool in read_json(tools_config_path) if tool.get("enabled", False)]
    if not tools:
        print("错误: 没有启用任何工具，请先修改 tools 配置", file=sys.stderr)
        return 1

    repos = read_jsonl(repo_list_path, args.limit)
    if not repos:
        print("错误: 仓库清单为空", file=sys.stderr)
        return 1

    prompt_template = prompt_path.read_text(encoding="utf-8")
    run_dir = build_run_dir(Path(args.output_root))

    tool_parallelism = args.tool_parallelism or len(tools)
    tool_parallelism = max(1, min(tool_parallelism, len(tools)))
    repo_parallelism = min(args.repo_parallelism, len(repos))

    for tool in tools:
        tool_dir = run_dir / sanitize_name(tool["name"])
        tool_dir.mkdir(parents=True, exist_ok=False)

    total = len(tools) * len(repos)
    done = 0
    all_rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()

    print(
        f"开始评测，共 {len(repos)} 个仓库，{len(tools)} 个工具，"
        f"仓库并发 {repo_parallelism}，每仓库工具并发 {tool_parallelism}"
    )

    with ThreadPoolExecutor(max_workers=repo_parallelism) as repo_executor:
        future_map = {
            repo_executor.submit(
                run_repo_tools,
                repo_obj,
                tools,
                prompt_template,
                prompt_path,
                run_dir,
                args.timeout,
                tool_parallelism,
            ): get_repository_name(repo_obj)
            for repo_obj in repos
        }

        for future in as_completed(future_map):
            repository = future_map[future]
            try:
                repo_rows = future.result()
            except Exception as exc:
                print(f"仓库任务异常: {repository}: {exc}", file=sys.stderr)
                continue

            with rows_lock:
                for row in repo_rows:
                    done += 1
                    all_rows.append(row)
                    write_incremental_outputs(run_dir, all_rows)
                    print(
                        f"[{done}/{total}] {row['tool']} | {row['repository']} | "
                        f"{format_status(row)} | {row['duration_sec']}s"
                    )

    write_incremental_outputs(run_dir, all_rows)
    print(f"原始结果已写入: {run_dir / 'raw_results.jsonl'}")
    print(f"run 汇总已写入: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
