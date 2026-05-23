#!/usr/bin/env python3
"""
Batch-evaluate the success rate of multiple CLI agents on repository setup tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.phase2_pipeline import build_external_tool_setup_history, run_phase2_review
from src.environment_manager import EnvironmentManager
from src.verifier_agent import VerifierAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CLI repository-setup benchmark")
    parser.add_argument(
        "--tools-config",
        default="experiment/configs/tools.json",
        help="Path to the tools configuration JSON",
    )
    parser.add_argument(
        "--repo-list",
        default="data/python329.jsonl",
        help="Path to the repository list JSONL",
    )
    parser.add_argument(
        "--prompt-file",
        default="experiment/prompts/repo_setup_task.txt",
        help="Path to the task prompt template",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of repositories to evaluate")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds for a single run")
    parser.add_argument(
        "--output-root",
        default="experiment/results",
        help="Root directory for results",
    )
    parser.add_argument(
        "--tool-parallelism",
        type=int,
        default=0,
        help="How many CLI tools to run concurrently per repository; 0 means auto-set to the number of enabled tools",
    )
    return parser.parse_args()


def build_run_dir(output_root: Path) -> Path:
    # Use microseconds to avoid directory collisions when running multiple times within the same second.
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def get_repository_name(repo_obj: dict[str, Any]) -> str:
    repository = str(repo_obj.get("repository") or repo_obj.get("repo") or "").strip()
    if not repository:
        raise KeyError("repository")
    return repository


def get_repository_url(repo_obj: dict[str, Any]) -> str:
    repo_url = str(repo_obj.get("repo_url") or "").strip()
    if repo_url:
        return repo_url
    return build_repo_url(get_repository_name(repo_obj))


def build_repo_url(repository: str) -> str:
    if repository.startswith("http://") or repository.startswith("https://"):
        return repository
    return f"https://github.com/{repository}.git"


def sanitize_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def build_repo_dir(run_dir: Path, tool_name: str, repository: str) -> Path:
    repo_dir = run_dir / sanitize_name(tool_name) / sanitize_name(repository)
    repo_dir.mkdir(parents=True, exist_ok=False)
    return repo_dir


def build_repo_dir_path(run_dir: Path, tool_name: str, repository: str) -> Path:
    return run_dir / sanitize_name(tool_name) / sanitize_name(repository)


def render_template(template: str, mapping: dict[str, str]) -> str:
    try:
        return template.format(**mapping)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Template is missing variable: {missing}") from exc


def judge_success(
    judge_mode: str,
    return_code: int,
    output_text: str,
    success_patterns: list[str],
) -> bool:
    has_pattern = any(pattern in output_text for pattern in success_patterns) if success_patterns else False

    if judge_mode == "return_code":
        return return_code == 0
    if judge_mode == "pattern":
        return has_pattern
    if judge_mode == "return_code_and_pattern":
        return return_code == 0 and has_pattern
    raise ValueError(f"Unsupported judge_mode: {judge_mode}")


def extract_pattern(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Failed to extract {label} from the output, pattern={pattern}")
    if match.groups():
        return match.group(1).strip()
    return match.group(0).strip()

def run_phase2(
    env: EnvironmentManager,
    tool: dict[str, Any],
    command: str,
    output_text: str,
    return_code: int,
    verify_result: Any,
) -> dict[str, Any]:
    if not tool.get("phase2_enabled", True):
        return {
            "phase2_attempted": False,
            "phase2_success": verify_result.success,
            "phase2_verdict": None,
            "phase2_reason": "",
            "prosecution": None,
            "phase2_error": "",
        }

    setup_history = build_external_tool_setup_history(
        tool_name=tool["name"],
        command=command,
        output_text=output_text,
        return_code=return_code,
    )
    phase2_meta = run_phase2_review(
        env=env,
        setup_history=setup_history,
        verify_messages=verify_result.messages,
    )
    return {
        "phase2_attempted": True,
        "phase2_success": phase2_meta["success"],
        "phase2_verdict": phase2_meta["verdict"],
        "phase2_reason": phase2_meta["reason"],
        "prosecution": phase2_meta["prosecution_dict"],
        "phase2_error": "",
    }

def run_verify(tool: dict[str, Any], output_text: str, command: str, return_code: int) -> dict[str, Any]:
    output_mode = tool.get("output_mode", "plain")
    if not tool.get("verify_enabled", False):
        return {
            "verify_attempted": False,
            "verify_success": None,
            "verify_result": None,
            "verify_error": "",
        }

    env: EnvironmentManager | None = None
    cleanup_mode = "none"
    try:
        if output_mode == "container":
            pattern = tool.get("container_id_pattern", "").strip()
            if not pattern:
                raise ValueError("output_mode=container but container_id_pattern is not configured")
            container_id = extract_pattern(pattern, output_text, "container_id")
            work_dir = tool.get("container_work_dir", "/workspace")
            env = EnvironmentManager.from_container(container_id, work_dir=work_dir)
            verify_hint = f"The project root inside the container is {work_dir}; first explore its structure there and do not assume a fixed path."
            cleanup_mode = "destroy" if tool.get("destroy_container_after_verify", False) else "none"
        elif output_mode == "dockerfile":
            pattern = tool.get("dockerfile_dir_pattern", "").strip()
            if not pattern:
                raise ValueError("output_mode=dockerfile but dockerfile_dir_pattern is not configured")
            dockerfile_dir = extract_pattern(pattern, output_text, "dockerfile_dir")
            work_dir = tool.get("dockerfile_work_dir", "/repo")
            env = EnvironmentManager.from_dockerfile(dockerfile_dir, work_dir=work_dir)
            verify_hint = f"The project root inside the container is {work_dir}; first explore its structure there and do not assume a fixed path."
            cleanup_mode = "destroy"
        else:
            raise ValueError(f"output_mode={output_mode} is not supported when verify_enabled=true")

        verifier = VerifierAgent(
            env,
            setup_summary=f"tool={tool['name']}, command={command}. The project root inside the container is {work_dir}.",
            hint=verify_hint,
        )
        verify_result = verifier.verify()
        phase2_meta = run_phase2(
            env=env,
            tool=tool,
            command=command,
            output_text=output_text,
            return_code=return_code,
            verify_result=verify_result,
        )
        return {
            "verify_attempted": True,
            "verify_success": verify_result.success,
            "verify_result": verify_result.to_dict(),
            "verify_error": "",
            **phase2_meta,
        }
    except Exception as exc:
        return {
            "verify_attempted": True,
            "verify_success": False,
            "verify_result": None,
            "verify_error": str(exc),
        }
    finally:
        if env is not None and cleanup_mode == "destroy":
            env.destroy()


def run_one(
    tool: dict[str, Any],
    repo_obj: dict[str, Any],
    prompt_template: str,
    prompt_path: Path,
    repo_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    repository = get_repository_name(repo_obj)
    revision = repo_obj.get("revision", "HEAD")
    repo_url = get_repository_url(repo_obj)
    task_prompt = render_template(
        prompt_template,
        {
            "repository": repository,
            "repo_url": repo_url,
            "revision": revision,
            "task_prompt_path": str(prompt_path),
        },
    )

    command = render_template(
        tool["command_template"],
        {
            "repository": repository,
            "repo_url": repo_url,
            "revision": revision,
            "task_prompt_path": str(prompt_path),
            "task_prompt": task_prompt,
            "repo_dir": str(repo_dir),
        },
    )

    log_path = repo_dir / "run.log"

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        duration_sec = round(time.time() - start_time, 2)
        output_text = (result.stdout or "") + "\n" + (result.stderr or "")
        success = judge_success(
            tool.get("judge_mode", "return_code"),
            result.returncode,
            output_text,
            tool.get("success_patterns", []),
        )
        timeout_hit = False
        return_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        duration_sec = round(time.time() - start_time, 2)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        output_text = stdout + "\n" + stderr + "\n[TIMEOUT]"
        success = False
        timeout_hit = True
        return_code = -1

    verify_meta = run_verify(tool, output_text, command, return_code)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(output_text)

    if verify_meta["verify_attempted"]:
        if verify_meta.get("phase2_attempted"):
            success = verify_meta.get("phase2_success")
        else:
            success = verify_meta["verify_success"]
    else:
        success = judge_success(
            tool.get("judge_mode", "return_code"),
            return_code,
            output_text,
            tool.get("success_patterns", []),
        )

    row = {
        "tool": tool["name"],
        "repository": repository,
        "revision": revision,
        "repo_url": repo_url,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "success": success,
        "timeout": timeout_hit,
        "return_code": return_code,
        "judge_mode": tool.get("judge_mode", "return_code"),
        "log_path": str(log_path),
        "command": command,
        "repo_dir": str(repo_dir),
        "verify_attempted": verify_meta["verify_attempted"],
        "verify_success": verify_meta["verify_success"],
        "verify_result": verify_meta["verify_result"],
        "verify_error": verify_meta["verify_error"],
        "phase2_attempted": verify_meta.get("phase2_attempted", False),
        "phase2_success": verify_meta.get("phase2_success"),
        "phase2_verdict": verify_meta.get("phase2_verdict"),
        "phase2_reason": verify_meta.get("phase2_reason", ""),
        "prosecution": verify_meta.get("prosecution"),
        "phase2_error": verify_meta.get("phase2_error", ""),
    }
    result_path = repo_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    row["result_path"] = str(result_path)
    return row


def build_crash_row(
    tool: dict[str, Any],
    repo_obj: dict[str, Any],
    repo_dir: Path,
    prompt_path: Path,
    started_at: str,
    duration_sec: float,
    error_text: str,
) -> dict[str, Any]:
    repository = get_repository_name(repo_obj)
    revision = repo_obj.get("revision", "HEAD")
    repo_url = get_repository_url(repo_obj)
    command = render_template(
        tool["command_template"],
        {
            "repository": repository,
            "repo_url": repo_url,
            "revision": revision,
            "task_prompt_path": str(prompt_path),
            "task_prompt": "",
            "repo_dir": str(repo_dir),
        },
    )
    return {
        "tool": tool["name"],
        "repository": repository,
        "revision": revision,
        "repo_url": repo_url,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "success": False,
        "timeout": False,
        "return_code": -2,
        "judge_mode": tool.get("judge_mode", "return_code"),
        "log_path": str(repo_dir / "run.log"),
        "command": command,
        "repo_dir": str(repo_dir),
        "verify_attempted": False,
        "verify_success": False,
        "verify_result": None,
        "verify_error": error_text,
        "phase2_attempted": False,
        "phase2_success": False,
        "phase2_verdict": None,
        "phase2_reason": "",
        "prosecution": None,
        "phase2_error": "",
    }


def persist_row_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    repo_dir = Path(row["repo_dir"])
    repo_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(row["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        error_text = row.get("verify_error") or row.get("phase2_error") or "Run exited abnormally"
        log_path.write_text(error_text + "\n", encoding="utf-8")

    result_path = repo_dir / "result.json"
    payload = dict(row)
    payload.pop("result_path", None)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    row["result_path"] = str(result_path)
    return row


def run_one_safe(
    tool: dict[str, Any],
    repo_obj: dict[str, Any],
    prompt_template: str,
    prompt_path: Path,
    repo_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    try:
        return run_one(
            tool=tool,
            repo_obj=repo_obj,
            prompt_template=prompt_template,
            prompt_path=prompt_path,
            repo_dir=repo_dir,
            timeout=timeout,
        )
    except Exception as exc:
        duration_sec = round(time.time() - start_time, 2)
        error_text = (
            f"Run error: {exc}\n"
            f"{traceback.format_exc()}"
        )
        row = build_crash_row(
            tool=tool,
            repo_obj=repo_obj,
            repo_dir=repo_dir,
            prompt_path=prompt_path,
            started_at=started_at,
            duration_sec=duration_sec,
            error_text=error_text,
        )
        return persist_row_artifacts(row)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success = sum(1 for row in rows if row["success"] is True)
    failed = sum(1 for row in rows if row["success"] is False)
    unknown = sum(1 for row in rows if row["success"] is None)
    timeout = sum(1 for row in rows if row["timeout"])
    avg_duration = round(sum(row["duration_sec"] for row in rows) / total, 2) if total else 0.0
    success_rate = round(success / total, 4) if total else 0.0
    resolved = success + failed
    resolved_success_rate = round(success / resolved, 4) if resolved else None
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "unknown": unknown,
        "timeout": timeout,
        "avg_duration_sec": avg_duration,
        "success_rate": success_rate,
        "resolved_success_rate": resolved_success_rate,
    }


def build_tool_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tool"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for tool_name, tool_rows in sorted(grouped.items()):
        summaries.append({"tool": tool_name, **summarize_rows(tool_rows)})
    return summaries


def write_incremental_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    raw_path = run_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tool_summaries = build_tool_summaries(rows)
    for item in tool_summaries:
        tool_summary_path = run_dir / sanitize_name(item["tool"]) / "summary.json"
        with tool_summary_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    run_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "overall": summarize_rows(rows),
        "tools": tool_summaries,
    }
    run_summary_path = run_dir / "summary.json"
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)


def format_status(row: dict[str, Any]) -> str:
    if row["success"] is True:
        return "success"
    if row["success"] is False:
        return "failed"
    return "error/undecided"


def main() -> int:
    args = parse_args()
    tools_config_path = Path(args.tools_config)
    repo_list_path = Path(args.repo_list)
    prompt_path = Path(args.prompt_file)

    if not tools_config_path.exists():
        print(f"Error: tools configuration file not found {tools_config_path}", file=sys.stderr)
        return 1
    if not repo_list_path.exists():
        print(f"Error: repository list not found {repo_list_path}", file=sys.stderr)
        return 1
    if not prompt_path.exists():
        print(f"Error: prompt template not found {prompt_path}", file=sys.stderr)
        return 1

    tools = [tool for tool in read_json(tools_config_path) if tool.get("enabled", False)]
    if not tools:
        print("Error: no tools are enabled; please edit tools.json first", file=sys.stderr)
        return 1

    repos = read_jsonl(repo_list_path, args.limit)
    if not repos:
        print("Error: the repository list is empty", file=sys.stderr)
        return 1

    prompt_template = prompt_path.read_text(encoding="utf-8")

    run_dir = build_run_dir(Path(args.output_root))

    rows: list[dict[str, Any]] = []
    total = len(tools) * len(repos)
    done = 0

    tool_parallelism = args.tool_parallelism or len(tools)
    tool_parallelism = max(1, min(tool_parallelism, len(tools)))

    for tool in tools:
        tool_dir = run_dir / sanitize_name(tool["name"])
        tool_dir.mkdir(parents=True, exist_ok=False)

    print(f"Starting evaluation: {len(repos)} repositories, {len(tools)} tools, {tool_parallelism} tools concurrently per repository")

    try:
        for repo_obj in repos:
            repository = get_repository_name(repo_obj)
            print(f"Starting repository: {repository}")
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
                        args.timeout,
                    )
                    future_map[future] = tool["name"]

                for future in as_completed(future_map):
                    tool_name = future_map[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        repo_dir = build_repo_dir_path(run_dir, tool_name, repository)
                        error_text = f"Concurrent task error: {exc}\n{traceback.format_exc()}"
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

                    done += 1
                    rows.append(row)
                    write_incremental_outputs(run_dir, rows)
                    print(
                        f"[{done}/{total}] {row['tool']} | {row['repository']} | "
                        f"{format_status(row)} | {row['duration_sec']}s"
                    )
    finally:
        write_incremental_outputs(run_dir, rows)

    raw_path = run_dir / "raw_results.jsonl"
    run_summary_path = run_dir / "summary.json"
    print(f"Raw results written to: {raw_path}")
    print(f"Run summary written to: {run_summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
