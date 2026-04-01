#!/usr/bin/env python3
"""
复用 claude_code 基础镜像，为单个仓库启动独立容器并运行 Claude Code。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import docker
from dotenv import load_dotenv
from docker.errors import ImageNotFound


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 Docker 中运行 claude code")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--task-file", required=True)
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def get_env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise ValueError(f"缺少环境变量，至少需要配置其中之一: {', '.join(names)}")


def to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_exec_timeout_sec() -> int:
    raw = os.getenv("CLAUDE_CODE_EXEC_TIMEOUT_SEC", "").strip()
    if not raw:
        return 1800
    try:
        value = int(raw)
    except ValueError:
        return 1800
    return value if value > 0 else 1800


def ensure_base_image(client: docker.DockerClient, image_tag: str, rebuild: bool) -> None:
    if not rebuild:
        try:
            client.images.get(image_tag)
            return
        except ImageNotFound:
            pass

    print(f"构建 Claude Code 基础镜像: {image_tag}", file=sys.stderr)
    build_env = os.environ.copy()
    build_env["DOCKER_BUILDKIT"] = "0"
    build_command = [
        "docker",
        "build",
        "--network",
        "host",
        "-t",
        image_tag,
        "--build-arg",
        f"CLAUDE_CODE_CLI_NPM_SPEC={os.getenv('CLAUDE_CODE_CLI_NPM_SPEC', '@anthropic-ai/claude-code')}",
        "--build-arg",
        f"CLAUDE_CODE_NPM_REGISTRY={os.getenv('CLAUDE_CODE_NPM_REGISTRY', 'https://registry.npmjs.org')}",
        str(CLAUDE_DIR),
    ]
    build_result = subprocess.run(
        build_command,
        text=True,
        capture_output=True,
        env=build_env,
        check=False,
    )
    if build_result.stdout:
        print(build_result.stdout, end="", file=sys.stderr)
    if build_result.stderr:
        print(build_result.stderr, end="", file=sys.stderr)
    if build_result.returncode != 0:
        raise RuntimeError(f"构建基础镜像失败，退出码={build_result.returncode}")

    image = client.images.get(image_tag)
    print(f"基础镜像构建完成: {image.id[:12]}", file=sys.stderr)


def run_exec(container_id: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "exec",
        container_id,
        "npm",
        "run",
        "benchmark-internal",
        "--",
        *args,
    ]
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            return_code = process.wait(timeout=get_exec_timeout_sec())
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise RuntimeError(
                f"claude code 执行超时，超过 {get_exec_timeout_sec()} 秒仍未结束"
            ) from exc

        stdout_file.flush()
        stderr_file.flush()
        stdout_file.seek(0)
        stderr_file.seek(0)
        return subprocess.CompletedProcess(
            args=command,
            returncode=return_code,
            stdout=stdout_file.read(),
            stderr=stderr_file.read(),
        )


def collect_container_env() -> dict[str, str]:
    env = {
        "ANTHROPIC_AUTH_TOKEN": get_env_value("ANTHROPIC_AUTH_TOKEN", "OPENCODE_API_KEY"),
        "ANTHROPIC_BASE_URL": get_env_value("ANTHROPIC_BASE_URL", "OPENCODE_BASE_URL"),
        "BENCHMARK_WORKSPACE_DIR": os.getenv("BENCHMARK_WORKSPACE_DIR", "/workspace"),
    }

    model = get_env_value("CLAUDE_CODE_MODEL", "ANTHROPIC_MODEL", "OPENCODE_MODEL")
    env["CLAUDE_CODE_MODEL"] = model
    env["ANTHROPIC_MODEL"] = model

    optional_names = [
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SMALL_FAST_MODEL",
        "CLAUDE_CODE_MAX_TURNS",
        "CLAUDE_CODE_OUTPUT_FORMAT",
    ]
    for name in optional_names:
        value = os.getenv(name, "").strip()
        if value:
            env[name] = value

    if "ANTHROPIC_SMALL_FAST_MODEL" not in env:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = os.getenv("OPENCODE_MODEL", "").strip() or model
    if "CLAUDE_CODE_SMALL_FAST_MODEL" not in env:
        env["CLAUDE_CODE_SMALL_FAST_MODEL"] = env["ANTHROPIC_SMALL_FAST_MODEL"]

    return env


def main() -> int:
    args = parse_args()
    task_prompt = Path(args.task_file).read_text(encoding="utf-8")
    image_tag = os.getenv("CLAUDE_CODE_BASE_IMAGE", "claude-code-benchmark:latest")
    rebuild_image = to_bool(os.getenv("CLAUDE_CODE_REBUILD_IMAGE"), default=False)

    client = docker.from_env()
    ensure_base_image(client, image_tag, rebuild_image)

    container = client.containers.run(
        image_tag,
        command="sleep infinity",
        detach=True,
        working_dir="/runner",
        environment=collect_container_env(),
        network_mode="host",
        user="node",
    )

    try:
        exec_result = run_exec(
            container.id,
            [
                "--repository",
                args.repository,
                "--repo-url",
                args.repo_url,
                "--revision",
                args.revision,
                "--task-prompt",
                task_prompt,
            ],
        )
        if exec_result.stdout:
            print(exec_result.stdout, end="")
        if exec_result.stderr:
            print(exec_result.stderr, end="", file=sys.stderr)
        if exec_result.returncode != 0:
            raise RuntimeError(f"claude code 执行失败，退出码={exec_result.returncode}")

        print(f"container_id={container.id}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        container.remove(force=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
