#!/usr/bin/env python3
"""
复用 qwen_code 基础镜像，为单个仓库启动独立容器并运行 Qwen Code。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import docker
from dotenv import load_dotenv
from docker.errors import ImageNotFound


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 Docker 中运行 qwen code")
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


def to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_base_image(client: docker.DockerClient, image_tag: str, rebuild: bool) -> None:
    if not rebuild:
        try:
            client.images.get(image_tag)
            return
        except ImageNotFound:
            pass

    print(f"构建 Qwen Code 基础镜像: {image_tag}", file=sys.stderr)
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
        f"QWEN_CODE_CLI_NPM_SPEC={os.getenv('QWEN_CODE_CLI_NPM_SPEC', '@qwen-code/qwen-code@latest')}",
        str(QWEN_DIR),
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
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    args = parse_args()
    task_prompt = Path(args.task_file).read_text(encoding="utf-8")
    image_tag = os.getenv(
        "QWEN_CODE_BASE_IMAGE",
        os.getenv("QWEN_CODE_SANDBOX_IMAGE", "qwen-code-benchmark:latest"),
    )
    rebuild_image = to_bool(os.getenv("QWEN_CODE_REBUILD_IMAGE"), default=False)

    env = {
        "QWEN_CODE_API_KEY": require_env("QWEN_CODE_API_KEY"),
        "QWEN_CODE_BASE_URL": require_env("QWEN_CODE_BASE_URL"),
        "QWEN_CODE_MODEL": os.getenv("QWEN_CODE_MODEL", "qwen3-coder-plus"),
    }
    if os.getenv("QWEN_CODE_CLI_PATH"):
        env["QWEN_CODE_CLI_PATH"] = os.getenv("QWEN_CODE_CLI_PATH", "")

    client = docker.from_env()
    ensure_base_image(client, image_tag, rebuild_image)

    container = client.containers.run(
        image_tag,
        command="sleep infinity",
        detach=True,
        working_dir="/runner",
        environment=env,
        network_mode="host",
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
            raise RuntimeError(f"qwen code 执行失败，退出码={exec_result.returncode}")

        print(f"container_id={container.id}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        container.remove(force=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
