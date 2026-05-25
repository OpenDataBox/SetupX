"""
Re-adjudicate ExecutionAgent baseline results with our Prosecutor + Judge criteria.

Usage:
    .venv/bin/python scripts/eval_execagent.py [--repos repo1,repo2,...] [--workers N]

Workflow:
    1. Read the Dockerfile (+ install.sh) from ExecutionAgent_success_output/
    2. Build the image with docker build
    3. Start the container with docker run
    4. Run install.sh (if present)
    5. Investigate the environment with ProsecutorAgent
    6. Adjudicate with JudgeAgent (if prosecuted)
    7. Aggregate the report
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment_manager import EnvironmentManager
from src.prosecutor_agent import ProsecutorAgent
from src.judge_agent import JudgeAgent
from src.logger import get_logger

logger = get_logger("eval_execagent")

EXEC_AGENT_OUTPUT = Path(__file__).resolve().parent.parent / "xpu-par" / "ExecutionAgent_success_output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

# Thread-safe lock for writing results
_results_lock = threading.Lock()

# Base-image signatures for Python projects (FROM python:* or FROM ubuntu:*)
PYTHON_BASE_IMAGES = {"python", "ubuntu"}


def find_python_projects() -> dict[str, Path]:
    """Scan ExecutionAgent_success_output/ and return Python projects as {name: dir_path}."""
    projects = {}
    for d in sorted(EXEC_AGENT_OUTPUT.iterdir()):
        if not d.is_dir():
            continue
        dockerfile = d / "Dockerfile"
        if not dockerfile.exists():
            continue
        # Check whether the base image is Python-related
        first_line = dockerfile.read_text().splitlines()[0] if dockerfile.read_text() else ""
        base_image = first_line.replace("FROM ", "").split(":")[0].strip().lower()
        if base_image in PYTHON_BASE_IMAGES:
            projects[d.name] = d
    return projects


def build_image(project_name: str, project_dir: Path) -> str | None:
    """Build the Docker image and return the image name, or None."""
    image_name = f"execagent_eval/{project_name}".lower()

    # Check whether the image already exists
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project_name}] image already exists: {image_name}")
        return image_name

    logger.info(f"[{project_name}] starting image build: {image_name}")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(project_dir)],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{project_name}] build timed out (30 minutes)")
        return None

    if result.returncode != 0:
        logger.warning(f"[{project_name}] build failed:\n{result.stderr[-500:]}")
        return None

    logger.info(f"[{project_name}] image built successfully")
    return image_name


def start_container(image_name: str) -> str | None:
    """Start a container and return its container_id."""
    result = subprocess.run(
        ["docker", "run", "-d", image_name, "sleep", "infinity"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Failed to start container: {result.stderr}")
        return None
    cid = result.stdout.strip()
    logger.info(f"Container started: {cid[:12]}")
    return cid


def stop_container(container_id: str) -> None:
    """Stop and remove the container."""
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
    logger.info(f"Container destroyed: {container_id[:12]}")


def exec_in_container(container_id: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    """Run a command inside the container."""
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def find_repo_dir(container_id: str, project_name: str) -> str | None:
    """Find the repository directory inside the container."""
    # ExecAgent's Dockerfile usually clones to /app/<project_name>
    for candidate in [f"/app/{project_name}", f"/app"]:
        exit_code, output = exec_in_container(
            container_id, f"test -d {candidate}/.git && echo FOUND", timeout=10,
        )
        if "FOUND" in output:
            return candidate

    # Fallback: search under /app
    exit_code, output = exec_in_container(
        container_id, "find /app -maxdepth 2 -name '.git' -type d 2>/dev/null | head -1", timeout=10,
    )
    if output.strip():
        return output.strip().replace("/.git", "")

    return "/app"


def build_setup_history(project_name: str, project_dir: Path) -> list[dict]:
    """Convert the Dockerfile + install.sh into a pseudo setup_history."""
    dockerfile_content = (project_dir / "Dockerfile").read_text()
    install_sh = ""
    if (project_dir / "install.sh").exists():
        install_sh = (project_dir / "install.sh").read_text()
    elif (project_dir / "install_and_test.sh").exists():
        install_sh = (project_dir / "install_and_test.sh").read_text()

    context = f"=== ExecutionAgent Dockerfile for {project_name} ===\n{dockerfile_content}"
    if install_sh:
        context += f"\n\n=== install.sh ===\n{install_sh}"

    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "The following is the Dockerfile and install script generated by ExecutionAgent.",
            "content": {},
        },
        "result": {
            "exit_code": 0,
            "stdout": context,
            "stderr": "",
        },
    }]


def build_verify_messages(project_name: str, project_dir: Path) -> list[dict]:
    """Build verify_messages."""
    # Read test_results.txt (if present)
    test_content = ""
    for name in ["test_results.txt", "test_reuslts.txt"]:  # note: they have a typo
        p = project_dir / name
        if p.exists():
            test_content = p.read_text()
            if len(test_content) > 3000:
                test_content = test_content[:1500] + "\n...[truncated]...\n" + test_content[-1500:]
            break

    msgs = [{
        "role": "user",
        "content": (
            "This is an environment configured by ExecutionAgent. "
            "ExecutionAgent is an AutoGPT-based automated environment configuration tool.\n\n"
            "Please investigate using our criteria: are the core dependencies importable? Can the tests actually run?"
        ),
    }]

    if test_content:
        msgs.append({
            "role": "user",
            "content": f"ExecutionAgent test output:\n{test_content}",
        })

    return msgs


def evaluate_repo(project_name: str, project_dir: Path) -> dict:
    """Evaluate a single project."""
    result = {
        "repo": project_name,
        "build_ok": False,
        "install_ok": False,
        "prosecute": None,
        "charges": [],
        "verdict": None,
        "reason": "",
    }

    # 1. Build the image
    image_name = build_image(project_name, project_dir)
    if not image_name:
        result["reason"] = "Docker build failed"
        return result
    result["build_ok"] = True

    # 2. Start the container
    container_id = start_container(image_name)
    if not container_id:
        result["reason"] = "Failed to start container"
        return result

    try:
        # 3. Find the repository directory
        repo_dir = find_repo_dir(container_id, project_name)
        logger.info(f"[{project_name}] repository directory: {repo_dir}")

        # 4. Run install.sh (if present)
        install_sh = project_dir / "install.sh"
        if not install_sh.exists():
            install_sh = project_dir / "install_and_test.sh"

        if install_sh.exists():
            logger.info(f"[{project_name}] running {install_sh.name}")
            # Copy into the container and run it
            subprocess.run(
                ["docker", "cp", str(install_sh), f"{container_id}:{repo_dir}/{install_sh.name}"],
                capture_output=True, check=True,
            )
            exec_in_container(container_id, f"chmod +x {repo_dir}/{install_sh.name}", timeout=10)
            # Run with source to preserve venv/activate and other environment variables.
            # Key point: install.sh may use set -e, which overrides the outer set +e and kills the whole shell on failure,
            # so the later env dump would never run. Fix: use trap EXIT to guarantee the env dump runs.
            exit_code, output = exec_in_container(
                container_id,
                f"cd {repo_dir} && trap 'env > /tmp/_install_env.txt' EXIT && source {install_sh.name} 2>&1; echo EXIT_CODE=$?",
                timeout=1200,
            )
            result["install_ok"] = True
            logger.info(f"[{project_name}] install.sh finished, exit_code={exit_code}")
        else:
            result["install_ok"] = True  # No install script; the Dockerfile already completed installation
            logger.info(f"[{project_name}] no install.sh; the Dockerfile already completed installation")

        # 5. Take over the container
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)

        # Inject environment variables from after install.sh ran (if any)
        _inject_install_env(container_id, env)

        setup_history = build_setup_history(project_name, project_dir)
        verify_messages = build_verify_messages(project_name, project_dir)

        # 6. Prosecutor investigation
        logger.info(f"[{project_name}] prosecutor starting investigation")
        prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges"] = prosecution.charges
        logger.info(f"[{project_name}] prosecutor conclusion: prosecute={prosecution.prosecute}, charges={len(prosecution.charges)}")

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reason"] = "Prosecutor did not prosecute"
        else:
            # 7. Judge adjudication
            logger.info(f"[{project_name}] judge starting adjudication")
            judgment = JudgeAgent(
                setup_history,
                verify_messages,
                prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment["verdict"]
            result["reason"] = judgment["reasoning"]
            logger.info(f"[{project_name}] judge verdict: {judgment['verdict']}")

    except Exception as e:
        logger.warning(f"[{project_name}] evaluation error: {e}")
        result["reason"] = f"evaluation error: {e}"
    finally:
        stop_container(container_id)

    return result


def _inject_install_env(container_id: str, env: EnvironmentManager) -> None:
    """Inject key variables from the environment snapshot taken after install.sh ran."""
    IMPORTANT_VARS = {
        "PATH", "PYTHONPATH", "PYTHONHOME",
        "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
    }

    try:
        exit_code, output = exec_in_container(
            container_id, "cat /tmp/_install_env.txt", timeout=10,
        )
        if exit_code != 0:
            return
    except Exception:
        return

    injected = 0
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in IMPORTANT_VARS and value.strip():
            env.set_env(key, value)
            injected += 1

    if injected:
        logger.info(f"Injected {injected} variables from the install environment snapshot")


def _save_results(results: list[dict], output_path: Path) -> None:
    """Write the results in a thread-safe way."""
    with _results_lock:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def _run_one(
    project_name: str,
    project_dir: Path,
    results: list[dict],
    output_path: Path,
    total: int,
) -> dict:
    """Evaluation entry point for a single project."""
    try:
        result = evaluate_repo(project_name, project_dir)
    except Exception as e:
        logger.error(f"[{project_name}] uncaught exception: {e}")
        result = {
            "repo": project_name, "build_ok": False, "install_ok": False,
            "prosecute": None, "charges": [], "verdict": None,
            "reason": f"uncaught exception: {e}",
        }

    with _results_lock:
        results.append(result)
        count = len(results)

    _save_results(results, output_path)

    status = ("✅ not_guilty" if result.get("verdict") == "not_guilty" else
              "❌ guilty" if result.get("verdict") == "guilty" else
              "⚠️ " + result.get("reason", "?"))
    print(f"  [{count}/{total}] {project_name}: {status}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Re-adjudicate ExecutionAgent baseline results")
    parser.add_argument("--repos", type=str, default=None,
                        help="comma-separated project names; default is all Python projects")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                        help="output file path")
    args = parser.parse_args()

    # Scan Python projects
    all_projects = find_python_projects()
    logger.info(f"Found {len(all_projects)} Python projects: {list(all_projects.keys())}")

    if args.repos:
        selected = [r.strip() for r in args.repos.split(",")]
        projects = {k: v for k, v in all_projects.items() if k in selected}
    else:
        projects = all_projects

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "execagent_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results (resume support)
    existing_results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for r in json.load(f):
                    existing_results[r["repo"]] = r
            logger.info(f"Loaded {len(existing_results)} historical results")
        except Exception as e:
            logger.warning(f"Failed to load historical results: {e}")

    results = list(existing_results.values())
    pending = {k: v for k, v in projects.items() if k not in existing_results}
    logger.info(f"{len(pending)} projects pending ({len(existing_results)} historical results already present), workers={args.workers}")

    if args.workers <= 1:
        for name, path in pending.items():
            _run_one(name, path, results, output_path, len(projects))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_one, name, path, results, output_path, len(projects)): name
                for name, path in pending.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[{name}] worker error: {e}")

    _save_results(results, output_path)

    # Summary
    print(f"\n{'='*60}")
    print("ExecutionAgent evaluation summary")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    install_ok = sum(1 for r in results if r.get("install_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    print(f"Total: {total}")
    print(f"Builds succeeded: {build_ok}")
    print(f"Installs succeeded: {install_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"No verdict: {no_verdict}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
