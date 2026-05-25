"""
Evaluate the 33 additional EnvBench repos with Prosecutor + Judge.

Data source: envbench_33/{owner__{repo}@{commit}}/Dockerfile + SETUP_AND_INSTALL.sh

Usage:
    .venv/bin/python scripts/eval_envbench_33.py [--workers N] [--repos a,b,c]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment_manager import EnvironmentManager
from src.prosecutor_agent import ProsecutorAgent
from src.judge_agent import JudgeAgent
from src.logger import get_logger

logger = get_logger("eval_envbench_33")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVBENCH_33_DIR = PROJECT_ROOT / "envbench_33"
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OVERLAP_FILE = PROJECT_ROOT / "data" / "ours_44_overlap.jsonl"

_results_lock = threading.Lock()


def discover_projects() -> list[dict]:
    """Scan the envbench_33 directory and return the list of projects."""
    # Load the 44-repo list for matching
    overlap_repos = set()
    with open(OVERLAP_FILE) as f:
        for line in f:
            d = json.loads(line)
            overlap_repos.add(d["repository"])

    projects = []
    for entry in sorted(ENVBENCH_33_DIR.iterdir()):
        if not entry.is_dir():
            continue
        # Format: owner__repo@commit
        dir_name = entry.name
        if "@" not in dir_name:
            continue
        name_part, commit = dir_name.rsplit("@", 1)
        owner_repo = name_part.replace("__", "/")

        dockerfile = entry / "Dockerfile"
        install_sh = entry / "SETUP_AND_INSTALL.sh"
        if not dockerfile.exists():
            logger.warning(f"[{owner_repo}] no Dockerfile, skipping")
            continue

        repo_name = owner_repo.split("/")[1]
        projects.append({
            "name": repo_name,
            "full": owner_repo,
            "repo_url": f"https://github.com/{owner_repo}",
            "commit": commit,
            "dir": entry,
            "dockerfile": dockerfile,
            "install_sh": install_sh if install_sh.exists() else None,
            "test_results": entry / "TEST_RESULTS.txt" if (entry / "TEST_RESULTS.txt").exists() else None,
        })

    return projects


def build_image(project: dict) -> str | None:
    """Build the Docker image."""
    image_name = f"envbench33_eval/{project['name']}".lower()

    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project['name']}] image already exists: {image_name}")
        return image_name

    logger.info(f"[{project['name']}] starting image build: {image_name}")
    try:
        # The EnvBench Dockerfile's COPY SETUP_AND_INSTALL.sh requires it in the build context
        # Use project['dir'] directly as the build context
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(project["dir"])],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{project['name']}] build timed out (30 minutes)")
        return None

    if result.returncode != 0:
        logger.warning(f"[{project['name']}] build failed:\n{result.stderr[-500:]}")
        return None

    logger.info(f"[{project['name']}] image built successfully")
    return image_name


def start_container(image_name: str) -> str | None:
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
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
    logger.info(f"Container destroyed: {container_id[:12]}")


def exec_in_container(container_id: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def _inject_env_from_container(container_id: str, env: EnvironmentManager, project: dict) -> None:
    """Probe the actually-available Python environment in the container (pyenv/conda/poetry venv) and inject PATH and related variables.

    The EnvBench Dockerfile runs SETUP_AND_INSTALL.sh in a RUN layer, but the RUN layer's
    conda activate / pyenv global / export PATH do not persist into the container runtime.
    Here we dynamically probe inside the container and restore the correct environment.
    """
    IMPORTANT_VARS = {
        "PATH", "PYTHONPATH", "PYTHONHOME",
        "PYENV_VERSION", "PYENV_ROOT", "PYENV_SHELL",
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CONDA_EXE",
        "CONDA_PYTHON_EXE", "CONDA_SHLVL",
        "VIRTUAL_ENV",
        "POETRY_HOME", "POETRY_VIRTUALENVS_IN_PROJECT",
        "LD_LIBRARY_PATH", "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH",
    }

    # Generic probe script: restore the Python environment used at install time
    # Priority: poetry venv > conda non-base env > pyenv (later-activated PATH comes first)
    probe_script = r"""
# 1. pyenv: set up first (lowest priority, placed at the front)
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" 2>/dev/null
fi

# 2. conda: activate the first non-base environment
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
    CONDA_ENV=$(conda env list 2>/dev/null | grep -v '^#' | grep -v '^base ' | grep -v '^\s*$' | head -1 | awk '{print $1}')
    if [ -n "$CONDA_ENV" ]; then
        conda activate "$CONDA_ENV" 2>/dev/null
    fi
fi

# 3. poetry venv: highest priority
if command -v poetry >/dev/null 2>&1; then
    VENV_PATH=$(cd /data/project 2>/dev/null && poetry env info --path 2>/dev/null)
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate" 2>/dev/null
    fi
fi

# 4. Common venv directories
for vdir in /data/project/.venv /data/project/venv /data/project/.tox; do
    if [ -f "$vdir/bin/activate" ]; then
        source "$vdir/bin/activate" 2>/dev/null
        break
    fi
done

env
"""
    try:
        exit_code, output = exec_in_container(container_id, probe_script, timeout=30)
    except Exception as e:
        logger.warning(f"Environment probe error: {e}")
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
            logger.debug(f"Injected environment variable: {key}={value[:80]}")

    logger.info(f"[{project['name']}] environment probe injected {injected} variables")


def find_repo_dir(container_id: str) -> str:
    """EnvBench always clones to /data/project."""
    exit_code, output = exec_in_container(
        container_id, "test -d /data/project/.git && echo FOUND", timeout=10,
    )
    if "FOUND" in output:
        return "/data/project"

    # Fallback search
    exit_code, output = exec_in_container(
        container_id,
        "find / -maxdepth 4 -name '.git' -type d "
        "! -path '/proc/*' ! -path '/sys/*' 2>/dev/null | head -1",
        timeout=15,
    )
    git_dir = output.strip()
    if git_dir and git_dir.startswith("/"):
        return git_dir.replace("/.git", "")

    return "/data/project"


def build_setup_history(project: dict) -> list[dict]:
    dockerfile_content = project["dockerfile"].read_text()
    install_content = ""
    if project["install_sh"]:
        install_content = project["install_sh"].read_text(errors="replace")
        if len(install_content) > 3000:
            install_content = install_content[:1500] + "\n...[truncated]...\n" + install_content[-1500:]

    test_content = ""
    if project["test_results"]:
        test_content = project["test_results"].read_text(errors="replace")
        if len(test_content) > 3000:
            test_content = test_content[:1500] + "\n...[truncated]...\n" + test_content[-1500:]

    context = f"=== EnvBench Dockerfile for {project['full']} ===\n{dockerfile_content}"
    if install_content:
        context += f"\n\n=== SETUP_AND_INSTALL.sh ===\n{install_content}"
    if test_content:
        context += f"\n\n=== EnvBench TEST_RESULTS ===\n{test_content}"

    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "The following is the Dockerfile and install script generated by the EnvBench Agent.",
            "content": {},
        },
        "result": {
            "exit_code": 0,
            "stdout": context,
            "stderr": "",
        },
    }]


def build_verify_messages(project: dict) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "This is an environment configured by the EnvBench Agent. "
            "EnvBench is an LLM-based automated repository environment configuration tool.\n\n"
            "Please investigate using our criteria: are the core dependencies importable? Can the tests actually run?"
        ),
    }]


def evaluate_repo(project: dict) -> dict:
    result = {
        "repo": project["name"],
        "full": project["full"],
        "repo_url": project["repo_url"],
        "build_ok": False,
        "install_ok": False,
        "prosecute": None,
        "charges": [],
        "verdict": None,
        "reason": "",
    }

    name = project["name"]

    # 1. Build the image
    image_name = build_image(project)
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
        # The EnvBench Dockerfile has already completed all installation
        result["install_ok"] = True

        # 3. Locate the repository directory
        repo_dir = find_repo_dir(container_id)
        logger.info(f"[{name}] repository directory: {repo_dir}")

        # 4. Take over the container
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)

        # Restore the pyenv/conda/poetry environment variables from SETUP_AND_INSTALL.sh
        # The EnvBench Dockerfile's RUN-layer environment variables do not persist
        _inject_env_from_container(container_id, env, project)

        setup_history = build_setup_history(project)
        verify_messages = build_verify_messages(project)

        # 5. Prosecutor investigation
        logger.info(f"[{name}] prosecutor starting investigation")
        prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges"] = prosecution.charges
        logger.info(f"[{name}] prosecutor conclusion: prosecute={prosecution.prosecute}, charges={len(prosecution.charges)}")

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reason"] = "Prosecutor did not prosecute"
        else:
            # 6. Judge adjudication
            logger.info(f"[{name}] judge starting adjudication")
            judgment = JudgeAgent(
                setup_history,
                verify_messages,
                prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment["verdict"]
            result["reason"] = judgment["reasoning"]
            logger.info(f"[{name}] judge verdict: {judgment['verdict']}")

    except Exception as e:
        logger.warning(f"[{name}] evaluation error: {e}")
        result["reason"] = f"evaluation error: {e}"
    finally:
        stop_container(container_id)

    return result


def _save_results(results: list[dict], output_path: Path) -> None:
    with _results_lock:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def _run_one(project: dict, results: list[dict], output_path: Path, total: int) -> dict:
    name = project["name"]
    try:
        result = evaluate_repo(project)
    except Exception as e:
        logger.error(f"[{name}] uncaught exception: {e}")
        result = {
            "repo": name, "full": project["full"],
            "repo_url": project["repo_url"],
            "build_ok": False, "install_ok": False,
            "prosecute": None, "charges": [], "verdict": None,
            "reason": f"uncaught exception: {e}",
        }

    with _results_lock:
        results.append(result)
        count = len(results)

    _save_results(results, output_path)

    status = ("not_guilty" if result.get("verdict") == "not_guilty" else
              "guilty" if result.get("verdict") == "guilty" else
              result.get("reason", "?"))
    print(f"  [{count}/{total}] {name}: {status}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the 33 additional EnvBench repos")
    parser.add_argument("--repos", type=str, default=None,
                        help="comma-separated project names; default is all")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                        help="output file path")
    args = parser.parse_args()

    all_projects = discover_projects()
    logger.info(f"Found {len(all_projects)} EnvBench projects")

    if args.repos:
        selected = {r.strip() for r in args.repos.split(",")}
        all_projects = [p for p in all_projects if p["name"] in selected]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "envbench_33_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_results = {}
    if output_path.exists():
        try:
            for r in json.load(open(output_path, encoding="utf-8")):
                existing_results[r["repo"]] = r
            logger.info(f"Loaded {len(existing_results)} historical results")
        except Exception as e:
            logger.warning(f"Failed to load historical results: {e}")

    results = list(existing_results.values())
    pending = [p for p in all_projects if p["name"] not in existing_results]
    logger.info(f"{len(pending)} projects pending ({len(existing_results)} historical results already present), workers={args.workers}")

    if args.workers <= 1:
        for project in pending:
            _run_one(project, results, output_path, len(all_projects))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_one, project, results, output_path, len(all_projects)): project["name"]
                for project in pending
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
    print("EnvBench 33-Repo evaluation summary")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    print(f"Total: {total}")
    print(f"Builds succeeded: {build_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"No verdict: {no_verdict}")
    print(f"Pass rate: {not_guilty}/{total} = {not_guilty/total*100:.0f}%" if total else "N/A")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
