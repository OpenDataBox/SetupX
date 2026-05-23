"""
Re-adjudicate the full ExecAgent 50-repo results with Prosecutor + Judge.

Data source: execagent_full/runs/top50_xpu_v2_b{001-005}/

Usage:
    .venv/bin/python scripts/eval_execagent_full.py [--workers N] [--only-success] [--repos a,b,c]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment_manager import EnvironmentManager
from src.prosecutor_agent import ProsecutorAgent
from src.judge_agent import JudgeAgent
from src.logger import get_logger

logger = get_logger("eval_execagent_full")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "execagent_full" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
BATCHES = [
    "top50_xpu_v2_b001",
    "top50_xpu_v2_b002",
    "top50_xpu_v2_b003",
    "top50_xpu_v2_b004",
    "top50_xpu_v2_b005",
]

_results_lock = threading.Lock()


def _find_install_sh(repo_dir: Path, name: str, ws_dir: Path) -> Path | None:
    """Find SETUP_AND_INSTALL.sh: prefer the workspace, fall back to the highest version number under files/."""
    ws_sh = ws_dir / "SETUP_AND_INSTALL.sh"
    if ws_sh.exists():
        return ws_sh

    files_dir = repo_dir / "experimental_setups" / "experiment_1" / "files" / name
    if not files_dir.exists():
        return None

    candidates = sorted(files_dir.glob("SETUP_AND_INSTALL.sh_*"))
    if not candidates:
        return None

    # Sort by version number and take the highest
    def version_key(p: Path) -> int:
        m = re.search(r"_(\d+)$", p.name)
        return int(m.group(1)) if m else 0

    return max(candidates, key=version_key)


def _find_test_results(project: dict) -> str:
    """Find TEST_RESULTS.txt_N under files/ and take the one with the highest version number."""
    name = project["name"]
    files_dir = project["repo_dir"] / "experimental_setups" / "experiment_1" / "files" / name
    if not files_dir.exists():
        return ""

    candidates = sorted(files_dir.glob("TEST_RESULTS.txt_*"))
    if not candidates:
        return ""

    def version_key(p: Path) -> int:
        m = re.search(r"_(\d+)$", p.name)
        return int(m.group(1)) if m else 0

    best = max(candidates, key=version_key)
    content = best.read_text(errors="replace")
    if len(content) > 3000:
        content = content[:1500] + "\n...[truncated]...\n" + content[-1500:]
    return content


def discover_projects() -> list[dict]:
    """Scan the 5 batches and return the deduplicated list of projects."""
    projects = {}
    for batch in BATCHES:
        batch_dir = DATA_ROOT / batch
        progress = batch_dir / "progress.jsonl"
        if not progress.exists():
            logger.warning(f"{progress} not found")
            continue

        for line in progress.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            name = d["project"]
            if name in projects:
                continue  # deduplicate

            repo_dir = batch_dir / name
            ws_dir = repo_dir / "execution_agent_workspace" / name
            dockerfile = ws_dir / "Dockerfile"

            if not dockerfile.exists():
                logger.warning(f"[{name}] no Dockerfile, skipping")
                continue

            projects[name] = {
                "name": name,
                "batch": batch,
                "repo_url": d["repo_url"],
                "execagent_status": d["status"],
                "repo_dir": repo_dir,
                "ws_dir": ws_dir,
                "dockerfile": dockerfile,
                "install_sh": _find_install_sh(repo_dir, name, ws_dir),
            }

    return list(projects.values())


def build_image(project: dict) -> str | None:
    """Build the Docker image (use a temporary context to avoid sending .git)."""
    image_name = f"execagent50_eval/{project['name']}".lower()

    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project['name']}] image already exists: {image_name}")
        return image_name

    logger.info(f"[{project['name']}] starting image build: {image_name}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(project["dockerfile"], Path(tmpdir) / "Dockerfile")
            result = subprocess.run(
                ["docker", "build", "-t", image_name, tmpdir],
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


def _detect_shell(container_id: str) -> str:
    """Detect the available shell in the container (prefer bash, fall back to sh)."""
    r = subprocess.run(
        ["docker", "exec", container_id, "sh", "-c", "command -v bash"],
        capture_output=True, text=True, timeout=10,
    )
    return "bash" if r.returncode == 0 else "sh"


def exec_in_container(container_id: str, cmd: str, timeout: int = 600, shell: str = None) -> tuple[int, str]:
    sh = shell or _detect_shell(container_id)
    result = subprocess.run(
        ["docker", "exec", container_id, sh, "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def find_repo_dir(container_id: str, project_name: str) -> str:
    """Find the repository directory in the container; return / if not found."""
    # Common locations (including /opt, /home, etc.)
    candidates = [
        f"/app/{project_name}", "/app",
        f"/workspace/{project_name}", "/workspace",
        f"/opt/{project_name}", f"/home/{project_name}",
        f"/root/{project_name}", f"/{project_name}",
    ]
    for candidate in candidates:
        exit_code, output = exec_in_container(
            container_id, f"test -d {candidate}/.git && echo FOUND", timeout=10,
        )
        if "FOUND" in output:
            return candidate

    # Search the entire filesystem (excluding /proc /sys)
    exit_code, output = exec_in_container(
        container_id,
        "find / -maxdepth 4 -name '.git' -type d "
        "! -path '/proc/*' ! -path '/sys/*' ! -path '/tmp/*' 2>/dev/null | head -1",
        timeout=30,
    )
    git_dir = output.strip().split("\n")[0].strip()
    if git_dir and git_dir.startswith("/") and git_dir.endswith("/.git"):
        return git_dir[:-5]  # strip /.git

    # Fall back to / rather than /workspace (/workspace most likely does not exist)
    logger.warning(f"[{project_name}] no .git directory found, falling back to /")
    return "/"


def build_setup_history(project: dict) -> list[dict]:
    dockerfile_content = project["dockerfile"].read_text()
    install_content = project["install_sh"].read_text() if project["install_sh"] else ""
    test_results = _find_test_results(project)

    context = f"=== ExecAgent Dockerfile for {project['name']} ===\n{dockerfile_content}"
    if install_content:
        context += f"\n\n=== SETUP_AND_INSTALL.sh ===\n{install_content}"
    if test_results:
        context += f"\n\n=== ExecAgent TEST_RESULTS ===\n{test_results}"

    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "The following is the Dockerfile and install script generated by ExecAgent.",
            "content": {},
        },
        "result": {
            "exit_code": 0,
            "stdout": context,
            "stderr": "",
        },
    }]


def build_verify_messages(project: dict) -> list[dict]:
    test_content = _find_test_results(project)

    msgs = [{
        "role": "user",
        "content": (
            "This is an environment configured by ExecAgent. "
            "ExecAgent is an AutoGPT-based automated environment configuration tool.\n\n"
            "Please investigate using our criteria: are the core dependencies importable? Can the tests actually run?"
        ),
    }]

    if test_content:
        msgs.append({
            "role": "user",
            "content": f"ExecAgent test output:\n{test_content}",
        })

    return msgs


def _inject_env_from_container(container_id: str, env: EnvironmentManager, repo_dir: str, name: str) -> None:
    """Probe the actually-available Python environment in the container (venv/uv/poetry/conda/pyenv) and inject PATH and related variables.

    install.sh may install dependencies in a venv/conda/uv, but a fresh docker exec shell cannot see them.
    Here we first try to read the environment snapshot saved by trap EXIT, then use in-container probing as a supplement.
    """
    IMPORTANT_VARS = {
        "PATH", "PYTHONPATH", "PYTHONHOME",
        "PYENV_VERSION", "PYENV_ROOT", "PYENV_SHELL",
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CONDA_EXE",
        "CONDA_PYTHON_EXE", "CONDA_SHLVL",
        "VIRTUAL_ENV",
        "POETRY_HOME", "POETRY_VIRTUALENVS_IN_PROJECT",
        "LD_LIBRARY_PATH", "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH",
        "UV_CACHE_DIR",
    }

    # Method 1: read the environment snapshot saved by trap EXIT (final state after install.sh ran)
    env_snapshot = {}
    try:
        exit_code, output = exec_in_container(
            container_id, "cat /tmp/_install_env.txt", timeout=10,
        )
        if exit_code == 0:
            for line in output.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key in IMPORTANT_VARS and value.strip():
                    env_snapshot[key] = value
    except Exception:
        pass

    if env_snapshot.get("PATH") and (".venv" in env_snapshot["PATH"] or "conda" in env_snapshot["PATH"]):
        # The trap snapshot already has a venv/conda PATH, use it directly
        for key, value in env_snapshot.items():
            env.set_env(key, value)
        logger.info(f"[{name}] injected {len(env_snapshot)} variables from the install environment snapshot")
        return

    # Method 2: dynamic in-container probing (covers uv/poetry/venv/conda/pyenv)
    # The path in repo_dir needs to be escaped
    probe_script = f"""
# 1. pyenv (lowest priority, activated first -> PATH last)
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" 2>/dev/null
fi

# 2. conda: activate the first non-base environment
for conda_sh in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh /root/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh" 2>/dev/null
        CONDA_ENV=$(conda env list 2>/dev/null | grep -v '^#' | grep -v '^base ' | grep -v '^\\s*$' | head -1 | awk '{{print $1}}')
        if [ -n "$CONDA_ENV" ]; then
            conda activate "$CONDA_ENV" 2>/dev/null
        fi
        break
    fi
done

# 3. uv: look for the uv-managed .venv
REPO_DIR="{repo_dir}"
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate" 2>/dev/null
elif command -v uv >/dev/null 2>&1 && [ -f "$REPO_DIR/uv.lock" ]; then
    # The .venv created by uv sync may be in a subdirectory
    for d in "$REPO_DIR" "$REPO_DIR"/*/; do
        if [ -f "$d/.venv/bin/activate" ]; then
            source "$d/.venv/bin/activate" 2>/dev/null
            break
        fi
    done
fi

# 4. poetry venv
if command -v poetry >/dev/null 2>&1; then
    VENV_PATH=$(cd "$REPO_DIR" 2>/dev/null && poetry env info --path 2>/dev/null)
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate" 2>/dev/null
    fi
fi

# 5. Common venv directories
for vdir in "$REPO_DIR/.venv" "$REPO_DIR/venv" "$REPO_DIR/.tox" /app/.venv /app/venv; do
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
        logger.warning(f"[{name}] environment probe error: {e}")
        # Fall back to the trap snapshot (even without a venv PATH, it is better than nothing)
        for key, value in env_snapshot.items():
            env.set_env(key, value)
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

    logger.info(f"[{name}] environment probe injected {injected} variables")


def evaluate_repo(project: dict) -> dict:
    """Evaluate a single project."""
    result = {
        "repo": project["name"],
        "batch": project["batch"],
        "repo_url": project["repo_url"],
        "execagent_status": project["execagent_status"],
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
        # 3. Run SETUP_AND_INSTALL.sh (if present)
        #    First cp to /tmp (always present); install.sh may include its own git clone
        install_sh = project["install_sh"]
        if install_sh:
            logger.info(f"[{name}] running {install_sh.name}")
            subprocess.run(
                ["docker", "cp", str(install_sh), f"{container_id}:/tmp/SETUP_AND_INSTALL.sh"],
                capture_output=True, check=True,
            )
            exec_in_container(container_id, "chmod +x /tmp/SETUP_AND_INSTALL.sh", timeout=10)
            exit_code, output = exec_in_container(
                container_id,
                "trap 'env > /tmp/_install_env.txt' EXIT && "
                "cd / && source /tmp/SETUP_AND_INSTALL.sh 2>&1; echo EXIT_CODE=$?",
                timeout=1200,
            )
            result["install_ok"] = True
            logger.info(f"[{name}] SETUP_AND_INSTALL.sh finished, exit_code={exit_code}")
        else:
            result["install_ok"] = True
            logger.info(f"[{name}] no SETUP_AND_INSTALL.sh; the Dockerfile already completed installation")

        # 4. Find the repository directory after install.sh ran (install.sh may include its own git clone)
        repo_dir = find_repo_dir(container_id, name)
        logger.info(f"[{name}] repository directory: {repo_dir}")

        # 5. Take over the container
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)
        _inject_env_from_container(container_id, env, repo_dir, name)

        setup_history = build_setup_history(project)
        verify_messages = build_verify_messages(project)

        # 6. Prosecutor investigation
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
            # 7. Judge adjudication
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
            "repo": name, "batch": project["batch"],
            "repo_url": project["repo_url"],
            "execagent_status": project["execagent_status"],
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
    parser = argparse.ArgumentParser(description="Re-adjudicate the full ExecAgent 50-repo results")
    parser.add_argument("--repos", type=str, default=None,
                        help="comma-separated project names; default is all")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    parser.add_argument("--only-success", action="store_true",
                        help="evaluate only repos that ExecAgent marked as SUCCESS")
    parser.add_argument("--output", type=str, default=None,
                        help="output file path")
    args = parser.parse_args()

    all_projects = discover_projects()
    logger.info(f"Found {len(all_projects)} projects")

    if args.only_success:
        all_projects = [p for p in all_projects if p["execagent_status"] == "SUCCESS"]
        logger.info(f"--only-success: {len(all_projects)} after filtering")

    if args.repos:
        selected = {r.strip() for r in args.repos.split(",")}
        all_projects = [p for p in all_projects if p["name"] in selected]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "execagent_full_eval.json"
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
    print("ExecAgent 50-Repo evaluation summary")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    install_ok = sum(1 for r in results if r.get("install_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    success_repos = [r for r in results if r.get("execagent_status") == "SUCCESS"]
    fail_repos = [r for r in results if r.get("execagent_status") == "FAILED"]

    print(f"Total: {total}")
    print(f"Builds succeeded: {build_ok}")
    print(f"Installs succeeded: {install_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"No verdict: {no_verdict}")

    if success_repos:
        sg = sum(1 for r in success_repos if r.get("verdict") == "guilty")
        sng = sum(1 for r in success_repos if r.get("verdict") == "not_guilty")
        print(f"\nSUCCESS repos ({len(success_repos)}): guilty={sg}, not_guilty={sng}, other={len(success_repos)-sg-sng}")

    if fail_repos:
        fg = sum(1 for r in fail_repos if r.get("verdict") == "guilty")
        fng = sum(1 for r in fail_repos if r.get("verdict") == "not_guilty")
        print(f"FAILED repos ({len(fail_repos)}): guilty={fg}, not_guilty={fng}, other={len(fail_repos)-fg-fng}")

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
