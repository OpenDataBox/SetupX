"""
Re-adjudicate Repo2Run (R2R) baseline outputs with our Prosecutor + Judge.

Data source: per-repo Dockerfile under $R2R_OUTPUT_ROOT/{owner}/{repo}/.
Point $R2R_OUTPUT_ROOT at a local clone of Repo2run-XPU's
output-baseline-no-xpu branch before running.

Usage:
    R2R_OUTPUT_ROOT=/path/to/Repo2run-XPU \\
        python eval/eval_r2r.py [--workers N] [--repos a,b,c]
"""

import argparse
import json
import os
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

logger = get_logger("eval_r2r")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
R2R_ROOT = Path(os.environ.get("R2R_OUTPUT_ROOT", "")).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OVERLAP_FILE = PROJECT_ROOT / "data" / "ours_44_overlap.jsonl"

if not R2R_ROOT or not R2R_ROOT.exists():
    raise SystemExit(
        "eval_r2r.py: set R2R_OUTPUT_ROOT to a local clone of Repo2run-XPU "
        "(branch output-baseline-no-xpu) before running."
    )

_results_lock = threading.Lock()


def discover_projects() -> list[dict]:
    """Load the 44 repos from ours_44_overlap.jsonl and match them to R2R outputs."""
    projects = []
    with open(OVERLAP_FILE) as f:
        for line in f:
            d = json.loads(line)
            full = d["repository"]  # owner/repo
            owner, name = full.split("/")
            repo_dir = R2R_ROOT / owner / name
            if not repo_dir.exists():
                # Try searching other owner directories
                found = False
                for d_name in os.listdir(R2R_ROOT):
                    p = R2R_ROOT / d_name / name
                    if p.is_dir() and (p / "Dockerfile").exists():
                        repo_dir = p
                        owner = d_name
                        found = True
                        break
                if not found:
                    logger.warning(f"[{name}] R2R output does not exist, skipping")
                    continue

            dockerfile = repo_dir / "Dockerfile"
            if not dockerfile.exists():
                logger.warning(f"[{name}] no Dockerfile, skipping")
                continue

            test_txt = repo_dir / "test.txt"
            projects.append({
                "name": name,
                "owner": owner,
                "full": f"{owner}/{name}",
                "repo_url": f"https://github.com/{full}",
                "r2r_dir": repo_dir,
                "dockerfile": dockerfile,
                "test_txt": test_txt if test_txt.exists() else None,
            })

    return projects


def build_image(project: dict) -> str | None:
    """Build the Docker image."""
    image_name = f"r2r_eval/{project['name']}".lower()

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
            # The R2R Dockerfile has COPY code_edit.py / search_patch, so create empty files
            (Path(tmpdir) / "code_edit.py").write_text("")
            (Path(tmpdir) / "search_patch").write_text("")
            # Copy the Dockerfile; the COPY lines can stay (empty files suffice)
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


def exec_in_container(container_id: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def find_repo_dir(container_id: str) -> str:
    """R2R always clones to /repo."""
    exit_code, output = exec_in_container(
        container_id, "test -d /repo/.git && echo FOUND", timeout=10,
    )
    if "FOUND" in output:
        return "/repo"

    # Fallback search
    exit_code, output = exec_in_container(
        container_id,
        "find / -maxdepth 3 -name '.git' -type d "
        "! -path '/proc/*' ! -path '/sys/*' 2>/dev/null | head -1",
        timeout=15,
    )
    git_dir = output.strip()
    if git_dir and git_dir.startswith("/"):
        return git_dir.replace("/.git", "")

    return "/repo"


def build_setup_history(project: dict) -> list[dict]:
    dockerfile_content = project["dockerfile"].read_text()
    test_content = ""
    if project["test_txt"]:
        test_content = project["test_txt"].read_text(errors="replace")
        if len(test_content) > 3000:
            test_content = test_content[:1500] + "\n...[truncated]...\n" + test_content[-1500:]

    context = f"=== Repo2Run Dockerfile for {project['name']} ===\n{dockerfile_content}"
    if test_content:
        context += f"\n\n=== R2R test.txt (list of test cases) ===\n{test_content}"

    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "The following is the Dockerfile and test list generated by Repo2Run (R2R).",
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
            "This is an environment configured by Repo2Run (R2R). "
            "R2R is an LLM-based automated repository environment configuration tool.\n\n"
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
        # R2R needs no extra install.sh; the Dockerfile has completed all installation
        result["install_ok"] = True

        # 3. Locate the repository directory
        repo_dir = find_repo_dir(container_id)
        logger.info(f"[{name}] repository directory: {repo_dir}")

        # 4. Take over the container
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)

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
    parser = argparse.ArgumentParser(description="Evaluate the 44 overlapping R2R repos")
    parser.add_argument("--repos", type=str, default=None,
                        help="comma-separated project names; default is all")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                        help="output file path")
    args = parser.parse_args()

    all_projects = discover_projects()
    logger.info(f"Found {len(all_projects)} R2R projects")

    if args.repos:
        selected = {r.strip() for r in args.repos.split(",")}
        all_projects = [p for p in all_projects if p["name"] in selected]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "r2r_overlap44_eval.json"
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
    print("R2R 44-Repo evaluation summary")
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
