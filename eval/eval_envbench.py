"""
评估 envBench baseline 结果：用我们的 Prosecutor + Judge 标准重新裁决。

用法:
    .venv/bin/python scripts/eval_envbench.py [--repos repo1,repo2,...] [--workers N]

流程:
    1. 从 envBench/output/scripts.jsonl 读取 bootstrap 脚本
    2. 启动 envbench-python 容器
    3. 容器内 clone 仓库 + 执行 bootstrap 脚本
    4. 用 ProsecutorAgent 调查环境
    5. 用 JudgeAgent 裁决（如果起诉）
    6. 汇总报告
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment_manager import EnvironmentManager
from src.prosecutor_agent import ProsecutorAgent
from src.judge_agent import JudgeAgent
from src.logger import get_logger

logger = get_logger("eval_envbench")

ENVBENCH_IMAGE = "envbench-python:local"
ENVBENCH_DOCKERFILE = Path(__file__).resolve().parent.parent / "envBench" / "dockerfiles" / "python.Dockerfile"
SCRIPTS_JSONL = Path(__file__).resolve().parent.parent / "envBench" / "output" / "scripts.jsonl"
ENVBENCH_EVAL_RESULTS = Path(__file__).resolve().parent.parent / "envBench" / "output" / "100xpu_eval" / "results.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

# 线程安全的结果写入锁
_results_lock = threading.Lock()


def ensure_base_image() -> None:
    """确保 envbench-python 基础镜像存在，不存在则构建"""
    result = subprocess.run(
        ["docker", "images", "-q", ENVBENCH_IMAGE],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"基础镜像已存在: {ENVBENCH_IMAGE}")
        return

    logger.info(f"构建 envbench-python 基础镜像（首次运行，约需 10 分钟）...")
    build_ctx = ENVBENCH_DOCKERFILE.parent
    result = subprocess.run(
        ["docker", "build", "-t", ENVBENCH_IMAGE, "-f", str(ENVBENCH_DOCKERFILE), str(build_ctx)],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        logger.error(f"镜像构建失败:\n{result.stderr[-1000:]}")
        sys.exit(1)
    logger.info("基础镜像构建完成")


def load_scripts() -> dict[str, dict]:
    """加载 scripts.jsonl，返回 {repository: {repository, revision, script}}"""
    data = {}
    with open(SCRIPTS_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data[rec["repository"]] = rec
    return data


def load_envbench_eval() -> dict[str, dict]:
    """加载 envBench 原始评估结果，用于对比"""
    data = {}
    with open(ENVBENCH_EVAL_RESULTS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data[rec["repo_name"]] = rec
    return data


def start_container(image: str) -> str | None:
    """启动容器，返回 container_id"""
    result = subprocess.run(
        ["docker", "run", "-d", image, "sleep", "infinity"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"启动容器失败: {result.stderr}")
        return None
    cid = result.stdout.strip()
    logger.info(f"容器已启动: {cid[:12]}")
    return cid


def stop_container(container_id: str) -> None:
    """停止并删除容器"""
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
    logger.info(f"容器已销毁: {container_id[:12]}")


def exec_in_container(container_id: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    """在容器中执行命令，返回 (exit_code, output)"""
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr
    return result.returncode, output


def build_setup_history_from_script(repo: str, script: str) -> list[dict]:
    """把 bootstrap 脚本转换成伪 setup_history，让 Prosecutor 了解装了什么"""
    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "以下是 envBench Agent 生成的 bootstrap 脚本，展示了环境配置过程",
            "content": {},
        },
        "result": {
            "exit_code": 0,
            "stdout": f"=== envBench Bootstrap Script for {repo} ===\n{script}",
            "stderr": "",
        },
    }]


def build_verify_messages(repo: str, envbench_result: dict | None) -> list[dict]:
    """构造 verify_messages，告知 Prosecutor 背景信息"""
    msgs = [
        {
            "role": "user",
            "content": (
                "【重要】这是 envBench Agent 配置的环境。"
                "envBench 的成功标准是 pyright reportMissingImports 为 0（静态分析无缺失导入），"
                "不要求测试实际执行通过。\n\n"
                "请用我们的标准调查：核心依赖是否可导入？测试是否能实际运行？"
            ),
        },
    ]

    if envbench_result:
        eb_exit = envbench_result.get("exit_code", "?")
        eb_issues = envbench_result.get("issues_count", "?")
        msgs.append({
            "role": "user",
            "content": f"envBench 原始评估: exit_code={eb_exit}, reportMissingImports={eb_issues}",
        })

    return msgs


def _inject_bootstrap_env(container_id: str, env: EnvironmentManager) -> None:
    """从 bootstrap 执行后 dump 的环境变量快照中恢复关键变量。

    bootstrap 脚本通过 `source` 执行，其 conda activate / pyenv global / export PATH
    等效果只存在于那个 shell 里。exec_run 开新 shell 时这些全丢了。
    解决：bootstrap 结束时 `env > /tmp/_bootstrap_env.txt`，这里读回来注入。
    """
    # 需要注入的关键变量（bootstrap 脚本常修改的）
    IMPORTANT_VARS = {
        "PATH", "PYTHONPATH", "PYTHONHOME",
        # pyenv
        "PYENV_VERSION", "PYENV_ROOT", "PYENV_SHELL",
        # conda
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CONDA_EXE",
        "CONDA_PYTHON_EXE", "CONDA_SHLVL",
        # virtualenv / venv
        "VIRTUAL_ENV",
        # poetry
        "POETRY_HOME", "POETRY_VIRTUALENVS_IN_PROJECT",
        # 其他
        "LD_LIBRARY_PATH", "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH",
    }

    try:
        exit_code, output = exec_in_container(
            container_id, "cat /tmp/_bootstrap_env.txt", timeout=10,
        )
        if exit_code != 0:
            logger.warning(f"读取 bootstrap 环境变量失败: {output[:200]}")
            return
    except Exception as e:
        logger.warning(f"读取 bootstrap 环境变量异常: {e}")
        return

    injected = 0
    for line in output.splitlines():
        # env 输出格式: KEY=VALUE（VALUE 可能含 =）
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in IMPORTANT_VARS and value.strip():
            env.set_env(key, value)
            injected += 1
            logger.debug(f"注入环境变量: {key}={value[:80]}")

    logger.info(f"从 bootstrap 环境快照注入 {injected} 个变量")


def evaluate_repo(
    repo: str,
    revision: str,
    script: str,
    envbench_result: dict | None,
) -> dict:
    """评估单个仓库"""
    result = {
        "repo": repo,
        "revision": revision,
        "container_ok": False,
        "script_ok": False,
        "prosecute": None,
        "charges": [],
        "verdict": None,
        "reason": "",
        "envbench_exit_code": envbench_result.get("exit_code") if envbench_result else None,
        "envbench_issues": envbench_result.get("issues_count") if envbench_result else None,
    }

    # 1. 启动容器
    container_id = start_container(ENVBENCH_IMAGE)
    if not container_id:
        result["reason"] = "容器启动失败"
        return result
    result["container_ok"] = True

    try:
        # 2. 在容器内 clone 仓库
        # envBench 容器内路径: /data/project/{org__repo@sha}/
        safe_name = repo.replace("/", "__")
        repo_dir_name = f"{safe_name}@{revision}"
        container_repo_path = f"/data/project/{repo_dir_name}"

        logger.info(f"[{repo}] clone 仓库到 {container_repo_path}")
        clone_cmd = (
            f"mkdir -p {container_repo_path} && "
            f"git clone --depth 1 https://github.com/{repo}.git {container_repo_path} && "
            f"cd {container_repo_path} && git fetch --depth 1 origin {revision} && git checkout {revision}"
        )
        exit_code, output = exec_in_container(container_id, clone_cmd, timeout=300)
        if exit_code != 0:
            # 有些 sha 可能 depth 1 拿不到，尝试完整 clone
            logger.info(f"[{repo}] 浅 clone 失败，尝试完整 clone")
            clone_cmd2 = (
                f"rm -rf {container_repo_path} && "
                f"git clone https://github.com/{repo}.git {container_repo_path} && "
                f"cd {container_repo_path} && git checkout {revision}"
            )
            exit_code, output = exec_in_container(container_id, clone_cmd2, timeout=600)
            if exit_code != 0:
                result["reason"] = f"clone 失败: {output[-500:]}"
                stop_container(container_id)
                return result

        # 3. 执行 bootstrap 脚本
        logger.info(f"[{repo}] 执行 bootstrap 脚本 ({len(script)} chars)")
        # 将脚本写入容器，然后 source 执行
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
            tmp.write(script)
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["docker", "cp", tmp_path, f"{container_id}:{container_repo_path}/bootstrap_script.sh"],
                capture_output=True, check=True,
            )
        finally:
            os.unlink(tmp_path)

        # 用 bash source 执行脚本（和 envBench 的 python_build.sh 一致）
        # set +e 防止单条命令失败导致整体退出
        # 执行完后 dump 环境变量到文件，供检察官继承
        # trap EXIT 确保即使 bootstrap 脚本含 set -e 导致提前退出，env dump 也会执行
        exec_cmd = (
            f"cd {container_repo_path} && trap 'env > /tmp/_bootstrap_env.txt' EXIT && "
            f"source bootstrap_script.sh 2>&1; echo EXIT_CODE=$?"
        )
        exit_code, output = exec_in_container(container_id, exec_cmd, timeout=600)
        # 脚本本身的 exit code 从输出尾部提取
        script_exit = exit_code
        if "EXIT_CODE=" in output:
            try:
                script_exit = int(output.rsplit("EXIT_CODE=", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        result["script_ok"] = True
        logger.info(f"[{repo}] bootstrap 脚本执行完成, exit_code={script_exit}")

        # 4. 接管容器，运行 Prosecutor + Judge
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=container_repo_path)

        # 从 bootstrap 执行后的环境变量快照中恢复关键变量
        # 这样检察官的 exec_run 能继承 conda activate / pyenv global / export PATH 等效果
        _inject_bootstrap_env(container_id, env)

        setup_history = build_setup_history_from_script(repo, script)
        verify_messages = build_verify_messages(repo, envbench_result)

        # 5. 检察官调查
        logger.info(f"[{repo}] 检察官开始调查")
        prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges"] = prosecution.charges
        logger.info(f"[{repo}] 检察官结论: prosecute={prosecution.prosecute}, 指控数={len(prosecution.charges)}")

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reason"] = "检察官未起诉"
        else:
            # 6. 法官裁决
            logger.info(f"[{repo}] 法官开始裁决")
            judgment = JudgeAgent(
                setup_history,
                verify_messages,
                prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment["verdict"]
            result["reason"] = judgment["reasoning"]
            logger.info(f"[{repo}] 法官裁决: {judgment['verdict']}")

    except Exception as e:
        logger.warning(f"[{repo}] 评估异常: {e}")
        result["reason"] = f"评估异常: {e}"
    finally:
        stop_container(container_id)

    return result


def _save_results(results: list[dict], output_path: Path) -> None:
    """线程安全地写入结果文件"""
    with _results_lock:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def _run_one(
    repo: str,
    scripts: dict[str, dict],
    envbench_eval: dict[str, dict],
    results: list[dict],
    output_path: Path,
    total: int,
) -> dict:
    """单个仓库的评估入口（供线程池调用）"""
    rec = scripts[repo]
    eb_result = envbench_eval.get(repo)

    logger.info(f"评估: {repo}")
    try:
        result = evaluate_repo(repo, rec["revision"], rec["script"], eb_result)
    except Exception as e:
        logger.error(f"[{repo}] 未捕获异常: {e}")
        result = {
            "repo": repo, "revision": rec["revision"],
            "container_ok": False, "script_ok": False,
            "prosecute": None, "charges": [], "verdict": None,
            "reason": f"未捕获异常: {e}",
        }

    with _results_lock:
        results.append(result)
        count = len(results)

    _save_results(results, output_path)

    status = ("✅ not_guilty" if result.get("verdict") == "not_guilty" else
              "❌ guilty" if result.get("verdict") == "guilty" else
              "⚠️ " + result.get("reason", "?"))
    eb_info = f" (envBench: exit={result.get('envbench_exit_code')}, issues={result.get('envbench_issues')})"
    print(f"  [{count}/{total}] {repo}: {status}{eb_info}")
    return result


def main():
    parser = argparse.ArgumentParser(description="评估 envBench baseline 结果")
    parser.add_argument("--repos", type=str, default=None,
                        help="逗号分隔的仓库名（org/repo），默认全部")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数（默认 1，建议 Mac 上不超过 3）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径，默认 results/envbench_100_eval.json")
    args = parser.parse_args()

    # 确保基础镜像存在
    ensure_base_image()

    # 加载数据
    scripts = load_scripts()
    envbench_eval = load_envbench_eval()
    logger.info(f"加载 {len(scripts)} 个仓库的 bootstrap 脚本")

    # 确定待评估列表
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",")]
    else:
        repos = sorted(scripts.keys())

    # 输出路径
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "envbench_100_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有结果（断点续跑）
    existing_results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for r in json.load(f):
                    existing_results[r["repo"]] = r
            logger.info(f"已加载 {len(existing_results)} 条历史结果")
        except Exception as e:
            logger.warning(f"加载历史结果失败: {e}")

    results = list(existing_results.values())

    # 过滤出待跑的仓库
    pending = [r for r in repos if r in scripts and r not in existing_results]
    skipped = [r for r in repos if r not in scripts]
    for r in skipped:
        logger.warning(f"[{r}] 未找到 bootstrap 脚本，跳过")

    logger.info(f"待评估 {len(pending)} 个仓库（已有 {len(existing_results)} 条历史结果），workers={args.workers}")

    if args.workers <= 1:
        # 串行
        for repo in pending:
            _run_one(repo, scripts, envbench_eval, results, output_path, len(repos))
    else:
        # 并行
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_one, repo, scripts, envbench_eval, results, output_path, len(repos)): repo
                for repo in pending
            }
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[{repo}] worker 异常: {e}")

    # 最终写入
    _save_results(results, output_path)

    # 打印汇总
    print(f"\n{'='*60}")
    print("评估汇总")
    print(f"{'='*60}")
    total = len(results)
    container_ok = sum(1 for r in results if r.get("container_ok"))
    script_ok = sum(1 for r in results if r.get("script_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    print(f"总数: {total}")
    print(f"容器启动成功: {container_ok}")
    print(f"脚本执行成功: {script_ok}")
    print(f"not_guilty (我们的标准也通过): {not_guilty}")
    print(f"guilty (我们的标准不通过): {guilty}")
    print(f"无裁决: {no_verdict}")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()
