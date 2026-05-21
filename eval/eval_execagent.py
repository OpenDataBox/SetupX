"""
评估 ExecutionAgent baseline 结果：用我们的 Prosecutor + Judge 标准重新裁决。

用法:
    .venv/bin/python scripts/eval_execagent.py [--repos repo1,repo2,...] [--workers N]

流程:
    1. 从 ExecutionAgent_success_output/ 读取 Dockerfile（+ install.sh）
    2. docker build 构建镜像
    3. docker run 启动容器
    4. 执行 install.sh（如果存在）
    5. 用 ProsecutorAgent 调查环境
    6. 用 JudgeAgent 裁决（如果起诉）
    7. 汇总报告
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

# 线程安全的结果写入锁
_results_lock = threading.Lock()

# Python 项目的基础镜像特征（FROM python:* 或 FROM ubuntu:*）
PYTHON_BASE_IMAGES = {"python", "ubuntu"}


def find_python_projects() -> dict[str, Path]:
    """扫描 ExecutionAgent_success_output/，返回 Python 项目 {name: dir_path}"""
    projects = {}
    for d in sorted(EXEC_AGENT_OUTPUT.iterdir()):
        if not d.is_dir():
            continue
        dockerfile = d / "Dockerfile"
        if not dockerfile.exists():
            continue
        # 检查基础镜像是否是 Python 相关
        first_line = dockerfile.read_text().splitlines()[0] if dockerfile.read_text() else ""
        base_image = first_line.replace("FROM ", "").split(":")[0].strip().lower()
        if base_image in PYTHON_BASE_IMAGES:
            projects[d.name] = d
    return projects


def build_image(project_name: str, project_dir: Path) -> str | None:
    """构建 Docker 镜像，返回镜像名或 None"""
    image_name = f"execagent_eval/{project_name}".lower()

    # 检查镜像是否已存在
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project_name}] 镜像已存在: {image_name}")
        return image_name

    logger.info(f"[{project_name}] 开始构建镜像: {image_name}")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(project_dir)],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{project_name}] 构建超时（30分钟）")
        return None

    if result.returncode != 0:
        logger.warning(f"[{project_name}] 构建失败:\n{result.stderr[-500:]}")
        return None

    logger.info(f"[{project_name}] 镜像构建成功")
    return image_name


def start_container(image_name: str) -> str | None:
    """启动容器，返回 container_id"""
    result = subprocess.run(
        ["docker", "run", "-d", image_name, "sleep", "infinity"],
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
    """在容器中执行命令"""
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def find_repo_dir(container_id: str, project_name: str) -> str | None:
    """在容器中找到仓库目录"""
    # ExecAgent 的 Dockerfile 一般 clone 到 /app/<project_name>
    for candidate in [f"/app/{project_name}", f"/app"]:
        exit_code, output = exec_in_container(
            container_id, f"test -d {candidate}/.git && echo FOUND", timeout=10,
        )
        if "FOUND" in output:
            return candidate

    # 兜底：在 /app 下搜索
    exit_code, output = exec_in_container(
        container_id, "find /app -maxdepth 2 -name '.git' -type d 2>/dev/null | head -1", timeout=10,
    )
    if output.strip():
        return output.strip().replace("/.git", "")

    return "/app"


def build_setup_history(project_name: str, project_dir: Path) -> list[dict]:
    """把 Dockerfile + install.sh 转换成伪 setup_history"""
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
            "thought": "以下是 ExecutionAgent 生成的 Dockerfile 和安装脚本",
            "content": {},
        },
        "result": {
            "exit_code": 0,
            "stdout": context,
            "stderr": "",
        },
    }]


def build_verify_messages(project_name: str, project_dir: Path) -> list[dict]:
    """构造 verify_messages"""
    # 读取 test_results.txt（如果有）
    test_content = ""
    for name in ["test_results.txt", "test_reuslts.txt"]:  # 注意他们有 typo
        p = project_dir / name
        if p.exists():
            test_content = p.read_text()
            if len(test_content) > 3000:
                test_content = test_content[:1500] + "\n...[截断]...\n" + test_content[-1500:]
            break

    msgs = [{
        "role": "user",
        "content": (
            "这是 ExecutionAgent 配置的环境。"
            "ExecutionAgent 是一个基于 AutoGPT 的自动化环境配置工具。\n\n"
            "请用我们的标准调查：核心依赖是否可导入？测试是否能实际运行？"
        ),
    }]

    if test_content:
        msgs.append({
            "role": "user",
            "content": f"ExecutionAgent 的测试输出:\n{test_content}",
        })

    return msgs


def evaluate_repo(project_name: str, project_dir: Path) -> dict:
    """评估单个项目"""
    result = {
        "repo": project_name,
        "build_ok": False,
        "install_ok": False,
        "prosecute": None,
        "charges": [],
        "verdict": None,
        "reason": "",
    }

    # 1. 构建镜像
    image_name = build_image(project_name, project_dir)
    if not image_name:
        result["reason"] = "Docker 构建失败"
        return result
    result["build_ok"] = True

    # 2. 启动容器
    container_id = start_container(image_name)
    if not container_id:
        result["reason"] = "容器启动失败"
        return result

    try:
        # 3. 找到仓库目录
        repo_dir = find_repo_dir(container_id, project_name)
        logger.info(f"[{project_name}] 仓库目录: {repo_dir}")

        # 4. 执行 install.sh（如果存在）
        install_sh = project_dir / "install.sh"
        if not install_sh.exists():
            install_sh = project_dir / "install_and_test.sh"

        if install_sh.exists():
            logger.info(f"[{project_name}] 执行 {install_sh.name}")
            # 复制到容器内执行
            subprocess.run(
                ["docker", "cp", str(install_sh), f"{container_id}:{repo_dir}/{install_sh.name}"],
                capture_output=True, check=True,
            )
            exec_in_container(container_id, f"chmod +x {repo_dir}/{install_sh.name}", timeout=10)
            # 用 source 执行以保留 venv/activate 等环境变量。
            # 关键：install.sh 可能含 set -e，会覆盖外层 set +e 并在失败时杀掉整个 shell，
            # 导致后面的 env dump 永远不执行。解决：用 trap EXIT 确保 env dump 一定执行。
            exit_code, output = exec_in_container(
                container_id,
                f"cd {repo_dir} && trap 'env > /tmp/_install_env.txt' EXIT && source {install_sh.name} 2>&1; echo EXIT_CODE=$?",
                timeout=1200,
            )
            result["install_ok"] = True
            logger.info(f"[{project_name}] install.sh 执行完成, exit_code={exit_code}")
        else:
            result["install_ok"] = True  # 没有安装脚本，Dockerfile 已完成安装
            logger.info(f"[{project_name}] 无 install.sh，Dockerfile 已完成安装")

        # 5. 接管容器
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)

        # 注入 install.sh 执行后的环境变量（如果有）
        _inject_install_env(container_id, env)

        setup_history = build_setup_history(project_name, project_dir)
        verify_messages = build_verify_messages(project_name, project_dir)

        # 6. 检察官调查
        logger.info(f"[{project_name}] 检察官开始调查")
        prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges"] = prosecution.charges
        logger.info(f"[{project_name}] 检察官结论: prosecute={prosecution.prosecute}, 指控数={len(prosecution.charges)}")

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reason"] = "检察官未起诉"
        else:
            # 7. 法官裁决
            logger.info(f"[{project_name}] 法官开始裁决")
            judgment = JudgeAgent(
                setup_history,
                verify_messages,
                prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment["verdict"]
            result["reason"] = judgment["reasoning"]
            logger.info(f"[{project_name}] 法官裁决: {judgment['verdict']}")

    except Exception as e:
        logger.warning(f"[{project_name}] 评估异常: {e}")
        result["reason"] = f"评估异常: {e}"
    finally:
        stop_container(container_id)

    return result


def _inject_install_env(container_id: str, env: EnvironmentManager) -> None:
    """从 install.sh 执行后的环境变量快照注入关键变量"""
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
        logger.info(f"从 install 环境快照注入 {injected} 个变量")


def _save_results(results: list[dict], output_path: Path) -> None:
    """线程安全写入结果"""
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
    """单个项目评估入口"""
    try:
        result = evaluate_repo(project_name, project_dir)
    except Exception as e:
        logger.error(f"[{project_name}] 未捕获异常: {e}")
        result = {
            "repo": project_name, "build_ok": False, "install_ok": False,
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
    print(f"  [{count}/{total}] {project_name}: {status}")
    return result


def main():
    parser = argparse.ArgumentParser(description="评估 ExecutionAgent baseline 结果")
    parser.add_argument("--repos", type=str, default=None,
                        help="逗号分隔的项目名，默认全部 Python 项目")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    args = parser.parse_args()

    # 扫描 Python 项目
    all_projects = find_python_projects()
    logger.info(f"发现 {len(all_projects)} 个 Python 项目: {list(all_projects.keys())}")

    if args.repos:
        selected = [r.strip() for r in args.repos.split(",")]
        projects = {k: v for k, v in all_projects.items() if k in selected}
    else:
        projects = all_projects

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "execagent_eval.json"
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
    pending = {k: v for k, v in projects.items() if k not in existing_results}
    logger.info(f"待评估 {len(pending)} 个项目（已有 {len(existing_results)} 条历史结果），workers={args.workers}")

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
                    logger.error(f"[{name}] worker 异常: {e}")

    _save_results(results, output_path)

    # 汇总
    print(f"\n{'='*60}")
    print("ExecutionAgent 评估汇总")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    install_ok = sum(1 for r in results if r.get("install_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    print(f"总数: {total}")
    print(f"构建成功: {build_ok}")
    print(f"安装成功: {install_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"无裁决: {no_verdict}")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()
