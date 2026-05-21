"""
评估 EnvBench 补充的 33 个 repo：用 Prosecutor + Judge 裁决。

数据来源：envbench_33/{owner__{repo}@{commit}}/Dockerfile + SETUP_AND_INSTALL.sh

用法:
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
    """扫描 envbench_33 目录，返回项目列表"""
    # 加载 44 repo 列表用于匹配
    overlap_repos = set()
    with open(OVERLAP_FILE) as f:
        for line in f:
            d = json.loads(line)
            overlap_repos.add(d["repository"])

    projects = []
    for entry in sorted(ENVBENCH_33_DIR.iterdir()):
        if not entry.is_dir():
            continue
        # 格式: owner__repo@commit
        dir_name = entry.name
        if "@" not in dir_name:
            continue
        name_part, commit = dir_name.rsplit("@", 1)
        owner_repo = name_part.replace("__", "/")

        dockerfile = entry / "Dockerfile"
        install_sh = entry / "SETUP_AND_INSTALL.sh"
        if not dockerfile.exists():
            logger.warning(f"[{owner_repo}] 无 Dockerfile，跳过")
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
    """构建 Docker 镜像"""
    image_name = f"envbench33_eval/{project['name']}".lower()

    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project['name']}] 镜像已存在: {image_name}")
        return image_name

    logger.info(f"[{project['name']}] 开始构建镜像: {image_name}")
    try:
        # EnvBench Dockerfile 的 COPY SETUP_AND_INSTALL.sh 需要在构建上下文中
        # 直接用 project['dir'] 作为构建上下文
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(project["dir"])],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{project['name']}] 构建超时（30分钟）")
        return None

    if result.returncode != 0:
        logger.warning(f"[{project['name']}] 构建失败:\n{result.stderr[-500:]}")
        return None

    logger.info(f"[{project['name']}] 镜像构建成功")
    return image_name


def start_container(image_name: str) -> str | None:
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
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
    logger.info(f"容器已销毁: {container_id[:12]}")


def exec_in_container(container_id: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def _inject_env_from_container(container_id: str, env: EnvironmentManager, project: dict) -> None:
    """探测容器中实际可用的 Python 环境（pyenv/conda/poetry venv），注入 PATH 等变量。

    EnvBench Dockerfile 在 RUN 层执行 SETUP_AND_INSTALL.sh，但 RUN 层的
    conda activate / pyenv global / export PATH 不会持久化到容器运行时。
    这里在容器中动态探测并恢复正确的环境。
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

    # 通用探测脚本：恢复安装时的 Python 环境
    # 优先级：poetry venv > conda 非 base 环境 > pyenv（后激活的 PATH 在最前面）
    probe_script = r"""
# 1. pyenv: 先设置（优先级最低，放最前面）
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" 2>/dev/null
fi

# 2. conda: 激活第一个非 base 环境
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
    CONDA_ENV=$(conda env list 2>/dev/null | grep -v '^#' | grep -v '^base ' | grep -v '^\s*$' | head -1 | awk '{print $1}')
    if [ -n "$CONDA_ENV" ]; then
        conda activate "$CONDA_ENV" 2>/dev/null
    fi
fi

# 3. poetry venv: 最高优先级
if command -v poetry >/dev/null 2>&1; then
    VENV_PATH=$(cd /data/project 2>/dev/null && poetry env info --path 2>/dev/null)
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate" 2>/dev/null
    fi
fi

# 4. 常见 venv 目录
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
        logger.warning(f"环境探测异常: {e}")
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
            logger.debug(f"注入环境变量: {key}={value[:80]}")

    logger.info(f"[{project['name']}] 环境探测注入 {injected} 个变量")


def find_repo_dir(container_id: str) -> str:
    """EnvBench 统一 clone 到 /data/project"""
    exit_code, output = exec_in_container(
        container_id, "test -d /data/project/.git && echo FOUND", timeout=10,
    )
    if "FOUND" in output:
        return "/data/project"

    # 回退搜索
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
            install_content = install_content[:1500] + "\n...[截断]...\n" + install_content[-1500:]

    test_content = ""
    if project["test_results"]:
        test_content = project["test_results"].read_text(errors="replace")
        if len(test_content) > 3000:
            test_content = test_content[:1500] + "\n...[截断]...\n" + test_content[-1500:]

    context = f"=== EnvBench Dockerfile for {project['full']} ===\n{dockerfile_content}"
    if install_content:
        context += f"\n\n=== SETUP_AND_INSTALL.sh ===\n{install_content}"
    if test_content:
        context += f"\n\n=== EnvBench TEST_RESULTS ===\n{test_content}"

    return [{
        "step": 0,
        "action": {
            "action_type": "CONTEXT",
            "thought": "以下是 EnvBench Agent 生成的 Dockerfile 和安装脚本",
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
            "这是 EnvBench Agent 配置的环境。"
            "EnvBench 是一个基于 LLM 的自动化仓库环境配置工具。\n\n"
            "请用我们的标准调查：核心依赖是否可导入？测试是否能实际运行？"
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

    # 1. 构建镜像
    image_name = build_image(project)
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
        # EnvBench Dockerfile 已完成所有安装
        result["install_ok"] = True

        # 3. 找仓库目录
        repo_dir = find_repo_dir(container_id)
        logger.info(f"[{name}] 仓库目录: {repo_dir}")

        # 4. 接管容器
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)

        # 恢复 SETUP_AND_INSTALL.sh 中的 pyenv/conda/poetry 环境变量
        # EnvBench Dockerfile 的 RUN 层执行完后环境变量不会持久化
        _inject_env_from_container(container_id, env, project)

        setup_history = build_setup_history(project)
        verify_messages = build_verify_messages(project)

        # 5. 检察官调查
        logger.info(f"[{name}] 检察官开始调查")
        prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges"] = prosecution.charges
        logger.info(f"[{name}] 检察官结论: prosecute={prosecution.prosecute}, 指控数={len(prosecution.charges)}")

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reason"] = "检察官未起诉"
        else:
            # 6. 法官裁决
            logger.info(f"[{name}] 法官开始裁决")
            judgment = JudgeAgent(
                setup_history,
                verify_messages,
                prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment["verdict"]
            result["reason"] = judgment["reasoning"]
            logger.info(f"[{name}] 法官裁决: {judgment['verdict']}")

    except Exception as e:
        logger.warning(f"[{name}] 评估异常: {e}")
        result["reason"] = f"评估异常: {e}"
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
        logger.error(f"[{name}] 未捕获异常: {e}")
        result = {
            "repo": name, "full": project["full"],
            "repo_url": project["repo_url"],
            "build_ok": False, "install_ok": False,
            "prosecute": None, "charges": [], "verdict": None,
            "reason": f"未捕获异常: {e}",
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
    parser = argparse.ArgumentParser(description="评估 EnvBench 补充的 33 个 repo")
    parser.add_argument("--repos", type=str, default=None,
                        help="逗号分隔的项目名，默认全部")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    args = parser.parse_args()

    all_projects = discover_projects()
    logger.info(f"发现 {len(all_projects)} 个 EnvBench 项目")

    if args.repos:
        selected = {r.strip() for r in args.repos.split(",")}
        all_projects = [p for p in all_projects if p["name"] in selected]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "envbench_33_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑
    existing_results = {}
    if output_path.exists():
        try:
            for r in json.load(open(output_path, encoding="utf-8")):
                existing_results[r["repo"]] = r
            logger.info(f"已加载 {len(existing_results)} 条历史结果")
        except Exception as e:
            logger.warning(f"加载历史结果失败: {e}")

    results = list(existing_results.values())
    pending = [p for p in all_projects if p["name"] not in existing_results]
    logger.info(f"待评估 {len(pending)} 个项目（已有 {len(existing_results)} 条历史结果），workers={args.workers}")

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
                    logger.error(f"[{name}] worker 异常: {e}")

    _save_results(results, output_path)

    # 汇总
    print(f"\n{'='*60}")
    print("EnvBench 33 Repo 评估汇总")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    print(f"总数: {total}")
    print(f"构建成功: {build_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"无裁决: {no_verdict}")
    print(f"通过率: {not_guilty}/{total} = {not_guilty/total*100:.0f}%" if total else "N/A")
    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    main()
