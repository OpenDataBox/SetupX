"""
评估 ExecAgent 50 repo 完整结果：用 Prosecutor + Judge 重新裁决。

数据来源：execagent_full/runs/top50_xpu_v2_b{001-005}/

用法:
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
    """查找 SETUP_AND_INSTALL.sh：优先 workspace，回退到 files/ 最高版本号"""
    ws_sh = ws_dir / "SETUP_AND_INSTALL.sh"
    if ws_sh.exists():
        return ws_sh

    files_dir = repo_dir / "experimental_setups" / "experiment_1" / "files" / name
    if not files_dir.exists():
        return None

    candidates = sorted(files_dir.glob("SETUP_AND_INSTALL.sh_*"))
    if not candidates:
        return None

    # 按版本号排序，取最大
    def version_key(p: Path) -> int:
        m = re.search(r"_(\d+)$", p.name)
        return int(m.group(1)) if m else 0

    return max(candidates, key=version_key)


def _find_test_results(project: dict) -> str:
    """从 files/ 中找 TEST_RESULTS.txt_N，取版本号最大的"""
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
        content = content[:1500] + "\n...[截断]...\n" + content[-1500:]
    return content


def discover_projects() -> list[dict]:
    """扫描 5 个 batch，返回去重后的项目列表"""
    projects = {}
    for batch in BATCHES:
        batch_dir = DATA_ROOT / batch
        progress = batch_dir / "progress.jsonl"
        if not progress.exists():
            logger.warning(f"未找到 {progress}")
            continue

        for line in progress.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            name = d["project"]
            if name in projects:
                continue  # 去重

            repo_dir = batch_dir / name
            ws_dir = repo_dir / "execution_agent_workspace" / name
            dockerfile = ws_dir / "Dockerfile"

            if not dockerfile.exists():
                logger.warning(f"[{name}] 无 Dockerfile，跳过")
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
    """构建 Docker 镜像（临时上下文避免发送 .git）"""
    image_name = f"execagent50_eval/{project['name']}".lower()

    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"[{project['name']}] 镜像已存在: {image_name}")
        return image_name

    logger.info(f"[{project['name']}] 开始构建镜像: {image_name}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(project["dockerfile"], Path(tmpdir) / "Dockerfile")
            result = subprocess.run(
                ["docker", "build", "-t", image_name, tmpdir],
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


def _detect_shell(container_id: str) -> str:
    """检测容器中可用的 shell（bash 优先，回退 sh）"""
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
    """在容器中找到仓库目录，找不到返回 /"""
    # 常见位置（含 /opt、/home 等）
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

    # 在整个文件系统搜索（排除 /proc /sys）
    exit_code, output = exec_in_container(
        container_id,
        "find / -maxdepth 4 -name '.git' -type d "
        "! -path '/proc/*' ! -path '/sys/*' ! -path '/tmp/*' 2>/dev/null | head -1",
        timeout=30,
    )
    git_dir = output.strip().split("\n")[0].strip()
    if git_dir and git_dir.startswith("/") and git_dir.endswith("/.git"):
        return git_dir[:-5]  # 去掉 /.git

    # 回退到 / 而非 /workspace（/workspace 大概率不存在）
    logger.warning(f"[{project_name}] 未找到 .git 目录，回退到 /")
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
            "thought": "以下是 ExecAgent 生成的 Dockerfile 和安装脚本",
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
            "这是 ExecAgent 配置的环境。"
            "ExecAgent 是一个基于 AutoGPT 的自动化环境配置工具。\n\n"
            "请用我们的标准调查：核心依赖是否可导入？测试是否能实际运行？"
        ),
    }]

    if test_content:
        msgs.append({
            "role": "user",
            "content": f"ExecAgent 的测试输出:\n{test_content}",
        })

    return msgs


def _inject_env_from_container(container_id: str, env: EnvironmentManager, repo_dir: str, name: str) -> None:
    """探测容器中实际可用的 Python 环境（venv/uv/poetry/conda/pyenv），注入 PATH 等变量。

    install.sh 可能在 venv/conda/uv 中安装依赖，但 docker exec 新开 shell 看不到。
    这里先尝试读取 trap EXIT 保存的环境快照，再用容器内探测作为补充。
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

    # 方法 1：读取 trap EXIT 保存的环境快照（install.sh 执行后的最终状态）
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
        # trap 快照中已有 venv/conda PATH，直接用
        for key, value in env_snapshot.items():
            env.set_env(key, value)
        logger.info(f"[{name}] 从 install 环境快照注入 {len(env_snapshot)} 个变量")
        return

    # 方法 2：容器内动态探测（覆盖 uv/poetry/venv/conda/pyenv）
    # repo_dir 中的路径需要转义
    probe_script = f"""
# 1. pyenv（优先级最低，最先激活 → PATH 最后）
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" 2>/dev/null
fi

# 2. conda：激活第一个非 base 环境
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

# 3. uv：查找 uv 管理的 .venv
REPO_DIR="{repo_dir}"
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate" 2>/dev/null
elif command -v uv >/dev/null 2>&1 && [ -f "$REPO_DIR/uv.lock" ]; then
    # uv sync 创建的 .venv 可能在子目录
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

# 5. 常见 venv 目录
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
        logger.warning(f"[{name}] 环境探测异常: {e}")
        # 回退到 trap 快照（即使不含 venv PATH，也比没有好）
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

    logger.info(f"[{name}] 环境探测注入 {injected} 个变量")


def evaluate_repo(project: dict) -> dict:
    """评估单个项目"""
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
        # 3. 执行 SETUP_AND_INSTALL.sh（如果存在）
        #    先 cp 到 /tmp（一定存在），install.sh 里可能自带 git clone
        install_sh = project["install_sh"]
        if install_sh:
            logger.info(f"[{name}] 执行 {install_sh.name}")
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
            logger.info(f"[{name}] SETUP_AND_INSTALL.sh 执行完成, exit_code={exit_code}")
        else:
            result["install_ok"] = True
            logger.info(f"[{name}] 无 SETUP_AND_INSTALL.sh，Dockerfile 已完成安装")

        # 4. install.sh 执行后再找仓库目录（install.sh 可能自带 git clone）
        repo_dir = find_repo_dir(container_id, name)
        logger.info(f"[{name}] 仓库目录: {repo_dir}")

        # 5. 接管容器
        env = EnvironmentManager()
        env.attach(container_id, repo_dir=repo_dir)
        _inject_env_from_container(container_id, env, repo_dir, name)

        setup_history = build_setup_history(project)
        verify_messages = build_verify_messages(project)

        # 6. 检察官调查
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
            # 7. 法官裁决
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
            "repo": name, "batch": project["batch"],
            "repo_url": project["repo_url"],
            "execagent_status": project["execagent_status"],
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
    parser = argparse.ArgumentParser(description="评估 ExecAgent 50 repo 完整结果")
    parser.add_argument("--repos", type=str, default=None,
                        help="逗号分隔的项目名，默认全部")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数")
    parser.add_argument("--only-success", action="store_true",
                        help="只评估 ExecAgent 标记为 SUCCESS 的 repo")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    args = parser.parse_args()

    all_projects = discover_projects()
    logger.info(f"发现 {len(all_projects)} 个项目")

    if args.only_success:
        all_projects = [p for p in all_projects if p["execagent_status"] == "SUCCESS"]
        logger.info(f"--only-success: 筛选后 {len(all_projects)} 个")

    if args.repos:
        selected = {r.strip() for r in args.repos.split(",")}
        all_projects = [p for p in all_projects if p["name"] in selected]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "execagent_full_eval.json"
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
    print("ExecAgent 50 Repo 评估汇总")
    print(f"{'='*60}")
    total = len(results)
    build_ok = sum(1 for r in results if r.get("build_ok"))
    install_ok = sum(1 for r in results if r.get("install_ok"))
    not_guilty = sum(1 for r in results if r.get("verdict") == "not_guilty")
    guilty = sum(1 for r in results if r.get("verdict") == "guilty")
    no_verdict = total - not_guilty - guilty

    success_repos = [r for r in results if r.get("execagent_status") == "SUCCESS"]
    fail_repos = [r for r in results if r.get("execagent_status") == "FAILED"]

    print(f"总数: {total}")
    print(f"构建成功: {build_ok}")
    print(f"安装成功: {install_ok}")
    print(f"not_guilty: {not_guilty}")
    print(f"guilty: {guilty}")
    print(f"无裁决: {no_verdict}")

    if success_repos:
        sg = sum(1 for r in success_repos if r.get("verdict") == "guilty")
        sng = sum(1 for r in success_repos if r.get("verdict") == "not_guilty")
        print(f"\nSUCCESS repos ({len(success_repos)}): guilty={sg}, not_guilty={sng}, 其他={len(success_repos)-sg-sng}")

    if fail_repos:
        fg = sum(1 for r in fail_repos if r.get("verdict") == "guilty")
        fng = sum(1 for r in fail_repos if r.get("verdict") == "not_guilty")
        print(f"FAILED repos ({len(fail_repos)}): guilty={fg}, not_guilty={fng}, 其他={len(fail_repos)-fg-fng}")

    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    main()
