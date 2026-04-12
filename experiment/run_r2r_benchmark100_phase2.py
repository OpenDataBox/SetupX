#!/usr/bin/env python3
"""
跑 Repo2Run（不带 XPU）部署 benchmark100 的 100 个仓库，
部署完成后用 setUpAgentOurs 的 Phase 2 verifier（Prosecutor + Judge）
在 Repo2Run 产出的 Dockerfile 上做独立审判。

流水线（每仓库串行）：
  1. subprocess 调 Repo2Run main.py（无 --enable_xpu / --online_xpu）
     - Repo2Run 内置 3600s 超时 + 100 步限制
  2. 检查 Repo2Run 是否生成 Dockerfile
     - 不存在：标记 integrate_failed，跳过 phase2
  3. EnvironmentManager.from_dockerfile(...) 构建镜像并起容器
  4. ProsecutorAgent → 调查
  5. JudgeAgent → 裁决
  6. env.cleanup() 删容器和镜像
  7. 汇总到 result.jsonl

约束：
  - Verifier 不设墙钟限制，走默认步数限制
  - Docker 资源回收失败也要继续下一个仓库
  - Repo2Run 和 setUpAgentOurs 分别使用各自的 .env
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# 确保能导入 setUpAgentOurs
SETUP_AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SETUP_AGENT_ROOT))

# 加载 setUpAgentOurs 的 .env（LLM + DB 配置）
from dotenv import load_dotenv
load_dotenv(SETUP_AGENT_ROOT / ".env", override=True)

from src.environment_manager import EnvironmentManager
from src.prosecutor_agent import ProsecutorAgent
from src.judge_agent import JudgeAgent
from src.logger import get_logger

logger = get_logger("r2r_bench100")

REPO2RUN_ROOT = Path("/home/zihang/Repo2Run")
BENCH_FILE = SETUP_AGENT_ROOT / "data" / "benchmark100.jsonl"
P329_FILE = REPO2RUN_ROOT / "python329.jsonl"

# Repo2Run 自带 3600s 守护线程，再额外留 300s 给 subprocess 兜底
R2R_TIMEOUT = 3900
LLM_MODEL = "qwen3.5-plus"
# Repo2Run 要在自己的 Python 环境里跑（依赖 pexpect 等）；
# setUpAgentOurs 的 .venv 不包含这些依赖，必须显式指定解释器。
REPO2RUN_PYTHON = "/home/zihang/miniconda3/envs/torch_gpu/bin/python"


def load_tasks() -> list[dict]:
    """读 benchmark100 + 通过 shortname 匹配 python329 拿 sha"""
    p329_short = {}
    with open(P329_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p329_short[d["repository"].split("/")[-1].lower()] = (
                d["repository"], d["revision"],
            )

    tasks = []
    with open(BENCH_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b = json.loads(line)
            short = b["repo"].lower()
            if short not in p329_short:
                logger.warning(f"benchmark 中 {b['repo']} 在 python329 中缺失 sha，跳过")
                continue
            full, sha = p329_short[short]
            tasks.append({
                "repo": b["repo"],
                "full_name": full,
                "sha": sha,
            })
    return tasks


def run_repo2run(full_name: str, sha: str, r2r_log_path: Path) -> tuple[bool, str]:
    """调 Repo2Run main.py 部署单个仓库。返回 (是否正常退出, 简要说明)"""
    cmd = [
        REPO2RUN_PYTHON, "build_agent/main.py",
        "--full_name", full_name,
        "--sha", sha,
        "--root_path", ".",
        "--llm", LLM_MODEL,
    ]
    logger.info(f"启动 Repo2Run: {' '.join(cmd)}")

    start = time.time()
    try:
        with open(r2r_log_path, "w") as logfp:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO2RUN_ROOT),
                stdout=logfp,
                stderr=subprocess.STDOUT,
                timeout=R2R_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, f"subprocess 超时 ({elapsed:.0f}s > {R2R_TIMEOUT}s)"

    elapsed = time.time() - start
    if proc.returncode == 0:
        return True, f"正常退出 ({elapsed:.0f}s)"
    if proc.returncode == 1:
        # Repo2Run 内部 3600s 守护线程 os._exit(1)
        return False, f"Repo2Run 超时退出 ({elapsed:.0f}s, exit=1)"
    return False, f"Repo2Run 异常退出 (exit={proc.returncode}, {elapsed:.0f}s)"


def dockerfile_dir_for(full_name: str) -> Path:
    author, repo = full_name.split("/")
    return REPO2RUN_ROOT / "output" / author / repo


CODE_EDIT_SRC = REPO2RUN_ROOT / "build_agent" / "tools" / "code_edit.py"


def run_phase2(dockerfile_dir: Path) -> dict:
    """对 Repo2Run 构建的 Dockerfile 做 Prosecutor + Judge 审判。"""
    # Repo2Run 的 Dockerfile 会 COPY code_edit.py 和 search_patch，
    # 需要确保它们在 build context 里
    dst = dockerfile_dir / "code_edit.py"
    if not dst.exists() and CODE_EDIT_SRC.exists():
        shutil.copy2(CODE_EDIT_SRC, dst)
    sp_dst = dockerfile_dir / "search_patch"
    if not sp_dst.exists():
        sp_dst.mkdir(exist_ok=True)  # 空目录占位，COPY 不会报错

    result = {
        "build_ok": False,
        "prosecute": None,
        "charges_count": 0,
        "verdict": None,
        "reasoning": "",
        "error": "",
    }

    env = None
    try:
        env = EnvironmentManager.from_dockerfile(
            str(dockerfile_dir),
            work_dir="/repo",
        )
        result["build_ok"] = True

        prosecutor = ProsecutorAgent(
            env=env,
            setup_history=[],      # 无原始轨迹，交给 Prosecutor 从容器里独立取证
            verify_messages=[],
        )
        prosecution = prosecutor.investigate()
        result["prosecute"] = prosecution.prosecute
        result["charges_count"] = len(prosecution.charges)
        result["charges"] = prosecution.charges

        if not prosecution.prosecute:
            result["verdict"] = "not_guilty"
            result["reasoning"] = "Prosecutor 未发现实质问题"
        else:
            judgment = JudgeAgent(
                setup_history=[],
                verify_messages=[],
                prosecution=prosecution,
                env=env,
            ).rule()
            result["verdict"] = judgment.get("verdict", "error")
            result["reasoning"] = judgment.get("reasoning", "")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"Phase 2 异常: {result['error']}\n{traceback.format_exc()}")
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception as e:
                logger.warning(f"env.cleanup 失败: {e}")

    return result


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = SETUP_AGENT_ROOT / "experiment" / f"r2r_benchmark100_phase2_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "result.jsonl"
    r2r_log_dir = out_root / "r2r_logs"
    r2r_log_dir.mkdir(exist_ok=True)

    logger.info(f"输出目录: {out_root}")

    tasks = load_tasks()
    logger.info(f"待处理仓库数: {len(tasks)}")

    with open(result_path, "w") as rf:
        for idx, task in enumerate(tasks, 1):
            full = task["full_name"]
            sha = task["sha"]
            logger.info(f"\n===== [{idx}/{len(tasks)}] {full} @ {sha[:10]} =====")

            record = {
                "index": idx,
                "repo": task["repo"],
                "full_name": full,
                "sha": sha,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

            # 1. Repo2Run 部署
            r2r_log = r2r_log_dir / f"{full.replace('/', '__')}.log"
            r2r_start = time.time()
            r2r_ok, r2r_note = run_repo2run(full, sha, r2r_log)
            record["r2r_ok"] = r2r_ok
            record["r2r_note"] = r2r_note
            record["r2r_elapsed_sec"] = round(time.time() - r2r_start, 1)

            # 2. 检查 Dockerfile
            df_dir = dockerfile_dir_for(full)
            dockerfile = df_dir / "Dockerfile"
            if not dockerfile.exists():
                record["phase2"] = {
                    "skipped": True,
                    "reason": "Dockerfile 未生成",
                }
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")
                rf.flush()
                logger.warning(f"{full}: Dockerfile 不存在，跳过 phase2")
                continue

            # 3. Phase 2 审判
            phase2_start = time.time()
            phase2 = run_phase2(df_dir)
            phase2["elapsed_sec"] = round(time.time() - phase2_start, 1)
            phase2["skipped"] = False
            record["phase2"] = phase2

            rf.write(json.dumps(record, ensure_ascii=False) + "\n")
            rf.flush()

            logger.info(
                f"{full}: r2r={r2r_note} | phase2_verdict={phase2.get('verdict')} "
                f"(prosecute={phase2.get('prosecute')}, charges={phase2.get('charges_count')}) "
                f"phase2_elapsed={phase2['elapsed_sec']}s"
            )

    # 汇总
    logger.info(f"\n全部完成，汇总到 {result_path}")
    summarize(result_path)


def summarize(result_path: Path) -> None:
    counts = {"not_guilty": 0, "guilty": 0, "error": 0, "skipped": 0, "total": 0}
    with open(result_path) as f:
        for line in f:
            r = json.loads(line)
            counts["total"] += 1
            p2 = r.get("phase2", {})
            if p2.get("skipped"):
                counts["skipped"] += 1
                continue
            v = p2.get("verdict") or "error"
            counts[v] = counts.get(v, 0) + 1
    logger.info(f"汇总: {counts}")


if __name__ == "__main__":
    main()
