"""
命令行入口 — 三阶段流程编排
阶段1: Setup（Agent 推理，最多50步）
阶段2: Phase 2 诉讼裁决（仅 Setup 主动 FINISH 时运行）
阶段3: Report（结果输出）
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .logger import get_logger
from .agent import SpeculativeSetupAgent
from .prosecutor_agent import ProsecutorAgent
from .judge_agent import JudgeAgent
from .models import ProsecutionResult

logger = get_logger("main")


def _build_traj_from_history(history: list[dict]) -> list[dict]:
    """将 agent history 转为 JSONL 格式，保留完整信息供 LLM 提取经验"""
    traj = []
    for entry in history:
        action = entry.get("action", {})
        result = entry.get("result", {})

        thought = action.get("thought", "")
        cmd = action.get("content", {}).get("command")
        if cmd:
            parts = []
            if thought:
                parts.append(f"思考: {thought}")
            parts.append(f"```bash\n{cmd}\n```")
            traj.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        if result:
            exit_code = result.get("exit_code", "?")
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            parts = [f"exit_code={exit_code}"]
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            output = "\n".join(parts)
            if output.strip() != f"exit_code={exit_code}":
                traj.append({
                    "role": "system",
                    "content": output,
                })
    return traj


# NOTE: 这函数 _store_xpu_experience 未被调用。
# XPU 经验提取与入库由 scripts/run_xpu_pipeline_from_logs.py 负责。
# def _store_xpu_experience(
#     xpu_client: Any,
#     setup_result: Any,
#     prosecution: "ProsecutionResult | None",
#     judgment: "dict | None",
# ) -> None:
#     """Phase 2 结束后统一触发 XPU 经验提取与入库。
#     completed=True：正常提取，phase2_context 含检察官/法官信号。
#     completed=False（超时）：也提取，phase2_context=None——
#         超时轨迹往往含最有价值的失败模式，不提取是浪费。
#     """
#     from .xpu_client import VectorXPUClient
#     if not isinstance(xpu_client, VectorXPUClient):
#         return
#     if not setup_result.history:
#         logger.debug("[XPU Store] 轨迹为空，跳过经验存储")
#         return
#
#     try:
#         traj = _build_traj_from_history(setup_result.history)
#
#         # 构造 phase2_context
#         phase2_context = None
#         if prosecution is not None:
#             verifier_msgs = setup_result.last_verify_messages or []
#             verifier_text = ""
#             for m in verifier_msgs:
#                 verifier_text += str(m.get("content", ""))
#
#             # 构建检察官调查轨迹摘要
#             prosecutor_investigation = ""
#             if prosecution.messages:
#                 inv_parts = []
#                 for msg in prosecution.messages:
#                     role = msg.get("role", "")
#                     content = msg.get("content", "")
#                     if role == "system":
#                         continue
#                     if role == "assistant":
#                         inv_parts.append(f"[检察官] {content[:400]}")
#                     elif role == "user":
#                         inv_parts.append(f"[取证结果] {content[:400]}")
#                 prosecutor_investigation = "\n".join(inv_parts[-30:])
#
#             phase2_context = {
#                 "prosecution_charges": prosecution.charges,
#                 "verdict": judgment["verdict"] if judgment else None,
#                 "judge_reasoning": judgment["reasoning"] if judgment else "",
#                 "verifier_summary": verifier_text[:1000],
#                 "prosecutor_investigation": prosecutor_investigation[:4000],
#             }
#
#         tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_agent_"))
#         try:
#             repo_path = setup_result.repo_url.rstrip("/")
#             if "github.com/" in repo_path:
#                 repo_path = repo_path.split("github.com/")[-1]
#             safe_name = repo_path.replace("/", "__")
#
#             traj_dir = tmp_dir / "trajs"
#             traj_dir.mkdir()
#             jsonl_path = traj_dir / f"{safe_name}@HEAD.jsonl"
#
#             with open(jsonl_path, "w", encoding="utf-8") as f:
#                 for step in traj:
#                     f.write(json.dumps(step, ensure_ascii=False) + "\n")
#
#             extracted_file = tmp_dir / "extracted.jsonl"
#             from .xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
#             extract_xpu_from_trajs(jsonl_path, extracted_file, phase2_context=phase2_context)
#
#             xpu_objects = []
#             if extracted_file.exists():
#                 with open(extracted_file, "r", encoding="utf-8") as f:
#                     for line in f:
#                         if not line.strip():
#                             continue
#                         rec = json.loads(line)
#                         if rec.get("llm_decision") == "xpu" and rec.get("xpu"):
#                             xpu_objects.append(rec["xpu"])
#
#             if not xpu_objects:
#                 logger.debug("[XPU Store] LLM 决定跳过，无有效经验存储")
#                 return
#
#             logger.info(f"[XPU Store] LLM 提取出 {len(xpu_objects)} 条经验，逐条入库")
#
#             from .xpu.xpu_adapter import XpuEntry, XpuAtom
#             from .xpu.xpu_vector_store import build_xpu_text, text_to_embedding
#             from .xpu.xpu_dedup import dedup_and_store
#
#             for i, xpu_obj in enumerate(xpu_objects):
#                 auto_id = f"xpu_{int(time.time())}_{os.urandom(3).hex()}"
#                 advice = xpu_obj.get("advice_nl", [])
#                 logger.info(f"[XPU Store] 经验[{i+1}] auto_id={auto_id} advice={advice}")
#
#                 atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
#                          for a in xpu_obj.get("atoms", [])]
#                 xpu_entry = XpuEntry(
#                     id=auto_id,
#                     context=xpu_obj.get("context", {}),
#                     signals=xpu_obj.get("signals", {}),
#                     advice_nl=xpu_obj.get("advice_nl", []),
#                     atoms=atoms,
#                 )
#                 text = build_xpu_text(xpu_entry)
#                 embedding = text_to_embedding(text)
#                 dedup_result = dedup_and_store(xpu_client._store, xpu_entry, embedding, use_llm=True)
#                 logger.info(f"[XPU Store] [{i+1}/{len(xpu_objects)}] {dedup_result['action']}: {dedup_result['reason']}")
#
#         finally:
#             shutil.rmtree(tmp_dir, ignore_errors=True)
#
#     except Exception as e:
#         logger.warning(f"[XPU Store] 经验存储失败（不影响任务结果）: {e}")


def main() -> int:
    """主入口函数"""
    if len(sys.argv) < 2:
        print("用法: python -m src.main <git_repo_url> [max_iterations]", file=sys.stderr)
        print("示例: python -m src.main https://github.com/user/repo", file=sys.stderr)
        return 1

    repo_url = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    logger.info("启动三阶段流程")
    logger.info(f"目标仓库: {repo_url}")
    logger.info(f"最大迭代次数: {max_iterations}")

    # ── 阶段1: Setup ──
    logger.info("=== 阶段1: Setup（Agent 推理）===")
    agent = SpeculativeSetupAgent(repo_url, max_iterations)
    setup_result = agent.run()
    logger.info(
        f"Setup 完成: completed={setup_result.completed}, "
        f"steps={setup_result.steps_taken}, container={setup_result.container_id[:12]}"
    )

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    safe_name = repo_url.rstrip("/").split("/")[-1]

    # ── 阶段2: Phase 2 诉讼裁决 ──
    logger.info("=== 阶段2: Phase 2 诉讼裁决 ===")

    phase2_success: bool | None
    phase2_reason: str
    prosecution_dict = None
    prosecution = None
    judgment = None

    try:
        if setup_result.completed:
            logger.info("Setup Agent 主动 FINISH，启动 Phase 2 诉讼模型")

            # 检察官调查
            logger.info("--- 检察官调查阶段 ---")
            prosecutor = ProsecutorAgent(
                agent.env,
                setup_result.history,
                setup_result.last_verify_messages,
            )
            prosecution = prosecutor.investigate()
            prosecution_dict = {
                "prosecute": prosecution.prosecute,
                "charges": prosecution.charges,
            }
            logger.info(f"检察官调查完成: prosecute={prosecution.prosecute}, 指控数={len(prosecution.charges)}")

            if not prosecution.prosecute:
                phase2_success = True
                phase2_reason = "Prosecutor 未发现实质问题"
                logger.info("检察官选择不起诉，直接判定 success=True")
            else:
                # 法官裁决
                logger.info("--- 法官裁决阶段 ---")
                judgment = JudgeAgent(
                    setup_result.history,
                    setup_result.last_verify_messages,
                    prosecution,
                    env=agent.env,
                ).rule()

                verdict = judgment["verdict"]
                phase2_reason = judgment["reasoning"]
                if verdict == "not_guilty":
                    phase2_success = True
                elif verdict == "guilty":
                    phase2_success = False
                else:
                    phase2_success = None
                    phase2_reason = f"[异常] {phase2_reason}"

                logger.info(
                    f"法官裁决: verdict={verdict}, "
                    f"reasoning={phase2_reason[:100]}"
                )
        else:
            phase2_success = False
            phase2_reason = f"Setup Agent 超时（{setup_result.steps_taken} 步），未主动 FINISH"
            logger.info(f"Setup 未完成，跳过 Phase 2: {phase2_reason}")

    except Exception as e:
        logger.error(f"Phase 2 执行异常: {e}")
        phase2_success = None
        phase2_reason = f"[异常] Phase 2 执行出错: {e}"

    finally:
        # ── 清理：关闭客户端 + 销毁容器 ──
        if hasattr(agent._xpu, "close"):
            agent._xpu.close()
        try:
            agent.env.destroy()
            logger.info("容器已销毁")
        except Exception as e:
            logger.warning(f"容器销毁失败: {e}")

    # ── 阶段3: Report ──
    logger.info("=== 阶段3: Report（结果输出）===")
    report = {
        "repo_url": repo_url,
        "setup": setup_result.to_dict(),
        "phase2": {
            "success": phase2_success,
            "reason": phase2_reason,
            "prosecution": prosecution_dict,
            "judgment": judgment,
        },
    }

    output_path = log_dir / f"{safe_name}_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已写入: {output_path}")

    # 屏幕输出摘要
    verdict_str = "通过" if phase2_success is True else ("失败" if phase2_success is False else "异常")
    print(f"\n{'='*50}")
    print(f"仓库: {repo_url}")
    print(f"Setup: {'完成' if setup_result.completed else '未完成'} ({setup_result.steps_taken} 步)")
    print(f"Phase2: {verdict_str}")
    print(f"裁决原因: {phase2_reason}")
    print(f"详细结果: {output_path}")
    print(f"{'='*50}")

    if phase2_success is None:
        return 2  # 异常退出码
    return 0 if phase2_success else 1


if __name__ == "__main__":
    sys.exit(main())
