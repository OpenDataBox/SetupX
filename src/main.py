"""
命令行入口 — 三阶段流程编排
阶段1: Setup（Agent 推理）
阶段2: Verify（pytest 验证）
阶段3: Report（结果输出）
"""

import json
import sys
from pathlib import Path

from .logger import get_logger
from .agent import SpeculativeSetupAgent
from .verifier_agent import VerifierAgent

logger = get_logger("main")


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

    # ── 阶段2: Verify ──
    logger.info("=== 阶段2: Verify（pytest 验证）===")
    verifier = VerifierAgent(agent.env)
    verify_result = verifier.verify()
    logger.info(
        f"Verify 完成: success={verify_result.success}, "
        f"framework={verify_result.test_framework}, "
        f"collected={verify_result.collect_count}, exit_code={verify_result.exit_code}"
    )

    # ── 阶段3: Report ──
    logger.info("=== 阶段3: Report（结果输出）===")
    report = {
        "repo_url": repo_url,
        "setup": setup_result.to_dict(),
        "verify": verify_result.to_dict(),
    }

    # 写入 JSON 文件
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    safe_name = repo_url.rstrip("/").split("/")[-1]
    output_path = log_dir / f"{safe_name}_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已写入: {output_path}")

    # 屏幕输出摘要
    print(f"\n{'='*50}")
    print(f"仓库: {repo_url}")
    print(f"Setup: {'完成' if setup_result.completed else '未完成'} ({setup_result.steps_taken} 步)")
    print(f"验证框架: {verify_result.test_framework}")
    print(f"测试收集: {verify_result.collect_count} 个")
    print(f"验证结果: {'通过' if verify_result.success else '失败'} (exit_code={verify_result.exit_code})")
    print(f"详细结果: {output_path}")
    print(f"{'='*50}")

    # ── 清理 ──
    agent.env.destroy()
    logger.info("容器已销毁，流程结束")

    return 0 if verify_result.success else 1


if __name__ == "__main__":
    sys.exit(main())
