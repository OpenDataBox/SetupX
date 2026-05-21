"""
experiment 专用的 Phase 2 诉讼裁决编排。
"""

from __future__ import annotations

from typing import Any

from src.environment_manager import EnvironmentManager
from src.judge_agent import JudgeAgent
from src.models import ProsecutionResult
from src.prosecutor_agent import ProsecutorAgent
from experiment.trajectory_parser import (
    parse_claude_setup_history,
    parse_opencode_setup_history,
    parse_qwen_setup_history,
)


def serialize_prosecution(prosecution: ProsecutionResult | None) -> dict[str, Any] | None:
    if prosecution is None:
        return None
    return {
        "prosecute": prosecution.prosecute,
        "charges": prosecution.charges,
    }


def run_phase2_review(
    env: EnvironmentManager,
    setup_history: list[dict],
    verify_messages: list[dict],
) -> dict[str, Any]:
    prosecutor = ProsecutorAgent(env, setup_history, verify_messages)
    prosecution = prosecutor.investigate()

    if not prosecution.prosecute:
        judgment = {"verdict": "not_guilty", "reasoning": "检察官未发现实质问题"}
    else:
        judgment = JudgeAgent(
            setup_history=setup_history,
            verify_messages=verify_messages,
            prosecution=prosecution,
            env=env,
        ).rule()

    return {
        "success": judgment.get("verdict") == "not_guilty",
        "reason": judgment.get("reasoning", ""),
        "verdict": judgment.get("verdict"),
        "prosecution": prosecution,
        "prosecution_dict": serialize_prosecution(prosecution),
        "judgment": judgment,
    }


def build_external_tool_setup_history(
    tool_name: str,
    command: str,
    output_text: str,
    return_code: int,
) -> list[dict[str, Any]]:
    history = [
        {
            "step": 1,
            "action": {
                "action_type": "EXTERNAL_TOOL_RUN",
                "thought": f"调用外部工具 {tool_name} 进行仓库配置",
                "content": {
                    "tool": tool_name,
                    "command": command,
                },
            },
            "result": {
                "exit_code": return_code,
                "stdout": output_text[:4000],
                "stderr": "",
            },
        }
    ]

    parsed_entries: list[dict[str, Any]] = []
    if tool_name in {"open_code", "opencode"}:
        parsed_entries = parse_opencode_setup_history(tool_name, output_text)
    elif tool_name in {"claude_code", "claude"}:
        parsed_entries = parse_claude_setup_history(tool_name, output_text)
    elif tool_name in {"qwen_code", "qwen"}:
        parsed_entries = parse_qwen_setup_history(tool_name, output_text)

    for offset, entry in enumerate(parsed_entries, start=2):
        entry["step"] = offset
        history.append(entry)

    return history
