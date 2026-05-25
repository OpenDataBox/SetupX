"""
Parse external CLI tool logs into a setup_history that Phase 2 can consume.
"""

from __future__ import annotations

import json
from typing import Any


def _iter_json_lines(output_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


def _collect_opencode_parts(parts: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    reasoning: list[str] = []
    text_blocks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        part_type = str(part.get("type", ""))
        if part_type == "reasoning":
            text = str(part.get("text", "")).strip()
            if text:
                reasoning.append(text)
        elif part_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                text_blocks.append(text)
        elif part_type == "tool":
            state = part.get("state") or {}
            state_status = state.get("status")
            metadata = state.get("metadata") or {}
            output = ""
            if state_status == "completed":
                output = str(state.get("output", "")).strip()
            if not output:
                output = str(state.get("raw", "")).strip()
            tool_calls.append({
                "tool": part.get("tool"),
                "call_id": part.get("callID"),
                "status": state_status,
                "title": str(state.get("title", "")).strip()[:500],
                "metadata": metadata,
                "output": output[:2000],
            })

    return "\n\n".join(reasoning), "\n\n".join(text_blocks), tool_calls


def _build_entry(
    step: int,
    action_type: str,
    thought: str,
    content: dict[str, Any],
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
) -> dict[str, Any]:
    return {
        "step": step,
        "action": {
            "action_type": action_type,
            "thought": thought[:1000],
            "content": content,
        },
        "result": {
            "exit_code": exit_code,
            "stdout": stdout[:4000],
            "stderr": stderr[:4000],
        },
    }


def parse_opencode_setup_history(tool_name: str, output_text: str) -> list[dict[str, Any]]:
    """Parse session_message/result/command_executed events from the newer OpenCode logs."""
    entries: list[dict[str, Any]] = []
    step = 1

    for event in _iter_json_lines(output_text):
        event_type = str(event.get("type", ""))

        if event_type == "session_message" and event.get("role") == "assistant":
            parts = event.get("parts") or []
            if not isinstance(parts, list):
                continue

            thought, text_content, tool_calls = _collect_opencode_parts(parts)
            if not (thought or text_content or tool_calls):
                continue

            action_type = "EXTERNAL_TOOL_CALL" if tool_calls else "EXTERNAL_TOOL_MESSAGE"
            stdout_parts: list[str] = []
            if text_content:
                stdout_parts.append(text_content[:4000])
            if tool_calls:
                stdout_parts.append(json.dumps(tool_calls, ensure_ascii=False)[:2000])

            entries.append(_build_entry(
                step=step,
                action_type=action_type,
                thought=thought,
                content={
                    "tool": tool_name,
                    "session_id": event.get("session_id"),
                    "message_id": event.get("message_id"),
                    "tool_calls": tool_calls,
                },
                stdout="\n\n".join(stdout_parts),
            ))
            step += 1

        elif event_type == "command_executed":
            command_name = str(event.get("name", "")).strip()
            command_args = event.get("arguments")
            if not command_name and not command_args:
                continue

            stdout = json.dumps(
                {
                    "name": command_name,
                    "arguments": command_args,
                },
                ensure_ascii=False,
            )
            entries.append(_build_entry(
                step=step,
                action_type="EXTERNAL_TOOL_COMMAND",
                thought="OpenCode command-executed event",
                content={
                    "tool": tool_name,
                    "session_id": event.get("session_id"),
                    "message_id": event.get("message_id"),
                    "name": command_name,
                    "arguments": command_args,
                },
                stdout=stdout,
            ))
            step += 1

        elif event_type == "result":
            message = event.get("message") or {}
            parts = message.get("parts") or []
            if not isinstance(message, dict) or not isinstance(parts, list):
                continue

            thought, text_content, tool_calls = _collect_opencode_parts(parts)
            if not (thought or text_content or tool_calls):
                continue

            finish_stdout = text_content
            completed_outputs = [str(call.get("output", "")).strip() for call in tool_calls if str(call.get("output", "")).strip()]
            if completed_outputs:
                suffix = "\n\n".join(completed_outputs)[:2000]
                finish_stdout = f"{finish_stdout}\n\n{suffix}".strip()

            entries.append(_build_entry(
                step=step,
                action_type="EXTERNAL_TOOL_FINISH",
                thought=thought,
                content={
                    "tool": tool_name,
                    "session_id": event.get("session_id"),
                    "message_id": (message.get("info") or {}).get("id"),
                    "tool_calls": tool_calls,
                },
                stdout=finish_stdout,
            ))
            step += 1

    return entries


def _collect_claude_blocks(blocks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    text_blocks: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in blocks:
        block_type = str(block.get("type", ""))
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                text_blocks.append(text)
        elif block_type == "tool_use":
            tool_uses.append({
                "tool": block.get("name"),
                "call_id": block.get("id"),
                "input": block.get("input"),
            })
        elif block_type == "tool_result":
            tool_results.append({
                "call_id": block.get("tool_use_id"),
                "content": str(block.get("content", "")).strip()[:2000],
                "is_error": bool(block.get("is_error", False)),
            })

    return "\n\n".join(text_blocks), tool_uses, tool_results


def parse_claude_setup_history(tool_name: str, output_text: str) -> list[dict[str, Any]]:
    """Parse Claude Code stream-json logs."""
    entries: list[dict[str, Any]] = []
    step = 1

    for event in _iter_json_lines(output_text):
        event_type = str(event.get("type", ""))

        if event_type in {"assistant", "user"}:
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            text_content, tool_uses, tool_results = _collect_claude_blocks(
                [block for block in content if isinstance(block, dict)]
            )
            if not (text_content or tool_uses or tool_results):
                continue

            if event_type == "assistant":
                action_type = "EXTERNAL_TOOL_CALL" if tool_uses else "EXTERNAL_TOOL_MESSAGE"
                stdout_parts: list[str] = []
                if text_content:
                    stdout_parts.append(text_content)
                if tool_uses:
                    stdout_parts.append(json.dumps(tool_uses, ensure_ascii=False))
                entries.append(_build_entry(
                    step=step,
                    action_type=action_type,
                    thought=text_content or "Claude Code assistant event",
                    content={
                        "tool": tool_name,
                        "session_id": event.get("session_id"),
                        "message_id": message.get("id"),
                        "tool_uses": tool_uses,
                    },
                    stdout="\n\n".join(stdout_parts),
                ))
                step += 1
            else:
                stdout_parts = [json.dumps(tool_results, ensure_ascii=False)] if tool_results else []
                if text_content:
                    stdout_parts.append(text_content)
                entries.append(_build_entry(
                    step=step,
                    action_type="EXTERNAL_TOOL_RESULT",
                    thought="Claude Code tool execution result",
                    content={
                        "tool": tool_name,
                        "session_id": event.get("session_id"),
                        "message_id": message.get("id"),
                        "tool_results": tool_results,
                    },
                    stdout="\n\n".join(stdout_parts),
                ))
                step += 1

        elif event_type == "result":
            result_text = str(event.get("result", "")).strip()
            subtype = str(event.get("subtype", "")).strip()
            if not result_text and not subtype:
                continue
            entries.append(_build_entry(
                step=step,
                action_type="EXTERNAL_TOOL_FINISH",
                thought="Claude Code final result",
                content={
                    "tool": tool_name,
                    "session_id": event.get("session_id"),
                    "subtype": subtype,
                    "cost_usd": event.get("total_cost_usd"),
                    "duration_ms": event.get("duration_ms"),
                },
                stdout=result_text or subtype,
            ))
            step += 1

    return entries


def _collect_qwen_blocks(blocks: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    reasoning_blocks: list[str] = []
    text_blocks: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in blocks:
        block_type = str(block.get("type", ""))
        if block_type == "thinking":
            text = str(block.get("thinking", "")).strip()
            if text:
                reasoning_blocks.append(text)
        elif block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                text_blocks.append(text)
        elif block_type == "tool_use":
            tool_uses.append({
                "tool": block.get("name"),
                "call_id": block.get("id"),
                "input": block.get("input"),
            })
        elif block_type == "tool_result":
            tool_results.append({
                "call_id": block.get("tool_use_id"),
                "content": str(block.get("content", "")).strip()[:2000],
                "is_error": bool(block.get("is_error", False)),
            })

    return "\n\n".join(reasoning_blocks), "\n\n".join(text_blocks), tool_uses, tool_results


def parse_qwen_setup_history(tool_name: str, output_text: str) -> list[dict[str, Any]]:
    """Parse the streaming JSON logs from Qwen Code's query()."""
    entries: list[dict[str, Any]] = []
    step = 1

    for event in _iter_json_lines(output_text):
        event_type = str(event.get("type", ""))

        if event_type in {"assistant", "user"}:
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            thought, text_content, tool_uses, tool_results = _collect_qwen_blocks(
                [block for block in content if isinstance(block, dict)]
            )
            if not (thought or text_content or tool_uses or tool_results):
                continue

            if event_type == "assistant":
                action_type = "EXTERNAL_TOOL_CALL" if tool_uses else "EXTERNAL_TOOL_MESSAGE"
                stdout_parts: list[str] = []
                if text_content:
                    stdout_parts.append(text_content)
                if tool_uses:
                    stdout_parts.append(json.dumps(tool_uses, ensure_ascii=False))
                entries.append(_build_entry(
                    step=step,
                    action_type=action_type,
                    thought=thought or text_content or "Qwen Code assistant event",
                    content={
                        "tool": tool_name,
                        "session_id": event.get("session_id"),
                        "message_id": message.get("id"),
                        "tool_uses": tool_uses,
                    },
                    stdout="\n\n".join(stdout_parts),
                ))
                step += 1
            else:
                stdout_parts = [json.dumps(tool_results, ensure_ascii=False)] if tool_results else []
                if text_content:
                    stdout_parts.append(text_content)
                entries.append(_build_entry(
                    step=step,
                    action_type="EXTERNAL_TOOL_RESULT",
                    thought="Qwen Code tool execution result",
                    content={
                        "tool": tool_name,
                        "session_id": event.get("session_id"),
                        "message_id": message.get("id"),
                        "tool_results": tool_results,
                    },
                    stdout="\n\n".join(stdout_parts),
                ))
                step += 1

        elif event_type == "result":
            result_text = str(event.get("result", "")).strip()
            subtype = str(event.get("subtype", "")).strip()
            if not result_text and not subtype:
                continue
            entries.append(_build_entry(
                step=step,
                action_type="EXTERNAL_TOOL_FINISH",
                thought="Qwen Code final result",
                content={
                    "tool": tool_name,
                    "session_id": event.get("session_id"),
                    "subtype": subtype,
                    "duration_ms": event.get("duration_ms"),
                    "num_turns": event.get("num_turns"),
                },
                stdout=result_text or subtype,
            ))
            step += 1

    return entries
