"""XPU 经验去重与智能合并模块。

当新经验与数据库中已有经验的 embedding 相似度 >= 阈值时，
通过 LLM 判断是否为同一问题，若是则智能合并 advice_nl。

流程：
  embedding → search similar
    ├─ < 0.85 → 直接新增
    └─ ≥ 0.85 → LLM Call 1: 判断是否本质相同
         ├─ LLM 说「不同」→ 仍作为新经验插入
         └─ LLM 说「相同」→ LLM Call 2: 智能合并 advice_nl → update_advice()
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .extract_xpu_from_trajs_mvp import (
    openai_compatible_chat_completions,
    parse_llm_json,
    load_llm_config_from_env,
    get_env_or_raise,
)
from ..logger import get_logger

logger = get_logger("xpu.dedup")

MERGE_SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# LLM 调用工具
# ---------------------------------------------------------------------------

def _call_llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """调用 LLM 并返回 parsed JSON，复用 XPU 提取的同一套环境变量配置。"""
    cfg = load_llm_config_from_env()
    api_key = get_env_or_raise(cfg["api_key_env_var"])
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    raw = openai_compatible_chat_completions(
        model=cfg["llm_model"],
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=cfg["timeout_sec"],
        response_format_json=True,
    )
    content = raw["choices"][0]["message"]["content"]
    return parse_llm_json(content)


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _build_judgment_prompt(
    existing_entry: Dict[str, Any],
    new_entry_dict: Dict[str, Any],
) -> List[Dict[str, str]]:
    """构造判断 prompt：两条经验是否解决本质相同的问题。"""
    system_text = (
        "你是一名资深 Python 环境配置专家。"
        "\n现在给你两条环境经验（XPU），请判断它们是否在解决本质上相同的问题。"
        "\n注意：即使具体的包名或版本不同，只要根因和修复思路相同，就视为同一问题。"
        "\n回答必须是严格的 JSON 对象，不包含任何多余文字。"
    )

    user_payload = {
        "task": "判断两条经验是否为同一问题",
        "existing_experience": {
            "id": existing_entry.get("id"),
            "context": existing_entry.get("context"),
            "signals": existing_entry.get("signals"),
            "advice_nl": existing_entry.get("advice_nl"),
        },
        "new_experience": {
            "id": new_entry_dict.get("id"),
            "context": new_entry_dict.get("context"),
            "signals": new_entry_dict.get("signals"),
            "advice_nl": new_entry_dict.get("advice_nl"),
        },
        "output_requirement": (
            "请输出 JSON: {\"same_problem\": true/false, \"reason\": \"简要说明判断依据\"}"
            "\nsame_problem=true 表示两条经验本质上解决同一类问题；"
            "\nsame_problem=false 表示虽然文本相似但实际是不同的问题。"
        ),
    }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _build_merge_prompt(
    existing_advice: List[str],
    new_advice: List[str],
) -> List[Dict[str, str]]:
    """构造合并 prompt：智能合并两组 advice_nl。"""
    system_text = (
        "你是一名资深 Python 环境配置专家。"
        "\n现在给你两组环境配置建议（advice_nl），它们来自针对同一问题的两条经验。"
        "\n请将它们智能合并为一组改进后的建议列表。"
        "\n合并要求："
        "\n1. 去除重复或含义相同的建议"
        "\n2. 保留所有独特的见解和修复方案"
        "\n3. 如果两条建议可以合为一条更完整的表述，请合并"
        "\n4. 最终建议条数控制在 1-7 条"
        "\n5. 保持简体中文"
        "\n回答必须是严格的 JSON 对象，不包含任何多余文字。"
    )

    user_payload = {
        "task": "合并两组建议",
        "existing_advice_nl": existing_advice,
        "new_advice_nl": new_advice,
        "output_requirement": (
            "请输出 JSON: {\"merged_advice_nl\": [\"建议1\", \"建议2\", ...], "
            "\"merge_summary\": \"简要说明合并了哪些内容\"}"
        ),
    }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def judge_and_merge(
    existing_entry: Dict[str, Any],
    new_entry_dict: Dict[str, Any],
) -> Tuple[str, Optional[List[str]]]:
    """
    LLM 判断两条经验是否本质相同，若相同则智能合并 advice_nl。

    Returns:
        (action, merged_advice)
        action ∈ {"different", "same_merged", "same_no_change"}
        merged_advice: 合并后的 advice 列表（仅 action=="same_merged" 时非 None）
    """
    # --- LLM Call 1: 判断 ---
    judgment_messages = _build_judgment_prompt(existing_entry, new_entry_dict)
    judgment_result = _call_llm_json(judgment_messages)

    same_problem = judgment_result.get("same_problem", False)
    judgment_reason = judgment_result.get("reason", "")

    logger.info(
        "[Dedup] LLM 判断: same_problem=%s, reason=%s",
        same_problem, judgment_reason,
    )

    if not same_problem:
        return "different", None

    # --- LLM Call 2: 智能合并 ---
    existing_advice = existing_entry.get("advice_nl") or []
    new_advice = new_entry_dict.get("advice_nl") or []

    merge_messages = _build_merge_prompt(existing_advice, new_advice)
    merge_result = _call_llm_json(merge_messages)

    merged_advice = merge_result.get("merged_advice_nl")
    merge_summary = merge_result.get("merge_summary", "")

    logger.info("[Dedup] LLM 合并完成: %s", merge_summary)

    if not merged_advice or not isinstance(merged_advice, list):
        # LLM 返回无效结果 → fallback 到简单追加
        logger.warning("[Dedup] LLM 合并结果无效，回退到简单追加")
        merged_advice = _simple_merge(existing_advice, new_advice)

    # 检查合并后是否与原来一致
    if merged_advice is None or set(merged_advice) == set(existing_advice):
        return "same_no_change", None

    return "same_merged", merged_advice


def _simple_merge(
    existing_advice: List[str],
    new_advice: List[str],
) -> Optional[List[str]]:
    """简单去重追加（fallback 逻辑，与之前的行为一致）。"""
    merged = list(existing_advice)
    added = 0
    for adv in new_advice:
        if adv not in merged:
            merged.append(adv)
            added += 1
    return merged if added > 0 else None


def dedup_and_store(
    store,
    entry,
    embedding: List[float],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    去重并存储一条 XPU 经验（Online / Offline 共用的顶层编排函数）。

    Args:
        store: XpuVectorStore 实例
        entry: XpuEntry 对象
        embedding: 预计算的 embedding 向量
        use_llm: 是否使用 LLM 智能去重（False 时回退到简单追加）

    Returns:
        {"action": str, "xpu_id": str, "reason": str}
        action ∈ {"new", "different_inserted", "merged", "duplicate"}
    """
    from .xpu_adapter import XpuContext

    ctx_lang = (
        entry.context.get("lang", "python")
        if isinstance(entry.context, dict)
        else "python"
    )
    similar_entries = store.search(
        query_embedding=embedding,
        ctx=XpuContext(lang=ctx_lang),
        k=3,
        min_similarity=MERGE_SIMILARITY_THRESHOLD,
    )

    # --- 无相似经验 → 直接新增 ---
    if not similar_entries:
        store.upsert_entry(entry, embedding)
        return {
            "action": "new",
            "xpu_id": entry.id,
            "reason": "成功提取并存储（新经验）",
        }

    existing = similar_entries[0]
    existing_id = existing["id"]
    similarity = existing.get("similarity", 0)

    # --- 同 ID → 覆盖更新 ---
    if existing_id == entry.id:
        store.upsert_entry(entry, embedding)
        return {
            "action": "new",
            "xpu_id": entry.id,
            "reason": f"覆盖更新已有经验 {entry.id}",
        }

    # --- 不同 ID 但相似 → 去重逻辑 ---
    new_entry_dict = {
        "id": entry.id,
        "context": entry.context,
        "signals": entry.signals,
        "advice_nl": entry.advice_nl,
    }

    if use_llm:
        try:
            action, merged_advice = judge_and_merge(existing, new_entry_dict)
        except Exception as e:
            logger.warning("[Dedup] LLM 调用失败，回退到简单追加: %s", e)
            action = "same_merged"
            merged_advice = _simple_merge(
                existing.get("advice_nl") or [],
                entry.advice_nl or [],
            )
    else:
        action = "same_merged"
        merged_advice = _simple_merge(
            existing.get("advice_nl") or [],
            entry.advice_nl or [],
        )

    if action == "different":
        # LLM 判定为不同问题 → 作为新经验插入
        store.upsert_entry(entry, embedding)
        return {
            "action": "different_inserted",
            "xpu_id": entry.id,
            "reason": (
                f"LLM 判定与 {existing_id} (sim={similarity:.3f}) "
                f"为不同问题，作为新经验插入"
            ),
        }
    elif action == "same_merged" and merged_advice is not None:
        store.update_advice(existing_id, merged_advice)
        return {
            "action": "merged",
            "xpu_id": existing_id,
            "reason": f"智能合并到已有经验 {existing_id} (sim={similarity:.3f})",
        }
    else:  # same_no_change
        return {
            "action": "duplicate",
            "xpu_id": existing_id,
            "reason": f"去重：已有经验 {existing_id} 完全覆盖 (sim={similarity:.3f})",
        }
