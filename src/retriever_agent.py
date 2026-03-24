"""
Retriever Agent — XPU 知识检索子 Agent

作为 Setup Agent 的子 Agent，职责是检索最合适的 XPU 经验，
并在独立上下文中完成精读筛选和事后审计。

核心设计（参考 Coding Agent 的 file search 范式）：
  1. 独立上下文：检索过程不污染主 Agent 的上下文，主 Agent 只看到最终建议
  2. 两层检索：
     - 第一层：向量粗筛（pgvector 余弦相似度，Top-N 候选）
     - 第二层：LLM 精读筛选（从 N 条候选中选出 K 条最相关的）
  3. 延迟审计：每次检索时顺便审计上一次用过的 XPU 效果
  4. 软过滤：不在数据库层硬过滤负面反馈，由 LLM 根据 telemetry 动态判断

两个接口：
  - 对数据库接口：通过 XpuVectorStore 读写 XPU 条目和 telemetry
  - 对主 Agent 接口：接收当前情境，返回筛选后的 XPU 建议
"""

import json
import re
from typing import Any

from .logger import get_logger
from .models import XPUSuggestion

logger = get_logger("retriever_agent")


# =============================================================================
# 审计结果数据结构
# =============================================================================

class AuditVerdict:
    """XPU 使用审计结果

    Retriever Agent 在每次检索时，顺便审计上一次推荐的 XPU 效果。
    审计结果包含判定（success/failure/neutral）和连续分数。

    Attributes:
        xpu_id: 被审计的 XPU ID
        verdict: 判定结果（success / failure / neutral）
        score: 连续分数（0.0 ~ 1.0，负面为 -1.0 ~ 0.0）
        reason: LLM 给出的判定理由
    """

    def __init__(self, xpu_id: str, verdict: str, score: float, reason: str):
        self.xpu_id = xpu_id
        self.verdict = verdict
        self.score = score
        self.reason = reason

    def __repr__(self):
        return f"AuditVerdict({self.xpu_id}, {self.verdict}, score={self.score:.2f})"


# =============================================================================
# 上次 XPU 使用记录（用于延迟审计）
# =============================================================================

class LastXPURecord:
    """记录上一次推荐的 XPU 信息，供延迟审计使用

    由 retrieve() 返回建议时自动创建，不依赖主 Agent 显式调用。
    这样无论主 Agent 选 TRY_XPU_SUGGESTION 还是 SHELL_COMMAND（参考 XPU），
    都能在下次 retrieve() 时审计效果。

    Attributes:
        xpu_ids: 上次推荐的 XPU ID 列表（可能推荐了多条）
        descriptions: 每条 XPU 的描述（advice_nl 或 retriever reason）
        situation: 推荐时的情境描述
        state_before: 推荐时的错误信息
        step_index: 推荐时 history 的长度（锚点，用于定位后续步骤）
    """

    def __init__(
        self,
        xpu_ids: list[str],
        descriptions: list[str],
        situation: str,
        state_before: str,
        step_index: int,
    ):
        self.xpu_ids = xpu_ids
        self.descriptions = descriptions
        self.situation = situation
        self.state_before = state_before
        self.step_index = step_index


# =============================================================================
# Retriever Agent 核心类
# =============================================================================

# LLM 精读筛选 Prompt
REFINE_PROMPT = """你是一个 XPU 经验检索助手。你的任务是从候选 XPU 经验列表中，筛选出与当前部署情境最匹配的 Top-K 条建议。

## 当前部署情境
{situation}

## 候选 XPU 经验列表（共 {n_candidates} 条，按向量相似度排序）
{candidates_text}

## 筛选规则
1. 精确匹配优先：XPU 的 advice_nl 直接解决当前问题的，排最前
2. telemetry 参考：关注每条 XPU 的历史命中/成功/失败次数，但不要因为失败多就直接排除——
   要判断之前失败的场景是否和当前场景相似。如果不相似，这条 XPU 可能仍然有效
3. 不相关的直接排除：如果 XPU 的 advice 跟当前问题完全无关，不要选
4. 最多选 {k} 条

你必须以 JSON 格式回复：
{{
  "selected": [
    {{
      "xpu_id": "...",
      "relevance_reason": "为什么这条 XPU 和当前情境匹配",
      "confidence": 0.85
    }}
  ]
}}
"""

# LLM 审计 Prompt
AUDIT_PROMPT = """你是一个 XPU 经验审计助手。你的任务是判断上次推荐给 Agent 的 XPU 经验是否对部署起到了帮助。

注意：XPU 建议被推荐给 Agent 后，Agent 可能直接执行了建议的命令（TRY_XPU_SUGGESTION），
也可能只是参考了建议思路自己生成了命令（SHELL_COMMAND）。你需要从后续步骤中判断
Agent 是否采纳了 XPU 的建议，以及这个建议是否有效。

## 上次推荐的 XPU 列表
{xpu_list}

## 推荐时的情境（Agent 当时遇到的问题）
{situation}

## 推荐后的后续步骤（Agent 接下来做了什么）
{subsequent_steps}

## 判定规则（对每条 XPU 分别判定）
- success: Agent 采纳了该 XPU 的建议思路（可能用了不同命令但思路一致），且后续步骤表明问题得到解决或明显改善
- failure: Agent 采纳了该 XPU 的建议思路，但问题没有解决或引入了新问题
- neutral: 无法判断 Agent 是否采纳了该建议，或者该建议与后续步骤无关

## 判定示例

### 示例1：success（Agent 参考建议自己生成命令，问题解决）
- XPU 建议: "Docker 容器未预装 Python3，需要通过包管理器安装"
- 情境: "bash: python3: command not found"
- 后续步骤:
  [SHELL_COMMAND] apt-get update && apt-get install -y python3 python3-pip → exit=0
  [SHELL_COMMAND] python3 --version → exit=0: Python 3.10.12
→ 判定: success, score=0.90
→ 理由: Agent 采纳了"通过包管理器安装 Python3"的思路，执行 apt-get install 成功，python3 命令可用

### 示例2：success（思路一致但命令不同，问题改善）
- XPU 建议: "安装 Poetry 工具来管理项目依赖"
- 情境: "poetry: command not found"
- 后续步骤:
  [SHELL_COMMAND] pip3 install poetry → exit=0
  [SHELL_COMMAND] poetry install → exit=0
→ 判定: success, score=0.75
→ 理由: XPU 建议安装 Poetry，Agent 用 pip3 安装（不是 XPU 建议的 curl 方式），但思路一致且问题解决

### 示例3：failure（采纳建议但引入新问题）
- XPU 建议: "降级 setuptools 版本解决兼容性问题"
- 情境: "AttributeError: module 'setuptools' has no attribute 'setup'"
- 后续步骤:
  [SHELL_COMMAND] pip install setuptools==58.0.0 → exit=0
  [SHELL_COMMAND] pip install -e . → exit=1: ERROR: Could not build wheels
→ 判定: failure, score=-0.60
→ 理由: Agent 按建议降级了 setuptools，但导致项目无法构建，引入新问题

### 示例4：neutral（建议与后续动作无关）
- XPU 建议: "安装 Redis 作为缓存后端"
- 情境: "ConnectionRefusedError: Redis server not available"
- 后续步骤:
  [SHELL_COMMAND] pip install pytest-cov → exit=0
  [SHELL_COMMAND] pytest tests/ -x → exit=1: ImportError: No module named flask
→ 判定: neutral, score=0.0
→ 理由: Agent 没有安装 Redis，后续步骤与 XPU 建议完全无关

### 示例5：neutral（后续步骤太少，无法判断）
- XPU 建议: "通过 apt 安装系统级依赖 libxml2-dev"
- 情境: "error: command 'gcc' failed"
- 后续步骤:
  [SET_ENV] PATH=/usr/local/bin:$PATH
→ 判定: neutral, score=0.0
→ 理由: 后续只有一步环境变量设置，无法判断 XPU 建议是否被采纳或是否有效

### 示例6：success 但帮助有限（score 0.2~0.5 区间）
- XPU 建议: "项目使用 tox 运行测试，需要先安装 tox"
- 情境: "tox: command not found"
- 后续步骤:
  [SHELL_COMMAND] pip install tox → exit=0
  [SHELL_COMMAND] tox -e py310 → exit=1: ERROR: missing dependency numpy
→ 判定: success, score=0.35
→ 理由: Agent 采纳了安装 tox 的建议，tox 安装成功，但后续运行仍有其他依赖问题，建议只解决了部分问题

### 示例7：failure 但危害不大（score -0.2~-0.5 区间）
- XPU 建议: "用 pip install -r requirements.txt 安装依赖"
- 情境: "ModuleNotFoundError: No module named 'yaml'"
- 后续步骤:
  [SHELL_COMMAND] pip install -r requirements.txt → exit=1: ERROR: No matching distribution found for some-internal-pkg
  [SHELL_COMMAND] pip install pyyaml → exit=0
→ 判定: failure, score=-0.30
→ 理由: Agent 尝试了建议的 requirements.txt 安装但失败了，最终自己用单独 pip install 解决，建议未能帮助但也没造成严重后果

## 评分标准（连续区间 -1.0 ~ 1.0，无空隙）

score 是一个连续数值，代表该 XPU 建议对当前部署的帮助程度：

| 分数区间 | verdict | 含义 |
|---------|---------|------|
| 0.8 ~ 1.0 | success | 建议被完全采纳，问题彻底解决 |
| 0.5 ~ 0.8 | success | 思路一致，问题明显改善（可能命令不同） |
| 0.2 ~ 0.5 | success | 部分采纳或间接帮助，有一定正面效果 |
| -0.2 ~ 0.2 | neutral | 无法判断是否采纳，或建议与后续步骤无关 |
| -0.5 ~ -0.2 | failure | 采纳了建议但效果不佳，问题未解决 |
| -0.8 ~ -0.5 | failure | 建议无效，浪费了步骤 |
| -1.0 ~ -0.8 | failure | 建议直接导致了新的严重问题 |

注意：verdict 字段应与 score 一致，但系统最终以 score 数值为准（score > 0.2 算 success，score < -0.2 算 failure，其他算neutral）

你必须以 JSON 格式回复：
{{
  "verdicts": [
    {{
      "xpu_id": "...",
      "verdict": "success | failure | neutral",
      "score": 0.8,
      "reason": "简短说明判定理由"
    }}
  ]
}}
"""


class RetrieverAgent:
    """XPU 知识检索子 Agent

    独立上下文运行，不污染主 Agent 的 prompt。
    主 Agent 只看到最终的检索结果（XPUSuggestion 列表）。

    架构：
      RetrieverAgent
      ├── XpuVectorStore：向量数据库（第一层粗筛）
      ├── LLMClientBase：LLM 客户端（第二层精读 + 审计）
      └── LastXPURecord：上次 XPU 使用记录（延迟审计）
    """

    def __init__(self, vector_store, llm_client):
        """初始化 Retriever Agent

        Args:
            vector_store: XpuVectorStore 实例（向量数据库）
            llm_client: LLMClientBase 实例（LLM 客户端，用于精读和审计）
        """
        self._store = vector_store
        self._llm = llm_client
        # 上次 XPU 使用记录，用于延迟审计
        self._last_xpu_record: LastXPURecord | None = None
        logger.info("RetrieverAgent 初始化完成")

    # =========================================================================
    # 对主 Agent 接口：检索 XPU 建议
    # =========================================================================

    def retrieve(
        self,
        situation: str,
        exclude_ids: list[str] | None = None,
        full_history: list[dict] | None = None,
        k: int = 3,
        n_candidates: int = 10,
    ) -> list[XPUSuggestion]:
        """检索最合适的 XPU 建议（两层检索 + 延迟审计）

        完整流程：
        1. 如果有上次 XPU 记录，先做延迟审计（从 full_history 中按锚点提取后续步骤）
        2. 第一层：向量粗筛，获取 Top-N 候选
        3. 第二层：LLM 精读筛选，从 N 条中选出 K 条
        4. 自动记录本次推荐的 XPU，供下次延迟审计

        Args:
            situation: 当前部署情境描述（做了什么/在做什么/遇到什么问题）
            exclude_ids: 已尝试过的 XPU ID 列表
            full_history: 主 Agent 的完整历史记录（用于延迟审计时按锚点提取后续步骤）
            k: 最终返回的建议数（默认 3）
            n_candidates: 第一层粗筛的候选数（默认 10）

        Returns:
            筛选后的 XPUSuggestion 列表（最多 k 条）
        """
        history = full_history or []

        # === 步骤 0：延迟审计上一次推荐的 XPU ===
        if self._last_xpu_record:
            self._do_delayed_audit(history)

        # === 步骤 1：第一层向量粗筛 ===
        logger.info(f"[第一层] 向量粗筛，候选数 N={n_candidates}")
        try:
            from .xpu.xpu_vector_store import text_to_embedding
            embedding = text_to_embedding(situation)
            candidates = self._store.search(
                embedding,
                k=n_candidates,
                exclude_ids=exclude_ids,
            )
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
            return []

        if not candidates:
            logger.info("[第一层] 未找到候选 XPU")
            return []

        logger.info(f"[第一层] 找到 {len(candidates)} 条候选")

        # === 步骤 2：第二层 LLM 精读筛选 ===
        suggestions = self._refine_with_llm(situation, candidates, k)

        # 批量更新 hits 计数（所有最终返回的 XPU）
        if suggestions:
            try:
                self._store.increment_telemetry(
                    [s.id for s in suggestions], "hits"
                )
            except Exception as e:
                logger.warning(f"更新 hits 计数失败: {e}")

        # === 步骤 3：自动记录本次推荐，供下次延迟审计 ===
        if suggestions:
            self._last_xpu_record = LastXPURecord(
                xpu_ids=[s.id for s in suggestions],
                descriptions=[s.description for s in suggestions],
                situation=situation,
                state_before=situation,  # situation 包含了当前错误信息
                step_index=len(history),  # 锚点：当前 history 长度
            )
            logger.info(
                f"记录推荐 XPU: {[s.id for s in suggestions]}，"
                f"锚点 step_index={len(history)}，等待下次审计"
            )

        logger.info(f"[第二层] 最终返回 {len(suggestions)} 条建议")
        return suggestions

    # =========================================================================
    # 内部方法：LLM 精读筛选
    # =========================================================================

    def _refine_with_llm(
        self,
        situation: str,
        candidates: list[dict],
        k: int,
    ) -> list[XPUSuggestion]:
        """第二层：LLM 精读筛选候选 XPU

        将所有候选打包成一个 prompt，一次 LLM 调用完成筛选。

        Args:
            situation: 当前部署情境
            candidates: 第一层粗筛的候选列表
            k: 最终选取数

        Returns:
            筛选后的 XPUSuggestion 列表
        """
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands

        # 构造候选列表文本
        candidate_lines = []
        candidate_map = {}  # xpu_id → 原始候选数据
        for i, c in enumerate(candidates):
            telemetry = c.get("telemetry") or {}
            hits = telemetry.get("hits", 0)
            successes = telemetry.get("successes", 0)
            failures = telemetry.get("failures", 0)

            advice = c.get("advice_nl") or []
            advice_text = " | ".join(advice) if isinstance(advice, list) else str(advice)

            similarity = c.get("similarity", 0)
            composite = c.get("composite_score", similarity)
            tier = c.get("tier", "normal")
            tier_label = {"golden": "★ 高质量", "normal": "普通", "cold": "▽ 低活跃"}.get(tier, tier)

            line = (
                f"[{i+1}] ID: {c['id']}  质量等级: {tier_label}\n"
                f"    建议: {advice_text}\n"
                f"    相似度: {similarity:.3f}, 复合分: {composite:.3f}\n"
                f"    历史统计: 命中 {hits} 次, 成功 {successes} 次, 失败 {failures} 次"
            )
            candidate_lines.append(line)
            candidate_map[c["id"]] = c

        candidates_text = "\n\n".join(candidate_lines)

        # 构造 LLM 消息
        prompt = REFINE_PROMPT.format(
            situation=situation,
            n_candidates=len(candidates),
            candidates_text=candidates_text,
            k=k,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请从候选列表中筛选最匹配的 XPU 建议。"},
        ]

        logger.info(f"[第二层] 调用 LLM 精读 {len(candidates)} 条候选")

        try:
            response = self._llm.chat(messages, json_mode=True)
            data = self._parse_json_response(response)
        except Exception as e:
            logger.warning(f"LLM 精读失败，回退到粗筛结果: {e}")
            # 回退：直接用粗筛的前 k 条
            return self._candidates_to_suggestions(candidates[:k])

        # 解析 LLM 选择结果
        selected = data.get("selected", [])
        if not selected:
            logger.warning("LLM 未选择任何候选，回退到粗筛结果")
            return self._candidates_to_suggestions(candidates[:k])

        # 按 LLM 选择顺序构造 XPUSuggestion
        suggestions = []
        for item in selected[:k]:
            xpu_id = item.get("xpu_id", "")
            if xpu_id not in candidate_map:
                continue

            c = candidate_map[xpu_id]
            reason = item.get("relevance_reason", "")
            confidence = float(item.get("confidence", c.get("composite_score", 0.5)))

            # 渲染 atoms 为可执行命令
            atoms = c.get("atoms") or []
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(render_atom_to_commands(atom))

            advice = c.get("advice_nl") or []
            advice_text = "\n".join(advice)
            if reason and advice_text:
                description = f"[Retriever] {reason}\n建议: {advice_text}"
            elif advice_text:
                description = advice_text
            else:
                description = f"[Retriever] {reason}" if reason else ""

            suggestions.append(XPUSuggestion(
                id=xpu_id,
                description=description,
                commands=commands,
                confidence=confidence,
                source="retriever_agent",
            ))

        return suggestions

    # =========================================================================
    # 内部方法：延迟审计
    # =========================================================================

    def _do_delayed_audit(self, full_history: list[dict]) -> None:
        """延迟审计上一次推荐的 XPU

        从 full_history 中按 step_index 锚点提取推荐后的后续步骤（最多 5 步），
        调用 LLM 判断每条 XPU 建议是否有效。

        Args:
            full_history: 主 Agent 的完整历史记录
        """
        record = self._last_xpu_record
        if not record:
            return

        logger.info(f"[审计] 延迟审计 XPU: {record.xpu_ids}")

        # 从锚点位置开始，提取后续最多 5 步
        subsequent_entries = full_history[record.step_index : record.step_index + 5]

        if not subsequent_entries:
            logger.info("[审计] 推荐后无后续步骤，跳过审计")
            self._last_xpu_record = None
            return

        # 将每一步提取为摘要文本
        subsequent = []
        for entry in subsequent_entries:
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_type = action.get("action_type", "unknown")
            cmd = action.get("content", {}).get("command", "")
            exit_code = result.get("exit_code", "?")
            stdout = (result.get("stdout") or "")[:150]
            subsequent.append(f"[{action_type}] {cmd} → exit={exit_code}: {stdout}")

        # 构造 XPU 列表文本
        xpu_lines = []
        for xpu_id, desc in zip(record.xpu_ids, record.descriptions):
            xpu_lines.append(f"- ID: {xpu_id}\n  建议: {desc[:300]}")
        xpu_list_text = "\n".join(xpu_lines)

        # 构造审计 prompt
        prompt = AUDIT_PROMPT.format(
            xpu_list=xpu_list_text,
            situation=record.situation[:500],
            subsequent_steps="\n".join(subsequent),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请对每条 XPU 建议分别判定效果。"},
        ]

        try:
            response = self._llm.chat(messages, json_mode=True)
            data = self._parse_json_response(response)

            verdicts_data = data.get("verdicts", [])
            for v in verdicts_data:
                verdict = AuditVerdict(
                    xpu_id=v.get("xpu_id", ""),
                    verdict=v.get("verdict", "neutral"),
                    score=float(v.get("score", 0.0)),
                    reason=v.get("reason", ""),
                )
                logger.info(f"[审计] {verdict}")
                self._update_telemetry_from_audit(verdict)

        except Exception as e:
            logger.warning(f"[审计] LLM 审计失败: {e}")

        # 清除记录，避免重复审计
        self._last_xpu_record = None

    def _update_telemetry_from_audit(self, verdict: AuditVerdict) -> None:
        """根据审计结果更新 XPU 的 telemetry（基于 score 数值判定，而非 verdict 字符串）

        判定规则：
        - score > 0.2  → successes +1
        - score < -0.2 → failures +1
        - 其他         → neutral，不更新

        Args:
            verdict: LLM 审计判定结果
        """
        if not verdict.xpu_id:
            return
        try:
            if verdict.score > 0.2:
                self._store.increment_telemetry([verdict.xpu_id], "successes")
                logger.info(f"[审计] XPU {verdict.xpu_id}: score={verdict.score:.2f} → successes +1")
            elif verdict.score < -0.2:
                self._store.increment_telemetry([verdict.xpu_id], "failures")
                logger.info(f"[审计] XPU {verdict.xpu_id}: score={verdict.score:.2f} → failures +1")
            else:
                logger.info(f"[审计] XPU {verdict.xpu_id}: score={verdict.score:.2f} → neutral，不更新 telemetry")
        except Exception as e:
            logger.warning(f"[审计] telemetry 更新失败: {e}")

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _candidates_to_suggestions(self, candidates: list[dict]) -> list[XPUSuggestion]:
        """将原始候选数据转为 XPUSuggestion（回退用）

        当 LLM 精读失败时，直接用粗筛结果兜底。

        Args:
            candidates: 原始候选列表

        Returns:
            XPUSuggestion 列表
        """
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands

        suggestions = []
        for c in candidates:
            atoms = c.get("atoms") or []
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(render_atom_to_commands(atom))

            advice = c.get("advice_nl") or []
            similarity = float(c.get("similarity", 0.5))
            composite = float(c.get("composite_score", similarity))

            suggestions.append(XPUSuggestion(
                id=c["id"],
                description="\n".join(advice) if isinstance(advice, list) else str(advice),
                commands=commands,
                confidence=composite,
                source="vector_db_fallback",
            ))

        return suggestions

    def _parse_json_response(self, response: str) -> dict:
        """解析 LLM 的 JSON 响应（兼容 markdown 代码块）

        Args:
            response: LLM 原始输出

        Returns:
            解析后的字典

        Raises:
            ValueError: 无法解析时抛出
        """
        # 剥离 <think> 标签
        if "<think>" in response:
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"无法解析 LLM 响应为 JSON: {response[:200]}")

    def close(self, full_history: list[dict] | None = None) -> None:
        """关闭资源，执行最后一次未完成的审计

        Args:
            full_history: 主 Agent 的完整历史记录，用于执行最终审计
        """
        if self._last_xpu_record:
            if full_history:
                logger.info("[关闭] 执行最终审计...")
                self._do_delayed_audit(full_history)
            else:
                logger.info("[关闭] 存在未审计的 XPU 记录，但无历史信息，跳过")
                self._last_xpu_record = None
