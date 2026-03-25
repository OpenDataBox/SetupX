"""
法官 Agent（逐条指控验证模式）
职责：对检察官的每条指控做针对性验证，作出最终裁决
- 不做开放式探索（那是检察官的事）
- 对每条 charge 执行 1~2 条验证命令，确认证据是否属实
- 有权驳回不合理的指控
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("judge")

# 每条指控最多 2 次验证命令
MAX_VERIFY_PER_CHARGE = 2

SYSTEM_PROMPT = """\
你是法官，负责**逐条验证**检察官的指控，然后作出裁决。

你不是检察官——你不做开放式调查。你的职责是：
对检察官提出的每条指控，执行 1~2 条验证命令，确认该指控是否属实。

## 审判流程

你会收到检察官的指控列表。对每条指控，你需要：

1. **阅读指控内容和证据**
2. **设计一条验证命令**来复现检察官的发现（如检察官说 import X 失败，你也跑一次 import X）
3. **根据验证结果判定**该指控是否成立

## 判定标准

**指控成立**：你的验证结果与检察官一致（如依赖确实不可导入、编译确实失败）
**指控驳回**：
- 你的验证结果与检察官矛盾（如依赖实际可导入）
- 检察官对项目类型判断错误（如对 C++/Java/JS 项目要求 Python 依赖）
- 依赖仅在可选 extras 中，非核心依赖
- 失败原因是外部服务/网络/测试逻辑 bug，非 Setup Agent 失职
- 检察官用错了环境（如系统 python3 而非项目的 venv/conda）

## 最终裁决

- 有 ≥1 条指控经你验证确认成立 → **guilty**
- 所有指控均被驳回 → **not_guilty**

## 输出格式（每步必须输出一个合法 JSON 对象）

对每条指控，先验证再判定。全部验证完后输出最终裁决：

{"thought": "验证指控N：...", "action": "exec_run", "args": {"command": "验证命令"}}
{"thought": "所有指控验证完毕", "action": "verdict", "args": {
  "verdict": "guilty 或 not_guilty",
  "reasoning": "逐条说明：指控1 成立/驳回（原因），指控2 ...，综合裁决",
  "charges_review": [
    {"charge_index": 1, "upheld": true, "reason": "验证确认依赖 X 不可导入"},
    {"charge_index": 2, "upheld": false, "reason": "依赖实际可导入，检察官用了系统 python 而非 venv"}
  ]
}}

## 硬性约束

- **不安装任何包**，不修改环境，只读取证
- **不做开放式探索**：只围绕检察官的具体指控验证，不自己找新问题
- **独立判断**：检察官说有罪不代表有罪，你的验证结果才是依据
"""


class JudgeAgent:
    """法官：逐条验证检察官指控，有限容器访问"""

    def __init__(
        self,
        setup_history: list[dict],
        verify_messages: list[dict],
        prosecution: ProsecutionResult,
        env: "EnvironmentManager | None" = None,
    ):
        self._setup_history = setup_history
        self._verify_messages = verify_messages
        self._prosecution = prosecution
        self._env = env
        self._llm = self._build_llm_client()

    def _build_llm_client(self):
        config = get_config()
        if config.llm_provider == "ark":
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai 但未配置 OPENAI_API_KEY")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.llm_provider}")

    def rule(self) -> dict:
        """执行审判，返回 {"verdict": "guilty"|"not_guilty", "reasoning": "..."}"""
        if not self._prosecution.prosecute:
            logger.info("检察官未起诉，裁定 not_guilty")
            self._llm.close()
            return {"verdict": "not_guilty", "reasoning": "检察官未起诉，未发现实质性问题"}

        if self._env is None:
            logger.warning("法官无容器访问权，退化为纸面审判")
            return self._paper_trial()

        logger.info(f"法官开始逐条验证（{len(self._prosecution.charges)} 条指控）")
        return self._verify_charges()

    def _verify_charges(self) -> dict:
        """逐条验证模式：对每条指控做针对性验证"""
        charges = self._prosecution.charges
        # 最大步数 = 每条指控 2 次验证 + 最终裁决的余量
        max_steps = len(charges) * MAX_VERIFY_PER_CHARGE + 3

        # 环境快照：让法官在验证前感知容器状态（可能发现检察官遗漏的 venv 等）
        env_snapshot = self._env.get_env_snapshot()
        logger.info(f"法官环境快照:\n{env_snapshot[:300]}")

        prosecution_summary = self._format_prosecution()
        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()

        user_content = (
            f"## 容器环境快照\n\n```\n{env_snapshot}\n```\n\n"
            f"## Setup Agent 执行轨迹（最近20步）\n\n{setup_summary}\n\n"
            f"## Verifier 验证记录\n\n{verify_summary}\n\n"
            f"## 检察官起诉书（共 {len(charges)} 条指控）\n\n{prosecution_summary}\n\n"
            f"请逐条验证以上指控。每条指控你可以执行最多 {MAX_VERIFY_PER_CHARGE} 条验证命令。"
            f"验证完所有指控后，输出最终裁决。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        effective_step = 0
        api_failures = 0
        for step in range(1, max_steps * 3 + 1):  # 给足重试空间
            if effective_step >= max_steps:
                break
            logger.info(f"=== Judge Step {effective_step+1}/{max_steps} (raw={step}) ===")

            try:
                raw = self._llm.chat(messages, json_mode=True)
            except Exception as e:
                api_failures += 1
                logger.warning(f"Judge LLM 调用失败（不计步数）: {e}，累计API失败={api_failures}")
                if api_failures >= 5:
                    logger.error("Judge API 连续失败过多，中止验证")
                    break
                continue
            effective_step += 1

            logger.info(f"LLM 输出: {raw[:300]}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                messages.append({"role": "user", "content": f"JSON 解析失败: {e}，请重新输出。"})
                continue

            action = parsed.get("action", "")
            args = parsed.get("args", {})

            if action == "verdict":
                verdict = args.get("verdict", "not_guilty")
                reasoning = args.get("reasoning", "")
                charges_review = args.get("charges_review", [])
                upheld = sum(1 for c in charges_review if c.get("upheld"))
                dismissed = len(charges_review) - upheld
                logger.info(
                    f"法官裁决: {verdict} "
                    f"(指控成立={upheld}, 驳回={dismissed})"
                )
                self._llm.close()
                return {"verdict": verdict, "reasoning": reasoning}

            elif action == "exec_run":
                cmd = args.get("command", "")
                if not cmd:
                    messages.append({"role": "user", "content": "错误：exec_run 缺少 command"})
                else:
                    result = self._env.exec_run(cmd)
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    messages.append({"role": "user", "content": f"验证结果:\n{obs}"})

            else:
                messages.append({"role": "user", "content": f"未知 action='{action}'，只能用 exec_run / verdict"})

        # 达到步数上限，强制要求裁决
        logger.warning("法官达到步数上限，请求最终裁决")
        messages.append({
            "role": "user",
            "content": "已达到验证步数上限。请立即根据已有验证结果输出最终裁决（verdict action）。",
        })
        try:
            raw = self._llm.chat(messages, json_mode=True)
            parsed = self._parse_json(raw)
            if parsed.get("action") == "verdict":
                self._llm.close()
                return {
                    "verdict": parsed["args"].get("verdict", "guilty"),
                    "reasoning": parsed["args"].get("reasoning", "步数上限"),
                }
        except Exception as e:
            logger.error(f"强制裁决失败: {e}")

        self._llm.close()
        # 法官无法完成验证 → 标记异常，不做 guilty/not_guilty 判定
        return {"verdict": "error", "reasoning": "法官验证未完成（API失败或步数上限）"}

    def _paper_trial(self) -> dict:
        """纸面审判（无容器，向后兼容）"""
        setup_summary = self._format_setup_history()
        prosecution_summary = self._format_prosecution()

        paper_prompt = (
            "你是法官。根据以下材料作出裁决。\n"
            "注意：你没有容器访问权，只能根据书面材料判断。\n"
            "如果检察官的证据不充分或指控不合理，应裁定 not_guilty。\n\n"
            "应当驳回指控的情形：\n"
            "- 检察官对项目类型判断错误（如对 C++ 项目检查 Python 依赖）\n"
            "- 依赖仅在可选 extras 中，非核心依赖\n"
            "- 失败原因是外部服务/网络/测试逻辑 bug\n"
            "- 检察官可能用错了环境（系统 python3 vs 项目 venv）\n\n"
            "输出格式（合法 JSON 对象）：{\"verdict\": \"guilty\"|\"not_guilty\", \"reasoning\": \"裁决依据\"}"
        )

        user_content = (
            f"## Setup Agent 执行轨迹\n\n{setup_summary}\n\n"
            f"## 检察官调查报告\n\n{prosecution_summary}\n\n"
            "请作出裁决。"
        )

        messages = [
            {"role": "system", "content": paper_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._llm.chat(messages, json_mode=True)
        logger.info(f"法官纸面裁决: {raw[:500]}")
        self._llm.close()

        try:
            result = self._parse_json(raw)
            return {"verdict": result.get("verdict", "error"), "reasoning": result.get("reasoning", "")}
        except Exception as e:
            logger.error(f"裁决解析失败: {e}")
            return {"verdict": "error", "reasoning": f"裁决解析失败: {e}"}

    def _format_setup_history(self) -> str:
        recent = self._setup_history[-20:]
        lines = []
        for entry in recent:
            step = entry.get("step", "?")
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_type = action.get("action_type", "?")
            content = action.get("content", {})
            thought = action.get("thought", "")[:100]
            exit_code = result.get("exit_code", "?")
            stdout = (result.get("stdout") or "")[:200]
            lines.append(
                f"[步骤{step}] {action_type} | thought: {thought}\n"
                f"  内容: {json.dumps(content, ensure_ascii=False)[:150]}\n"
                f"  结果: exit_code={exit_code}, stdout: {stdout}"
            )
        return "\n\n".join(lines) if lines else "（无历史）"

    def _format_verify_messages(self) -> str:
        if not self._verify_messages:
            return "（无 Verifier 对话记录）"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def _format_prosecution(self) -> str:
        if not self._prosecution.prosecute:
            return "检察官选择不起诉：未发现实质性问题。"
        lines = ["检察官提起诉讼，指控如下：\n"]
        for i, charge in enumerate(self._prosecution.charges, 1):
            claim = charge.get("claim", "")
            evidence = charge.get("evidence", "")
            lines.append(f"**指控{i}**：{claim}\n证据：\n{evidence}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"无法提取 JSON: {raw[:200]}")
