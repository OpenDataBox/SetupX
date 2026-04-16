"""
LLM 推理核心（按 blueprint 1.3 节定义）
支持 ARK 和 OpenAI 兼容接口，记录完整的输入输出日志

核心职责：
  1. 封装 LLM API 调用（支持字节 ARK 和 OpenAI 兼容接口）
  2. 构造 System Prompt（包含动作定义 + XPU 建议 + 当前观测）
  3. 将 LLM 输出解析为结构化的 AgentAction
  4. XPU 建议适配：根据经验思路 + 当前上下文，LLM 生成适配命令

模块结构：
  LLMClientBase（抽象基类）
  ├── ARKClient：字节 ARK API 客户端
  └── OpenAICompatibleClient：OpenAI 兼容 API 客户端
  LLMEngine：推理引擎，编排 prompt 构造 + API 调用 + 响应解析
"""

import json  # JSON 序列化与解析
import re    # 正则表达式，用于从 markdown 中提取 JSON
from abc import ABC, abstractmethod  # 抽象基类
from typing import Any  # 类型标注

import httpx  # HTTP 客户端，用于调用 LLM API

from .config import get_config, ARKConfig, OpenAIConfig  # 项目配置
from .logger import get_logger  # 统一日志系统
from .models import AgentAction, ActionType, XPUSuggestion  # 数据模型

logger = get_logger("llm")  # LLM 模块专用日志


# =============================================================================
# LLM 客户端抽象层
# =============================================================================

class LLMClientBase(ABC):
    """LLM 客户端抽象基类

    定义统一的 chat 接口，所有 LLM 后端都通过此接口调用。
    """

    @abstractmethod
    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """发送聊天请求并返回 LLM 生成的文本

        Args:
            messages: 对话消息列表（role: system/user/assistant）
            json_mode: 是否要求 LLM 以 JSON 格式输出

        Returns:
            LLM 生成的文本内容
        """
        pass


class ARKClient(LLMClientBase):
    """字节 ARK API 客户端

    ARK 是字节跳动的大模型服务平台，API 与 OpenAI 兼容但使用
    独立的 base_url 和 deployment（模型部署名）。
    """

    def __init__(self, config: ARKConfig):
        """初始化 ARK 客户端

        Args:
            config: ARK 配置对象，包含 api_key、base_url、deployment 等
        """
        self._config = config
        # 创建 HTTP 客户端，超时 300 秒（精读/审计 prompt 较长，需要更多时间）
        self._client = httpx.Client(timeout=300)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """调用 ARK API 发送聊天请求

        Args:
            messages: 对话消息列表
            json_mode: 是否要求 JSON 格式输出

        Returns:
            LLM 生成的文本内容
        """
        # 构造 HTTP 请求头（Bearer Token 认证）
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        # 构造请求体
        payload: dict[str, Any] = {
            "model": self._config.deployment,  # ARK 使用 deployment 作为模型标识
            "messages": messages,
        }

        # 可选：要求 JSON 格式输出
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # 发送 POST 请求到 ARK 的 chat/completions 端点
        response = self._client.post(
            f"{self._config.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()  # 非 2xx 状态码直接抛出异常

        # 从响应中提取生成的文本内容
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        """关闭 HTTP 客户端连接"""
        self._client.close()


class OpenAICompatibleClient(LLMClientBase):
    """OpenAI 兼容 API 客户端

    支持所有遵循 OpenAI Chat Completions API 格式的后端，包括：
    - 官方 OpenAI API
    - 各种开源模型的 API 服务（如 vLLM、Ollama 等）
    - 推理模型（如 qwen、glm-4.6），额外处理 <think> 标签和 reasoning_content
    """

    def __init__(self, config: OpenAIConfig):
        """初始化 OpenAI 兼容客户端

        Args:
            config: OpenAI 配置对象，包含 api_key、base_url、model 等
        """
        self._config = config
        # 创建 HTTP 客户端，超时 300 秒（精读/审计 prompt 较长，需要更多时间）
        self._client = httpx.Client(timeout=300)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """调用 OpenAI 兼容 API 发送聊天请求

        特殊处理：
        1. 推理模型兼容：优先取 content，若为空则取 reasoning_content
        2. <think> 标签剥离：qwen 等推理模型会在 content 中包含思考过程，需要去掉

        Args:
            messages: 对话消息列表
            json_mode: 是否要求 JSON 格式输出

        Returns:
            LLM 生成的文本内容（已去除思考标签）
        """
        # 构造 HTTP 请求头
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        # 构造请求体
        payload: dict[str, Any] = {
            "model": self._config.model,  # OpenAI 使用 model 字段
            "messages": messages,
            "max_tokens": 4096,  # 限制最大输出长度
        }

        # 可选：要求 JSON 格式输出
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # 发送 POST 请求
        response = self._client.post(
            f"{self._config.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        msg = data["choices"][0]["message"]

        # 推理模型兼容处理：有些模型将主要内容放在 reasoning_content 字段
        content = msg.get("content")
        if not content:
            content = msg.get("reasoning_content", "")

        # 剥离 <think>...</think> 标签
        # qwen 等推理模型会在 content 中嵌入思考过程（<think>推理过程</think>实际输出）
        # 需要去掉思考过程，只保留实际输出
        if content and "<think>" in content:
            import re as _re
            content = _re.sub(r"<think>.*?</think>\s*", "", content, flags=_re.DOTALL)
        return content

    def close(self) -> None:
        """关闭 HTTP 客户端连接"""
        self._client.close()


# =============================================================================
# LLM 推理引擎
# =============================================================================

class LLMEngine:
    """LLM 推理引擎（按 blueprint 1.3 节定义）

    负责：
    1. 根据配置选择 LLM 后端（ARK 或 OpenAI）
    2. 构造 System Prompt（嵌入 XPU 建议、当前观测等上下文）
    3. 调用 LLM 生成动作决策
    4. 解析 LLM 输出为结构化的 AgentAction
    5. 适配 XPU 命令（将通用建议适配到具体仓库环境）
    """

    # =========================================================================
    # System Prompt 模板（按 blueprint 3.2 节定义）
    # =========================================================================
    # 此模板定义了 Agent 可用的所有动作类型及其使用规则。
    # 占位符 {cwd}、{os_info}、{formatted_xpu_suggestions} 在运行时填充。
    # 使用 {{ }} 转义大括号，避免被 .format() 误解析。
    SYSTEM_PROMPT_TEMPLATE = """You are an expert DevOps agent tailored for environment setup.
You have access to a Linux terminal and an external eXPerience Unit (XPU).

Current Status:
- WorkDir: {cwd}
- OS: {os_info}

XPU Suggestions (Proven solutions from history):
{formatted_xpu_suggestions}

## Action Types — Purpose and When to Use

### SHELL_COMMAND
Execute any shell command directly in the container.
- **Use for**: installing packages, setting PYTHONPATH, exploring repo structure, running
  diagnostic commands (e.g. `pytest --co -q` to inspect collection errors), fixing configs.
- **This is your default action.** Use it whenever you are still diagnosing or fixing.
- To check if dependencies are installed, run: `pip list | grep <pkg>` or `python -c "import X"`.
- To understand pytest errors without triggering full verification, run: `pytest --co -q 2>&1 | head -50`.

### TRY_XPU_SUGGESTION
Apply a proven fix from the XPU knowledge base inside a snapshot sandbox.
- **Use for**: applying an "Executable XPU Fix" that is relevant to the current error.
  YOU must adapt the XPU advice to the current environment and write the concrete shell
  commands yourself in the "command" field (semicolon-separated, like SHELL_COMMAND).
  The container is snapshotted before execution and auto-rolled back on failure,
  making it safer than SHELL_COMMAND.
- **Do NOT use** if "Executable XPU Fixes" is absent from the XPU section (means commands are empty).

### SET_ENV
Persist an environment variable across all subsequent commands.
- **Use for**: setting variables like PYTHONPATH, JAVA_HOME that must survive across steps.
- Prefer this over `export VAR=...` in a SHELL_COMMAND, which only lasts for that single command.

### ROLLBACK_ENV
Roll the container back to the most recent snapshot.
- **Use for**: recovering from a broken environment state — e.g. after multiple failed attempts
  left the container in an inconsistent state, or after a bad TRY_XPU_SUGGESTION result.
- This is a recovery escape hatch, not a routine action. Use sparingly.

### VERIFY
Trigger the full pytest verification pipeline. This is an **expensive, final-stage action**:
it spins up a sub-agent that probes the project structure and runs `pytest --co -q` then
`pytest -x -q`. It is designed to confirm that setup is complete, NOT to diagnose problems.
- **ONLY call VERIFY when you genuinely believe all dependencies are installed and the
  environment is fully configured.**
- **NEVER call VERIFY to probe the environment, discover missing packages, or get pytest
  output for diagnosis.** Use `SHELL_COMMAND` with `pytest --co -q` for that purpose instead.
- If VERIFY succeeds → call FINISH immediately.
- If VERIFY fails → analyze the output and continue fixing with SHELL_COMMAND.

### FINISH
Signal that the task is complete. **ONLY call after a successful VERIFY.**

---

## Decision Instructions

1. Analyze the Last Error carefully before choosing an action.
2. If "Executable XPU Fixes" lists a fix relevant to the current error, **prefer
   TRY_XPU_SUGGESTION** — read the XPU advice, adapt it to your current environment,
   and write the concrete command yourself. The command is snapshot-protected with
   auto-rollback on failure.
3. Use SHELL_COMMAND when no relevant XPU fix is available, or when you need to run
   diagnostic/exploratory commands (e.g., ls, cat, pip list).
4. Default to SHELL_COMMAND when in doubt.
5. Only call VERIFY when you are confident the environment is ready. Until then, diagnose
   with SHELL_COMMAND.
6. Always explain WHY you chose the action in the "thought" field.

---

You MUST respond in JSON format with this schema:
{{
  "thought": "分析当前状态和错误原因，解释为什么选择该动作...",
  "action_type": "SHELL_COMMAND" | "TRY_XPU_SUGGESTION" | "SET_ENV" | "ROLLBACK_ENV" | "VERIFY" | "FINISH",
  "content": {{
    // 如果是 SHELL_COMMAND:
    "command": "pip install numpy",

    // 如果是 TRY_XPU_SUGGESTION:
    "xpu_suggestion_id": "suggestion_123",
    "command": "pip install numpy==1.23.5",
    "reasoning": "XPU 建议降级 numpy 版本，这与报错信息高度吻合"

    // 如果是 SET_ENV:
    "env_key": "VAR_NAME",
    "env_value": "value"

    // 如果是 ROLLBACK_ENV / VERIFY / FINISH:
    // （ROLLBACK_ENV、VERIFY 无需额外字段）
    // FINISH 需要: "message": "环境配置完成"
  }}
}}
"""

    # =========================================================================
    # 情境描述 Prompt（用于 XPU 向量检索）
    # =========================================================================
    SITUATION_PROMPT = (
        "你是环境配置 Agent 的情境感知模块。"
        "根据当前工作历史，用2-3句中文描述当前情境，内容要便于检索相关经验：\n"
        "1. 项目特征：语言、包管理器（pip/poetry/conda）、依赖文件类型\n"
        "2. 已完成的操作和当前卡点（有错误则描述错误类型，无错误则描述在做什么）\n"
        "3. 下一步意图\n"
        "只输出纯文本描述，不输出 JSON，不超过150字。\n"
        "【严格约束】只描述历史中实际执行过的命令和观察到的文件，禁止推断未见过的工具名。"
        "例如：只有在历史中确认运行过 poetry 命令或观察到 poetry.lock 时，才能写\"使用 Poetry\"；"
        "否则只写\"使用 pip\"或\"包管理器未知\"。"
    )

    def __init__(self):
        """初始化 LLM 推理引擎

        根据项目配置（LLM_PROVIDER 环境变量）选择对应的 LLM 后端：
        - "ark"：使用字节 ARK API（通过 ARKClient）
        - "openai"：使用 OpenAI 兼容 API（通过 OpenAICompatibleClient）
        """
        config = get_config()

        # 根据配置选择 LLM 后端
        if config.llm_provider == "ark":
            self._client = ARKClient(config.ark)
            logger.info("使用 ARK LLM 客户端")
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai 但未配置 OPENAI_API_KEY")
            self._client = OpenAICompatibleClient(config.openai)
            logger.info("使用 OpenAI 兼容 LLM 客户端")
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.llm_provider}")

    def describe_situation(
        self,
        history: list[dict],
        cwd: str,
        os_info: str,
        last_error: str | None,
    ) -> str:
        """用 LLM 生成当前情境描述，用于 XPU 向量检索

        将最近 5 条历史记录 + 当前状态发给 LLM，生成 2-3 句中文描述。
        失败时回退到截断的 last_error 文本。
        """
        recent = history[-5:]
        lines = []
        for entry in recent:
            if "action" in entry:
                a = entry["action"]
                lines.append(f"动作: {a.get('action_type', '')} {a.get('command', '')[:80]}")
            if "result" in entry:
                r = entry["result"]
                out = (r.get("stdout") or "")[:100]
                err = (r.get("stderr") or "")[:100]
                lines.append(f"结果(exit={r.get('exit_code', '')}): {out or err}")

        history_text = "\n".join(lines) if lines else "（无历史记录，任务刚开始）"
        error_text = f"\n当前错误: {last_error[:200]}" if last_error else ""

        messages = [
            {"role": "system", "content": self.SITUATION_PROMPT},
            {"role": "user", "content": (
                f"工作目录: {cwd}\nOS: {os_info}{error_text}\n\n"
                f"最近操作历史:\n{history_text}"
            )},
        ]

        try:
            situation = self._client.chat(messages, json_mode=False).strip()
        except Exception as e:
            logger.warning(f"情境描述生成失败: {e}")
            situation = last_error[:150] if last_error else "python 项目环境配置"

        logger.info(f"[情境描述] {situation[:100]}")
        return situation

    # =========================================================================
    # XPU 建议格式化
    # =========================================================================

    def _format_xpu_suggestions(
        self,
        suggestions: list[XPUSuggestion],
        tried_ids: set[str],
    ) -> str:
        """格式化 XPU 建议为两层文本，嵌入 System Prompt

        两层结构设计意图：
        - Layer 1（Reference Knowledge）：所有建议的自然语言描述
          LLM 可以参考这些思路自行编写 SHELL_COMMAND
        - Layer 2（Executable Fixes）：commands 非空的可执行建议
          LLM 可以直接通过 TRY_XPU_SUGGESTION 调用执行

        Args:
            suggestions: 从 XPU 知识库检索到的建议列表
            tried_ids: 已尝试的建议 ID 集合（跳过这些建议）

        Returns:
            格式化后的文本，嵌入 System Prompt 的 {formatted_xpu_suggestions} 位置
        """
        if not suggestions:
            return "No XPU knowledge available."

        ref_lines = []    # Layer 1：所有建议的自然语言参考（无论 commands 是否为空）
        exec_lines = []   # Layer 2：commands 非空的可执行建议

        for s in suggestions:
            if s.id in tried_ids:
                continue  # 跳过已尝试的建议，避免 LLM 重复选择

            # Layer 1：始终展示自然语言建议（description 是 advice_nl 的拼接）
            ref_lines.append(f"- [{s.id}] {s.description}")

            # Layer 2：只有 commands 非空时才展示为可执行选项
            if s.commands:
                exec_lines.append(
                    f"  [ID: {s.id}] Commands: {s.commands} (confidence: {s.confidence:.2f})"
                )

        # 组装两层文本
        parts = []
        if ref_lines:
            parts.append(
                "XPU Reference Knowledge (use to inform your SHELL_COMMAND):\n"
                + "\n".join(ref_lines)
            )
        if exec_lines:
            parts.append(
                "Executable XPU Fixes (use TRY_XPU_SUGGESTION, snapshot-protected):\n"
                + "\n".join(exec_lines)
            )
        return "\n\n".join(parts) if parts else "No applicable XPU knowledge."

    # =========================================================================
    # 主决策接口
    # =========================================================================

    def generate_action(
        self,
        history: list[dict],
        xpu_suggestions: list[XPUSuggestion],
        cwd: str = "/workspace/repo",
        os_info: str = "Ubuntu 22.04",
        last_error: str | None = None,
        tried_suggestion_ids: set[str] | None = None,
    ) -> AgentAction:
        """生成下一步动作（按 blueprint 1.3 节定义）

        完整流程：
        1. 构造 System Prompt（填充 cwd、os_info、XPU 建议）
        2. 将最近 10 条历史记录添加为 assistant/user 消息对
        3. 添加当前观测（last_error）作为最终 user 消息
        4. 调用 LLM API（JSON mode）
        5. 解析 LLM 响应为 AgentAction

        Args:
            history: Agent 历史记录（action + result 对）
            xpu_suggestions: 当前检索到的 XPU 建议
            cwd: 当前工作目录
            os_info: 操作系统信息
            last_error: 最近一次错误信息
            tried_suggestion_ids: 已尝试的建议 ID 集合（无论成功失败）

        Returns:
            结构化的 AgentAction 对象
        """
        if tried_suggestion_ids is None:
            tried_suggestion_ids = set()

        # === 1. 构造 System Prompt ===
        # 将 XPU 建议、当前目录、OS 信息等填入模板
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            cwd=cwd,
            os_info=os_info,
            formatted_xpu_suggestions=self._format_xpu_suggestions(
                xpu_suggestions, tried_suggestion_ids
            ),
        )

        messages = [{"role": "system", "content": system_prompt}]

        # === 2. 添加历史记录 ===
        # 取最近 10 条历史，转为 assistant（动作）+ user（执行结果）消息对
        for entry in history[-10:]:
            # assistant 消息：Agent 之前的动作决策
            if "action" in entry:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(entry["action"], ensure_ascii=False),
                })
            # user 消息：命令执行结果（模拟为 user 输入供 LLM 参考）
            if "result" in entry:
                result = entry["result"]
                content = f"命令执行结果:\n退出码: {result.get('exit_code', 'N/A')}\n"
                if result.get("stdout"):
                    content += f"输出: {result['stdout']}\n"
                if result.get("stderr"):
                    content += f"错误: {result['stderr']}\n"
                messages.append({"role": "user", "content": content})

        # === 3. 添加当前观测 ===
        # 最终 user 消息：如果有错误则包含错误信息，否则提示分析当前状态
        user_content = "请分析当前状态并决定下一步动作。"
        if last_error:
            user_content = f"Last Error:\n{last_error}\n\n请分析错误原因并决定下一步动作。"

        messages.append({"role": "user", "content": user_content})

        # === 记录 LLM 完整输入（调试用）===
        logger.info("=" * 60)
        logger.info("LLM 输入 (Full Prompt)")
        logger.info("=" * 60)
        for i, msg in enumerate(messages):
            logger.info(f"[{i}] role={msg['role']}")
            # 对于过长的消息进行截断显示，避免日志文件过大
            content = msg["content"]
            if len(content) > 2000:
                logger.info(f"    content (truncated): {content[:1000]}...")
                logger.info(f"    ... ({len(content)} chars total)")
            else:
                logger.info(f"    content: {content}")
        logger.info("=" * 60)

        # === 4. 调用 LLM ===
        # json_mode=True 要求 LLM 以 JSON 格式输出
        response = self._client.chat(messages, json_mode=True)

        # === 记录 LLM 完整输出（调试用）===
        logger.info("=" * 60)
        logger.info("LLM 输出 (Raw Response)")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)

        # === 5. 解析响应 ===
        return self._parse_response(response, xpu_suggestions)

    # =========================================================================
    # 响应解析
    # =========================================================================

    def _parse_response(
        self,
        response: str,
        xpu_suggestions: list[XPUSuggestion],
    ) -> AgentAction:
        """解析 LLM 响应为结构化的 AgentAction

        支持两种格式：
        1. 纯 JSON 文本（理想情况）
        2. 包裹在 ```json ... ``` markdown 代码块中的 JSON

        Args:
            response: LLM 原始输出文本
            xpu_suggestions: 当前的 XPU 建议列表（供上下文关联）

        Returns:
            解析后的 AgentAction 对象

        Raises:
            ValueError: 无法解析为 JSON 时抛出
        """
        # 尝试直接解析 JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # 回退方案：从 markdown 代码块中提取 JSON
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                raise ValueError(f"无法解析 LLM 响应为 JSON: {response[:200]}")

        # 提取 LLM 输出的三个核心字段
        action_type_str = data.get("action_type", "SHELL_COMMAND")  # 动作类型
        content = data.get("content", {})  # 动作内容（命令、建议 ID 等）
        thought = data.get("thought", "")  # LLM 的思考过程

        # 根据动作类型字符串映射为对应的 AgentAction 对象
        if action_type_str == "SHELL_COMMAND":
            return AgentAction(
                action_type=ActionType.SHELL_COMMAND,
                thought=thought,
                command=content.get("command"),  # 要执行的 shell 命令
            )
        elif action_type_str == "TRY_XPU_SUGGESTION":
            return AgentAction(
                action_type=ActionType.TRY_XPU_SUGGESTION,
                thought=thought,
                xpu_suggestion_id=content.get("xpu_suggestion_id"),  # XPU 建议 ID
                command=content.get("command"),  # 主 agent 适配后的命令
                reasoning=content.get("reasoning"),  # 选择该建议的理由
            )
        elif action_type_str == "FINISH":
            return AgentAction(
                action_type=ActionType.FINISH,
                thought=thought,
                message=content.get("message", "任务完成"),  # 完成消息
            )
        elif action_type_str == "SET_ENV":
            return AgentAction(
                action_type=ActionType.SET_ENV,
                thought=thought,
                env_key=content.get("env_key"),    # 环境变量名
                env_value=content.get("env_value"), # 环境变量值
            )
        elif action_type_str == "ROLLBACK_ENV":
            return AgentAction(
                action_type=ActionType.ROLLBACK_ENV,
                thought=thought,
            )
        elif action_type_str == "VERIFY":
            return AgentAction(
                action_type=ActionType.VERIFY,
                thought=thought,
                verify_hint=content.get("hint"),
            )
        else:
            # 未知动作类型：降级为 SHELL_COMMAND 处理
            logger.warning(f"未知动作类型: {action_type_str}，默认作为 SHELL_COMMAND")
            return AgentAction(
                action_type=ActionType.SHELL_COMMAND,
                thought=thought,
                command=content.get("command") or data.get("command"),
            )

    # =========================================================================
    # XPU 建议适配（方案 A：LLM 根据经验思路生成命令）
    # =========================================================================

    # XPU 适配专用 System Prompt
    # 告诉 LLM：参考历史经验的修复思路，结合当前仓库的具体情况，生成适配后的命令
    ADAPT_XPU_PROMPT = """你是一名资深 DevOps 工程师。现在给你一条来自历史经验库的环境修复建议（advice_nl），以及当前仓库的具体错误信息和环境状态。

你的任务：参考建议思路，结合当前仓库的具体情况（错误信息、OS、工作目录等），生成**适配后的可直接执行的 shell 命令**。

注意：
1. 建议思路是通用的，你需要根据当前实际错误信息调整具体的包名、版本号等参数
2. 每条命令必须是完整可执行的 shell 命令
3. 命令按执行顺序排列
4. 不要生成与修复无关的命令（如 echo、注释等）

你必须以 JSON 格式回复：
{{"commands": ["cmd1", "cmd2", ...]}}
"""

    def adapt_xpu_commands(
        self,
        advice_nl: list[str],
        last_error: str,
        cwd: str,
        os_info: str,
    ) -> list[str]:
        """根据 XPU 建议思路 + 当前上下文，让 LLM 生成适配后的命令列表

        核心理念：XPU 知识库中存储的是通用修复思路（advice_nl），
        但具体的包名、版本号等需要根据当前仓库的实际情况调整。
        本方法让 LLM 扮演"适配器"角色，将通用思路转化为具体可执行的命令。

        示例：
        - advice_nl: ["降级 numpy 版本以兼容旧 API"]
        - last_error: "numpy 1.24 has no attribute 'float'"
        - LLM 生成: ["pip install numpy==1.23.5"]

        Args:
            advice_nl: XPU 经验的自然语言修复建议列表
            last_error: 当前仓库的具体错误信息
            cwd: 当前工作目录
            os_info: 操作系统信息

        Returns:
            LLM 生成的适配命令列表。解析失败时返回空列表。
        """
        # 构造 user 消息：将 advice_nl + 上下文打包为 JSON
        user_payload = json.dumps({
            "advice_nl": advice_nl,
            "current_error": last_error[:3000] if last_error else "",  # 截断过长错误信息
            "cwd": cwd,
            "os_info": os_info,
        }, ensure_ascii=False)

        messages = [
            {"role": "system", "content": self.ADAPT_XPU_PROMPT},
            {"role": "user", "content": user_payload},
        ]

        # 记录 LLM 输入（调试用）
        logger.info("=" * 60)
        logger.info("LLM 适配 XPU 命令 (输入)")
        logger.info(f"  advice_nl: {advice_nl}")
        logger.info(f"  error: {(last_error or '')[:200]}...")
        logger.info("=" * 60)

        # 调用 LLM 生成适配命令
        response = self._client.chat(messages, json_mode=True)

        # 记录 LLM 输出（调试用）
        logger.info("=" * 60)
        logger.info("LLM 适配 XPU 命令 (输出)")
        logger.info(response)
        logger.info("=" * 60)

        # 解析 LLM 输出为命令列表
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # 回退：从 markdown 代码块中提取 JSON
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                logger.warning(f"适配命令解析失败，返回空列表: {response[:200]}")
                return []

        commands = data.get("commands", [])
        if not isinstance(commands, list):
            logger.warning(f"适配命令格式异常: {commands}")
            return []

        logger.info(f"LLM 生成 {len(commands)} 条适配命令: {commands}")
        return commands

    # =========================================================================
    # 生命周期管理
    # =========================================================================

    def close(self) -> None:
        """关闭 LLM 客户端（释放 HTTP 连接）"""
        if hasattr(self._client, "close"):
            self._client.close()
