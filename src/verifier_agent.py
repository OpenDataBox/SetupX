"""
验证 Sub-Agent（轻量 ReAct 循环）

职责：纯检验——判断 Setup Agent 配置的环境是否合格
- 只运行测试、观察结果、做出判断；不安装包、不修改环境
- 失败原因是 setup 遗留问题 → success=False
- 失败原因是项目固有限制（外部服务/测试 bug）→ success=True
- 对 Setup Agent 完全黑箱，仅返回 VerifyResult

本模块实现了一个独立的 ReAct（Reasoning + Acting）Agent：
每步 LLM 输出一个 JSON 动作（exec_run / write_file / finish），
执行后将观察结果反馈给 LLM，循环直到 LLM 输出 finish 或达到最大步数。
"""

import base64  # 用于将文件内容编码为 base64，绕过 shell 转义问题
import json  # JSON 解析
import re  # 正则表达式，用于解析 LLM 输出中的 JSON

from .llm_engine import ARKClient, OpenAICompatibleClient  # LLM 客户端（支持 ARK 和 OpenAI 兼容接口）
from .config import get_config  # 全局配置
from .environment_manager import EnvironmentManager  # Docker 环境管理器
from .logger import get_logger  # 统一日志系统
from .models import VerifyResult  # 验证结果数据类

logger = get_logger("verifier_agent")  # 创建 verifier_agent 模块专用日志记录器

# Verifier 最大执行步数，防止无限循环
MAX_STEPS = 30

# ============================================================================
# Verifier Agent 的系统提示词
# 定义了 Verifier 的角色、流程、判断标准、可用工具和硬性约束
# ============================================================================
SYSTEM_PROMPT = """\
你是一个在 Docker 容器中工作的验证 Agent。
你的任务是：**检验** Setup Agent 配置的环境是否合格，然后如实汇报结果。

你的角色是**检察官**，不是**修复工**。
你不负责修任何东西，你只负责运行测试、观察结果、做出判断。

## 检验流程

1. 探索项目，找到测试套件（pytest / unittest / tox 等）
2. 按项目原本的方式运行测试，收集结果
3. 分析失败原因，做出判断（见下）
4. 如果项目完全没有测试，在 /tmp/ 写一个 smoke test 验证基本环境可用性

## 判断标准：success=True 还是 False

运行测试后，逐条分析失败/错误的根因：

**success=False（Setup 遗留问题）**：
- 缺少 Python 包（ImportError、ModuleNotFoundError）
- 路径、PYTHONPATH 配置错误
- 项目未正确安装（editable install 缺失等）
- 任何"Setup Agent 本应处理但没处理"的问题

**success=True（项目固有限制，不是 Setup 的责任）**：
- 测试逻辑 bug（断言错误、平台特定问题）
- 依赖外部服务（数据库、API、网络）无法在容器内运行
- 可选依赖的测试被跳过（skipif）
- 测试数据缺失（非 setup 阶段可解决）

判断必须有证据。hint 里要写：运行了什么命令、看到了什么输出、为什么得出这个结论。

## 工具（每步只能调用一个，必须响应合法 JSON）

{"thought": "当前观察和下一步推理", "action": "exec_run", "args": {"command": "shell 命令"}}
{"thought": "...", "action": "write_file", "args": {"path": "/tmp/xxx.py", "content": "文件内容"}}
{"thought": "...", "action": "finish", "args": {"success": true, "hint": "简要说明", "test_framework": "pytest", "collect_count": 12}}

## 硬性约束（违反则判定无效）

- **不安装任何包**：禁止 pip install、apt install、apt-get install、conda install 等一切包安装操作
- **不修改任何环境配置**：禁止 export 环境变量、禁止修改 .bashrc/.profile/PATH 等
- **不修改 /workspace/repo 下的任何文件**
- **write_file 只能写 /tmp/ 路径**（仅用于 smoke test，不得用于 monkeypatch 或绕过测试）

如果测试因为缺包失败，**正确做法是报告 success=False**，写清楚缺什么包，而不是去安装它。
"""


class VerifierAgent:
    """轻量 ReAct 验证 sub-agent

    采用 ReAct（Reasoning + Acting）模式工作：
    1. LLM 思考当前状态（thought）
    2. 选择一个动作执行（exec_run / write_file / finish）
    3. 将执行结果（observation）反馈给 LLM
    4. 重复直到 LLM 输出 finish 或达到最大步数
    """

    def __init__(self, env: EnvironmentManager, max_steps: int = MAX_STEPS, setup_summary: str = ""):
        """初始化 Verifier Agent

        Args:
            env: Docker 环境管理器，用于在容器内执行命令
            max_steps: 最大执行步数（默认 30），防止 LLM 无限循环
            setup_summary: Setup Agent 的交接信息（可选），告诉 Verifier 环境是怎么搭建的
        """
        self._env = env  # Docker 环境管理器实例
        self._max_steps = max_steps  # 最大步数限制
        self._setup_summary = setup_summary  # Setup Agent 的交接摘要
        self._llm = self._build_llm_client()  # 初始化 LLM 客户端

    def _build_llm_client(self):
        """根据配置创建 LLM 客户端

        复用 llm_engine 的客户端类型，不重复实现。
        根据 config.llm_provider 选择 ARK 或 OpenAI 兼容客户端。
        """
        config = get_config()
        if config.llm_provider == "ark":
            # 使用火山引擎 ARK 客户端
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            # 使用 OpenAI 兼容接口（也可用于其他兼容 API，如 qwen）
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai 但未配置 OPENAI_API_KEY")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.llm_provider}")

    def verify(self) -> VerifyResult:
        """ReAct 主循环，执行验证流程并返回 VerifyResult

        核心流程：
        1. 构造初始消息（系统提示 + Setup 交接信息）
        2. 循环：LLM 输出动作 → 执行 → 观察 → 反馈
        3. LLM 输出 finish 时返回验证结果
        4. 达到最大步数仍未完成则返回 success=False

        Returns:
            VerifyResult 包含 success、test_framework、collect_count 等信息
        """
        logger.info("verifier sub-agent 启动")

        # 如果有 Setup Agent 的交接信息，作为第一条用户消息告知 Verifier
        # 注明"仅供参考，你仍需独立验证"，避免 Verifier 盲目信任
        if self._setup_summary:
            logger.info(f"[Verifier] 收到 Setup 交接信息: {self._setup_summary}")
            first_user_msg = (
                f"Setup Agent 交接信息（仅供参考，你仍需独立验证）：\n{self._setup_summary}\n\n请开始验证。"
            )
        else:
            first_user_msg = "请开始验证。"

        # 构造 LLM 对话消息列表
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示词
            {"role": "user", "content": first_user_msg},  # 第一条用户消息
        ]

        # ============================================================
        # ReAct 主循环：每步让 LLM 输出一个动作，执行后反馈观察结果
        # ============================================================
        for step in range(1, self._max_steps + 1):
            logger.info(f"=== Verifier Step {step}/{self._max_steps} ===")

            # 调用 LLM 获取下一步动作（要求 JSON 模式输出）
            raw = self._llm.chat(messages, json_mode=True)
            logger.info(f"LLM 输出: {raw}")
            # 将 LLM 输出追加到对话历史
            messages.append({"role": "assistant", "content": raw})

            # 解析 LLM 输出中的 JSON（支持多种格式：裸 JSON、markdown 代码块等）
            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                # JSON 解析失败：告知 LLM 重新输出合法 JSON
                obs = f"JSON 解析失败: {e}，请重新输出合法 JSON。"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})
                continue  # 跳过本步，让 LLM 重新输出

            # 从 JSON 中提取动作、参数和思考过程
            action = parsed.get("action", "")  # 动作类型
            args = parsed.get("args", {})  # 动作参数
            thought = parsed.get("thought", "")  # LLM 的思考过程
            logger.info(f"action={action}, thought={thought[:80]}")

            # ── finish 动作：验证完成，返回结果 ──
            if action == "finish":
                success = bool(args.get("success", False))  # 是否验证通过
                hint = str(args.get("hint", ""))  # 验证说明/证据
                collect_count = int(args.get("collect_count", 0))  # 收集到的测试数量
                test_framework = str(args.get("test_framework", "unknown"))  # 使用的测试框架
                logger.info(f"验证完成: success={success}, hint={hint}")
                self._llm.close()  # 关闭 LLM 客户端
                return VerifyResult(
                    success=success,
                    test_framework=test_framework,
                    collect_count=collect_count,
                    command=args.get("command", ""),
                    exit_code=0 if success else 1,
                    stdout=hint,  # hint 作为 stdout 返回
                    stderr="",
                )

            # ── exec_run 动作：在容器中执行命令 ──
            elif action == "exec_run":
                cmd = args.get("command", "")  # 要执行的命令
                if not cmd:
                    obs = "错误：exec_run 缺少 command 参数"
                else:
                    # 调用 EnvironmentManager 在容器中执行命令
                    result = self._env.exec_run(cmd)
                    # 将执行结果格式化为观察文本
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    logger.debug(f"exec_run [{cmd}] → exit_code={result.exit_code}")
                # 将观察结果作为用户消息反馈给 LLM
                messages.append({"role": "user", "content": f"命令结果:\n{obs}"})

            # ── write_file 动作：写文件（仅限 /tmp/ 目录，用于 smoke test）──
            elif action == "write_file":
                path = args.get("path", "")  # 目标文件路径
                content = args.get("content", "")  # 文件内容
                # 安全检查：只允许写入 /tmp/ 目录
                if not path.startswith("/tmp/"):
                    obs = "错误：write_file 只允许写入 /tmp/ 目录"
                else:
                    ok = self._write_file(path, content)  # 通过 base64 安全写入
                    obs = f"write_file {'成功' if ok else '失败'}: {path}"
                    logger.debug(obs)
                messages.append({"role": "user", "content": obs})

            # ── 未知动作 ──
            else:
                obs = f"未知 action='{action}'，只能使用 exec_run / write_file / finish"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        # 达到最大步数仍未完成验证
        logger.warning("verifier 达到最大步数，未能完成验证")
        self._llm.close()  # 关闭 LLM 客户端
        return VerifyResult(
            success=False,
            test_framework="unknown",
            collect_count=0,
            command="",
            exit_code=-1,
            stdout="",
            stderr=f"verifier 达到最大步数 {self._max_steps}",
        )

    def _write_file(self, path: str, content: str) -> bool:
        """用 base64 编码安全写入文件到容器

        直接用 echo 或 heredoc 写入多行 Python 文件容易遇到 shell 转义问题
        （引号、反斜杠、特殊字符等），所以先将内容编码为 base64，
        在容器内用 Python 解码并写入，彻底避免转义问题。

        Args:
            path: 容器内文件路径（必须以 /tmp/ 开头）
            content: 文件内容

        Returns:
            是否写入成功
        """
        # 将文件内容编码为 base64 字符串
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        # 在容器内用 Python 解码 base64 并写入文件
        cmd = (
            f"python3 -c \""
            f"import base64; "
            f"open('{path}', 'w').write(base64.b64decode('{b64}').decode('utf-8'))"
            f"\""
        )
        result = self._env.exec_run(cmd)
        return result.success

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """宽松解析 LLM 输出中的 JSON

        LLM 输出的 JSON 可能有多种格式：
        - 裸 JSON 对象
        - 包裹在 ```json ... ``` 中
        - 包裹在 <think>...</think> 后跟 JSON
        - 混有其他文本的 JSON

        此方法依次尝试多种解析策略，提取出 JSON 字典。

        Args:
            raw: LLM 的原始输出文本

        Returns:
            解析出的 JSON 字典

        Raises:
            ValueError: 无法从输出中提取 JSON
        """
        raw = raw.strip()
        # 剥离 qwen 等模型输出的 <think>...</think> 推理过程
        if "<think>" in raw:
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        # 策略 1：直接解析整个字符串为 JSON
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        # 策略 2：提取 ```json ... ``` 代码块中的 JSON
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # 策略 3：提取裸 { ... } JSON 对象
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        # 所有策略失败，抛出异常
        raise ValueError(f"无法提取 JSON: {raw[:200]}")
