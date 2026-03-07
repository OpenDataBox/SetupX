"""
验证 Sub-Agent（轻量 ReAct）
职责：纯检验——判断 Setup Agent 配置的环境是否合格
- 只运行测试、观察结果、做出判断；不安装包、不修改环境
- 失败原因是 setup 遗留问题 → success=False
- 失败原因是项目固有限制（外部服务/测试 bug）→ success=True
对 setup agent 完全黑箱，仅返回 VerifyResult
"""

import base64
import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .environment_manager import EnvironmentManager
from .logger import get_logger
from .models import VerifyResult

logger = get_logger("verifier_agent")

MAX_STEPS = 30

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
    """轻量 ReAct 验证 sub-agent"""

    def __init__(self, env: EnvironmentManager, max_steps: int = MAX_STEPS, setup_summary: str = ""):
        self._env = env
        self._max_steps = max_steps
        self._setup_summary = setup_summary
        self._llm = self._build_llm_client()

    def _build_llm_client(self):
        """复用 llm_engine 的客户端，不重复实现"""
        config = get_config()
        if config.llm_provider == "ark":
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai 但未配置 OPENAI_API_KEY")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.llm_provider}")

    def verify(self) -> VerifyResult:
        """ReAct 主循环，返回 VerifyResult"""
        logger.info("verifier sub-agent 启动")

        if self._setup_summary:
            logger.info(f"[Verifier] 收到 Setup 交接信息: {self._setup_summary}")
            first_user_msg = (
                f"Setup Agent 交接信息（仅供参考，你仍需独立验证）：\n{self._setup_summary}\n\n请开始验证。"
            )
        else:
            first_user_msg = "请开始验证。"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_msg},
        ]

        for step in range(1, self._max_steps + 1):
            logger.info(f"=== Verifier Step {step}/{self._max_steps} ===")

            raw = self._llm.chat(messages, json_mode=True)
            logger.info(f"LLM 输出: {raw}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                obs = f"JSON 解析失败: {e}，请重新输出合法 JSON。"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})
                continue

            action = parsed.get("action", "")
            args = parsed.get("args", {})
            thought = parsed.get("thought", "")
            logger.info(f"action={action}, thought={thought[:80]}")

            # ── finish ──
            if action == "finish":
                success = bool(args.get("success", False))
                hint = str(args.get("hint", ""))
                collect_count = int(args.get("collect_count", 0))
                test_framework = str(args.get("test_framework", "unknown"))
                logger.info(f"验证完成: success={success}, hint={hint}")
                self._llm.close()
                return VerifyResult(
                    success=success,
                    test_framework=test_framework,
                    collect_count=collect_count,
                    command=args.get("command", ""),
                    exit_code=0 if success else 1,
                    stdout=hint,
                    stderr="",
                )

            # ── exec_run ──
            elif action == "exec_run":
                cmd = args.get("command", "")
                if not cmd:
                    obs = "错误：exec_run 缺少 command 参数"
                else:
                    result = self._env.exec_run(cmd)
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    logger.debug(f"exec_run [{cmd}] → exit_code={result.exit_code}")
                messages.append({"role": "user", "content": f"命令结果:\n{obs}"})

            # ── write_file ──
            elif action == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path.startswith("/tmp/"):
                    obs = "错误：write_file 只允许写入 /tmp/ 目录"
                else:
                    ok = self._write_file(path, content)
                    obs = f"write_file {'成功' if ok else '失败'}: {path}"
                    logger.debug(obs)
                messages.append({"role": "user", "content": obs})

            else:
                obs = f"未知 action='{action}'，只能使用 exec_run / write_file / finish"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        logger.warning("verifier 达到最大步数，未能完成验证")
        self._llm.close()
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
        """用 base64 绕过 shell 转义问题，安全写入多行 Python 文件"""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
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
        """宽松解析 LLM 输出中的 JSON"""
        raw = raw.strip()
        # 1. 直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 2. 提取 ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # 3. 提取裸 { ... }
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"无法提取 JSON: {raw[:200]}")
