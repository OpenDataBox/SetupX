"""
检察官 Agent（ReAct 风格）
职责：调查 Setup Agent 配置的环境是否存在实质性问题
- 有容器访问权，可执行命令取证
- 如果发现问题，提出带具体证据的指控
- 如果没有问题，选择不起诉
- 不安装包、不修改环境
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .environment_manager import EnvironmentManager
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("prosecutor")

MAX_STEPS = 30

SYSTEM_PROMPT = """\
你是检察官，核心任务是回答一个问题：**Setup Agent 配置的环境，能否满足该项目运行测试的基本要求？**

你不是审判 Verifier 的行为，你审判的是 Setup Agent 是否尽职。
你有容器访问权，可以执行命令取证，但不得安装任何包或修改环境。

## 强制调查流程（按顺序执行，不可跳过）

**第零步（必须）：识别项目语言与构建工具**

检查容器内项目目录的标记文件，确定项目类型：
- **Python**：pyproject.toml / setup.py / setup.cfg / requirements.txt → 后续按 Python 流程
- **C/C++**：CMakeLists.txt / Makefile / configure / meson.build → 后续按 C/C++ 流程
- **Java**：pom.xml / build.gradle → 后续按 Java 流程
- **JavaScript**：package.json → 后续按 JavaScript 流程
- **其他**：Cargo.toml (Rust) / go.mod (Go) 等 → 按对应语言流程

```
ls pyproject.toml setup.py setup.cfg requirements.txt CMakeLists.txt Makefile configure pom.xml build.gradle package.json meson.build Cargo.toml go.mod 2>/dev/null
```
确定语言后，后续所有步骤按该语言的标准执行。

**第一步（必须）：验证核心依赖可用**

### Python 项目
从 pyproject.toml / setup.cfg / requirements.txt 读取核心（非可选）依赖，逐一验证：
```
cd /workspace/repo && python3 -c "import 包名" 2>&1
```
**注意：pip 包名和 Python import 名经常不同！** 常见映射：
- beautifulsoup4 → `import bs4`
- GitPython / gitpython → `import git`
- Pillow / pillow → `import PIL`
- PyYAML / pyyaml → `import yaml`
- attrs → `import attr`
- scikit-learn → `import sklearn`
- opencv-python → `import cv2`
- python-dateutil → `import dateutil`
- python-dotenv → `import dotenv`
如果不确定，先 `pip show 包名` 查看安装位置。

### C/C++ 项目
读取 CMakeLists.txt 的 `find_package()` / `target_link_libraries()`，或 Makefile 的 `-l` 链接库：
```
apt list --installed 2>/dev/null | grep -i 关键词
pkg-config --exists 库名 && echo OK || echo MISSING
```
不需要 `import` 验证，只需确保编译时能找到头文件和库。

### Java 项目
```
mvn dependency:tree 2>&1 | tail -30
```
或 `gradle dependencies`，检查依赖树能否解析。

### JavaScript 项目
```
npm ls --depth=0 2>&1 | tail -30
```
或 `yarn list`，检查 node_modules 是否完整。

重点检查：
- Python：`ImportError` / `ModuleNotFoundError` → 核心依赖不可用则**必须起诉**
- C/C++：`fatal error: xxx.h: No such file` / `undefined reference` → **必须起诉**
- Java：`package does not exist` / `ClassNotFoundException` → **必须起诉**
- JS：`Cannot find module` → **必须起诉**
- 可选依赖不可用 → 可免责

**第二步（必须）：亲自运行测试套件**

### Python 项目
```
cd /workspace/repo && python3 -m pytest --tb=line -q --timeout=60 2>&1 | tail -60
```
或按项目标准方式（poetry run pytest、tox、虚拟环境内 pytest 等）。

### C/C++ 项目
若已构建，直接运行测试：
```
cd /workspace/repo && ctest --output-on-failure 2>&1 | tail -60
```
或 `make test`、`make check`。若未构建，先 `cmake . && make -j$(nproc)` 再测试。

### Java 项目
```
cd /workspace/repo && mvn test -q 2>&1 | tail -60
```
或 `gradle test`。

### JavaScript 项目
```
cd /workspace/repo && npm test 2>&1 | tail -60
```

记录：通过数、失败数、错误类型、是否被 Killed（exit_code=137/124）。

**第三步：对每类失败逐一判责**

| 失败类型 | 判责 |
|----------|------|
| **Python**: `ImportError`/`ModuleNotFoundError` + 包在核心依赖声明中 | **必须起诉** |
| **C/C++**: 编译错误（头文件/库缺失）或链接错误 | **必须起诉** |
| **Java**: 编译失败（package not found）或运行时 ClassNotFoundException | **必须起诉** |
| **JS**: `Cannot find module`（核心依赖）| **必须起诉** |
| 包已安装但版本不兼容，导致运行即崩溃 | **必须起诉** |
| 完整套件被 Killed（exit_code=137/124）且**子集测试也有依赖缺失错误** | **必须起诉** |
| 完整套件被 Killed，但小子集无依赖缺失，仅资源超限 | 可免责 |
| 外部服务不可用（数据库、Redis、网络） | 可免责 |
| 纯测试逻辑断言失败 | 可免责 |
| 可选依赖未安装，对应测试被跳过 | 可免责 |

**第四步：核查 Verifier 结论的可信度**
Verifier 声称 success=True，你的结果是否一致？
- 若 Verifier 使用了测试过滤（pytest `--ignore`/`-k`、CTest `-E`、Maven excludes 等），
  检查被过滤的测试是否存在核心依赖缺失。
  - 有核心依赖缺失 → 即使 Verifier 规避了，Setup Agent 仍应追责
  - 失败仅因外部服务不可用 → Verifier 的规避合理，不追责
- 若完整套件被 Killed，运行 10~20 个测试的小子集判断是否存在依赖缺失

## 起诉指控格式

每条指控必须包含：
- **指控对象**：Setup Agent 的哪个具体失职行为（如"未安装核心依赖 X"）
- **依赖声明证据**：该依赖在哪个文件的哪个字段中声明
- **取证命令和原始输出**：你亲自运行的命令 + 完整输出

## 工具（每步必须输出一个合法 JSON 对象）

{"thought": "当前观察和下一步推理", "action": "exec_run", "args": {"command": "shell 命令"}}
{"thought": "调查完毕，所有失败均属免责情形", "action": "finish", "args": {"prosecute": false}}
{"thought": "发现可追责问题，提出指控", "action": "finish", "args": {
  "prosecute": true,
  "charges": [
    {"claim": "Setup Agent 未安装核心依赖 X（来源：pyproject.toml [project.dependencies]）",
     "evidence": "命令: python3 -c 'import X'\\n输出: ModuleNotFoundError: No module named 'X'"}
  ]
}}

## 硬性约束

- **不安装任何包**：禁止 pip install、apt install 等
- **不修改任何环境配置和项目文件**
- **指控对象是 Setup Agent，不是 Verifier**：Verifier 用 --ignore 跳过测试是它的判断，
  你关心的是 Setup Agent 有没有让核心依赖可用，而不是 Verifier 有没有走捷径
"""


class ProsecutorAgent:
    """检察官 ReAct sub-agent，有容器访问权"""

    def __init__(
        self,
        env: EnvironmentManager,
        setup_history: list[dict],
        verify_messages: list[dict],
    ):
        self._env = env
        self._setup_history = setup_history
        self._verify_messages = verify_messages
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

    def investigate(self) -> ProsecutionResult:
        """执行调查，返回 ProsecutionResult"""
        logger.info("检察官开始调查")

        # 环境快照：让检察官在调查前感知容器的实际状态
        # （类似 Setup Agent 每步的 pwd/ls，避免盲目用系统 python3 检查 venv 内的依赖）
        env_snapshot = self._env.get_env_snapshot()
        logger.info(f"环境快照:\n{env_snapshot[:300]}")

        # 构造调查背景
        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()

        first_user_msg = (
            f"## 容器环境快照\n\n```\n{env_snapshot}\n```\n\n"
            f"## Setup Agent 执行轨迹（最近20步）\n\n{setup_summary}\n\n"
            f"## in-loop Verifier 验证对话\n\n{verify_summary}\n\n"
            "请开始调查，判断 Setup Agent 配置的环境是否存在实质性问题。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_msg},
        ]

        successful_steps = 0
        api_failures = 0

        for step in range(1, MAX_STEPS + 1):
            logger.info(f"=== Prosecutor Step {step}/{MAX_STEPS} ===")

            try:
                raw = self._llm.chat(messages, json_mode=True)
            except Exception as e:
                api_failures += 1
                logger.warning(f"Prosecutor LLM 调用失败（API 异常或超时）: {e}，跳过本步（累计失败 {api_failures} 次）")
                continue
            successful_steps += 1
            logger.info(f"LLM 输出: {raw[:300]}")
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

            if action == "finish":
                prosecute = bool(args.get("prosecute", False))
                charges = args.get("charges", [])
                logger.info(f"调查完成: prosecute={prosecute}, 指控数={len(charges)}")
                self._llm.close()
                return ProsecutionResult(
                    prosecute=prosecute,
                    charges=charges,
                    messages=list(messages),
                )

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

            else:
                obs = f"未知 action='{action}'，只能使用 exec_run / finish"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        if successful_steps == 0:
            logger.error(f"Prosecutor 全部 {MAX_STEPS} 步 LLM 调用均失败（API failures={api_failures}），标记为调查失败")
            self._llm.close()
            return ProsecutionResult(
                prosecute=True,
                charges=[{"claim": "检察官调查失败：LLM API 全部不可用，无法完成调查",
                          "evidence": f"共 {MAX_STEPS} 步全部因 API 异常跳过，无有效调查结果"}],
                messages=list(messages),
            )

        logger.warning(f"Prosecutor 达到最大步数（成功步数={successful_steps}），默认不起诉")
        self._llm.close()
        return ProsecutionResult(
            prosecute=False,
            charges=[],
            messages=list(messages),
        )

    def _format_setup_history(self) -> str:
        """格式化 Setup 历史（最近20步）"""
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
        """格式化 Verifier 对话"""
        if not self._verify_messages:
            return "（无 Verifier 对话记录）"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """宽松解析 LLM 输出中的 JSON"""
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
