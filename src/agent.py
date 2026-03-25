"""
主控制器（按 blueprint 1.4 和 2 节定义）
实现推测执行的 Setup Agent

核心职责：
  1. 感知（Perception）：通过 Docker 容器获取环境状态（pwd、os-release）
  2. 决策（Decision）：将观测 + 历史 + XPU 建议发送给 LLM，生成动作
  3. 执行（Execution）：在容器中执行命令、尝试 XPU 建议、设置环境变量等
  4. 观测（Observation）：收集执行结果，反馈给 LLM 做下一轮决策

主循环流程：
  观测 → 检索 XPU → LLM 决策 → 执行动作 → 记录历史 → 循环
  直到 LLM 输出 FINISH 或达到 max_steps 上限

推测执行（Speculative Execution）：
  当使用 TRY_XPU_SUGGESTION 时，先 docker commit 创建快照，
  执行失败后自动回滚到快照，不会污染环境。
"""

import time  # 时间戳，用于归因报告
from typing import Any  # 类型标注

from .config import get_config  # 项目配置加载
from .logger import get_logger  # 统一日志系统
from .models import (
    AgentState,        # Agent 内部状态（step、history、last_error 等）
    AgentAction,       # LLM 输出的动作对象
    ActionType,        # 动作类型枚举（SHELL_COMMAND、TRY_XPU_SUGGESTION 等）
    CommandResult,     # 命令执行结果（exit_code、stdout、stderr）
    AttributionReport, # XPU 归因报告（用于反馈 telemetry）
    XPUSuggestion,     # XPU 建议（id、description、commands、confidence）
    SetupResult,       # 最终输出结果
)
from .environment_manager import EnvironmentManager  # Docker 容器环境管理
from .xpu_client import create_xpu_client, XPUClientBase, VectorXPUClient  # XPU 知识库客户端
from .llm_engine import LLMEngine  # LLM 推理引擎
from .retriever_agent import RetrieverAgent  # XPU 检索子 Agent
from .verifier_agent import VerifierAgent  # pytest 验证子 Agent

logger = get_logger("agent")  # 主 Agent 专用日志


class SpeculativeSetupAgent:
    """推测执行的环境配置 Agent（按 blueprint 1.4 节定义）

    整体架构：
      Agent（本类）= 编排器
      ├── EnvironmentManager：管理 Docker 容器生命周期
      ├── XPUClientBase：查询 XPU 知识库获取历史修复经验
      ├── LLMEngine：调用大模型进行推理决策
      └── VerifierAgent：在验证阶段运行 pytest 确认环境正确
    """

    def __init__(self, repo_url: str, max_steps: int = 50):
        """初始化 Agent

        Args:
            repo_url: 目标 Git 仓库 URL（如 https://github.com/user/repo）
            max_steps: 最大迭代次数，防止无限循环（默认 50 步）
        """
        # 初始化 Agent 内部状态，跟踪当前步数、历史记录、错误信息等
        self._state = AgentState(
            repo_url=repo_url,
            max_steps=max_steps,
        )
        # 创建 Docker 容器环境管理器（负责容器的创建、命令执行、快照回滚）
        self._env: EnvironmentManager = EnvironmentManager()
        # 根据配置创建 XPU 客户端（VectorXPUClient / HTTPXPUClient / MockXPUClient）
        self._xpu: XPUClientBase = create_xpu_client()
        # 初始化 LLM 推理引擎（ARK 或 OpenAI 兼容接口）
        self._llm: LLMEngine = LLMEngine()

        # 初始化 Retriever Agent（仅在 VectorXPUClient 时启用）
        # Retriever Agent 在独立上下文中进行两层检索 + 延迟审计
        self._retriever: RetrieverAgent | None = None
        if isinstance(self._xpu, VectorXPUClient):
            self._retriever = RetrieverAgent(
                vector_store=self._xpu._store,
                llm_client=self._llm._client,
            )
            logger.info("RetrieverAgent 已启用（VectorXPUClient 模式）")

        # XPU 建议池：{id: (suggestion, step_last_seen)}，保留最近2步内见过的建议
        self._xpu_suggestion_pool: dict[str, tuple[XPUSuggestion, int]] = {}
        # 缓存当前步骤检索到的 XPU 建议，供 TRY_XPU_SUGGESTION 动作查找使用
        self._current_xpu_suggestions: list[XPUSuggestion] = []
        # 保存最后一次成功 verify 的 Verifier 对话轨迹，供 Phase 2 使用
        self._last_verify_messages: list[dict] = []

        logger.info(f"Agent 初始化完成，目标仓库: {repo_url}")

    @property
    def env(self) -> EnvironmentManager:
        """暴露环境管理器，供验证阶段复用同一容器

        主流程 run() 结束后不销毁容器，验证阶段通过此属性
        直接在同一容器上运行 pytest。
        """
        return self._env

    def run(self) -> SetupResult:
        """运行 Agent 主循环（按 blueprint 2 节实现）

        完整流程：
        1. 创建容器 + 克隆仓库
        2. 进入主循环（最多 max_steps 步）：
           a. 观测环境（pwd、os-release）
           b. 如果有错误，检索 XPU 知识库
           c. 将观测 + 历史 + XPU 建议发给 LLM 生成动作
           d. 执行动作（SHELL_COMMAND / TRY_XPU_SUGGESTION / SET_ENV / ...）
           e. 记录执行结果到历史
        3. 循环结束后关闭客户端连接，清理快照镜像
        4. 返回 SetupResult（保留容器供验证阶段使用）

        Returns:
            SetupResult 对象，包含任务完成状态、历史记录等
            注意：容器不会被销毁，调用方需在验证完成后调用 env.destroy()
        """
        logger.info("开始执行环境配置任务")

        # === 1. 初始化阶段 ===
        # 创建 Docker 容器并自动克隆仓库到 /workspace/repo
        container_id = self._env.create_container(self._state.repo_url)
        self._state.container_id = container_id  # 记录容器 ID 到状态

        # === 2. 主循环 ===
        while self._state.step < self._state.max_steps:
            self._state.step += 1  # 递增步数计数器
            logger.info(f"=== Step {self._state.step}/{self._state.max_steps} ===")

            # --- 2a. 观测（Observation）---
            # 获取当前工作目录（通常是 /workspace/repo）
            cwd = self._env.exec_run("pwd").stdout.strip()
            # 获取操作系统信息（Ubuntu/Debian 版本等），供 LLM 判断环境
            os_info = self._env.exec_run("cat /etc/os-release | head -2").stdout.strip()

            # --- 2b. 诊断与检索 ---
            # 有错误时才检索 XPU 建议
            self._current_xpu_suggestions = []
            if self._state.last_error:
                exclude = list(self._state.tried_suggestions) if self._state.tried_suggestions else None

                # 构建混合 situation：LLM 语义总结 + 原始命令/错误文本
                # LLM 总结提供语义匹配质量（命中 situation_triggers/advice_nl）
                # 原始文本保留关键词命中率（命中 keywords/regex）
                llm_summary = self._llm.describe_situation(
                    history=self._state.get_recent_history(),
                    cwd=cwd,
                    os_info=os_info,
                    last_error=self._state.last_error,
                )
                raw_situation = self._build_situation(self._state.last_error)
                situation = f"{llm_summary}\n\n{raw_situation}"

                if self._retriever:
                    self._current_xpu_suggestions = self._retriever.retrieve(
                        situation=situation,
                        exclude_ids=exclude,
                        full_history=self._state.history,
                    )
                else:
                    # 回退：非 VectorXPUClient 时，双路检索（情境 + error 原文），去重合并
                    suggestions_by_situation = self._xpu.query(
                        {"query": situation, "os_release": os_info},
                        exclude_ids=exclude,
                    )
                    suggestions_by_error = self._xpu.query(
                        {"error": self._state.last_error, "os_release": os_info},
                        exclude_ids=exclude,
                    )
                    seen_ids = {s.id for s in suggestions_by_situation}
                    for s in suggestions_by_error:
                        if s.id not in seen_ids:
                            suggestions_by_situation.append(s)
                    self._current_xpu_suggestions = suggestions_by_situation

            # 更新跨步建议池：新建议加入，过期（超过2步未见）的移除
            current_step = self._state.step
            for s in self._current_xpu_suggestions:
                self._xpu_suggestion_pool[s.id] = (s, current_step)
            self._xpu_suggestion_pool = {
                sid: (sg, step)
                for sid, (sg, step) in self._xpu_suggestion_pool.items()
                if current_step - step <= 1  # 保留本步和上一步的建议
            }
            self._current_xpu_suggestions = [sg for sg, _ in self._xpu_suggestion_pool.values()]

            # --- 2c. LLM 决策（Thought & Plan）---
            # 将历史记录 + XPU 建议 + 当前观测发给 LLM，生成下一步动作
            action = self._llm.generate_action(
                history=self._state.get_recent_history(),  # 最近 10 条历史
                xpu_suggestions=self._current_xpu_suggestions,  # 检索到的 XPU 建议
                cwd=cwd,  # 当前工作目录
                os_info=os_info,  # 操作系统信息
                last_error=self._state.last_error,  # 最近一次错误
                tried_suggestion_ids=self._state.tried_suggestions,  # 已尝试的建议 ID 集合（双重防御）
            )

            logger.info(f"决策: {action}")

            # --- 2d. 执行（Execution）---
            # 根据 LLM 输出的动作类型，分发到对应的处理函数
            if action.action_type == ActionType.SHELL_COMMAND:
                # 直接执行 shell 命令
                self._handle_shell_command(action)

            elif action.action_type == ActionType.TRY_XPU_SUGGESTION:
                # 推测执行 XPU 建议（带快照保护）
                self._handle_try_xpu_suggestion(action)

            elif action.action_type == ActionType.SET_ENV:
                # 设置环境变量（持久化到容器）
                self._handle_set_env(action)

            elif action.action_type == ActionType.ROLLBACK_ENV:
                # 回滚容器到最近快照
                self._handle_rollback_env(action)

            elif action.action_type == ActionType.VERIFY:
                # 运行 pytest 验证（昂贵操作，仅在 LLM 认为环境就绪时调用）
                verified = self._handle_verify(action)
                if verified:
                    break  # 验证通过，退出主循环

            elif action.action_type == ActionType.FINISH:
                # LLM 宣布任务完成
                self._handle_finish(action)
                break

        # === 3. 清理阶段 ===
        # 关闭 LLM 连接，XPU 连接保留（main.py 在 _store_xpu_experience 后关闭）
        self._llm.close()
        # RetrieverAgent 关闭前执行最终审计
        if self._retriever:
            self._retriever.close(full_history=self._state.history)
        # 清理快照镜像释放磁盘空间，验证阶段不再需要回滚能力
        self._env.cleanup_snapshots()

        # 如果主循环因达到 max_steps 而退出，标记为未完成
        if not self._state.completed:
            logger.warning("达到最大迭代次数，任务未完成")

        # 返回最终结果（容器保留，调用方负责后续销毁）
        return SetupResult(
            repo_url=self._state.repo_url,
            container_id=container_id,
            completed=self._state.completed,
            steps_taken=self._state.step,
            final_message=self._state.final_message or "达到最大迭代次数，任务未完成",
            history=self._state.history,
            last_verify_messages=self._last_verify_messages,
        )

    # =========================================================================
    # 动作处理函数
    # =========================================================================

    def _handle_shell_command(self, action: AgentAction) -> None:
        """处理 SHELL_COMMAND 动作：直接在容器中执行 shell 命令

        这是最常用的动作类型，Agent 默认使用此动作进行诊断和修复。
        命令在 /workspace/repo 目录下执行。

        Args:
            action: LLM 输出的动作对象，action.command 为要执行的命令
        """
        if not action.command:
            logger.warning("SHELL_COMMAND 缺少 command")
            return

        # 在容器中执行命令
        result = self._env.exec_run(action.command)

        # 更新错误状态：成功则清空，失败则记录错误信息
        if not result.success:
            self._state.last_error = result.stderr or result.stdout
        else:
            self._state.last_error = None  # 成功执行，清除上一次错误

        # 记录到历史，供后续 LLM 决策参考
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": result.to_dict(),
        })

    def _handle_try_xpu_suggestion(self, action: AgentAction) -> None:
        """处理 TRY_XPU_SUGGESTION 动作（推测执行模式）

        推测执行流程：
        A. 存档（Checkpoint）：docker commit 创建快照
        B. 试错（Trial）：逐条执行建议中的命令
        C. 验证与归因（Verification & Attribution）：评估执行效果
        D. 提交反馈（Feedback Loop）：向 XPU 知识库提交 telemetry
        E. 决策分支（Decision Branch）：成功保留 / 失败回滚

        Args:
            action: LLM 输出的动作对象，action.xpu_suggestion_id 为建议 ID
        """
        if not action.xpu_suggestion_id:
            logger.warning("TRY_XPU_SUGGESTION 缺少 xpu_suggestion_id")
            return

        # 在当前缓存的建议列表中查找对应 ID 的建议
        suggestion = None
        for s in self._current_xpu_suggestions:
            if s.id == action.xpu_suggestion_id:
                suggestion = s
                break

        # 建议不存在：可能已被过滤或 ID 拼写错误
        if not suggestion:
            logger.warning(f"未找到 XPU 建议: {action.xpu_suggestion_id}")
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": f"[XPU BLOCKED] 建议 {action.xpu_suggestion_id} 不存在或已被禁用，请改用 SHELL_COMMAND",
                    "stderr": "",
                },
            })
            return

        # 记录执行前的错误信息，用于归因对比
        error_before = self._state.last_error or ""

        # --- A. 存档（Checkpoint）---
        # 使用 docker commit 创建容器快照，失败时可回滚
        ckpt_tag = f"step_{self._state.step}_pre_xpu"
        self._env.create_checkpoint(ckpt_tag)
        logger.info(f"创建快照 {ckpt_tag}，开始推测执行 XPU 建议")

        # --- B. 试错（Trial）---
        # 如果建议的命令列表为空，跳过执行
        if not suggestion.commands:
            logger.warning(f"XPU 建议 {suggestion.id} 的 commands 为空，跳过执行")
            self._state.record_tried_suggestion(suggestion.id)
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": f"[XPU SKIP] {suggestion.id}：commands 为空，未执行",
                    "stderr": "",
                },
            })
            return

        # 逐条执行建议中的命令，任一命令失败即中断
        success = True
        logs: list[CommandResult] = []
        for cmd in suggestion.commands:
            result = self._env.exec_run(cmd)
            logs.append(result)
            if not result.success:
                success = False
                break

        # --- C. 验证与归因（Verification & Attribution）---
        # 获取执行后的错误信息（用于与执行前对比）
        error_after = ""
        if not success and logs:
            error_after = logs[-1].stderr or logs[-1].stdout

        # 计算归因分数：用于 XPU telemetry 反馈
        if success:
            attribution_score = 1.0   # 完全成功
            outcome = "SUCCESS"
        elif error_after and error_after != error_before:
            attribution_score = -1.0  # 产生了新错误，比之前更糟
            outcome = "FAIL"
        else:
            attribution_score = 0.0   # 无效果，错误信息没变
            outcome = "FAIL"

        # --- D. 反馈（Feedback Loop）---
        # RetrieverAgent 模式下，telemetry 由 RetrieverAgent 在 retrieve() 时
        # 自动延迟审计，无需主 Agent 显式调用。
        # 非 RetrieverAgent 模式回退到旧的即时反馈机制。
        if not self._retriever:
            report = AttributionReport(
                suggestion_id=suggestion.id,
                timestamp=time.time(),
                repo_context=self._state.repo_url,
                outcome=outcome,
                error_before=error_before,
                error_after=error_after,
                score=attribution_score,
                logs=logs,
            )
            self._xpu.submit_feedback(report)

        # --- E. 决策分支（Decision Branch）---
        if not success:
            # 执行失败：回滚到快照，恢复环境
            logger.info(f"XPU 建议 {suggestion.id} 执行失败，回滚中...")
            self._env.rollback_to_checkpoint()
            self._state.last_error = error_before  # 恢复执行前的错误状态
        else:
            # 执行成功：保留环境变更
            logger.info(f"XPU 建议 {suggestion.id} 验证通过")
            self._state.last_error = None  # 清除错误

        # 无论成功失败，都标记为"已尝试"，防止 LLM 对同一条建议无限循环重试
        self._state.record_tried_suggestion(suggestion.id)

        # 记录历史：将执行结果写入 history，供 LLM 后续参考
        cmd_outputs = "\n".join(
            f"$ {log.get('command', '')}\n{(log.get('stdout') or log.get('stderr') or '')[:300]}"
            for log in [l.to_dict() for l in logs[:3]]  # 最多记录前 3 条命令的输出
        )
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[XPU {outcome}] {suggestion.id}\n{cmd_outputs}",
                "stderr": "",
            },
        })

    def _handle_set_env(self, action: AgentAction) -> None:
        """处理 SET_ENV 动作：在容器中持久化设置环境变量

        与在 SHELL_COMMAND 中使用 export 不同，SET_ENV 设置的环境变量
        会持久化到后续所有命令的执行环境中（通过 EnvironmentManager 维护）。

        Args:
            action: LLM 输出的动作对象，需要 env_key 和 env_value
        """
        if not action.env_key or action.env_value is None:
            logger.warning("SET_ENV 缺少 env_key 或 env_value")
            return

        # 在容器环境中设置变量（后续 exec_run 都会带上该变量）
        self._env.set_env(action.env_key, action.env_value)
        # 同时在 Agent 状态中记录，方便追踪
        self._state.env_vars[action.env_key] = action.env_value

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[SET_ENV] {action.env_key}={action.env_value}",
                "stderr": "",
            },
        })
        # SET_ENV 是主动配置动作，清除之前的错误状态，避免 LLM 继续纠结旧错误
        self._state.last_error = None

    def _handle_rollback_env(self, action: AgentAction) -> None:
        """处理 ROLLBACK_ENV 动作：回滚容器到最近快照

        使用 EnvironmentManager 的快照栈（LIFO），弹出最近一个快照
        并用它恢复容器状态。通常在多次修复失败导致环境损坏时使用。

        Args:
            action: LLM 输出的动作对象（此动作无需额外参数）
        """
        # rollback_to_checkpoint 内部从快照栈弹出最近的快照并恢复
        success = self._env.rollback_to_checkpoint()
        status = "成功" if success else "失败（无可用快照）"
        logger.info(f"[ROLLBACK_ENV] 回滚 {status}")
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[ROLLBACK_ENV] {status}",
                "stderr": "",
            },
        })

    def _handle_verify(self, action: AgentAction) -> bool:
        """处理 VERIFY 动作：调用 VerifierAgent 运行 pytest

        验证是一个昂贵的操作——VerifierAgent 会启动一个 ReAct 子循环
        来探测项目结构、执行 pytest --co -q 和 pytest -x -q。
        LLM 应仅在确信环境已就绪时才调用此动作。

        验证通过后：自动标记任务完成，并触发经验存储。
        验证失败后：将 pytest 输出反馈给 LLM，继续修复循环。

        Args:
            action: LLM 输出的动作对象（此动作无需额外参数）

        Returns:
            True 表示验证通过（调用方应退出主循环）
            False 表示验证失败（调用方继续循环修复）
        """
        logger.info("[VERIFY] 开始 pytest 验证")
        # 创建 VerifierAgent，复用当前容器，传递 hint 给 Verifier
        hint = action.verify_hint or ""
        if hint:
            logger.info(f"[VERIFY] 传递 hint 给 Verifier: {hint}")
        verifier = VerifierAgent(self._env, hint=hint)
        result = verifier.verify()  # 运行 pytest 验证

        logger.info(
            f"[VERIFY] 结果: success={result.success}, "
            f"framework={result.test_framework}, "
            f"collected={result.collect_count}, exit_code={result.exit_code}"
        )

        # 组织验证结果摘要，无论成功失败都写入历史供 LLM 参考
        verify_summary = (
            f"验证框架: {result.test_framework}\n"
            f"收集测试数: {result.collect_count}\n"
            f"退出码: {result.exit_code}\n"
            f"输出:\n{result.stdout}\n"
            f"错误:\n{result.stderr}"
        )

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": result.exit_code,
                "stdout": (
                    f"[VERIFY] success={result.success}, framework={result.test_framework}, "
                    f"collected={result.collect_count}\n{result.stdout or ''}"
                )[:1000],  # 截断过长输出
                "stderr": (result.stderr or "")[:500],
            },
        })

        if result.success:
            # 验证通过：标记任务完成
            logger.info("[VERIFY] 验证通过，自动标记任务完成")
            self._state.completed = True
            self._state.final_message = f"pytest 验证通过，{result.collect_count} 个测试用例"
            self._state.last_error = None
            self._last_verify_messages = result.messages
            return True
        else:
            # 验证失败：将完整 pytest 输出反馈给 LLM，让它继续修复
            logger.warning("[VERIFY] 验证失败，将 pytest 输出反馈给 LLM 继续修复")
            self._state.last_error = verify_summary
            return False

    def _handle_finish(self, action: AgentAction) -> None:
        """处理 FINISH 动作：标记任务完成

        LLM 应仅在验证通过后才输出此动作。
        完成后会尝试将本次修复经验存入 XPU 知识库。

        Args:
            action: LLM 输出的动作对象，action.message 为完成消息
        """
        self._state.completed = True
        self._state.final_message = action.message or "任务完成"

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[FINISH] {self._state.final_message}",
                "stderr": "",
            },
        })

    # =========================================================================
    # 情境构建（供向量检索使用）
    # =========================================================================

    def _build_situation(self, last_error: str) -> str:
        """构建原始情境文本（命令历史 + 错误原文 + 仓库信息）

        与 describe_situation() 的 LLM 语义总结配合使用：
        - LLM 总结：语义化，匹配经验库的 situation_triggers 和 advice_nl
        - 原始文本：保留关键词，匹配经验库的 keywords 和 regex
        """
        parts = []

        recent = self._state.get_recent_history(3)
        if recent:
            done_steps = []
            for entry in recent:
                action = entry.get("action", {})
                cmd = action.get("content", {}).get("command", "")
                if cmd:
                    result = entry.get("result", {})
                    exit_code = result.get("exit_code", "?")
                    done_steps.append(f"  {cmd} (exit={exit_code})")
            if done_steps:
                parts.append("已执行的操作:\n" + "\n".join(done_steps))

        if last_error:
            error_text = last_error[:1500] if len(last_error) > 1500 else last_error
            parts.append(f"当前遇到的问题:\n{error_text}")

        parts.append(f"目标仓库: {self._state.repo_url}")

        return "\n\n".join(parts)

    # =========================================================================
    # 经验存储（Online Learning）
    # =========================================================================

    def _store_experience_if_applicable(self) -> None:
        """任务成功完成后，将本次修复经验存入 XPU 向量数据库（如果可用）

        Online Learning 流程：
        1. 将 Agent 的 history 转换为 Repo2Run 兼容的轨迹 JSONL 格式
        2. 写入临时文件
        3. 调用 extract_xpu_from_trajs 进行 LLM 提取（启发式筛选 + LLM 决策）
        4. 将提取出的 XPU 条目逐条去重入库

        前置条件：
        - XPU 客户端必须是 VectorXPUClient（直接连接向量数据库）
        - 任务必须已标记为完成（self._state.completed = True）

        注意：此方法不抛出异常，存储失败不影响主任务结果。
        """
        from .xpu_client import VectorXPUClient
        # 只有 VectorXPUClient 才能直接写入向量数据库
        if not isinstance(self._xpu, VectorXPUClient):
            return
        # 任务未完成则不存储
        if not self._state.completed:
            return

        try:
            import json
            import shutil
            import tempfile
            from pathlib import Path

            # === Step 1: 将 agent history 转换为轨迹 JSONL 格式 ===
            # 格式兼容 extract_xpu_from_trajs_mvp.py 中 "our_agent" 格式
            traj = []
            for entry in self._state.history:
                action = entry.get("action", {})
                result = entry.get("result", {})

                # assistant 消息：将命令包装为 bash 代码块（模拟 LLM 的输出格式）
                cmd = action.get("content", {}).get("command")
                if cmd:
                    traj.append({
                        "role": "assistant",
                        "content": f"执行命令:\n```bash\n{cmd}\n```"
                    })

                # system 消息：命令执行的输出/错误信息
                if result:
                    output = result.get("stderr") or result.get("stdout") or ""
                    if output:
                        traj.append({
                            "role": "system",
                            "content": output
                        })

            if not traj:
                logger.debug("[XPU Store] 轨迹为空，跳过经验存储")
                return

            # === Step 2: 写入临时 JSONL 文件 ===
            # 文件名格式遵循 Repo2Run 命名规范：{safe_name}@HEAD.jsonl
            tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_agent_"))
            try:
                # 从仓库 URL 提取安全的文件名（替换 / 为 __）
                repo_path = self._state.repo_url.rstrip("/")
                if "github.com/" in repo_path:
                    repo_path = repo_path.split("github.com/")[-1]
                safe_name = repo_path.replace("/", "__")

                traj_dir = tmp_dir / "trajs"
                traj_dir.mkdir()
                jsonl_path = traj_dir / f"{safe_name}@HEAD.jsonl"

                # 逐行写入 JSONL（每行一个 JSON 对象）
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for step in traj:
                        f.write(json.dumps(step, ensure_ascii=False) + "\n")

                # === Step 3: LLM 提取 ===
                # 复用离线提取管道：启发式筛选 → LLM 决策 → 结构化 XPU
                extracted_file = tmp_dir / "extracted.jsonl"
                from .xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
                extract_xpu_from_trajs(jsonl_path, extracted_file)

                # === Step 4: 收集所有有效经验 ===
                # 一条轨迹可能提炼出多条 XPU（不同的错误-修复对）
                xpu_objects = []
                if extracted_file.exists():
                    with open(extracted_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            rec = json.loads(line)
                            # 只保留 LLM 决策为 "xpu"（值得存储）的条目
                            if rec.get("llm_decision") == "xpu" and rec.get("xpu"):
                                xpu_objects.append(rec["xpu"])

                if not xpu_objects:
                    logger.debug("[XPU Store] LLM 决定跳过，无有效经验存储")
                    return

                logger.info(f"[XPU Store] LLM 提取出 {len(xpu_objects)} 条经验，逐条入库")

                # === Step 5 & 6: 逐条构建 XpuEntry 并去重入库 ===
                from .xpu.xpu_adapter import XpuEntry, XpuAtom
                from .xpu.xpu_vector_store import build_xpu_text, text_to_embedding
                from .xpu.xpu_dedup import dedup_and_store

                for i, xpu_obj in enumerate(xpu_objects):
                    # 将 JSON 中的 atoms 转换为 XpuAtom 对象
                    atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                             for a in xpu_obj.get("atoms", [])]
                    # 构建 XpuEntry 对象
                    xpu_entry = XpuEntry(
                        id=xpu_obj.get("id"),
                        context=xpu_obj.get("context", {}),
                        signals=xpu_obj.get("signals", {}),
                        advice_nl=xpu_obj.get("advice_nl", []),
                        atoms=atoms,
                    )

                    # 生成文本表示并计算 embedding 向量
                    text = build_xpu_text(xpu_entry)
                    embedding = text_to_embedding(text)
                    # 去重入库：检查是否与已有 XPU 重复，必要时智能合并
                    dedup_result = dedup_and_store(self._xpu._store, xpu_entry, embedding, use_llm=True)
                    logger.info(f"[XPU Store] [{i+1}/{len(xpu_objects)}] {dedup_result['action']}: {dedup_result['reason']}")

            finally:
                # 清理临时目录
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            # 经验存储失败不影响主任务结果，仅记录警告
            logger.warning(f"[XPU Store] 经验存储失败（不影响任务结果）: {e}")

    # =========================================================================
    # 生命周期管理
    # =========================================================================

    def _close_clients(self) -> None:
        """关闭 LLM 和 XPU 的 HTTP 连接（不销毁容器）

        在主循环结束后调用，释放 HTTP 连接资源。
        容器保留给验证阶段使用。
        """
        logger.info("关闭 LLM/XPU/Retriever 连接...")
        if self._retriever:
            self._retriever.close(full_history=self._state.history)  # 关闭前执行最终审计
        self._llm.close()  # 关闭 LLM 客户端的 httpx.Client
        if hasattr(self._xpu, "close"):
            self._xpu.close()  # 关闭 XPU 客户端的 HTTP 连接

    def __enter__(self):
        """上下文管理器入口：支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：自动关闭连接并清理容器

        使用 with SpeculativeSetupAgent(...) as agent: 时，
        退出 with 块会自动关闭所有资源并销毁容器。
        """
        self._close_clients()
        self._env.cleanup()  # 销毁容器并清理快照镜像
        return False  # 不吞掉异常
