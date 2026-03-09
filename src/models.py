"""
数据结构定义
按照 blueprint.md 规范
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    """Agent 动作类型（按 blueprint 3.1 节定义）"""
    SHELL_COMMAND = "SHELL_COMMAND"           # 直接执行命令
    TRY_XPU_SUGGESTION = "TRY_XPU_SUGGESTION"  # 推测执行 XPU 建议
    SET_ENV = "SET_ENV"                        # 设置环境变量
    ROLLBACK_ENV = "ROLLBACK_ENV"              # 回滚容器到最近快照
    VERIFY = "VERIFY"                          # 运行 pytest 验证当前环境
    FINISH = "FINISH"                          # 结束任务


@dataclass
class CommandResult:
    """命令执行结果（按 blueprint 1.1 节定义）"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }

    def __str__(self) -> str:
        status = "成功" if self.success else f"失败(退出码={self.exit_code})"
        output = self.stdout or self.stderr
        if self.truncated:
            output += "\n... [输出已截断]"
        return f"[{status}] {self.command}\n{output}"


@dataclass
class XPUSuggestion:
    """XPU 返回的诊断建议（按 blueprint 1.2 节定义）"""
    id: str                   # 建议 ID
    description: str          # 简短描述
    commands: list[str]       # 具体的 Shell 指令
    confidence: float         # 置信度 0-1
    source: str = "mock"      # 来源（mock/http）

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "commands": self.commands,
            "confidence": self.confidence,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"[ID: {self.id}] {self.description} (置信度: {self.confidence:.2f})"


@dataclass
class AttributionReport:
    """归因报告（按 blueprint 5 节定义）"""
    suggestion_id: str        # XPU 建议 ID
    timestamp: float          # 时间戳
    repo_context: str         # 当前仓库名/语言
    outcome: str              # "SUCCESS" | "FAIL" | "PARTIAL"
    error_before: str         # 采纳建议前的报错
    error_after: str          # 采纳建议后的报错
    score: float              # 1.0 (解决) -> 0.0 (无效) -> -1.0 (新错误)
    logs: list[CommandResult] = field(default_factory=list)  # 执行日志

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "timestamp": self.timestamp,
            "repo_context": self.repo_context,
            "outcome": self.outcome,
            "error_before": self.error_before,
            "error_after": self.error_after,
            "score": self.score,
            "logs": [log.to_dict() for log in self.logs],
        }

    def __str__(self) -> str:
        return (
            f"[归因报告] suggestion_id={self.suggestion_id}, "
            f"outcome={self.outcome}, score={self.score}"
        )


@dataclass
class AgentAction:
    """Agent 决策的动作（按 blueprint 3.1 节定义）"""
    action_type: ActionType
    thought: str = ""                         # LLM 的思考过程
    command: str | None = None                # SHELL_COMMAND 时的命令
    xpu_suggestion_id: str | None = None      # TRY_XPU_SUGGESTION 时的建议 ID
    reasoning: str | None = None              # 选择 XPU 建议的理由
    env_key: str | None = None                # SET_ENV 时的变量名
    env_value: str | None = None              # SET_ENV 时的变量值
    message: str | None = None                # FINISH 时的消息

    def to_dict(self) -> dict:
        result = {
            "thought": self.thought,
            "action_type": self.action_type.value,
            "content": {},
        }
        if self.action_type == ActionType.SHELL_COMMAND:
            result["content"]["command"] = self.command
        elif self.action_type == ActionType.TRY_XPU_SUGGESTION:
            result["content"]["xpu_suggestion_id"] = self.xpu_suggestion_id
            result["content"]["reasoning"] = self.reasoning
        elif self.action_type == ActionType.SET_ENV:
            result["content"]["env_key"] = self.env_key
            result["content"]["env_value"] = self.env_value
        elif self.action_type == ActionType.FINISH:
            result["content"]["message"] = self.message
        return result

    def __str__(self) -> str:
        if self.action_type == ActionType.SHELL_COMMAND:
            return f"[执行命令] {self.command}"
        elif self.action_type == ActionType.TRY_XPU_SUGGESTION:
            return f"[尝试XPU建议] id={self.xpu_suggestion_id}, 理由: {self.reasoning}"
        elif self.action_type == ActionType.SET_ENV:
            return f"[设置环境变量] {self.env_key}={self.env_value}"
        elif self.action_type == ActionType.ROLLBACK_ENV:
            return f"[回滚环境] {self.thought}"
        elif self.action_type == ActionType.VERIFY:
            return f"[验证环境] {self.thought}"
        elif self.action_type == ActionType.FINISH:
            return f"[结束] {self.message}"
        return f"[{self.action_type.value}]"


@dataclass
class AgentState:
    """Agent 运行状态"""
    repo_url: str
    container_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    step: int = 0
    max_steps: int = 50
    completed: bool = False
    final_message: str | None = None
    last_error: str | None = None             # 最近的错误信息
    failed_suggestions: set[str] = field(default_factory=set)  # 已失败的 XPU 建议 ID

    def add_to_history(self, entry: dict[str, Any]) -> None:
        """添加历史记录"""
        self.history.append({
            "step": self.step,
            "timestamp": time.time(),
            **entry,
        })

    def get_recent_history(self, n: int = 10) -> list[dict[str, Any]]:
        """获取最近 n 条历史记录"""
        return self.history[-n:]

    def record_failed_suggestion(self, suggestion_id: str) -> None:
        """记录失败的 XPU 建议，防止重试"""
        self.failed_suggestions.add(suggestion_id)

    def is_suggestion_failed(self, suggestion_id: str) -> bool:
        """检查建议是否已失败"""
        return suggestion_id in self.failed_suggestions


@dataclass
class SetupResult:
    """Setup 阶段结果"""
    repo_url: str
    container_id: str               # 保留的容器 ID
    completed: bool                  # LLM 是否主动 FINISH
    steps_taken: int
    final_message: str
    history: list[dict] = field(default_factory=list)  # 完整执行历史

    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "container_id": self.container_id,
            "completed": self.completed,
            "steps_taken": self.steps_taken,
            "final_message": self.final_message,
        }


@dataclass
class VerifyResult:
    """验证阶段结果"""
    success: bool                    # pytest 是否成功
    test_framework: str              # 检测到的框架 (pytest / unittest / none)
    collect_count: int               # 收集到多少测试用例
    command: str                     # 实际执行的验证命令
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "test_framework": self.test_framework,
            "collect_count": self.collect_count,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
