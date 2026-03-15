"""
XPU 知识接口（按 blueprint 1.2 节定义）
提供环境配置问题的诊断建议

本模块定义了 XPU（eXPerience Unit，经验单元）客户端的完整体系：
- XPUClientBase：抽象基类，定义 query（查询建议）和 submit_feedback（提交反馈）两个核心接口
- MockXPUClient：基于预定义关键词的本地 mock 客户端，用于开发调试
- HTTPXPUClient：通过 HTTP 调用远程 XPU 服务的客户端
- VectorXPUClient：基于 PostgreSQL + pgvector 向量数据库的客户端，用于生产环境
- NoopXPUClient：完全禁用 XPU 的空实现客户端

XPU 系统的核心流程：
  Agent 遇到错误 → 调用 query() 检索相关 XPU → 返回建议列表 → Agent 决策执行 → 调用 submit_feedback() 回写结果
"""

import json  # 用于 JSON 序列化归因报告
import uuid  # 用于生成 mock 建议的唯一 ID
from abc import ABC, abstractmethod  # 抽象基类机制
from typing import Any  # 类型标注

import httpx  # HTTP 客户端库，用于调用远程 XPU 服务

from .config import get_config  # 获取全局配置（决定使用哪种 XPU 客户端）
from .logger import get_logger  # 统一日志系统
from .models import XPUSuggestion, AttributionReport  # XPU 建议 和 归因报告 数据结构

logger = get_logger("xpu")  # 创建 xpu 模块专用的日志记录器


# ============================================================================
# 抽象基类：定义所有 XPU 客户端必须实现的接口
# ============================================================================

class XPUClientBase(ABC):
    """XPU 客户端抽象基类（按 blueprint 1.2 节定义）

    所有 XPU 客户端实现（Mock / HTTP / Vector / Noop）都必须继承此类，
    并实现 query() 和 submit_feedback() 两个方法。
    """

    @abstractmethod
    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        """查询诊断建议

        Args:
            context: 查询上下文字典，包含以下可能的键：
                - error / error_log: 当前遇到的错误日志文本
                - repo_metadata: 仓库元信息（语言、工具链等）
                - current_packages: 当前已安装的包列表
            exclude_ids: 需要排除的 XPU ID 列表（已尝试过的建议）

        Returns:
            XPUSuggestion 列表，按置信度从高到低排序，通常返回 Top-K（均为未尝试的）
        """
        pass

    @abstractmethod
    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告（按 blueprint 1.2 节定义）

        当 Agent 使用 TRY_XPU_SUGGESTION 执行 XPU 建议后，调用此方法
        将执行结果（成功/失败/无效果）回写到 XPU 系统，用于更新 telemetry。

        Args:
            report: 归因报告，包含 suggestion_id、outcome、score 等信息
        """
        pass


# ============================================================================
# Mock 客户端：基于预定义关键词匹配的本地知识库，用于开发调试
# ============================================================================

class MockXPUClient(XPUClientBase):
    """Mock XPU 客户端，预定义常见问题解决方案

    内置了常见的环境问题-解决方案映射（如 npm 缺失、Python 模块缺失等），
    通过关键词匹配来返回建议。不连接任何外部服务。
    """

    # 预定义的问题-解决方案映射表
    # 每条记录包含：keywords（匹配关键词列表）、description（描述）、commands（修复命令）、confidence（基础置信度）
    KNOWLEDGE_BASE: list[dict] = [
        {
            "keywords": ["command not found", "npm"],  # 匹配 npm 未安装的错误
            "description": "安装 Node.js 和 npm",
            "commands": [
                "apt-get update",
                "apt-get install -y nodejs npm",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["command not found", "python", "pip"],  # 匹配 Python/pip 未安装
            "description": "安装 Python 和 pip",
            "commands": [
                "apt-get update",
                "apt-get install -y python3 python3-pip python3-venv",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["ModuleNotFoundError", "No module named"],  # 匹配 Python 模块缺失
            "description": "安装缺失的 Python 依赖",
            "commands": [
                "pip install -r requirements.txt",
            ],
            "confidence": 0.8,
        },
        {
            "keywords": ["ENOENT", "package.json"],  # 匹配 Node.js 依赖未安装
            "description": "安装 Node.js 依赖",
            "commands": [
                "npm install",
            ],
            "confidence": 0.85,
        },
        {
            "keywords": ["permission denied"],  # 匹配权限不足错误
            "description": "修改文件权限",
            "commands": [
                "chmod +x ./script.sh",
            ],
            "confidence": 0.7,
        },
        {
            "keywords": ["EACCES", "npm", "global"],  # 匹配 npm 全局安装权限问题
            "description": "配置 npm 全局安装路径",
            "commands": [
                "npm config set prefix ~/.npm-global",
                "export PATH=~/.npm-global/bin:$PATH",
            ],
            "confidence": 0.8,
        },
        {
            "keywords": ["cargo", "command not found"],  # 匹配 Rust 工具链未安装
            "description": "安装 Rust 和 Cargo",
            "commands": [
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                "source $HOME/.cargo/env",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["go", "command not found"],  # 匹配 Go 语言未安装
            "description": "安装 Go 语言",
            "commands": [
                "apt-get update",
                "apt-get install -y golang",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["java", "command not found", "javac"],  # 匹配 Java 未安装
            "description": "安装 JDK",
            "commands": [
                "apt-get update",
                "apt-get install -y default-jdk",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["docker", "command not found"],  # 匹配 Docker 未安装
            "description": "安装 Docker",
            "commands": [
                "apt-get update",
                "apt-get install -y docker.io",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["make", "command not found"],  # 匹配 make 工具未安装
            "description": "安装构建工具 build-essential",
            "commands": [
                "apt-get update",
                "apt-get install -y build-essential",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["cmake", "command not found"],  # 匹配 cmake 未安装
            "description": "安装 CMake",
            "commands": [
                "apt-get update",
                "apt-get install -y cmake",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["libmysqlclient", "mysql_config"],  # 匹配 MySQL 开发库缺失
            "description": "安装 MySQL 客户端开发库",
            "commands": [
                "apt-get update",
                "apt-get install -y libmysqlclient-dev",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["libpq", "pg_config"],  # 匹配 PostgreSQL 开发库缺失
            "description": "安装 PostgreSQL 客户端开发库",
            "commands": [
                "apt-get update",
                "apt-get install -y libpq-dev",
            ],
            "confidence": 0.9,
        },
    ]

    def __init__(self):
        # 存储所有提交过的归因报告（仅用于调试和日志回溯）
        self._feedback_history: list[AttributionReport] = []

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        """基于关键词匹配查询建议

        从预定义知识库中逐条匹配，计算匹配分数，返回分数最高的 Top-3 建议。
        匹配公式：score = (匹配关键词数 / 总关键词数) × 基础置信度
        """
        # 从上下文中提取错误日志文本（兼容两种 key 名）
        error_log = context.get("error", "") or context.get("error_log", "")
        combined_text = f"{error_log}".lower()  # 转小写以便不区分大小写匹配

        suggestions = []  # 存储匹配到的建议（带分数）

        # 遍历知识库中的每条规则，尝试关键词匹配
        for entry in self.KNOWLEDGE_BASE:
            # 计算当前规则中有多少个关键词命中了错误日志
            matched_keywords = sum(
                1 for kw in entry["keywords"]
                if kw.lower() in combined_text
            )

            if matched_keywords > 0:
                # 匹配分数 = 命中关键词比例 × 该规则的基础置信度
                score = matched_keywords / len(entry["keywords"]) * entry["confidence"]
                if score > 0.3:  # 分数阈值：低于 0.3 的弱匹配不返回
                    suggestion = XPUSuggestion(
                        id=f"xpu_{uuid.uuid4().hex[:8]}",  # 生成随机唯一 ID
                        description=entry["description"],  # 问题描述
                        commands=entry["commands"],  # 修复命令列表
                        confidence=score,  # 最终置信度
                        source="mock",  # 标记来源为 mock 客户端
                    )
                    suggestions.append((score, suggestion))  # 保存（分数, 建议）元组

        # 按置信度从高到低排序，取前 3 条
        suggestions.sort(key=lambda x: x[0], reverse=True)
        result = [s[1] for s in suggestions[:3]]  # 只保留建议对象，丢弃分数

        # 记录查询结果日志
        if result:
            logger.info(f"XPU 查询返回 {len(result)} 条建议")
            for s in result:
                logger.info(f"  - {s}")
        else:
            logger.debug(f"XPU 未找到匹配建议")

        return result

    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告（Mock 实现：仅记录到日志，不回写数据库）"""
        self._feedback_history.append(report)  # 追加到历史列表

        # 打印详细的归因报告到日志，方便调试
        logger.info("=" * 60)
        logger.info("XPU 归因报告 (Attribution Report)")
        logger.info("=" * 60)
        logger.info(f"  suggestion_id: {report.suggestion_id}")  # 被评价的建议 ID
        logger.info(f"  timestamp: {report.timestamp}")  # 执行时间戳
        logger.info(f"  repo_context: {report.repo_context}")  # 仓库上下文
        logger.info(f"  outcome: {report.outcome}")  # 执行结果（SUCCESS/FAIL）
        logger.info(f"  score: {report.score}")  # 归因分数（1.0 成功 / -1.0 失败 / 0.0 无效果）
        logger.info(f"  error_before: {report.error_before[:200] if report.error_before else 'N/A'}...")  # 执行前错误
        logger.info(f"  error_after: {report.error_after[:200] if report.error_after else 'N/A'}...")  # 执行后错误
        logger.info(f"  执行日志:")
        for i, log in enumerate(report.logs):  # 逐条打印命令执行日志
            logger.info(f"    [{i+1}] {log.command} -> exit_code={log.exit_code}")
        logger.info("=" * 60)


# ============================================================================
# HTTP 客户端：连接远程 XPU 服务（适用于有中心化 XPU 服务器的场景）
# ============================================================================

class HTTPXPUClient(XPUClientBase):
    """HTTP XPU 客户端，连接真实 XPU 服务

    通过 HTTP POST 请求调用远程 XPU 服务的 /api/query 和 /api/feedback 接口。
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")  # 去除尾部斜杠，避免拼接出双斜杠
        self._client = httpx.Client(timeout=30)  # 创建 HTTP 客户端，30 秒超时

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        """调用远程 XPU 服务的 /api/query 接口进行检索"""
        try:
            # 向远程 XPU 服务发送 POST 请求，传入查询上下文
            response = self._client.post(
                f"{self._base_url}/api/query",
                json=context,
            )
            response.raise_for_status()  # 检查 HTTP 状态码，非 2xx 抛异常
            data = response.json()  # 解析 JSON 响应

            # 将响应中的建议列表转换为 XPUSuggestion 对象
            suggestions = []
            for item in data.get("suggestions", []):
                suggestions.append(XPUSuggestion(
                    id=item["id"],
                    description=item["description"],
                    commands=item.get("commands", []),
                    confidence=item.get("confidence", 0.5),  # 默认置信度 0.5
                    source="http",  # 标记来源为 HTTP 远程服务
                ))

            logger.info(f"XPU HTTP 查询返回 {len(suggestions)} 条建议")
            return suggestions

        except httpx.HTTPError as e:
            # HTTP 请求失败时记录警告并返回空列表（不中断 agent 主流程）
            logger.warning(f"XPU 服务调用失败: {e}")
            return []

    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告到远程 XPU 服务的 /api/feedback 接口"""
        try:
            response = self._client.post(
                f"{self._base_url}/api/feedback",
                json=report.to_dict(),  # 序列化归因报告为字典
            )
            response.raise_for_status()
            logger.info(f"归因报告已提交: {report.suggestion_id}")

            # 同时在本地日志中记录一份归因报告（方便回溯）
            logger.info("=" * 60)
            logger.info("XPU 归因报告 (Attribution Report)")
            logger.info(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            logger.info("=" * 60)

        except httpx.HTTPError as e:
            logger.warning(f"提交归因报告失败: {e}")

    def close(self) -> None:
        """关闭 HTTP 客户端连接"""
        self._client.close()


# ============================================================================
# Vector 客户端：基于 PostgreSQL + pgvector 向量数据库的生产环境客户端
# 这是最核心的 XPU 客户端实现，通过 embedding 向量相似度检索历史经验
# ============================================================================

class VectorXPUClient(XPUClientBase):
    """基于 PostgreSQL 向量数据库的 XPU 客户端（复用 xpu_standalone）

    工作流程：
    1. 将错误日志文本转为 embedding 向量
    2. 在 pgvector 数据库中做向量相似度检索
    3. 将检索到的 XPU 条目转换为 XPUSuggestion
    4. 同时递增被检索到的 XPU 的 hits 计数
    """

    def __init__(self, dns: str):
        # 延迟导入 XPU 相关模块（避免在不使用 Vector 客户端时也加载数据库依赖）
        from .xpu.xpu_vector_store import XpuVectorStore, text_to_embedding, build_xpu_text
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands
        self._store = XpuVectorStore(connection_string=dns)  # 初始化向量数据库连接
        self._text_to_embedding = text_to_embedding  # 文本转 embedding 向量的函数
        self._build_xpu_text = build_xpu_text  # XPU 条目转可搜索文本的函数
        self._render_atom_to_commands = render_atom_to_commands  # 将 XPU atom 渲染为可执行 bash 命令
        self._id_to_raw: dict[str, dict] = {}  # 缓存 suggestion_id → 原始检索结果的映射
        # 日志中隐藏连接串的敏感部分（只显示 @ 后面的主机信息）
        logger.info(f"VectorXPUClient 初始化完成（连接: {dns.split('@')[-1] if '@' in dns else '...'}）")

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        """向量相似度检索 XPU 经验

        核心检索流程：
        1. 从 context 中提取错误日志
        2. 调用 embedding API 将错误文本转为 1536 维向量
        3. 在 pgvector 数据库中做余弦相似度检索，返回 Top-3（排除已尝试的 ID）
        4. 将 atoms（结构化操作）渲染为可执行的 bash 命令
        5. 构造 XPUSuggestion 返回给 Agent

        Args:
            context: 查询上下文，包含 error/error_log 等字段
            exclude_ids: 需要排除的 XPU ID 列表（已尝试过的建议），保证返回 k 条均为未尝试的
        """
        # 补充导入，解决作用域问题（因为在 __init__ 中是局部导入）
        from .xpu.xpu_adapter import XpuAtom

        # 从上下文中提取错误日志文本（兼容 error 和 error_log 两种 key）
        error_text = context.get("error", "") or context.get("error_log", "")
        if not error_text:
            return []  # 没有错误文本就无法检索

        try:
            # 将错误文本转为 embedding 向量（调用 embedding API）
            embedding = self._text_to_embedding(error_text)
            # 在 pgvector 数据库中检索最相似的 3 条 XPU，排除已尝试的 ID
            results = self._store.search(embedding, k=3, exclude_ids=exclude_ids)
        except Exception as e:
            logger.warning(f"VectorXPUClient 查询失败: {e}")
            return []

        suggestions = []  # 最终返回的建议列表
        result_ids = []  # 记录所有被检索到的 XPU ID（用于更新 hits 计数）

        # 遍历检索结果，将每条 XPU 转换为 XPUSuggestion
        for res in results:
            xpu_id = res["id"]  # XPU 唯一标识
            advice_nl = res.get("advice_nl") or []  # 自然语言建议（中文）
            atoms = res.get("atoms") or []  # 结构化操作列表（如 pip_install, set_env 等）
            similarity = float(res.get("similarity", 0.5))  # 向量相似度分数

            # 将 atoms 转换为可执行的 bash 命令列表
            # 例如 {"name": "pip_install", "args": {"package": "numpy"}} → "pip install numpy"
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))  # 构造 XpuAtom 对象
                commands.extend(self._render_atom_to_commands(atom))  # 渲染为 bash 命令

            # [P0-3] 置信度分级：基于 composite_score 标注置信度等级
            composite = float(res.get("composite_score", similarity))
            if composite >= 0.8:
                confidence_level = "high"
            elif composite >= 0.7:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # 构造 XPUSuggestion 对象
            suggestion = XPUSuggestion(
                id=xpu_id,  # XPU ID
                description=f"[{confidence_level}] " + "\n".join(advice_nl),  # 置信度标签 + 建议文本
                commands=commands,  # 可执行的 bash 命令列表
                confidence=composite,  # 使用复合分数作为置信度
                source="vector_db",  # 标记来源为向量数据库
            )
            suggestions.append(suggestion)
            result_ids.append(xpu_id)
            self._id_to_raw[xpu_id] = res  # 缓存原始结果，供后续 feedback 时使用

        # 批量更新所有被检索到的 XPU 的 hits 计数（记录"被使用了几次"）
        if result_ids:
            try:
                self._store.increment_telemetry(result_ids, "hits")
            except Exception as e:
                logger.warning(f"遥测 hits 写入失败: {e}")

        # 记录查询结果日志
        logger.info(f"VectorXPU 查询返回 {len(suggestions)} 条建议")
        for s in suggestions:
            logger.info(f"  - [{s.confidence:.3f}] {s.id}: {s.description[:60]}...")

        return suggestions

    def submit_feedback(self, report: AttributionReport) -> None:
        """将归因结果写入遥测数据库

        根据 report.score 更新对应 XPU 的 telemetry：
        - score > 0（建议有效）→ successes + 1
        - score < 0（建议有害）→ failures + 1
        - score == 0（无效果）→ 不更新 successes/failures
        """
        try:
            if report.score > 0:
                # 建议执行成功：递增该 XPU 的 successes 计数
                self._store.increment_telemetry([report.suggestion_id], "successes")
            elif report.score < 0:
                # 建议执行失败且导致新错误：递增该 XPU 的 failures 计数
                self._store.increment_telemetry([report.suggestion_id], "failures")
        except Exception as e:
            logger.warning(f"遥测 feedback 写入失败: {e}")

        # 打印归因报告到日志
        logger.info("=" * 60)
        logger.info("XPU 归因报告 (VectorXPUClient)")
        logger.info(f"  suggestion_id: {report.suggestion_id}")
        logger.info(f"  outcome: {report.outcome}  score: {report.score}")
        logger.info(f"  error_before: {(report.error_before or '')[:200]}...")
        logger.info(f"  error_after:  {(report.error_after or '')[:200]}...")
        logger.info("=" * 60)

    def close(self) -> None:
        """关闭向量数据库连接池"""
        self._store.close()


# ============================================================================
# 工厂函数：根据配置创建对应的 XPU 客户端实例
# ============================================================================

def create_xpu_client() -> XPUClientBase:
    """创建 XPU 客户端实例

    根据配置文件中的 xpu 配置项决定使用哪种客户端：
    - config.disabled = True → NoopXPUClient（完全禁用）
    - config.vector_enabled = True 且有 db_dns → VectorXPUClient（向量数据库）
    - config.enabled = True → HTTPXPUClient（HTTP 远程服务）
    - 其他 → MockXPUClient（本地 mock）
    """
    config = get_config().xpu  # 获取 xpu 相关配置

    if config.disabled:
        # 完全禁用 XPU 功能
        logger.info("XPU 已禁用，使用 NoopXPU 客户端")
        return NoopXPUClient()
    if config.vector_enabled and config.db_dns:
        # 优先使用向量数据库客户端（生产环境推荐）
        logger.info("使用 VectorXPU 客户端（PostgreSQL 向量数据库）")
        return VectorXPUClient(config.db_dns)
    elif config.enabled:
        # 其次使用 HTTP 远程服务客户端
        logger.info(f"使用 HTTP XPU 客户端: {config.base_url}")
        return HTTPXPUClient(config.base_url)
    else:
        # 兜底使用 Mock 客户端（开发调试用）
        logger.info("使用 Mock XPU 客户端")
        return MockXPUClient()


# ============================================================================
# Noop 客户端：完全禁用 XPU，所有方法返回空
# ============================================================================

class NoopXPUClient(XPUClientBase):
    """完全禁用的 XPU 客户端：不提供建议，不回传反馈。

    当配置中 xpu.disabled = True 时使用此客户端，
    确保 Agent 主流程不受 XPU 系统影响。
    """

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        return []  # 永远返回空建议列表

    def submit_feedback(self, report: AttributionReport) -> None:
        return None  # 不做任何操作
