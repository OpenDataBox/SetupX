"""
XPU 知识接口（按 blueprint 1.2 节定义）
提供环境配置问题的诊断建议
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import get_config
from .logger import get_logger
from .models import XPUSuggestion, AttributionReport

logger = get_logger("xpu")


class XPUClientBase(ABC):
    """XPU 客户端抽象基类（按 blueprint 1.2 节定义）"""

    @abstractmethod
    def query(self, context: dict[str, Any]) -> list[XPUSuggestion]:
        """查询诊断建议

        Args:
            context: 包含 error_log, repo_metadata, current_packages 等

        Returns:
            Top-K 建议列表
        """
        pass

    @abstractmethod
    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告（按 blueprint 1.2 节定义）"""
        pass


class MockXPUClient(XPUClientBase):
    """Mock XPU 客户端，预定义常见问题解决方案"""

    # 预定义的问题-解决方案映射
    KNOWLEDGE_BASE: list[dict] = [
        {
            "keywords": ["command not found", "npm"],
            "description": "安装 Node.js 和 npm",
            "commands": [
                "apt-get update",
                "apt-get install -y nodejs npm",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["command not found", "python", "pip"],
            "description": "安装 Python 和 pip",
            "commands": [
                "apt-get update",
                "apt-get install -y python3 python3-pip python3-venv",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["ModuleNotFoundError", "No module named"],
            "description": "安装缺失的 Python 依赖",
            "commands": [
                "pip install -r requirements.txt",
            ],
            "confidence": 0.8,
        },
        {
            "keywords": ["ENOENT", "package.json"],
            "description": "安装 Node.js 依赖",
            "commands": [
                "npm install",
            ],
            "confidence": 0.85,
        },
        {
            "keywords": ["permission denied"],
            "description": "修改文件权限",
            "commands": [
                "chmod +x ./script.sh",
            ],
            "confidence": 0.7,
        },
        {
            "keywords": ["EACCES", "npm", "global"],
            "description": "配置 npm 全局安装路径",
            "commands": [
                "npm config set prefix ~/.npm-global",
                "export PATH=~/.npm-global/bin:$PATH",
            ],
            "confidence": 0.8,
        },
        {
            "keywords": ["cargo", "command not found"],
            "description": "安装 Rust 和 Cargo",
            "commands": [
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                "source $HOME/.cargo/env",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["go", "command not found"],
            "description": "安装 Go 语言",
            "commands": [
                "apt-get update",
                "apt-get install -y golang",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["java", "command not found", "javac"],
            "description": "安装 JDK",
            "commands": [
                "apt-get update",
                "apt-get install -y default-jdk",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["docker", "command not found"],
            "description": "安装 Docker",
            "commands": [
                "apt-get update",
                "apt-get install -y docker.io",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["make", "command not found"],
            "description": "安装构建工具 build-essential",
            "commands": [
                "apt-get update",
                "apt-get install -y build-essential",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["cmake", "command not found"],
            "description": "安装 CMake",
            "commands": [
                "apt-get update",
                "apt-get install -y cmake",
            ],
            "confidence": 0.95,
        },
        {
            "keywords": ["libmysqlclient", "mysql_config"],
            "description": "安装 MySQL 客户端开发库",
            "commands": [
                "apt-get update",
                "apt-get install -y libmysqlclient-dev",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["libpq", "pg_config"],
            "description": "安装 PostgreSQL 客户端开发库",
            "commands": [
                "apt-get update",
                "apt-get install -y libpq-dev",
            ],
            "confidence": 0.9,
        },
    ]

    def __init__(self):
        # 存储归因报告（用于调试和日志）
        self._feedback_history: list[AttributionReport] = []

    def query(self, context: dict[str, Any]) -> list[XPUSuggestion]:
        """基于关键词匹配查询建议"""
        error_log = context.get("error", "") or context.get("error_log", "")
        combined_text = f"{error_log}".lower()

        suggestions = []

        for entry in self.KNOWLEDGE_BASE:
            # 计算匹配分数
            matched_keywords = sum(
                1 for kw in entry["keywords"]
                if kw.lower() in combined_text
            )

            if matched_keywords > 0:
                score = matched_keywords / len(entry["keywords"]) * entry["confidence"]
                if score > 0.3:
                    suggestion = XPUSuggestion(
                        id=f"xpu_{uuid.uuid4().hex[:8]}",
                        description=entry["description"],
                        commands=entry["commands"],
                        confidence=score,
                        source="mock",
                    )
                    suggestions.append((score, suggestion))

        # 按置信度排序，返回 Top-K
        suggestions.sort(key=lambda x: x[0], reverse=True)
        result = [s[1] for s in suggestions[:3]]

        if result:
            logger.info(f"XPU 查询返回 {len(result)} 条建议")
            for s in result:
                logger.info(f"  - {s}")
        else:
            logger.debug(f"XPU 未找到匹配建议")

        return result

    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告（记录到日志）"""
        self._feedback_history.append(report)

        # 记录详细的归因报告到日志
        logger.info("=" * 60)
        logger.info("XPU 归因报告 (Attribution Report)")
        logger.info("=" * 60)
        logger.info(f"  suggestion_id: {report.suggestion_id}")
        logger.info(f"  timestamp: {report.timestamp}")
        logger.info(f"  repo_context: {report.repo_context}")
        logger.info(f"  outcome: {report.outcome}")
        logger.info(f"  score: {report.score}")
        logger.info(f"  error_before: {report.error_before[:200] if report.error_before else 'N/A'}...")
        logger.info(f"  error_after: {report.error_after[:200] if report.error_after else 'N/A'}...")
        logger.info(f"  执行日志:")
        for i, log in enumerate(report.logs):
            logger.info(f"    [{i+1}] {log.command} -> exit_code={log.exit_code}")
        logger.info("=" * 60)


class HTTPXPUClient(XPUClientBase):
    """HTTP XPU 客户端，连接真实 XPU 服务"""

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def query(self, context: dict[str, Any]) -> list[XPUSuggestion]:
        """调用远程 XPU 服务"""
        try:
            response = self._client.post(
                f"{self._base_url}/api/query",
                json=context,
            )
            response.raise_for_status()
            data = response.json()

            suggestions = []
            for item in data.get("suggestions", []):
                suggestions.append(XPUSuggestion(
                    id=item["id"],
                    description=item["description"],
                    commands=item.get("commands", []),
                    confidence=item.get("confidence", 0.5),
                    source="http",
                ))

            logger.info(f"XPU HTTP 查询返回 {len(suggestions)} 条建议")
            return suggestions

        except httpx.HTTPError as e:
            logger.warning(f"XPU 服务调用失败: {e}")
            return []

    def submit_feedback(self, report: AttributionReport) -> None:
        """提交归因报告到远程服务"""
        try:
            response = self._client.post(
                f"{self._base_url}/api/feedback",
                json=report.to_dict(),
            )
            response.raise_for_status()
            logger.info(f"归因报告已提交: {report.suggestion_id}")

            # 同时记录到本地日志
            logger.info("=" * 60)
            logger.info("XPU 归因报告 (Attribution Report)")
            logger.info(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            logger.info("=" * 60)

        except httpx.HTTPError as e:
            logger.warning(f"提交归因报告失败: {e}")

    def close(self) -> None:
        """关闭 HTTP 客户端"""
        self._client.close()


class VectorXPUClient(XPUClientBase):
    """基于 PostgreSQL 向量数据库的 XPU 客户端（复用 xpu_standalone）"""

    def __init__(self, dns: str):
        from .xpu.xpu_vector_store import XpuVectorStore, text_to_embedding, build_xpu_text
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands
        self._store = XpuVectorStore(connection_string=dns)
        self._text_to_embedding = text_to_embedding
        self._build_xpu_text = build_xpu_text
        self._render_atom_to_commands = render_atom_to_commands
        self._id_to_raw: dict[str, dict] = {}  # suggestion_id → raw search result
        logger.info(f"VectorXPUClient 初始化完成（连接: {dns.split('@')[-1] if '@' in dns else '...'}）")

    def query(self, context: dict[str, Any]) -> list[XPUSuggestion]:
        """向量相似度检索 XPU 经验"""
        # 补充导入，解决作用域问题
        from .xpu.xpu_adapter import XpuAtom

        error_text = context.get("error", "") or context.get("error_log", "")
        if not error_text:
            return []

        try:
            embedding = self._text_to_embedding(error_text)
            results = self._store.search(embedding, k=3, min_similarity=0.3)
        except Exception as e:
            logger.warning(f"VectorXPUClient 查询失败: {e}")
            return []

        suggestions = []
        result_ids = []

        for res in results:
            xpu_id = res["id"]
            advice_nl = res.get("advice_nl") or []
            atoms = res.get("atoms") or []
            similarity = float(res.get("similarity", 0.5))

            # atoms → commands：通过 render_atom_to_commands 转换为 bash 命令
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(self._render_atom_to_commands(atom))

            suggestion = XPUSuggestion(
                id=xpu_id,
                description="\n".join(advice_nl),
                commands=commands,
                confidence=similarity,
                source="vector_db",
            )
            suggestions.append(suggestion)
            result_ids.append(xpu_id)
            self._id_to_raw[xpu_id] = res

        if result_ids:
            try:
                self._store.increment_telemetry(result_ids, "hits")
            except Exception as e:
                logger.warning(f"遥测 hits 写入失败: {e}")

        logger.info(f"VectorXPU 查询返回 {len(suggestions)} 条建议")
        for s in suggestions:
            logger.info(f"  - [{s.confidence:.3f}] {s.id}: {s.description[:60]}...")

        return suggestions

    def submit_feedback(self, report: AttributionReport) -> None:
        """将归因结果写入遥测"""
        try:
            if report.score > 0:
                self._store.increment_telemetry([report.suggestion_id], "successes")
            elif report.score < 0:
                self._store.increment_telemetry([report.suggestion_id], "failures")
        except Exception as e:
            logger.warning(f"遥测 feedback 写入失败: {e}")

        logger.info("=" * 60)
        logger.info("XPU 归因报告 (VectorXPUClient)")
        logger.info(f"  suggestion_id: {report.suggestion_id}")
        logger.info(f"  outcome: {report.outcome}  score: {report.score}")
        logger.info(f"  error_before: {(report.error_before or '')[:200]}...")
        logger.info(f"  error_after:  {(report.error_after or '')[:200]}...")
        logger.info("=" * 60)

    def close(self) -> None:
        self._store.close()


def create_xpu_client() -> XPUClientBase:
    """创建 XPU 客户端实例"""
    config = get_config().xpu

    if config.disabled:
        logger.info("XPU 已禁用，使用 NoopXPU 客户端")
        return NoopXPUClient()
    if config.vector_enabled and config.db_dns:
        logger.info("使用 VectorXPU 客户端（PostgreSQL 向量数据库）")
        return VectorXPUClient(config.db_dns)
    elif config.enabled:
        logger.info(f"使用 HTTP XPU 客户端: {config.base_url}")
        return HTTPXPUClient(config.base_url)
    else:
        logger.info("使用 Mock XPU 客户端")
        return MockXPUClient()


class NoopXPUClient(XPUClientBase):
    """完全禁用的 XPU 客户端：不提供建议，不回传反馈。"""

    def query(self, context: dict[str, Any]) -> list[XPUSuggestion]:
        return []

    def submit_feedback(self, report: AttributionReport) -> None:
        return None
