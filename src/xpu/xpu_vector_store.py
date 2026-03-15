"""XPU 向量数据库存储层，基于 PostgreSQL + pgvector 扩展。

本模块负责 XPU 经验条目的持久化存储和向量相似度检索：

核心功能：
- 表结构管理：创建 xpu_entries 表和 IVFFlat 向量索引
- Embedding 生成：调用 OpenAI 兼容 API 将文本转为向量
- 文本构建：将 XPU 条目转为可搜索的文本表示
- CRUD 操作：插入/更新/查询/删除 XPU 条目
- 向量检索：基于余弦相似度的 Top-K 检索
- 遥测更新：原子操作更新 hits/successes/failures 计数

数据库表结构（xpu_entries）：
- id: TEXT PRIMARY KEY —— XPU 唯一标识
- context: JSONB —— 上下文信息（语言、工具链等）
- signals: JSONB —— 触发信号（正则、关键词）
- advice_nl: JSONB —— 自然语言建议列表
- atoms: JSONB —— 原子操作列表
- embedding: vector(1536) —— 文本 embedding 向量
- telemetry: JSONB —— 遥测数据（hits, successes, failures）
- created_at: TIMESTAMP —— 创建时间
"""

import json  # JSON 序列化/反序列化
import os  # 环境变量读取
from typing import Any, Dict, List, Optional  # 类型标注

import numpy as np  # 数值计算（虽然当前未直接使用，保留供后续扩展）
import psycopg2  # PostgreSQL 数据库驱动
from psycopg2.extras import execute_values  # 批量插入工具
from psycopg2.pool import ThreadedConnectionPool  # 线程安全的连接池

from .xpu_adapter import XpuEntry, XpuContext  # XPU 数据结构
from ..logger import get_logger  # 统一日志系统

logger = get_logger("xpu.vector_store")  # 创建向量存储模块专用日志

# Embedding 向量维度（默认 1536，对应 OpenAI text-embedding-3-small）
# 可通过环境变量 EMBEDDING_DIM 覆盖
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))


# ============================================================================
# 数据库连接与表结构
# ============================================================================

def get_db_connection_string() -> str:
    """从环境变量 dns 获取 PostgreSQL 连接串

    Returns:
        PostgreSQL 连接字符串

    Raises:
        RuntimeError: 环境变量 dns 未设置
    """
    dns = os.environ.get("dns")
    if not dns:
        raise RuntimeError("缺少必需的环境变量: dns（PostgreSQL 连接串）")
    return dns


def create_xpu_table(conn) -> None:
    """创建 XPU 表和向量索引（如不存在）

    包含三个操作：
    1. 启用 pgvector 扩展
    2. 创建 xpu_entries 表
    3. 创建 IVFFlat 向量索引（使用余弦距离）

    Args:
        conn: psycopg2 数据库连接对象
    """
    with conn.cursor() as cur:
        # 启用 pgvector 扩展（PostgreSQL 向量搜索扩展）
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 创建 XPU 条目表（如不存在）
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS xpu_entries (
                id TEXT PRIMARY KEY,
                context JSONB NOT NULL,
                signals JSONB NOT NULL,
                advice_nl JSONB NOT NULL,
                atoms JSONB NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                telemetry JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # 创建 IVFFlat 向量索引（加速余弦相似度检索）
        # lists=100：将向量空间划分为 100 个聚类，适合中小规模数据集
        # vector_cosine_ops：使用余弦距离作为度量
        cur.execute("""
            CREATE INDEX IF NOT EXISTS xpu_entries_embedding_idx
            ON xpu_entries
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        conn.commit()
        logger.info("XPU 表和索引已创建/验证")


# ============================================================================
# Embedding 生成
# ============================================================================

def text_to_embedding(text: str, model: str = None) -> List[float]:
    """调用 OpenAI 兼容 API 生成文本 embedding 向量

    配置优先级（支持独立 embedding 服务和 OpenAI 服务两种模式）：
    1. EMBEDDING_API_KEY + EMBEDDING_BASE_URL → 独立 embedding 服务
    2. OPENAI_API_K1EY + OPENAI_BASE_URL → 回退到 OpenAI 配置
    3. OPENAI_API_KEY → 使用 OpenAI 官方 API

    Args:
        text: 要生成 embedding 的文本
        model: 指定 embedding 模型（可选，默认 text-embedding-3-small）

    Returns:
        1536 维的 embedding 向量（浮点数列表）

    Raises:
        RuntimeError: 缺少 API Key
    """
    import openai  # 延迟导入，避免在不使用 embedding 时也加载

    # 优先检查独立 embedding 配置
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    if embedding_api_key:
        # 使用独立 embedding 服务
        api_key = embedding_api_key
        base_url = embedding_base_url
        model = model or embedding_model or "text-embedding-3-small"
        logger.info(f"使用 embedding API: {base_url or 'default'}, 模型: {model}")
    else:
        # 回退到 OpenAI 配置
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = model or "text-embedding-3-small"

        if not api_key:
            raise RuntimeError(
                "缺少 embedding 生成所需的 API Key，"
                "请设置 EMBEDDING_API_KEY 或 OPENAI_API_KEY"
            )

    # 构造 OpenAI 客户端参数
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    # 调用 embedding API
    client = openai.OpenAI(**client_kwargs)
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding  # 返回第一个（也是唯一的）embedding 向量


# ============================================================================
# 文本构建：将 XPU 条目转为可搜索的文本
# ============================================================================

def build_xpu_text(entry: XpuEntry) -> str:
    """构建 XPU 条目的可搜索文本表示

    将 XPU 的各字段拼接为一段文本，用于生成 embedding。
    拼接顺序：上下文（语言/工具/版本/OS）→ 信号（关键词/正则）→ 建议

    [P2 优化] signals.keywords 重复拼接 3 次以提升其在 embedding 中的权重，
    因为关键词是检索匹配最关键的信号。

    Args:
        entry: XPU 经验条目

    Returns:
        拼接后的文本，各部分用换行分隔
    """
    parts = []

    # 拼接上下文信息
    ctx = entry.context
    if ctx.get("lang"):
        parts.append(f"Language: {ctx['lang']}")
    if ctx.get("tools"):
        parts.append(f"Tools: {', '.join(ctx['tools'])}")
    if ctx.get("python"):
        parts.append(f"Python versions: {', '.join(map(str, ctx['python']))}")
    if ctx.get("os"):
        parts.append(f"OS: {', '.join(ctx['os'])}")

    # 拼接信号信息（关键词重复 3 次以加权）
    signals = entry.signals
    if signals.get("keywords"):
        keywords_str = ', '.join(signals['keywords'])
        # 重复 3 次使 keywords 在 embedding 空间中占更大权重
        parts.append(f"Keywords: {keywords_str}")
        parts.append(f"Error keywords: {keywords_str}")
        parts.append(f"Matching signals: {keywords_str}")
    if signals.get("regex"):
        parts.append(f"Error patterns: {', '.join(signals['regex'])}")

    # 拼接自然语言建议
    if entry.advice_nl:
        parts.append("Advice: " + " ".join(entry.advice_nl))

    return "\n".join(parts)


# ============================================================================
# XpuVectorStore：向量数据库存储核心类
# ============================================================================

class XpuVectorStore:
    """XPU 向量数据库存储

    封装了对 PostgreSQL + pgvector 数据库的所有操作，
    使用线程安全的连接池管理数据库连接。

    主要方法：
    - upsert_entry()：插入或更新 XPU 条目
    - search()：向量相似度检索
    - get_entry()：按 ID 获取单条记录
    - increment_telemetry()：原子递增遥测计数
    - update_advice()：更新建议内容
    """

    def __init__(self, connection_string: Optional[str] = None):
        """初始化向量存储

        Args:
            connection_string: PostgreSQL 连接串（不提供则从环境变量获取）
        """
        self.connection_string = connection_string or get_db_connection_string()
        # 创建线程安全的连接池（最小 1 个连接，最大 5 个连接）
        self.pool = ThreadedConnectionPool(1, 5, self.connection_string)
        self._ensure_table()  # 确保表和索引存在

    def _get_conn(self):
        """从连接池获取一个数据库连接"""
        return self.pool.getconn()

    def _put_conn(self, conn):
        """将数据库连接归还到连接池"""
        self.pool.putconn(conn)

    def _ensure_table(self) -> None:
        """确保 xpu_entries 表和索引存在（初始化时调用）"""
        conn = self._get_conn()
        try:
            create_xpu_table(conn)
        finally:
            self._put_conn(conn)

    # ========================================================================
    # CRUD 操作
    # ========================================================================

    def upsert_entry(self, entry: XpuEntry, embedding: List[float]) -> None:
        """插入或更新 XPU 条目（含 embedding 向量）

        使用 PostgreSQL 的 INSERT ... ON CONFLICT DO UPDATE 实现 upsert。
        如果 id 已存在则更新所有字段。

        Args:
            entry: XPU 经验条目
            embedding: 预计算的 embedding 向量

        Raises:
            ValueError: embedding 维度不匹配
        """
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding 维度不匹配: 期望 {EMBEDDING_DIM}，实际 {len(embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 将 Python 列表转为 pgvector 字符串格式: '[0.1,0.2,...,0.9]'
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

                # INSERT ... ON CONFLICT DO UPDATE：id 冲突时更新所有字段
                cur.execute("""
                    INSERT INTO xpu_entries (id, context, signals, advice_nl, atoms, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        context = EXCLUDED.context,
                        signals = EXCLUDED.signals,
                        advice_nl = EXCLUDED.advice_nl,
                        atoms = EXCLUDED.atoms,
                        embedding = EXCLUDED.embedding;
                """, (
                    entry.id,
                    json.dumps(entry.context),  # 序列化为 JSON 字符串
                    json.dumps(entry.signals),
                    json.dumps(entry.advice_nl),
                    json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),  # atoms 列表序列化
                    embedding_str,
                ))
                conn.commit()
        finally:
            self._put_conn(conn)

    # ========================================================================
    # 向量相似度检索
    # ========================================================================

    def search(
        self,
        query_embedding: List[float],
        ctx: Optional[XpuContext] = None,
        k: int = 3,
        min_similarity: float = 0.6,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """基于向量相似度搜索 XPU 条目，结合 telemetry 复合排序

        使用 pgvector 的余弦距离算子（<=>）计算相似度，
        支持可选的上下文过滤（语言、Python 版本、工具链）。

        复合排序公式：composite_score = similarity × (1 + success_rate)
        其中 success_rate = successes / max(hits, 1)

        过滤规则：
        - 最低相似度阈值：默认 0.6（低于此值的结果不返回）
        - 负面反馈过滤：failures > 3 且 success_rate < 0.2 的条目不返回
        - 排除 ID 列表：已尝试过的 XPU 不返回，保证返回 k 条均为未尝试的

        Args:
            query_embedding: 查询 embedding 向量（1536 维）
            ctx: 可选的上下文过滤条件
            k: 返回的最大结果数（默认 3）
            min_similarity: 最低相似度阈值（默认 0.6）
            exclude_ids: 需要排除的 XPU ID 列表（已尝试过的建议）

        Returns:
            匹配结果列表，每条包含 id、context、signals、advice_nl、atoms、similarity、composite_score

        Raises:
            ValueError: 查询 embedding 维度不匹配
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(f"查询 embedding 维度不匹配: 期望 {EMBEDDING_DIM}，实际 {len(query_embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 设置 IVFFlat 探测数为 10（默认是 1）
                # 数据量少时需要更多探测以避免索引漏检
                cur.execute("SET ivfflat.probes = 10")

                # ---- 构建 WHERE 子句（上下文过滤）----
                where_clauses = []
                where_params = []

                if ctx:
                    # 语言过滤（精确匹配）
                    if ctx.lang:
                        if isinstance(ctx.lang, (list, tuple, set)):
                            where_clauses.append("context->>'lang' = ANY(%s)")
                            where_params.append(list(ctx.lang))
                        else:
                            where_clauses.append("context->>'lang' = %s")
                            where_params.append(ctx.lang)
                    # Python 版本前缀匹配
                    if ctx.python:
                        py_list = ctx.python if isinstance(ctx.python, (list, tuple, set)) else [ctx.python]
                        py_conditions = []
                        for py_ver in py_list:
                            py_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'python') AS v WHERE v LIKE %s)")
                            where_params.append(f"{py_ver}%")
                        if py_conditions:
                            where_clauses.append(f"({' OR '.join(py_conditions)})")
                    # 工具链过滤（加权匹配：全匹配 > 部分匹配）
                    # 至少有一个工具匹配才纳入候选
                    if ctx.tools:
                        tool_conditions = []
                        for tool in ctx.tools:
                            tool_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'tools') AS t WHERE t = %s)")
                            where_params.append(tool)
                        if tool_conditions:
                            where_clauses.append(f"({' OR '.join(tool_conditions)})")

                # [P1-2] 负面反馈过滤：failures > 3 且 success_rate < 0.2 的条目排除
                where_clauses.append("""
                    NOT (
                        COALESCE((telemetry->>'failures')::int, 0) > 3
                        AND COALESCE((telemetry->>'successes')::int, 0)::float
                            / GREATEST(COALESCE((telemetry->>'hits')::int, 0), 1) < 0.2
                    )
                """)

                # 排除已尝试过的 XPU ID，保证返回的 k 条均为未尝试的
                if exclude_ids:
                    where_clauses.append("id != ALL(%s)")
                    where_params.append(list(exclude_ids))

                # 拼接 WHERE 子句
                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

                # 将 embedding 列表转为 pgvector 字符串格式
                embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

                # ---- 构建工具加权排序表达式 ----
                # [P2-2] 工具匹配加权：全匹配的 XPU 排序更靠前
                # tool_boost = 1.0 + 0.05 × 匹配工具数（注意：当前未启用，需传入 ctx.tools 才生效）
                tool_boost_expr = "1.0"
                tool_boost_params = []
                if ctx and ctx.tools:
                    # 统计查询工具与 XPU 条目工具的交集大小
                    tool_count_parts = []
                    for tool in ctx.tools:
                        tool_count_parts.append(
                            "CASE WHEN EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'tools') AS t WHERE t = %s) THEN 1 ELSE 0 END"
                        )
                        tool_boost_params.append(tool)
                    tool_boost_expr = f"(1.0 + 0.05 * ({' + '.join(tool_count_parts)}))"

                # ---- 执行向量检索查询 ----
                # [P0-1] 复合排序：similarity × (1 + success_rate) × tool_boost
                # success_rate = successes / max(hits, 1)
                query = f"""
                    SELECT
                        id,
                        context,
                        signals,
                        advice_nl,
                        atoms,
                        1 - (embedding <=> %s::vector) AS similarity,
                        telemetry,
                        (1 - (embedding <=> %s::vector))
                            * (1.0 + COALESCE((telemetry->>'successes')::float, 0)
                               / GREATEST(COALESCE((telemetry->>'hits')::int, 0), 1))
                            * {tool_boost_expr}
                        AS composite_score
                    FROM xpu_entries
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    {where_sql}
                    ORDER BY composite_score DESC
                    LIMIT %s;
                """
                params = (
                    [embedding_str, embedding_str]
                    + tool_boost_params
                    + [embedding_str, min_similarity]
                    + where_params
                    + [k]
                )

                cur.execute(query, params)
                rows = cur.fetchall()

                # 将查询结果转为字典列表
                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "context": row[1],
                        "signals": row[2],
                        "advice_nl": row[3],
                        "atoms": row[4],
                        "similarity": float(row[5]),
                        "telemetry": row[6] or {},
                        "composite_score": float(row[7]),
                    })

                return results
        finally:
            self._put_conn(conn)

    def get_entry(self, xpu_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取单条 XPU 条目

        Args:
            xpu_id: XPU 唯一标识

        Returns:
            XPU 条目字典，不存在返回 None
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, context, signals, advice_nl, atoms
                    FROM xpu_entries
                    WHERE id = %s;
                """, (xpu_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "context": row[1],
                    "signals": row[2],
                    "advice_nl": row[3],
                    "atoms": row[4],
                }
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """关闭连接池，释放所有数据库连接"""
        if self.pool and not self.pool.closed:
            self.pool.closeall()

    # ========================================================================
    # 遥测数据操作
    # ========================================================================

    # 允许更新的遥测字段白名单
    _TELEMETRY_FIELDS = {"hits", "successes", "failures"}

    def increment_telemetry(self, xpu_ids: List[str], field: str):
        """原子操作：给指定 ID 列表的 telemetry 某个字段 +1

        使用 PostgreSQL 的 jsonb_set + COALESCE 实现原子递增，
        避免并发更新时的竞态条件。

        Args:
            xpu_ids: 要更新的 XPU ID 列表
            field: 遥测字段名，只能是 'hits'、'successes'、'failures'

        Raises:
            ValueError: 字段名不在白名单中
        """
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"非法的 telemetry 字段: {field}，仅允许 {self._TELEMETRY_FIELDS}")
        if not xpu_ids: return  # ID 列表为空则跳过
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 使用 jsonb_set 原子更新 JSONB 中的特定字段
                # COALESCE 确保 telemetry 为 null 时也能正常操作
                sql = f"""
                    UPDATE xpu_entries
                    SET telemetry = jsonb_set(
                        COALESCE(telemetry, '{{}}'::jsonb),
                        '{{{field}}}',
                        (COALESCE(telemetry->>'{field}', '0')::int + 1)::text::jsonb
                    )
                    WHERE id = ANY(%s);
                """
                cur.execute(sql, (xpu_ids,))
                conn.commit()
        except Exception as e:
            logger.error(f"更新 telemetry ({field}) 失败: {e}")
        finally:
            self._put_conn(conn)

    def update_advice(self, xpu_id: str, new_advice: List[str]) -> None:
        """更新某条经验的 advice_nl 字段（用于去重合并后的新建议）

        Args:
            xpu_id: 要更新的 XPU ID
            new_advice: 合并后的新建议列表
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE xpu_entries SET advice_nl = %s WHERE id = %s;",
                    (json.dumps(new_advice), xpu_id),
                )
                conn.commit()
                logger.info(f"已更新经验 '{xpu_id}' 的 advice_nl（{len(new_advice)} 条建议）")
        except Exception as e:
            logger.error(f"更新经验 '{xpu_id}' 的 advice_nl 失败: {e}")
        finally:
            self._put_conn(conn)

    def update_telemetry_scores(self, updates: Dict[str, float], field: str = 'hits'):
        """批量更新遥测分数（支持小数增量，用于加权反馈）

        与 increment_telemetry 不同，此方法支持浮点数增量，
        适用于加权归因场景。

        Args:
            updates: { "xpu_id_1": 0.5, "xpu_id_2": 0.25 }
            field: 遥测字段名
        """
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"非法的 telemetry 字段: {field}，仅允许 {self._TELEMETRY_FIELDS}")
        if not updates: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for xpu_id, score in updates.items():
                    # 使用 to_jsonb 将计算结果转为 JSONB 类型
                    sql = f"""
                        UPDATE xpu_entries
                        SET telemetry = jsonb_set(
                            COALESCE(telemetry, '{{}}'::jsonb),
                            '{{{field}}}',
                            to_jsonb(COALESCE((telemetry->>'{field}')::numeric, 0) + %s)
                        )
                        WHERE id = %s;
                    """
                    cur.execute(sql, (score, xpu_id))
                conn.commit()
        except Exception as e:
            logger.error(f"批量更新 telemetry 分数失败: {e}")
        finally:
            self._put_conn(conn)
