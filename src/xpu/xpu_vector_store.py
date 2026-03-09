"""XPU 向量数据库存储层，基于 PostgreSQL + pgvector 扩展。"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

from .xpu_adapter import XpuEntry, XpuContext
from ..logger import get_logger

logger = get_logger("xpu.vector_store")

# Embedding 维度（可通过环境变量 EMBEDDING_DIM 覆盖）
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))


def get_db_connection_string() -> str:
    """从环境变量获取数据库连接串。"""
    dns = os.environ.get("dns")
    if not dns:
        raise RuntimeError("缺少必需的环境变量: dns（PostgreSQL 连接串）")
    return dns


def create_xpu_table(conn) -> None:
    """创建 XPU 表和向量索引（如不存在）。"""
    with conn.cursor() as cur:
        # 启用 pgvector 扩展
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 创建表
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

        # 创建向量相似度搜索索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS xpu_entries_embedding_idx
            ON xpu_entries
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        conn.commit()
        logger.info("XPU 表和索引已创建/验证")


def text_to_embedding(text: str, model: str = None) -> List[float]:
    """调用 OpenAI 兼容 API 生成文本 embedding。

    配置优先级：
    1. EMBEDDING_API_KEY + EMBEDDING_BASE_URL — 独立 embedding 服务
    2. OPENAI_API_KEY + OPENAI_BASE_URL — 回退到 OpenAI 配置
    3. OPENAI_API_KEY — 使用 OpenAI 官方 API
    """
    import openai

    # 优先检查独立 embedding 配置
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    if embedding_api_key:
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

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def build_xpu_text(entry: XpuEntry) -> str:
    """构建 XPU 条目的可搜索文本表示。"""
    parts = []

    ctx = entry.context
    if ctx.get("lang"):
        parts.append(f"Language: {ctx['lang']}")
    if ctx.get("tools"):
        parts.append(f"Tools: {', '.join(ctx['tools'])}")
    if ctx.get("python"):
        parts.append(f"Python versions: {', '.join(map(str, ctx['python']))}")
    if ctx.get("os"):
        parts.append(f"OS: {', '.join(ctx['os'])}")

    signals = entry.signals
    if signals.get("keywords"):
        parts.append(f"Keywords: {', '.join(signals['keywords'])}")
    if signals.get("regex"):
        parts.append(f"Error patterns: {', '.join(signals['regex'])}")

    if entry.advice_nl:
        parts.append("Advice: " + " ".join(entry.advice_nl))

    return "\n".join(parts)


class XpuVectorStore:
    """XPU 向量数据库存储。"""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or get_db_connection_string()
        self.pool = ThreadedConnectionPool(1, 5, self.connection_string)
        self._ensure_table()

    def _get_conn(self):
        """从连接池获取连接。"""
        return self.pool.getconn()

    def _put_conn(self, conn):
        """归还连接到连接池。"""
        self.pool.putconn(conn)

    def _ensure_table(self) -> None:
        """确保表存在。"""
        conn = self._get_conn()
        try:
            create_xpu_table(conn)
        finally:
            self._put_conn(conn)

    def upsert_entry(self, entry: XpuEntry, embedding: List[float]) -> None:
        """插入或更新 XPU 条目（含 embedding 向量）。"""
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding 维度不匹配: 期望 {EMBEDDING_DIM}，实际 {len(embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 将 embedding 列表转为 pgvector 字符串格式: '[0.1,0.2,...]'
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

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
                    json.dumps(entry.context),
                    json.dumps(entry.signals),
                    json.dumps(entry.advice_nl),
                    json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),
                    embedding_str,
                ))
                conn.commit()
        finally:
            self._put_conn(conn)

    def search(
        self,
        query_embedding: List[float],
        ctx: Optional[XpuContext] = None,
        k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """基于向量相似度搜索 XPU 条目。"""
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(f"查询 embedding 维度不匹配: 期望 {EMBEDDING_DIM}，实际 {len(query_embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 设置 IVFFlat 探测数，避免数据量少时索引漏检
                cur.execute("SET ivfflat.probes = 10")

                # 构建 WHERE 子句（上下文过滤）
                where_clauses = []
                where_params = []

                if ctx:
                    if ctx.lang:
                        if isinstance(ctx.lang, (list, tuple, set)):
                            where_clauses.append("context->>'lang' = ANY(%s)")
                            where_params.append(list(ctx.lang))
                        else:
                            where_clauses.append("context->>'lang' = %s")
                            where_params.append(ctx.lang)
                    if ctx.python:
                        py_list = ctx.python if isinstance(ctx.python, (list, tuple, set)) else [ctx.python]
                        # 匹配 Python 版本列表中的任意一个
                        py_conditions = []
                        for py_ver in py_list:
                            py_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'python') AS v WHERE v LIKE %s)")
                            where_params.append(f"{py_ver}%")
                        if py_conditions:
                            where_clauses.append(f"({' OR '.join(py_conditions)})")
                    if ctx.tools:
                        # 匹配工具列表中的任意一个
                        tool_conditions = []
                        for tool in ctx.tools:
                            tool_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'tools') AS t WHERE t = %s)")
                            where_params.append(tool)
                        if tool_conditions:
                            where_clauses.append(f"({' OR '.join(tool_conditions)})")

                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

                # 将 embedding 列表转为 pgvector 字符串格式
                embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

                query = f"""
                    SELECT
                        id,
                        context,
                        signals,
                        advice_nl,
                        atoms,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM xpu_entries
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                params = [embedding_str, embedding_str, min_similarity] + where_params + [embedding_str, k]

                cur.execute(query, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "context": row[1],
                        "signals": row[2],
                        "advice_nl": row[3],
                        "atoms": row[4],
                        "similarity": float(row[5]),
                    })

                return results
        finally:
            self._put_conn(conn)

    def get_entry(self, xpu_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取单条 XPU 条目。"""
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
        """关闭连接池。"""
        if self.pool and not self.pool.closed:
            self.pool.closeall()

    _TELEMETRY_FIELDS = {"hits", "successes", "failures"}

    def increment_telemetry(self, xpu_ids: List[str], field: str):
        """原子操作：给指定 ID 列表的 telemetry 某个字段 +1。
        field 只能是 'hits', 'successes', 'failures'
        """
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"非法的 telemetry 字段: {field}，仅允许 {self._TELEMETRY_FIELDS}")
        if not xpu_ids: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 使用 Postgres 的 jsonb_set 和 COALESCE 实现原子 +1
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
        """更新某条经验的 advice_nl 字段（用于合并后生成的新建议）。"""
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
        """批量更新遥测分数。
        updates: { "xpu_id_1": 0.5, "xpu_id_2": 0.25 }
        """
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"非法的 telemetry 字段: {field}，仅允许 {self._TELEMETRY_FIELDS}")
        if not updates: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for xpu_id, score in updates.items():
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
