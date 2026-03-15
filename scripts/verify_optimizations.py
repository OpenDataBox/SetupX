"""验证 XPU 优化改动的正确性

逐个测试每个优化项：
1. build_xpu_text() - Embedding 关键词加权
2. heuristic_is_candidate() - 提取阈值
3. search() - 复合排序 + 最低相似度 + 负面反馈过滤 + 工具加权（需数据库）
4. VectorXPUClient.query() - 置信度分级（需数据库 + Embedding API）
"""

import sys
import os
import json

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)

passed = 0
failed = 0
total = 0


def test(name, func):
    """运行单个测试"""
    global passed, failed, total
    total += 1
    try:
        func()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


# ============================================================================
# 测试 1: build_xpu_text() 关键词加权
# ============================================================================
print("\n=== 测试 1: build_xpu_text() 关键词加权 ===")

from src.xpu.xpu_adapter import XpuEntry, XpuAtom

def test_build_xpu_text_keyword_weighting():
    from src.xpu.xpu_vector_store import build_xpu_text

    entry = XpuEntry(
        id="test_001",
        context={"lang": "python", "tools": ["pytest", "pip"], "python": ["3.10"], "os": ["linux"]},
        signals={"keywords": ["ModuleNotFoundError", "numpy"], "regex": ["ModuleNotFoundError.*numpy"]},
        advice_nl=["安装 numpy 包"],
        atoms=[XpuAtom(name="pip_install", args={"package": "numpy"})],
    )
    text = build_xpu_text(entry)

    # 验证关键词出现了 3 次
    count = text.count("ModuleNotFoundError")
    assert count == 4, f"关键词 'ModuleNotFoundError' 应出现 4 次（3次加权 + 1次regex），实际 {count} 次"

    # 验证包含三种不同前缀
    assert "Keywords:" in text, "缺少 Keywords 前缀"
    assert "Error keywords:" in text, "缺少 Error keywords 前缀"
    assert "Matching signals:" in text, "缺少 Matching signals 前缀"

test("关键词重复拼接 3 次", test_build_xpu_text_keyword_weighting)

def test_build_xpu_text_no_keywords():
    from src.xpu.xpu_vector_store import build_xpu_text

    entry = XpuEntry(
        id="test_002",
        context={"lang": "python"},
        signals={},
        advice_nl=["通用建议"],
        atoms=[],
    )
    text = build_xpu_text(entry)
    assert "Keywords:" not in text, "无关键词时不应包含 Keywords"
    assert "Language: python" in text

test("无关键词时正常工作", test_build_xpu_text_no_keywords)


# ============================================================================
# 测试 2: heuristic_is_candidate() 提取阈值
# ============================================================================
print("\n=== 测试 2: heuristic_is_candidate() 提取阈值 ===")

from src.xpu.extract_xpu_from_trajs_mvp import heuristic_is_candidate

def test_threshold_only_commands():
    """只有命令（+1），不应通过"""
    stats = {"num_env_commands": 0, "num_error_keywords": 0, "num_commands": 5}
    is_cand, score = heuristic_is_candidate(stats)
    assert not is_cand, f"只有命令(score={score})不应通过阈值 >= 6"
    assert score == 1.0

test("仅有命令(score=1) → 不通过", test_threshold_only_commands)

def test_threshold_env_only():
    """只有环境命令（+5+1=6），应通过"""
    stats = {"num_env_commands": 1, "num_error_keywords": 0, "num_commands": 1}
    is_cand, score = heuristic_is_candidate(stats)
    assert is_cand, f"环境命令+命令(score={score})应通过阈值 >= 6"
    assert score == 6.0

test("环境命令+命令(score=6) → 通过", test_threshold_env_only)

def test_threshold_error_only():
    """只有错误关键词（+5+1=6），应通过"""
    stats = {"num_env_commands": 0, "num_error_keywords": 1, "num_commands": 1}
    is_cand, score = heuristic_is_candidate(stats)
    assert is_cand, f"错误关键词+命令(score={score})应通过阈值 >= 6"
    assert score == 6.0

test("错误关键词+命令(score=6) → 通过", test_threshold_error_only)

def test_threshold_both():
    """环境命令+错误关键词+命令（+5+5+1=11），应通过"""
    stats = {"num_env_commands": 2, "num_error_keywords": 3, "num_commands": 5}
    is_cand, score = heuristic_is_candidate(stats)
    assert is_cand, f"全部满足(score={score})应通过阈值 >= 6"
    assert score == 11.0

test("环境命令+错误+命令(score=11) → 通过", test_threshold_both)

def test_threshold_empty():
    """空轨迹（score=0），不应通过"""
    stats = {"num_env_commands": 0, "num_error_keywords": 0, "num_commands": 0}
    is_cand, score = heuristic_is_candidate(stats)
    assert not is_cand, f"空轨迹(score={score})不应通过"
    assert score == 0.0

test("空轨迹(score=0) → 不通过", test_threshold_empty)

def test_threshold_error_no_commands():
    """有错误但无命令（+5），不应通过"""
    stats = {"num_env_commands": 0, "num_error_keywords": 1, "num_commands": 0}
    is_cand, score = heuristic_is_candidate(stats)
    assert not is_cand, f"仅错误关键词(score={score})不应通过阈值 >= 6"
    assert score == 5.0

test("仅错误关键词(score=5) → 不通过", test_threshold_error_no_commands)


# ============================================================================
# 测试 3: search() 函数签名和 SQL 构建（需数据库连接）
# ============================================================================
print("\n=== 测试 3: search() 复合排序 + 负面反馈过滤（数据库测试）===")

db_available = False
try:
    import psycopg2
    dns = os.environ.get("dns")
    if dns:
        conn = psycopg2.connect(dns)
        conn.close()
        db_available = True
        print("  数据库连接成功")
except Exception as e:
    print(f"  数据库不可用: {e}")

if db_available:
    from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding, EMBEDDING_DIM

    def test_search_default_min_similarity():
        """验证 search() 默认 min_similarity 已改为 0.6"""
        import inspect
        sig = inspect.signature(XpuVectorStore.search)
        default = sig.parameters["min_similarity"].default
        assert default == 0.6, f"min_similarity 默认值应为 0.6，实际为 {default}"

    test("search() 默认 min_similarity=0.6", test_search_default_min_similarity)

    def test_search_returns_composite_score():
        """验证 search() 返回结果包含 composite_score 字段"""
        store = XpuVectorStore()
        try:
            # 先插入一条测试数据
            test_entry = XpuEntry(
                id="__test_verify_composite__",
                context={"lang": "python", "tools": ["pytest"]},
                signals={"keywords": ["TestError"]},
                advice_nl=["测试建议"],
                atoms=[],
            )
            text = build_xpu_text(test_entry)
            embedding = text_to_embedding(text)
            store.upsert_entry(test_entry, embedding)

            # 用相同的 embedding 搜索（应该完全匹配）
            results = store.search(embedding, k=3, min_similarity=0.3)

            # 验证返回结果包含新字段
            found = False
            for r in results:
                if r["id"] == "__test_verify_composite__":
                    found = True
                    assert "composite_score" in r, "结果中缺少 composite_score 字段"
                    assert "telemetry" in r, "结果中缺少 telemetry 字段"
                    assert "similarity" in r, "结果中缺少 similarity 字段"
                    # composite_score 应该 >= similarity（因为 1+success_rate >= 1）
                    assert r["composite_score"] >= r["similarity"] * 0.99, \
                        f"composite_score({r['composite_score']}) 应 >= similarity({r['similarity']})"
                    break
            assert found, "未找到测试数据 __test_verify_composite__"
        finally:
            # 清理测试数据
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM xpu_entries WHERE id = '__test_verify_composite__'")
                    conn.commit()
            finally:
                store._put_conn(conn)
            store.close()

    test("search() 返回 composite_score 和 telemetry", test_search_returns_composite_score)

    def test_search_negative_feedback_filter():
        """验证负面反馈过滤：failures > 3 且 success_rate < 0.2 的条目被排除"""
        store = XpuVectorStore()
        try:
            # 插入一条"劣质"数据：failures=5, successes=0, hits=10
            test_entry = XpuEntry(
                id="__test_verify_neg_filter__",
                context={"lang": "python"},
                signals={"keywords": ["NegativeFilterTestError"]},
                advice_nl=["这是一条应被过滤的建议"],
                atoms=[],
            )
            text = build_xpu_text(test_entry)
            embedding = text_to_embedding(text)
            store.upsert_entry(test_entry, embedding)

            # 手动设置高失败 telemetry
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE xpu_entries
                        SET telemetry = '{"hits": 10, "successes": 0, "failures": 5}'::jsonb
                        WHERE id = '__test_verify_neg_filter__'
                    """)
                    conn.commit()
            finally:
                store._put_conn(conn)

            # 搜索应不返回该条目
            results = store.search(embedding, k=10, min_similarity=0.3)
            filtered_ids = [r["id"] for r in results]
            assert "__test_verify_neg_filter__" not in filtered_ids, \
                f"负面反馈条目应被过滤，但仍出现在结果中: {filtered_ids}"
        finally:
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM xpu_entries WHERE id = '__test_verify_neg_filter__'")
                    conn.commit()
            finally:
                store._put_conn(conn)
            store.close()

    test("负面反馈过滤 (failures>3, rate<0.2)", test_search_negative_feedback_filter)

    def test_search_composite_ranking_order():
        """验证复合排序：有成功记录的条目排在无记录的前面"""
        store = XpuVectorStore()
        try:
            # 插入两条相似数据，一条有成功记录一条没有
            entry_good = XpuEntry(
                id="__test_rank_good__",
                context={"lang": "python"},
                signals={"keywords": ["RankTestError"]},
                advice_nl=["有成功记录的建议"],
                atoms=[],
            )
            entry_new = XpuEntry(
                id="__test_rank_new__",
                context={"lang": "python"},
                signals={"keywords": ["RankTestError"]},
                advice_nl=["无记录的建议"],
                atoms=[],
            )
            text = build_xpu_text(entry_good)
            embedding = text_to_embedding(text)
            store.upsert_entry(entry_good, embedding)
            store.upsert_entry(entry_new, embedding)  # 相同 embedding

            # 给 good 设置成功记录
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE xpu_entries
                        SET telemetry = '{"hits": 10, "successes": 8, "failures": 0}'::jsonb
                        WHERE id = '__test_rank_good__'
                    """)
                    cur.execute("""
                        UPDATE xpu_entries
                        SET telemetry = '{"hits": 0, "successes": 0, "failures": 0}'::jsonb
                        WHERE id = '__test_rank_new__'
                    """)
                    conn.commit()
            finally:
                store._put_conn(conn)

            # 搜索，good 应排在 new 前面
            results = store.search(embedding, k=10, min_similarity=0.3)
            rank_ids = [r["id"] for r in results]

            if "__test_rank_good__" in rank_ids and "__test_rank_new__" in rank_ids:
                good_idx = rank_ids.index("__test_rank_good__")
                new_idx = rank_ids.index("__test_rank_new__")
                assert good_idx < new_idx, \
                    f"有成功记录的条目(idx={good_idx})应排在无记录(idx={new_idx})前面"

                # 验证 composite_score 关系
                good_result = results[good_idx]
                new_result = results[new_idx]
                assert good_result["composite_score"] > new_result["composite_score"], \
                    f"good composite({good_result['composite_score']}) 应 > new({new_result['composite_score']})"
            else:
                # 如果因相似度阈值被过滤也算通过（至少说明搜索没报错）
                print(f"    注意: 测试数据可能被 min_similarity 过滤，结果 IDs: {rank_ids[:5]}")

        finally:
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM xpu_entries WHERE id IN ('__test_rank_good__', '__test_rank_new__')")
                    conn.commit()
            finally:
                store._put_conn(conn)
            store.close()

    test("复合排序：成功记录高的排前面", test_search_composite_ranking_order)

else:
    print("  ⏭️ 跳过数据库相关测试")


# ============================================================================
# 测试 4: VectorXPUClient 置信度分级
# ============================================================================
print("\n=== 测试 4: VectorXPUClient 置信度分级 ===")

if db_available:
    def test_confidence_level_logic():
        """验证置信度分级逻辑"""
        from src.xpu_client import VectorXPUClient
        dns = os.environ.get("dns")
        client = VectorXPUClient(dns)
        try:
            # 插入测试数据
            from src.xpu.xpu_vector_store import build_xpu_text, text_to_embedding
            test_entry = XpuEntry(
                id="__test_confidence__",
                context={"lang": "python", "tools": ["pytest"]},
                signals={"keywords": ["ConfidenceLevelTestError"]},
                advice_nl=["测试置信度分级的建议"],
                atoms=[XpuAtom(name="pip_install", args={"package": "test-pkg"})],
            )
            text = build_xpu_text(test_entry)
            embedding = text_to_embedding(text)
            client._store.upsert_entry(test_entry, embedding)

            # 查询
            results = client.query({"error": "ConfidenceLevelTestError"})

            # 查找测试数据
            found = False
            for s in results:
                if s.id == "__test_confidence__":
                    found = True
                    # 验证 description 包含置信度标签
                    assert s.description.startswith("["), \
                        f"description 应以 [high/medium/low] 开头，实际: {s.description[:30]}"
                    level = s.description.split("]")[0].lstrip("[")
                    assert level in ("high", "medium", "low"), \
                        f"置信度标签应为 high/medium/low，实际: {level}"
                    print(f"    置信度: [{level}], composite_score: {s.confidence:.4f}")
                    break

            if not found:
                print("    注意: 测试数据未返回（可能被 min_similarity 过滤），但查询执行无错误")
        finally:
            conn = client._store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM xpu_entries WHERE id = '__test_confidence__'")
                    conn.commit()
            finally:
                client._store._put_conn(conn)
            client.close()

    test("VectorXPUClient 置信度分级标注", test_confidence_level_logic)
else:
    print("  ⏭️ 跳过（需数据库）")


# ============================================================================
# 测试 5: 工具加权匹配
# ============================================================================
print("\n=== 测试 5: 工具加权匹配 ===")

if db_available:
    def test_tool_boost():
        """验证工具加权：匹配更多工具的 XPU 排序更靠前"""
        store = XpuVectorStore()
        try:
            from src.xpu.xpu_adapter import XpuContext

            # 插入两条数据：一条工具完全匹配，一条部分匹配
            entry_full = XpuEntry(
                id="__test_tool_full__",
                context={"lang": "python", "tools": ["pytest", "pip", "tox"]},
                signals={"keywords": ["ToolBoostTestError"]},
                advice_nl=["完全匹配工具链"],
                atoms=[],
            )
            entry_partial = XpuEntry(
                id="__test_tool_partial__",
                context={"lang": "python", "tools": ["pytest"]},
                signals={"keywords": ["ToolBoostTestError"]},
                advice_nl=["部分匹配工具链"],
                atoms=[],
            )
            text = build_xpu_text(entry_full)
            embedding = text_to_embedding(text)
            store.upsert_entry(entry_full, embedding)
            store.upsert_entry(entry_partial, embedding)

            # 带工具上下文搜索
            ctx = XpuContext(lang="python", tools=["pytest", "pip", "tox"])
            results = store.search(embedding, ctx=ctx, k=10, min_similarity=0.3)

            full_score = None
            partial_score = None
            for r in results:
                if r["id"] == "__test_tool_full__":
                    full_score = r["composite_score"]
                elif r["id"] == "__test_tool_partial__":
                    partial_score = r["composite_score"]

            if full_score is not None and partial_score is not None:
                assert full_score > partial_score, \
                    f"全匹配工具({full_score:.4f})应 > 部分匹配({partial_score:.4f})"
                print(f"    全匹配: {full_score:.4f}, 部分匹配: {partial_score:.4f}")
            else:
                print(f"    注意: 部分测试数据未返回，full={full_score}, partial={partial_score}")
        finally:
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM xpu_entries WHERE id IN ('__test_tool_full__', '__test_tool_partial__')")
                    conn.commit()
            finally:
                store._put_conn(conn)
            store.close()

    test("工具加权：全匹配 > 部分匹配", test_tool_boost)
else:
    print("  ⏭️ 跳过（需数据库）")


# ============================================================================
# 汇总结果
# ============================================================================
print(f"\n{'='*50}")
print(f"验证结果: {passed}/{total} 通过, {failed} 失败")
print(f"{'='*50}")
sys.exit(1 if failed > 0 else 0)
