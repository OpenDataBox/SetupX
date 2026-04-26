"""
将 XPU JSONL 批量导入 pgvector 数据库。
用法: .venv/bin/python scripts/import_xpu_jsonl.py [jsonl_path] [--clear]

--clear: 导入前清空 xpu_entries 表
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(str(Path(__file__).parent.parent))

from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_adapter import XpuEntry, XpuAtom


def import_jsonl(jsonl_path: str, clear: bool = False) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"文件不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    total = len([l for l in lines if l.strip()])
    print(f"共 {total} 条经验，开始导入...")

    store = XpuVectorStore()

    if clear:
        conn = store._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM xpu_entries;")
                conn.commit()
            print("已清空 xpu_entries 表")
        finally:
            store._put_conn(conn)

    ok, fail = 0, 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i}] JSON 解析失败，跳过: {e}")
            fail += 1
            continue

        atoms = [
            XpuAtom(name=a["name"], args=a.get("args", {}))
            for a in raw.get("atoms", [])
        ]

        entry = XpuEntry(
            id=raw["id"],
            context=raw.get("context", {}),
            signals=raw.get("signals", {}),
            advice_nl=raw.get("advice_nl", []),
            atoms=atoms,
            telemetry=raw.get("telemetry", {}),
        )

        try:
            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)

            # 带 telemetry 的 upsert
            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                    cur.execute("""
                        INSERT INTO xpu_entries (id, context, signals, advice_nl, atoms, embedding, telemetry)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            context = EXCLUDED.context,
                            signals = EXCLUDED.signals,
                            advice_nl = EXCLUDED.advice_nl,
                            atoms = EXCLUDED.atoms,
                            embedding = EXCLUDED.embedding,
                            telemetry = EXCLUDED.telemetry;
                    """, (
                        entry.id,
                        json.dumps(entry.context),
                        json.dumps(entry.signals),
                        json.dumps(entry.advice_nl),
                        json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),
                        embedding_str,
                        json.dumps(entry.telemetry),
                    ))
                    conn.commit()
            finally:
                store._put_conn(conn)

            print(f"[{i}/{total}] ✓ {entry.id}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{total}] ✗ {entry.id}: {e}")
            fail += 1

    store.close()
    print(f"\n导入完成：成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--clear"]
    jsonl_path = args[0] if args else "xpu_v1.jsonl"
    import_jsonl(jsonl_path, clear=clear)
