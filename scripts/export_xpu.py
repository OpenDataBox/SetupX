#!/usr/bin/env python3
"""
从 PostgreSQL 导出 XPU 经验到 JSONL 文件。

默认增量追加：只导出文件中尚未包含的条目。
加 --full 则全量覆盖。

用法:
  python scripts/export_xpu.py                        # 增量追加到 xpu_v2.jsonl
  python scripts/export_xpu.py -o backup.jsonl --full # 全量导出到指定文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.xpu.xpu_vector_store import XpuVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 PostgreSQL 导出 XPU 经验到 JSONL")
    parser.add_argument("-o", "--output", default="xpu_v2.jsonl", help="输出文件路径（默认 xpu_v2.jsonl）")
    parser.add_argument("--full", action="store_true", help="全量导出（覆盖已有文件），默认增量追加")
    return parser.parse_args()


def load_existing_ids(path: Path) -> set[str]:
    """读取已有 JSONL 文件中的 ID 集合"""
    ids = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def export_from_db(store: XpuVectorStore, exclude_ids: set[str]) -> list[dict]:
    """从数据库查出需要导出的条目"""
    conn = store._get_conn()
    try:
        with conn.cursor() as cur:
            if exclude_ids:
                cur.execute("""
                    SELECT id, context, signals, advice_nl, atoms, telemetry
                    FROM xpu_entries
                    WHERE id != ALL(%s)
                    ORDER BY created_at;
                """, (list(exclude_ids),))
            else:
                cur.execute("""
                    SELECT id, context, signals, advice_nl, atoms, telemetry
                    FROM xpu_entries
                    ORDER BY created_at;
                """)
            rows = cur.fetchall()
    finally:
        store._put_conn(conn)

    entries = []
    for row in rows:
        entries.append({
            "id": row[0],
            "context": row[1] or {},
            "signals": row[2] or {},
            "advice_nl": row[3] or [],
            "atoms": row[4] or [],
            "telemetry": row[5] or {},
        })
    return entries


def main() -> int:
    args = parse_args()
    output = Path(args.output)

    store = XpuVectorStore()

    if args.full:
        exclude_ids = set()
        mode = "w"
    else:
        exclude_ids = load_existing_ids(output)
        mode = "a"
        print(f"已有 {len(exclude_ids)} 条，增量导出中...")

    entries = export_from_db(store, exclude_ids)
    store.close()

    if not entries:
        print("无新增条目，文件未变更")
        return 0

    with open(output, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(exclude_ids) + len(entries) if not args.full else len(entries)
    print(f"导出 {len(entries)} 条到 {output}（文件共 {total} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
