"""
将 xpu_v1.jsonl 批量导入 XPU 向量数据库。
用法: .venv/bin/python scripts/import_xpu_jsonl.py [jsonl_path]
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(str(Path(__file__).parent.parent))

from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_adapter import XpuEntry, XpuAtom


def import_jsonl(jsonl_path: str) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"文件不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    total = len([l for l in lines if l.strip()])
    print(f"共 {total} 条经验，开始导入...")

    store = XpuVectorStore()
    ok, skip, fail = 0, 0, 0

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
            store.upsert_entry(entry, embedding)
            print(f"[{i}/{total}] ✓ {entry.id}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{total}] ✗ {entry.id}: {e}")
            fail += 1

    store.close()
    print(f"\n导入完成：成功 {ok}，失败 {fail}，跳过 {skip}")


if __name__ == "__main__":
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "xpu_v1.jsonl"
    import_jsonl(jsonl_path)
