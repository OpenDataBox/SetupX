#!/usr/bin/env python3
"""
从日志目录批量构建轨迹，并提取 XPU 经验后入库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# 必须先加载环境变量
load_dotenv(override=True)

# 确保能导入 src
sys.path.append(str(Path(__file__).parent.parent))

from scripts.convert_log_to_traj import convert_log_to_traj
from src.xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
from src.xpu.xpu_adapter import XpuEntry, XpuAtom
from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_dedup import dedup_and_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从日志批量提取 XPU 并入库")
    parser.add_argument("--log-dir", default="log", help="日志目录")
    parser.add_argument("--traj-dir", default="data/trajectories", help="轨迹输出目录")
    parser.add_argument("--extracted-dir", default="data/xpu_extracted", help="提取结果目录")
    parser.add_argument("--step1-dir", default="data/step1_logs", help="log→traj 中间产物目录")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个日志（0=全部）")
    return parser.parse_args()


def list_logs(log_dir: Path, limit: int) -> List[Path]:
    logs = sorted([p for p in log_dir.glob("*.log")], key=lambda p: p.stat().st_mtime)
    if limit > 0:
        logs = logs[:limit]
    return logs


def load_xpu_from_extracted(path: Path) -> List[Dict]:
    xpus: List[Dict] = []
    if not path.exists():
        return xpus
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("llm_decision") == "xpu" and rec.get("xpu"):
                xpus.append(rec["xpu"])
    return xpus


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    traj_dir = Path(args.traj_dir)
    extracted_dir = Path(args.extracted_dir)
    step1_dir = Path(args.step1_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    step1_dir.mkdir(parents=True, exist_ok=True)

    logs = list_logs(log_dir, args.limit)
    if not logs:
        print("未找到任何日志文件", file=sys.stderr)
        return 1

    store = XpuVectorStore()

    processed = 0
    inserted = 0
    skipped = 0

    for idx, log_path in enumerate(logs, start=1):
        print(f"[{idx}/{len(logs)}] 处理日志: {log_path.name}")
        traj_path = traj_dir / log_path.with_suffix(".jsonl").name
        extracted_path = extracted_dir / log_path.with_suffix(".jsonl").name
        step1_log_dir = step1_dir / log_path.stem
        step1_log_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: log -> traj，并保存中间产物与状态
        convert_log_to_traj(str(log_path), str(traj_path))
        status_path = step1_log_dir / "status.json"
        status = {
            "log": str(log_path),
            "traj": str(traj_path),
            "traj_exists": traj_path.exists(),
        }
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        if not traj_path.exists():
            skipped += 1
            continue

        extract_xpu_from_trajs(traj_path, extracted_path)
        xpu_objs = load_xpu_from_extracted(extracted_path)
        if not xpu_objs:
            skipped += 1
            continue

        for xpu_obj in xpu_objs:
            atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                     for a in xpu_obj.get("atoms", [])]
            entry = XpuEntry(
                id=xpu_obj.get("id"),
                context=xpu_obj.get("context", {}),
                signals=xpu_obj.get("signals", {}),
                advice_nl=xpu_obj.get("advice_nl", []),
                atoms=atoms,
            )

            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)
            result = dedup_and_store(store, entry, embedding, use_llm=True)
            if result.get("action") in ("new", "different_inserted", "merged"):
                inserted += 1

        processed += 1

    store.close()

    print(f"完成: 处理 {processed} 个日志，入库 {inserted} 条经验，跳过 {skipped} 个日志")
    return 0


if __name__ == "__main__":
    sys.exit(main())
