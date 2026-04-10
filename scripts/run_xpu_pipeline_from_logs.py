#!/usr/bin/env python3
"""
从日志目录批量构建轨迹，并提取 XPU 经验后入库。
支持多 worker 并发处理。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    parser.add_argument("--workers", type=int, default=4, help="并发 worker 数")
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


def process_one_log(
    log_path: Path,
    traj_dir: Path,
    extracted_dir: Path,
    step1_dir: Path,
    store: XpuVectorStore,
    lock: threading.Lock,
) -> Tuple[str, int, bool]:
    """处理单个日志文件，返回 (日志名, 入库条数, 是否成功)"""
    name = log_path.name
    traj_path = traj_dir / log_path.with_suffix(".jsonl").name
    extracted_path = extracted_dir / log_path.with_suffix(".jsonl").name
    step1_log_dir = step1_dir / log_path.stem
    step1_log_dir.mkdir(parents=True, exist_ok=True)
    local_inserted = 0

    # 跳过已提取过的日志（断点续跑）
    if extracted_path.exists() and extracted_path.stat().st_size > 0:
        return name, 0, True

    # Step 1: log -> traj
    convert_log_to_traj(str(log_path), str(traj_path))
    status_path = step1_log_dir / "status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump({"log": str(log_path), "traj": str(traj_path), "traj_exists": traj_path.exists()}, f, ensure_ascii=False, indent=2)

    if not traj_path.exists():
        return name, 0, False

    # Step 2: LLM 提取（带重试）
    for attempt in range(3):
        try:
            extract_xpu_from_trajs(traj_path, extracted_path)
            break
        except Exception as e:
            print(f"  [{name}] 提取失败（第 {attempt+1} 次）: {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                print(f"  [{name}] 3 次均失败，跳过")
                return name, 0, False

    # Step 3: 入库（带重试）
    xpu_objs = load_xpu_from_extracted(extracted_path)
    if not xpu_objs:
        return name, 0, True

    for xpu_obj in xpu_objs:
        atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                 for a in xpu_obj.get("atoms", [])]
        auto_id = xpu_obj.get("id") or f"xpu_{int(time.time())}_{os.urandom(3).hex()}"
        entry = XpuEntry(
            id=auto_id,
            context=xpu_obj.get("context", {}),
            signals=xpu_obj.get("signals", {}),
            advice_nl=xpu_obj.get("advice_nl", []),
            atoms=atoms,
        )

        for attempt in range(3):
            try:
                text = build_xpu_text(entry)
                embedding = text_to_embedding(text)
                with lock:
                    result = dedup_and_store(store, entry, embedding, use_llm=True)
                if result.get("action") in ("new", "different_inserted", "merged"):
                    local_inserted += 1
                break
            except Exception as e:
                print(f"  [{name}] 入库失败（第 {attempt+1} 次）: {e}")
                if attempt < 2:
                    time.sleep(10)

    return name, local_inserted, True


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

    total = len(logs)
    print(f"共 {total} 个日志，{args.workers} 个 worker 并发处理")

    store = XpuVectorStore()
    lock = threading.Lock()

    done = 0
    inserted = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_log, log_path, traj_dir, extracted_dir, step1_dir, store, lock): log_path
            for log_path in logs
        }

        for future in as_completed(futures):
            name, n_inserted, success = future.result()
            done += 1
            inserted += n_inserted
            if not success:
                skipped += 1
            processed = done - skipped
            print(f"[{done}/{total}] {name} → 入库 {n_inserted} 条 {'✓' if success else '✗'}")

    store.close()
    print(f"\n完成: 处理 {done - skipped} 个日志，入库 {inserted} 条经验，跳过 {skipped} 个日志")
    return 0


if __name__ == "__main__":
    sys.exit(main())
