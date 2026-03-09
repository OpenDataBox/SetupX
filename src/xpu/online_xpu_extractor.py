#!/usr/bin/env python3
"""
在线 XPU 提取器 - 复用离线管道脚本，实现每个仓库跑完后自动提取存储。

离线模式流程 (run_xpu_pipeline.py):
  1. convert_tracks() - 转换 track.json → jsonl
  2. extract_xpu_from_trajs_mvp.py - LLM 提取
  3. extract_xpu_to_v1.py - 过滤有效经验
  4. index_xpu_to_vector_db.py - 存入数据库

在线模式: 对单个仓库执行同样的流程
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

from ..logger import get_logger

logger = get_logger("xpu.online_extractor")


def online_extract_and_store(repo_name: str, output_dir: str, sha: str = "HEAD") -> dict:
    """
    在线模式主入口 - 复用离线管道脚本

    Args:
        repo_name: 仓库名称 (如 "owner/repo")
        output_dir: 仓库输出目录 (包含 track.json)
        sha: commit SHA

    Returns:
        结果字典
    """
    result = {
        "repo": repo_name,
        "extracted": False,
        "stored": False,
        "xpu_id": None,
        "reason": None
    }

    track_path = Path(output_dir) / "track.json"
    if not track_path.exists():
        result["reason"] = "track.json 不存在"
        return result

    logger.info(f" 在线提取 XPU: {repo_name}")

    # 创建临时工作目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_online_"))

    try:
        # ===== Step 1: 转换格式 (复用 run_xpu_pipeline.convert_tracks 的逻辑) =====
        safe_name = repo_name.replace('/', '__')
        jsonl_name = f"{safe_name}@{sha}.jsonl"
        traj_dir = tmp_dir / "trajs"
        traj_dir.mkdir()
        jsonl_path = traj_dir / jsonl_name

        with open(track_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for step in data:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")

        # ===== Step 2: LLM 提取 (复用 extract_xpu_from_trajs_mvp) =====
        extracted_file = tmp_dir / "extracted.jsonl"

        from .extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
        extract_xpu_from_trajs(jsonl_path, extracted_file)

        # ===== Step 3: 过滤有效经验 (复用 extract_xpu_to_v1 的逻辑) =====
        xpu_obj = None
        if extracted_file.exists():
            with open(extracted_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get('llm_decision') == 'xpu':
                        xpu_obj = entry.get('xpu')
                        break

        if not xpu_obj:
            result["reason"] = entry.get('llm_reason', 'LLM 决定跳过') if 'entry' in dir() else "无提取结果"
            return result

        result["extracted"] = True
        result["xpu_id"] = xpu_obj.get("id")

        # 保存本地副本
        local_xpu_path = Path(output_dir) / "extracted_xpu.json"
        with open(local_xpu_path, 'w', encoding='utf-8') as f:
            json.dump(xpu_obj, f, ensure_ascii=False, indent=2)

        # ===== Step 4: 存入数据库 (复用 index_xpu_to_vector_db 的逻辑) =====
        dns = os.environ.get("dns")
        if not dns:
            result["reason"] = "提取成功但缺少数据库连接 (dns)"
            return result

        try:
            from .xpu_adapter import XpuEntry, XpuAtom
            from .xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
            from .xpu_dedup import dedup_and_store

            # 构建 XpuEntry
            atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                     for a in xpu_obj.get("atoms", [])]
            entry = XpuEntry(
                id=xpu_obj.get("id"),
                context=xpu_obj.get("context", {}),
                signals=xpu_obj.get("signals", {}),
                advice_nl=xpu_obj.get("advice_nl", []),
                atoms=atoms
            )

            # 生成 embedding
            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)

            store = XpuVectorStore()

            # ===== 去重/合并逻辑（LLM 判断 + 智能合并） =====
            dedup_result = dedup_and_store(store, entry, embedding, use_llm=True)
            result["stored"] = True
            result["xpu_id"] = dedup_result["xpu_id"]
            result["reason"] = dedup_result["reason"]
            logger.info(f"[Dedup] {dedup_result['action']}: {dedup_result['reason']}")
            # ===== 去重/合并逻辑结束 =====

            store.close()

        except Exception as e:
            result["reason"] = f"提取成功但存储失败: {e}"
            logger.error(f"存储失败: {e}")

    except Exception as e:
        result["reason"] = f"异常: {str(e)}"
        logger.error(f"在线提取失败: {e}")

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="在线提取并存储 XPU")
    parser.add_argument("--repo", required=True, help="仓库名称 (owner/repo)")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--sha", default="HEAD", help="commit SHA")

    args = parser.parse_args()
    result = online_extract_and_store(args.repo, args.output_dir, args.sha)
    print(json.dumps(result, ensure_ascii=False, indent=2))
