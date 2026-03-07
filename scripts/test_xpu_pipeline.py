import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量 (必须在导入 src 模块之前，因为模块级变量依赖环境变量)
# 强制覆盖系统环境变量，以确保使用 .env 中的配置
load_dotenv(override=True)

# 将项目根目录加入 python path
sys.path.append(str(Path(__file__).parent.parent))

from src.xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_adapter import XpuEntry, XpuAtom
from src.xpu.xpu_dedup import dedup_and_store
from src.logger import get_logger

# 加载环境变量
load_dotenv()

# 配置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_pipeline")

def run_test():
    # 1. 确定输入输出路径
    input_jsonl = Path("data/trajectories/20260205-211732.jsonl")
    output_extracted = Path("data/trajectories/extracted_test.jsonl")
    
    if not input_jsonl.exists():
        print(f"Error: {input_jsonl} not found.")
        return

    print(f"\n=== Step 1: 从轨迹提取经验 (LLM Processing) ===")
    print(f"Input: {input_jsonl}")
    
    # 调用核心提取逻辑
    try:
        extract_xpu_from_trajs(input_jsonl, output_extracted)
    except Exception as e:
        print(f"提取过程出错: {e}")
        return
    
    if not output_extracted.exists():
        print("Extraction failed: Output file not created.")
        return
        
    print(f"\n=== Step 2: 解析提取结果 ===")
    xpu_obj = None
    with open(output_extracted, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                # 提取脚本会输出决策结果，我们需要 llm_decision == 'xpu' 的条目
                if entry.get('llm_decision') == 'xpu':
                    xpu_obj = entry.get('xpu')
                    break
                else:
                    print(f"LLM 决定不提取: {entry.get('llm_reason')}")
            except json.JSONDecodeError:
                continue
    
    if not xpu_obj:
        print("未找到有效的 XPU 经验条目。")
        return

    print(f"提取到的 XPU ID: {xpu_obj.get('id')}")
    print(f"Advice: {xpu_obj.get('advice_nl')}")
    
    print(f"\n=== Step 3: 存入向量数据库 (Vector Store) ===")
    
    try:
        # 构建内存对象
        atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                 for a in xpu_obj.get("atoms", [])]
        
        entry = XpuEntry(
            id=xpu_obj.get("id"),
            context=xpu_obj.get("context", {}),
            signals=xpu_obj.get("signals", {}),
            advice_nl=xpu_obj.get("advice_nl", []),
            atoms=atoms
        )

        # 生成 Embedding
        print("正在调用 Embedding API 生成向量...")
        text = build_xpu_text(entry)
        embedding = text_to_embedding(text)
        print(f"向量生成成功，维度: {len(embedding)}")
        
        # 初始化存储（会自动建表）
        store = XpuVectorStore()
        print("数据库连接成功，表结构已确认。")
        
        # 执行去重与存储
        print("正在执行去重与入库逻辑...")
        result = dedup_and_store(store, entry, embedding, use_llm=True)
        
        print(f"\n>>> 最终结果: {result['action']} <<<")
        print(f"详情: {result['reason']}")
        print(f"Final XPU ID: {result['xpu_id']}")
        
        store.close()
        print("\n流程测试完成！")
        
    except Exception as e:
        print(f"\n入库过程发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
