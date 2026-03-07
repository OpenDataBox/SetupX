import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量 (必须在导入 src 模块之前，因为模块级变量依赖环境变量)
load_dotenv(override=True)

# 将项目根目录加入 python path
sys.path.append(str(Path(__file__).parent.parent))

from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_adapter import XpuEntry, XpuAtom

def seed_data():
    print("开始预设 XPU 数据...")
    
    # 构造一条关于 build-essential 的经验
    # 当遇到 gcc 错误或 Python.h 缺失时使用
    entry = XpuEntry(
        id="xpu_env_build_essential_missing",
        context={
            "os": ["ubuntu", "debian"],
            "lang": "python",
            "desc": "缺少 C/C++ 编译环境，导致某些 Python 包无法安装"
        },
        signals={
            "keywords": ["gcc: command not found", "fatal error: Python.h: No such file or directory", "error: command 'gcc' failed"],
            "regex": [r"command 'gcc' failed with exit status 1", r"fatal error: Python.h: No such file or directory"]
        },
        advice_nl=[
            "缺少构建工具链，需安装 build-essential 和 python3-dev",
            "执行: apt-get update && apt-get install -y build-essential python3-dev"
        ],
        atoms=[
            XpuAtom(name="apt_install", args={"packages": ["build-essential", "python3-dev"]})
        ]
    )

    # 生成 Embedding
    print(f"正在生成 Embedding (ID: {entry.id})...")
    text = build_xpu_text(entry)
    embedding = text_to_embedding(text)
    
    # 存入数据库
    store = XpuVectorStore()
    store.upsert_entry(entry, embedding)
    store.close()
    
    print(f"成功预设经验: {entry.id}")
    print("数据准备完成！")

if __name__ == "__main__":
    seed_data()
