import json
import subprocess
import sys
from pathlib import Path

def run_batch(jsonl_path, limit=50):
    """
    读取 JSONL 列表并批量运行前 N 个仓库
    """
    if not Path(jsonl_path).exists():
        print(f"错误: 找不到文件 {jsonl_path}")
        return

    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            
            try:
                data = json.loads(line)
                repo_full_name = data.get("repository")
                if not repo_full_name:
                    continue
                
                repo_url = f"https://github.com/{repo_full_name}.git"
                print(f"
{'='*60}")
                print(f"正在运行第 {count+1}/{limit} 个仓库: {repo_url}")
                print(f"{'='*60}")

                # 调用 Agent 主入口
                # 使用 -u 确保实时输出日志
                cmd = [sys.executable, "-m", "src.main", repo_url, "50"]
                
                # 我们开启一个新的进程运行，避免子进程崩溃影响主脚本
                process = subprocess.run(cmd)
                
                print(f"仓库 {repo_full_name} 运行结束，退出码: {process.returncode}")
                count += 1
                
            except Exception as e:
                print(f"处理行时出错: {e}")
                continue

    print(f"
批量运行完成，共处理 {count} 个仓库。")

if __name__ == "__main__":
    # 默认读取 data/python329.jsonl
    jsonl_file = "data/python329.jsonl"
    run_batch(jsonl_file, limit=50)
