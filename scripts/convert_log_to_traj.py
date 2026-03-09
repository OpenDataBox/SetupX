import re
import json
import sys
from pathlib import Path

def parse_log_line(line):
    # 匹配格式: 2026-02-05 21:18:19 | INFO     | setup_agent.llm | [0] role=system
    # 或者:     2026-02-05 21:18:19 | INFO     | setup_agent.llm |     content: ...
    pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)\s+\| ([a-zA-Z0-9_.]+) \| (.*)$"
    match = re.match(pattern, line)
    if match:
        return match.groups()
    return None

def convert_log_to_traj(log_path, output_path=None):
    path = Path(log_path)
    if not path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    if output_path is None:
        # 默认保存到 data/trajectories 目录
        traj_dir = Path("data/trajectories")
        traj_dir.mkdir(parents=True, exist_ok=True)
        output_path = traj_dir / path.with_suffix('.jsonl').name

    messages = []
    
    # 状态机变量
    in_message_block = False
    current_role = None
    buffer = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parsed = parse_log_line(line)
            
            # 如果解析失败，可能是多行内容的后续部分
            if not parsed:
                if in_message_block:
                    # 过滤分割线
                    if "=====" in line: continue
                    buffer.append(line)
                continue
                
            timestamp, level, module, content = parsed
            
            # 只关注 setup_agent.llm 模块的日志
            if module != 'setup_agent.llm':
                continue

            # 检测新消息开始: [N] role=...
            role_match = re.match(r"^\[(\d+)\] role=(\w+)$", content)
            if role_match:
                # 保存上一条消息
                if current_role and buffer:
                    full_content = "\n".join(buffer).strip()
                    # 去除 "content: " 前缀
                    if full_content.startswith("content: "):
                        full_content = full_content[9:]
                    
                    # 尝试解析 JSON（针对 assistant）
                    if current_role == 'assistant':
                        try:
                            # 提取JSON部分
                            start_idx = full_content.find('{')
                            if start_idx != -1:
                                json_str = full_content[start_idx:]
                                # 验证是否为合法 JSON
                                json.loads(json_str) 
                                full_content = json_str
                        except:
                            pass # 保持原样

                    messages.append({"role": current_role, "content": full_content})
                
                # 开始新消息
                current_role = role_match.group(2)
                buffer = []
                in_message_block = True
                continue

            # 检测内容行
            if in_message_block:
                if "=====" in content: continue
                # 过滤 content: 前缀
                if content.startswith("content: "):
                    content = content[9:]
                
                buffer.append(content)

        # 保存最后一条消息
        if current_role and buffer:
            full_content = "\n".join(buffer).strip()
            if full_content.startswith("content: "):
                full_content = full_content[9:]
            
            if current_role == 'assistant':
                try:
                    start_idx = full_content.find('{')
                    if start_idx != -1:
                        json_str = full_content[start_idx:]
                        json.loads(json_str)
                        full_content = json_str
                except:
                    pass
            messages.append({"role": current_role, "content": full_content})

    if not messages:
        print("Warning: No LLM conversation found in log.")
        return

    print(f"Extracted {len(messages)} messages.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_log_to_traj.py <log_file> [output_file]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_log_to_traj(log_file, output_file)