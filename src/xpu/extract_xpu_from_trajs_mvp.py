"""从 EnvBench / Repo2Run 轨迹中抽取环境经验 XPU 的 MVP 脚本。

本模块是 XPU 提取流程的核心实现，支持从 Agent 执行轨迹（JSONL 格式）中
自动抽取可复用的环境配置经验。

提取流程（四阶段）：
1. 加载轨迹文件（iter_traj_files → load_traj）
2. 启发式筛选（heuristic_stats_for_traj → heuristic_is_candidate）
   - 统计环境命令数、错误关键词数等指标
   - 根据评分决定是否值得送 LLM 提取
3. LLM 提取（build_traj_prompt → openai_compatible_chat_completions）
   - 构造 prompt 发给 LLM，让 LLM 分析轨迹并生成结构化 XPU
4. 输出结果（JSONL 格式，每行一条 XPU）

支持的轨迹格式：
- EnvBench 格式：包含 node="commands_history" 的结构化轨迹
- Repo2Run 格式：Markdown 代码块中的 bash 命令
- 本项目 Agent 格式：JSON 格式的 SHELL_COMMAND 动作
"""

import argparse  # 命令行参数解析
import json  # JSON 序列化/反序列化
import os  # 环境变量
import re  # 正则表达式
from pathlib import Path  # 路径操作
from typing import Any, Dict, Iterable, List, Tuple  # 类型标注

import requests  # HTTP 请求（调用 LLM API）
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量
from tqdm import tqdm  # 进度条


# ============================================================================
# 默认配置
# ============================================================================

# 项目根目录（向上两级：extract_xpu_from_trajs_mvp.py → xpu/ → src/）
ROOT_DIR = Path(__file__).resolve().parents[1]
# 默认轨迹目录
DEFAULT_TRAJ_DIR = ROOT_DIR / "tmp" / "traj_py_subset_50_kimi"
# 默认输出文件路径
DEFAULT_OUTPUT = ROOT_DIR / "xpuExtract" / "outputs" / "traj_xpu_mvp.jsonl"

# LLM 调用相关默认配置（可通过环境变量覆盖）
DEFAULT_LLM_MODEL = os.environ.get("XPU_EXTRACT_MODEL", os.environ.get("MOONSHOT_MODEL", "gpt-4o-2024-05-13"))
DEFAULT_API_KEY_ENV = os.environ.get("XPU_EXTRACT_API_KEY_ENV", "OPENAI_API_KEY")  # API Key 对应的环境变量名
DEFAULT_BASE_URL_ENV = os.environ.get("XPU_EXTRACT_BASE_URL_ENV", "OPENAI_BASE_URL")  # Base URL 对应的环境变量名
DEFAULT_TIMEOUT_SEC = int(os.environ.get("XPU_EXTRACT_TIMEOUT", "60"))  # API 调用超时秒数

# 启发式关键词：用于判断轨迹中是否包含环境相关错误
ERROR_KEYWORDS = [
    "ModuleNotFoundError",  # Python 模块未找到
    "ImportError",  # Python 导入错误
    "No module named",  # Python 模块缺失
    "cannot import name",  # Python 导入名称错误
    "Could not find a version",  # pip 找不到版本
    "command not found",  # 系统命令未安装
    "Permission denied",  # 权限不足
    "error:",  # 通用错误标记
    "Error:",  # 通用错误标记（首字母大写）
    "Traceback",  # Python 异常回溯
    "failed with exit code"  # 命令执行失败
]

# 环境相关命令关键词：用于判断轨迹中是否执行了环境配置命令
ENV_CMD_KEYWORDS = [
    "pip install",  # pip 包安装
    "poetry install",  # poetry 依赖安装
    "apt-get install",  # 系统包安装
    "conda install",  # conda 包安装
    "python setup.py"  # setuptools 安装
]


# ============================================================================
# 工具函数
# ============================================================================

def get_env_or_raise(name: str) -> str:
    """获取必需的环境变量，不存在时抛出异常

    特殊降级逻辑：如果找不到 MOONSHOT_API_KEY，会尝试回退到 OPENAI_API_KEY。

    Args:
        name: 环境变量名

    Returns:
        环境变量值

    Raises:
        RuntimeError: 环境变量未设置
    """
    val = os.environ.get(name)
    if not val:
        # 降级尝试：Kimi 的 key 没找到时试试通用的 OpenAI key
        if name == "MOONSHOT_API_KEY":
            val = os.environ.get("OPENAI_API_KEY")
    if not val:
        raise RuntimeError(f"缺少必需的环境变量: {name}")
    return val


def openai_compatible_chat_completions(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_sec: int,
    response_format_json: bool = True,
) -> Dict[str, Any]:
    """调用 OpenAI 兼容的 Chat Completions API

    支持 OpenAI、ARK（火山引擎）、Kimi 等兼容 API。

    Args:
        model: 模型名称
        messages: 对话消息列表
        api_key: API Key
        base_url: API Base URL
        timeout_sec: 请求超时秒数
        response_format_json: 是否要求 JSON 格式输出

    Returns:
        API 响应的完整 JSON 字典

    Raises:
        RuntimeError: API 返回 HTTP 4xx/5xx 错误
    """
    # 修复 Base URL：ARK 使用 v3，不要强行加 v1
    if "v1" not in base_url and "v3" not in base_url and not base_url.endswith("/"):
        base_url += "/v1"
    url = base_url.rstrip("/") + "/chat/completions"  # 拼接完整 API 路径


    # 调试日志：打印请求信息（API Key 只显示前 8 位）
    masked_key = api_key[:8] + "..." if api_key else "None"
    print(f"[DEBUG] LLM Request URL: {url}")
    print(f"[DEBUG] LLM API Key: {masked_key}")
    print(f"[DEBUG] LLM Model: {model}")

    # 构造 HTTP 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # 构造请求体
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,  # 使用确定性输出
        "stream": False,  # 不使用流式输出
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}  # 要求 JSON 格式输出

    # 发送 HTTP POST 请求
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def parse_llm_json(s: str) -> Dict[str, Any]:
    """解析 LLM 输出中的 JSON

    处理多种 LLM 输出格式：
    - 带 BOM 前缀的 JSON
    - 包裹在 ```json ... ``` 中的 JSON
    - 纯 JSON 字符串
    """
    s = s.strip()
    # 移除 BOM（Byte Order Mark）前缀
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    # 移除 Markdown 代码块包裹
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[len("```json"):].strip()  # 去掉 ```json 前缀
        else:
            s = s[3:].strip()  # 去掉 ``` 前缀
        if s.endswith("```"):
            s = s[:-3].strip()  # 去掉 ``` 后缀
    return json.loads(s)


def truncate(text: Any, max_len: int) -> str:
    """截断过长文本，保留头尾各一半"""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    keep = max_len // 2  # 头尾各保留一半
    return text[:keep] + "\n... [TRUNCATED] ...\n" + text[-keep:]


def load_llm_config_from_env() -> Dict[str, Any]:
    """从环境变量加载 LLM 调用配置"""
    return {
        "llm_model": DEFAULT_LLM_MODEL,  # LLM 模型名称
        "api_key_env_var": DEFAULT_API_KEY_ENV,  # API Key 对应的环境变量名
        "base_url_env_var": DEFAULT_BASE_URL_ENV,  # Base URL 对应的环境变量名
        "timeout_sec": DEFAULT_TIMEOUT_SEC,  # 请求超时秒数
        "llm_language": "zh",  # 输出语言
    }


# ============================================================================
# 轨迹文件加载与解析
# ============================================================================

def iter_traj_files(traj_path: Path) -> List[Path]:
    """枚举轨迹文件列表

    支持单文件或目录。目录模式下只返回文件名中包含 "@" 的 .jsonl 文件
    （"@" 分隔 repo 名和 revision，如 "org__repo@abc123.jsonl"）。
    """
    if traj_path.is_file():
        return [traj_path]  # 单文件直接返回
    if traj_path.is_dir():
        return sorted([p for p in traj_path.glob("*.jsonl") if "@" in p.name])
    raise FileNotFoundError(str(traj_path))


def parse_repo_revision_from_name(path: Path) -> Tuple[str, str]:
    """从轨迹文件名解析仓库名和 revision

    文件名格式：org__repo@revision.jsonl
    例如："pytest-dev__pytest@abc123.jsonl" → ("pytest-dev/pytest", "abc123")
    """
    name = path.name
    if not (name.endswith(".jsonl") and "@" in name):
        return "unknown/repo", "unknown"
    base = name[: -len(".jsonl")]  # 去掉 .jsonl 后缀
    try:
        repo_part, rev = base.rsplit("@", 1)  # 按最后一个 "@" 分割
    except ValueError:
        return base, "unknown"
    repo = repo_part.replace("__", "/")  # 将 "__" 还原为 "/"
    return repo, rev


def load_traj(path: Path) -> List[Dict[str, Any]]:
    """加载轨迹文件（JSONL 格式，每行一个 JSON 对象）"""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue  # 跳过空行
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 跳过 JSON 解析失败的行
    return out


# ============================================================================
# 启发式筛选：从轨迹中提取统计信息并评分
# ============================================================================

def _iter_strings(obj: Any) -> Iterable[str]:
    """递归遍历嵌套数据结构中的所有字符串

    支持字符串、字典、列表的递归遍历。
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def extract_commands_history(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从轨迹中提取命令执行历史

    支持三种轨迹格式：
    1. EnvBench 格式：node="commands_history" 包含结构化命令列表
    2. 本项目 Agent 格式：JSON 格式的 SHELL_COMMAND 动作
    3. Repo2Run 格式：Markdown 代码块中的 bash 命令
    """
    cmds = []

    # 正则匹配 Markdown 中的 bash 代码块
    bash_pattern = re.compile(r"```bash\s+(.*?)\s+```", re.DOTALL)

    for item in traj:
        # 格式 1：EnvBench 的结构化命令历史节点
        if item.get("node") == "commands_history":
            raw = item.get("commands") or []
            if isinstance(raw, list):
                return raw  # 直接返回结构化命令列表

        content = item.get("content", "")  # 消息内容
        role = item.get("role", "")  # 消息角色

        # 只解析 assistant 角色的输出（agent 的决策）
        if role == "assistant" and content:
            # 格式 2：尝试解析 JSON 格式（本项目 Agent 的输出）
            try:
                if isinstance(content, str) and content.strip().startswith("{"):
                    data = json.loads(content)
                    cmd = None
                    if isinstance(data, dict):
                        inner_content = data.get("content")
                        if isinstance(inner_content, dict):
                            # 形式 A: {"action_type": "SHELL_COMMAND", "content": {"command": "..."}}
                            cmd = inner_content.get("command")
                        elif data.get("command"):
                            # 形式 B: {"command": "..."} （直接结构）
                            cmd = data.get("command")

                    if cmd:
                        cmds.append({"command": cmd, "exit_code": 0})
                        continue  # 成功解析出 JSON 命令，跳过正则匹配
            except json.JSONDecodeError:
                pass

            # 格式 3：正则匹配 Markdown bash 代码块（Repo2Run 兼容）
            matches = bash_pattern.findall(content)
            for match in matches:
                clean_cmd = match.strip()
                cmds.append({"command": clean_cmd, "exit_code": 0})

    return cmds


def heuristic_stats_for_traj(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计轨迹的启发式特征

    扫描轨迹中的所有文本和命令，统计：
    - num_agent_steps: agent 步骤数
    - num_error_keywords: 错误关键词出现次数
    - num_commands: 总命令数
    - num_env_commands: 环境相关命令数（pip install 等）
    """
    num_agent_steps = 0
    num_error_keywords = 0

    # 1. 扫描全文找错误关键词
    for item in traj:
        # 统计 agent 步骤数（兼容 role 和 node 两种格式）
        if item.get("role") == "assistant" or item.get("node") == "agent":
            num_agent_steps += 1

        # 递归遍历所有文本字段，查找错误关键词
        for text in _iter_strings(item):
            t_low = text.lower()
            if any(kw.lower() in t_low for kw in ERROR_KEYWORDS):
                num_error_keywords += 1
                # break # 不要 break，统计所有

    # 2. 提取并统计命令
    cmds = extract_commands_history(traj)
    num_commands = len(cmds)
    num_env_commands = 0

    # 统计环境相关命令数（pip install、apt-get install 等）
    for c in cmds:
        cmd_str = str(c.get("command", ""))
        cmd_low = cmd_str.lower()
        if any(kw.lower() in cmd_low for kw in ENV_CMD_KEYWORDS):
            num_env_commands += 1

    return {
        "num_agent_steps": num_agent_steps,
        "num_commands": num_commands,
        "num_env_commands": num_env_commands,
        "num_error_keywords": num_error_keywords,
    }


def heuristic_is_candidate(stats: Dict[str, Any]) -> Tuple[bool, float]:
    """判断轨迹是否值得送 LLM 提取 XPU

    评分规则：
    - 有环境命令（pip install 等）→ +5 分
    - 有错误关键词 → +5 分
    - 执行过任何命令 → +1 分

    当前阈值：score > 0 即为候选（非常宽松，只要执行过命令就会送 LLM）
    """
    score = 0.0

    # 有环境命令大幅加分（说明轨迹包含环境配置操作）
    if stats.get("num_env_commands", 0) >= 1:
        score += 5.0 # 大幅加分
    # 有错误关键词大幅加分（说明轨迹中遇到了问题）
    if stats.get("num_error_keywords", 0) >= 1:
        score += 5.0 # 大幅加分
    # 执行过命令小幅加分
    if stats.get("num_commands", 0) >= 1:
        score += 1.0

    print(f"[DEBUG] Heuristic Stats: {stats}, Score: {score}")

    # [P1 优化] 阈值从 score > 0 提升到 score >= 6
    # 要求至少同时包含环境命令(+5)和错误关键词(+5)，或其中之一加上命令(+1)
    # 这样避免把纯探索性轨迹（只有命令没有环境操作）送 LLM 浪费 token
    return score >= 6, score


# ============================================================================
# LLM 提取：构造 prompt 让 LLM 从轨迹中提取 XPU
# ============================================================================

def build_traj_prompt(
    repo: str,
    rev: str,
    traj: List[Dict[str, Any]],
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
    phase2_context: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """构造 XPU 提取的 LLM prompt

    将轨迹中的命令历史和错误日志整理为结构化的 prompt，
    让 LLM 分析轨迹并生成 XPU 经验条目。
    """
    # 提取并格式化命令历史
    cmds = extract_commands_history(traj)
    lines_cmds: List[str] = []
    for c in cmds:
        cmd_str = str(c.get("command", ""))
        lines_cmds.append(f"$ {cmd_str}")  # 用 $ 前缀标记命令
    commands_text = truncate("\n".join(lines_cmds), 4000)  # 截断到 4000 字符

    # 提取错误日志片段（只看 system 和 user 角色的内容）
    error_lines: List[str] = []
    for item in traj:
        # 我们的 Agent 将执行结果记录在 user 角色中
        if item.get("role") in ("system", "user"):
            text = item.get("content", "")
            if any(kw.lower() in text.lower() for kw in ERROR_KEYWORDS):
                error_lines.append(text)

    # 如果错误日志太多，保留头 15 条 + 尾 10 条
    if len(error_lines) > 30:
        error_lines = error_lines[:15] + ["... [TRUNCATED] ..."] + error_lines[-10:]
    errors_text = truncate("\n".join(error_lines), 4000)

    # 系统提示：定义 LLM 角色和任务要求
    system_text = (
        "你是一名资深 Python 项目环境配置与依赖问题专家。"
        "\n现在给你一个仓库在自动环境搭建时的完整 agent 轨迹（含执行的命令和报错日志）。"
        "\n"
        "\n你的任务："
        "\n1. 仔细分析整条轨迹，识别其中所有独立的环境问题（例如：缺少 Python 解释器、缺少系统库、pip 依赖冲突、权限问题等）。"
        "\n2. 对于每个独立问题，判断它是否值得提炼成一条可复用的环境经验 XPU。"
        "\n   - 值得提炼的标准：该问题具有通用性，其他仓库也可能遇到相同或类似的错误，且修复方案是确定的。"
        "\n   - 不值得提炼的情况：问题过于特定于该仓库（如仓库自身代码 bug），或者修复方案不明确。"
        "\n3. 对于每个值得提炼的问题，生成一条结构化的 XPU 条目，每条 XPU 应当聚焦于一个独立的根因，不要把多个不相关问题混在一条里。"
        "\n"
        "\n"
        "\n【提炼原则（必须遵守）】"
        "\n- 如果提供了 phase2_context，prosecution_charges 是因果关系最清晰的知识来源，优先从中提炼。"
        "\n- 即便 verdict=guilty，也应提炼其中可泛化的模式。"
        "\n- 允许记录三类经验（按优先级）："
        "\n  1.【工具链模式】构建工具/包管理器层面的规律"
        "\n  2.【包级安装模式】特定 Python 包的已知安装陷阱"
        "\n  3.【环境配置模式】系统级配置/权限/路径问题"
        "\n- 禁止记录纯粹的仓库特定事实，即【该仓库需要包 X】但不解释 WHY。"
        "\n  判断标准：如果去掉仓库名，这条经验对其他用到相同包/工具的仓库是否仍然有用？有用则记录，否则丢弃。"
        "\n- signals 中增加 situation_triggers（2-4条），描述经验适用场景，用于向量检索召回。"
        "\n- 一条 XPU 只解决一个根因，不混合多个不相关问题。"
        "\n- 不要生成 id 字段，系统会自动分配唯一 ID。"
        "\n"
        "\n回答必须是严格的 JSON 对象，不包含任何多余文字。"
    )

    # 用户输入：仓库信息 + 统计数据 + 命令历史 + 错误日志 + XPU schema
    user_payload: Dict[str, Any] = {
        "repository": repo,
        "revision": rev,
        "stats": stats,
        "commands_history_text": commands_text,  # 命令执行历史
        "error_snippets_text": errors_text,  # 错误日志片段
        "xpu_schema": {  # XPU 输出格式说明
            "id": "string，唯一标识，如 xpu_env_py_xxx",
            "context": {
                "lang": "例如 python",
                "os": ["相关操作系统，如 linux 等"],
                "python": ["相关 Python 版本前缀，如 3.8"],
                "tools": ["相关工具列表，如 pytest, pip 等"],
            },
            "signals": {
                "regex": ["匹配该错误的正则表达式"],
                "keywords": ["用于粗略检索的关键词"],
            },
            "advice_nl": ["1-5 条中文建议，解释问题根因和修复思路"],
            "atoms": [
                {
                    "name": "原子操作类型，如 pip_install / pip_pin / or_upgrade_pkg / set_env / set_umask 等",
                    "args": "一个字典，包含该原子需要的参数",
                }
            ],
        },
        "output_requirement": (
            "你必须输出一个 JSON 对象，形如：{decision, reason, xpus}。"
            "decision 只能是 'skip' 或 'xpu'。"
            "当 decision='skip' 时，表示整条轨迹没有任何值得提炼的经验，xpus 为空数组 []。"
            "当 decision='xpu' 时，xpus 是一个数组，包含一条或多条与 xpu_schema 兼容的 XPU 对象，"
            "每条 XPU 对应轨迹中一个独立的环境问题及其修复方案。"
            "每条 XPU 的 id 必须唯一（如 xpu_env_py_001, xpu_env_py_002）。"
            "所有说明性文字使用简体中文。"
        ),
        "language": cfg.get("llm_language", "zh"),
    }

    if phase2_context:
        user_payload["phase2_context"] = {
            "prosecution_charges": phase2_context.get("prosecution_charges", []),
            "verdict": phase2_context.get("verdict"),
            "judge_reasoning": phase2_context.get("judge_reasoning", ""),
            "verifier_summary": phase2_context.get("verifier_summary", ""),
            "prosecutor_investigation": phase2_context.get("prosecutor_investigation", ""),
        }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# ============================================================================
# 主提取流程
# ============================================================================

def extract_xpu_from_trajs(
    traj_path: Path,
    output_jsonl: Path,
    phase2_context: Dict[str, Any] | None = None,
) -> None:
    """从轨迹文件中批量提取 XPU 经验（主入口函数）

    完整流程：
    1. 加载 .env 环境变量
    2. 枚举所有轨迹文件
    3. 对每个轨迹文件：启发式筛选 → LLM 提取 → 写入输出文件
    4. 输出为 JSONL 格式（每行一条记录）
    """
    load_dotenv()  # 加载 .env 文件
    cfg = load_llm_config_from_env()  # 加载 LLM 配置

    api_key = get_env_or_raise(cfg["api_key_env_var"])  # 获取 API Key
    # 默认 Base URL 处理
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    files = iter_traj_files(traj_path)  # 枚举轨迹文件
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for path in tqdm(files, total=len(files), desc="从轨迹中提取 XPU"):
            # 从文件名解析仓库名和 revision
            repo, rev = parse_repo_revision_from_name(path)
            # 加载轨迹数据
            traj = load_traj(path)
            # 启发式统计和筛选
            stats = heuristic_stats_for_traj(traj)
            is_candidate, score = heuristic_is_candidate(stats)
            stats["heuristic_score"] = score
            stats["heuristic_is_candidate"] = is_candidate

            # 初始化 LLM 决策结果
            llm_decision: str = "heuristic_skip"  # 默认被启发式筛选跳过
            llm_reason: str | None = None
            xpu_obj: Dict[str, Any] | None = None
            usage: Dict[str, Any] = {}
            error_info: str | None = None

            # 通过启发式筛选的候选才送 LLM 提取
            if is_candidate:
                try:
                    # 构造 LLM prompt（含 Phase 2 上下文）
                    messages = build_traj_prompt(repo, rev, traj, stats, cfg, phase2_context=phase2_context)
                    # 调用 LLM API
                    raw = openai_compatible_chat_completions(
                        model=cfg["llm_model"],
                        messages=messages,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_sec=cfg["timeout_sec"],
                        response_format_json=True,
                    )
                    content = raw["choices"][0]["message"]["content"]  # 提取 LLM 输出
                    usage = raw.get("usage") or {}  # token 使用统计
                    parsed = parse_llm_json(content)  # 解析 JSON
                    llm_decision = str(parsed.get("decision") or "error")
                    llm_reason = parsed.get("reason")
                    if llm_decision == "xpu":
                        # 兼容新格式 xpus（数组）和旧格式 xpu（单条）
                        xpu_list = parsed.get("xpus") or []
                        if not xpu_list:
                            single = parsed.get("xpu")
                            if single:
                                xpu_list = [single]
                    elif llm_decision not in {"skip", "xpu"}:
                        llm_decision = "error"
                        error_info = f"非预期的 decision 值: {parsed!r}"
                except Exception as e:
                    llm_decision = "error"
                    error_info = str(e)

            # 写入输出文件：每条 XPU 独立一行（方便下游逐条处理）
            if llm_decision == "xpu" and xpu_list:
                for xpu_obj in xpu_list:
                    out_obj = {
                        "repository": repo,
                        "revision": rev,
                        "traj_path": str(path),
                        "heuristics": stats,
                        "llm_decision": "xpu",
                        "llm_reason": llm_reason,
                        "xpu": xpu_obj,
                        "llm_model": cfg.get("llm_model"),
                        "usage": usage,
                        "error": error_info,
                    }
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            else:
                # 未提取到 XPU 的也记录一行（保留筛选/决策日志）
                out_obj = {
                    "repository": repo,
                    "revision": rev,
                    "traj_path": str(path),
                    "heuristics": stats,
                    "llm_decision": llm_decision,
                    "llm_reason": llm_reason,
                    "xpu": None,
                    "llm_model": cfg.get("llm_model"),
                    "usage": usage,
                    "error": error_info,
                }
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


# ============================================================================
# 命令行入口
# ============================================================================

def main() -> None:
    """命令行入口：解析参数并执行 XPU 提取"""
    parser = argparse.ArgumentParser(description="从 EnvBench 轨迹中启发式筛选并通过 LLM 抽取 XPU 经验")
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ_DIR)  # 轨迹文件/目录
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)  # 输出路径
    args = parser.parse_args()

    extract_xpu_from_trajs(Path(args.traj), Path(args.output))


if __name__ == "__main__":
    main()
