"""XPU 数据结构定义与评分检索逻辑。

本模块定义了 XPU（eXPerience Unit，经验单元）系统的核心数据结构和检索逻辑：

数据结构：
- XpuAtom：原子操作（如 pip_install、set_env 等），可被渲染为 bash 命令
- XpuEntry：完整的 XPU 经验条目，包含上下文、信号、建议、原子操作和遥测数据
- XpuContext：查询时的上下文过滤条件

检索逻辑：
- score_xpu()：计算单条 XPU 与当前错误日志的匹配分数
- retrieve_xpu_candidates()：从 XPU 列表中选取 Top-K 最相关的条目

渲染逻辑：
- render_atom_to_commands()：将原子操作转为可执行 bash 命令
- render_candidates_block()：将候选 XPU 列表渲染为注入 LLM prompt 的文本块
"""

import json  # JSON 序列化/反序列化
import re  # 正则表达式匹配
from dataclasses import dataclass, field  # 数据类装饰器
from pathlib import Path  # 路径操作
from typing import Any, Dict, Iterable, List, Optional, Sequence  # 类型标注


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass
class XpuAtom:
    """XPU 原子操作（如 pip_install, set_env 等）

    原子操作是 XPU 中的最小可执行单元，通过 render_atom_to_commands() 可以
    将其转换为一条或多条 bash 命令。

    Attributes:
        name: 操作类型名称，如 "pip_install"、"set_env"、"apt_install" 等
        args: 操作参数字典，如 {"package": "numpy", "spec": ">=1.20"}
    """
    name: str  # 操作类型名称
    args: Dict[str, Any]  # 操作参数字典


@dataclass
class XpuEntry:
    """XPU 经验条目——完整描述一个可复用的环境配置经验

    一条 XPU 记录了"遇到什么问题（signals）、在什么环境下（context）、
    该怎么解决（advice_nl + atoms）"的完整信息。

    Attributes:
        id: 唯一标识符，如 "xpu_env_py_001"
        context: 适用的上下文环境，如 {"lang": "python", "tools": ["pytest"], "os": ["linux"]}
        signals: 触发信号，包含 regex（正则匹配）和 keywords（关键词匹配）
        advice_nl: 自然语言建议列表（中文），1-5 条
        atoms: 原子操作列表，可被渲染为具体 bash 命令
        telemetry: 遥测数据，记录使用次数（hits）、成功次数（successes）、失败次数（failures）
    """
    id: str  # 唯一标识符
    context: Dict[str, Any]  # 适用的上下文环境
    signals: Dict[str, Any]  # 触发信号（regex + keywords）
    advice_nl: List[str]  # 自然语言建议列表
    atoms: List[XpuAtom] = field(default_factory=list)  # 原子操作列表（可选）
    telemetry: Dict[str, Any] = field(default_factory=lambda: {"hits": 0, "successes": 0, "failures": 0})  # 遥测数据


@dataclass
class XpuContext:
    """XPU 查询上下文——用于检索时的上下文过滤条件

    Attributes:
        lang: 编程语言，如 "python"
        os: 操作系统，如 "linux"
        python: Python 版本前缀，如 "3.8"
        tools: 工具链列表，如 ("pytest", "pip")
    """
    lang: Optional[str] = None  # 编程语言
    os: Optional[str] = None  # 操作系统
    python: Optional[str] = None  # Python 版本
    tools: Sequence[str] = tuple()  # 工具链列表


# ============================================================================
# JSONL 文件加载
# ============================================================================

def _parse_xpu_line(obj: Dict[str, Any]) -> XpuEntry:
    """解析单行 JSON 字典为 XpuEntry 对象

    将 JSON 中的 atoms 列表转换为 XpuAtom 对象列表。

    Args:
        obj: 从 JSONL 文件中读取的一行 JSON 字典

    Returns:
        解析后的 XpuEntry 对象
    """
    atoms_raw = obj.get("atoms") or []  # 获取原子操作列表，不存在则为空
    # 将每个原子操作字典转为 XpuAtom 对象
    atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {})) for a in atoms_raw]
    return XpuEntry(
        id=obj.get("id", ""),
        context=obj.get("context", {}),
        signals=obj.get("signals", {}),
        advice_nl=list(obj.get("advice_nl") or []),
        atoms=atoms,
        telemetry=obj.get("telemetry", {"hits": 0, "successes": 0, "failures": 0})
    )


def load_xpu_entries(jsonl_path: Path) -> List[XpuEntry]:
    """从 JSONL 文件加载所有 XPU 条目

    JSONL 格式：每行一个 JSON 对象，代表一条 XPU 经验。

    Args:
        jsonl_path: JSONL 文件路径

    Returns:
        XpuEntry 列表
    """
    entries: List[XpuEntry] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            obj = json.loads(line)  # 解析 JSON
            entries.append(_parse_xpu_line(obj))  # 转为 XpuEntry 对象
    return entries


# ============================================================================
# 评分与检索逻辑
# ============================================================================

def _match_regex(log_snippet: str, patterns: Iterable[str]) -> bool:
    """检测日志片段是否匹配任一正则表达式

    Args:
        log_snippet: 错误日志文本片段
        patterns: 正则表达式列表（来自 XPU 的 signals.regex）

    Returns:
        是否有至少一个正则匹配成功
    """
    for p in patterns:
        try:
            if re.search(p, log_snippet):
                return True  # 有一个匹配就返回
        except re.error:
            continue  # 正则语法错误则跳过
    return False


def _keyword_score(log_snippet: str, keywords: Iterable[str]) -> int:
    """简单关键词重叠评分：统计日志片段中出现了多少个关键词

    Args:
        log_snippet: 错误日志文本片段
        keywords: 关键词列表（来自 XPU 的 signals.keywords）

    Returns:
        匹配到的关键词数量
    """
    text = log_snippet.lower()  # 转小写，不区分大小写匹配
    score = 0
    for kw in keywords:
        if not kw:
            continue  # 跳过空关键词
        if kw.lower() in text:
            score += 1  # 每匹配一个关键词加 1 分
    return score


def _context_match_score(entry: XpuEntry, ctx: XpuContext) -> int:
    """计算上下文匹配分数

    对查询上下文与 XPU 条目的上下文进行多维度匹配打分：
    - 语言匹配：+2 分
    - 工具交集：+2 分（有任一工具重叠即可）
    - Python 版本前缀匹配：+1 分
    - 操作系统匹配：+1 分

    Args:
        entry: XPU 经验条目
        ctx: 查询上下文

    Returns:
        上下文匹配分数（0-6）
    """
    score = 0
    ectx = entry.context  # XPU 条目的上下文

    # 语言匹配（精确匹配）
    if ctx.lang and ectx.get("lang") == ctx.lang:
        score += 2

    # 工具交集（ANY 匹配：只要有一个工具重叠就加分）
    tools_entry = set(ectx.get("tools") or [])  # XPU 的工具集
    tools_ctx = set(ctx.tools or [])  # 查询的工具集
    if tools_entry and tools_ctx and tools_entry & tools_ctx:
        score += 2  # 有交集就加 2 分

    # Python 版本前缀匹配（如 "3.8" 匹配 "3.8.10"）
    if ctx.python:
        py_list = ectx.get("python") or []
        for py in py_list:
            if str(py).startswith(str(ctx.python)):
                score += 1
                break  # 匹配到一个就够了

    # 操作系统匹配
    if ctx.os:
        os_list = ectx.get("os") or []
        if ctx.os in os_list:
            score += 1

    return score


def score_xpu(entry: XpuEntry, log_snippet: str, ctx: XpuContext) -> float:
    """计算单条 XPU 条目的综合评分

    评分公式：正则匹配（+10） + 关键词重叠（×1.0） + 上下文匹配（×1.5） + 有原子操作（+0.5）

    Args:
        entry: XPU 经验条目
        log_snippet: 当前错误日志文本
        ctx: 查询上下文

    Returns:
        综合评分（浮点数）
    """
    signals = entry.signals or {}
    regexes = signals.get("regex") or []  # 正则表达式列表
    keywords = signals.get("keywords") or []  # 关键词列表

    score = 0.0

    # 正则匹配：命中则大幅加分（+10），因为正则匹配精度最高
    if regexes and _match_regex(log_snippet, regexes):
        score += 10.0

    # 关键词重叠：每匹配一个关键词 +1 分
    score += 1.0 * _keyword_score(log_snippet, keywords)

    # 上下文匹配：每分 ×1.5 权重
    score += 1.5 * _context_match_score(entry, ctx)

    # 有原子操作的条目给小加分（+0.5），因为有可执行的修复方案更有价值
    if entry.atoms:
        score += 0.5

    return score


def retrieve_xpu_candidates(
    entries: Sequence[XpuEntry],
    log_snippet: str,
    ctx: XpuContext,
    *,
    k: int = 3,
    prefer_atoms: bool = True,
) -> List[XpuEntry]:
    """从 XPU 列表中选取 Top-K 最相关的条目

    评分基于正则 + 关键词 + 上下文；可选优先返回包含原子操作的条目。

    Args:
        entries: 所有 XPU 条目列表
        log_snippet: 当前错误日志文本
        ctx: 查询上下文
        k: 返回的最大条目数（默认 3）
        prefer_atoms: 是否优先返回包含原子操作的条目（默认 True）

    Returns:
        Top-K XPU 条目列表
    """
    if not entries:
        return []

    # 对所有条目计算评分
    scored: List[tuple[float, XpuEntry]] = []
    for e in entries:
        s = score_xpu(e, log_snippet=log_snippet, ctx=ctx)
        scored.append((s, e))

    if not scored:
        return []

    # 优先返回有原子操作的条目（因为有可执行命令的 XPU 更实用）
    if prefer_atoms:
        # 分为有原子操作和无原子操作两组
        with_atoms = [(s, e) for s, e in scored if e.atoms]
        without_atoms = [(s, e) for s, e in scored if not e.atoms]

        # 各自按分数排序
        with_atoms.sort(key=lambda x: x[0], reverse=True)
        without_atoms.sort(key=lambda x: x[0], reverse=True)

        # 先取有原子操作的，不够再用无原子操作的补充
        result: List[XpuEntry] = [e for _, e in with_atoms[:k]]
        if len(result) < k:
            result.extend(e for _, e in without_atoms[: k - len(result)])
        return result

    # 不优先原子操作时，直接按分数排序取 Top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


# ============================================================================
# 原子操作渲染：将结构化的 atom 转换为可执行的 bash 命令
# ============================================================================

def render_atom_to_commands(atom: XpuAtom) -> List[str]:
    """将单个原子操作渲染为一条或多条 bash 命令

    支持的原子类型：
    - pip_pin / pip_install：pip 包安装
    - set_pytest_flag：pytest 参数设置
    - set_env：环境变量设置
    - set_umask：umask 设置
    - set_django_setting：Django 配置修改
    - or_upgrade_pkg：包升级（pip/apt）
    - apt_install：apt 包安装
    - conda_install：conda 包安装
    - npm_install：npm 包安装
    - shell：自定义 shell 命令
    - adjust_command：调整后的命令

    Args:
        atom: XpuAtom 对象

    Returns:
        bash 命令列表（可能为空，表示无法渲染）
    """
    name = atom.name  # 原子操作类型
    args = atom.args or {}  # 操作参数

    # pip_pin：锁定特定版本安装
    if name == "pip_pin":
        pkg = args.get("name")
        spec = args.get("spec", "")  # 版本约束，如 "==1.0.0"
        if pkg is None:
            return []
        return [f"pip install '{pkg}{spec}'"]

    # pip_install：常规 pip 安装
    if name == "pip_install":
        pkg = args.get("name") or args.get("package")  # 兼容两种 key
        spec = args.get("spec", "")  # 版本约束
        flags = args.get("flags", [])  # 额外 pip 参数，如 ["--no-deps"]
        if pkg is None:
            return []
        flag_str = " ".join(flags) + " " if flags else ""
        return [f"pip install {flag_str}'{pkg}{spec}'"]

    # set_pytest_flag：设置 pytest 命令行参数
    if name == "set_pytest_flag":
        flag_name = args.get("name")  # 参数名，如 "--import-mode"
        value = args.get("value")  # 参数值，如 "append"
        if not flag_name or value is None:
            return []
        return [f"pytest {flag_name}={value}"]

    # set_env：设置环境变量
    if name == "set_env":
        key = args.get("key") or args.get("var")  # 变量名
        value = args.get("value")  # 变量值
        if not key or value is None:
            return []
        return [f"export {key}={value}"]

    # set_umask：设置 umask
    if name == "set_umask":
        value = args.get("value")
        if value is None:
            return []
        return [f"umask {value}"]

    # set_django_setting：修改 Django settings
    if name == "set_django_setting":
        key = args.get("key")  # settings 属性名
        value = args.get("value")  # 属性值
        if not key:
            return []
        # 生成 Python heredoc 脚本修改 Django settings
        return [
            "python - <<'PY'",
            "from django.conf import settings",
            f"settings.{key} = {repr(value)}",
            "PY",
        ]

    # or_upgrade_pkg：升级包（支持 pip 和 apt 两种包管理器）
    if name == "or_upgrade_pkg":
        pkg_manager = args.get("package_manager", "pip")  # 默认 pip
        if pkg_manager == "apt":
            pkg = args.get("package_name") or args.get("name")
            if not pkg:
                return []
            use_sudo = args.get("use_sudo", False)
            sudo = "sudo " if use_sudo else ""
            return [f"{sudo}apt-get update", f"{sudo}apt-get install -y {pkg}"]
        else:
            # pip 升级：安装 >= min_version 的版本
            pkg = args.get("name") or args.get("package_name")
            min_version = args.get("min_version")
            if not pkg or not min_version:
                return []
            return [f"pip install '{pkg}>={min_version}'"]

    # apt_install：apt 批量安装系统包
    if name == "apt_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]  # 兼容字符串和列表
        if not packages:
            return []
        return ["apt-get update", f"apt-get install -y {' '.join(packages)}"]

    # conda_install：conda 包安装
    if name == "conda_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]
        if not packages:
            return []
        return [f"conda install -y {' '.join(packages)}"]

    # npm_install：npm 包安装
    if name == "npm_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]
        if not packages:
            return ["npm install"]  # 无指定包时执行 npm install
        return [f"npm install {' '.join(packages)}"]

    # shell：自定义 shell 命令（直接执行）
    if name == "shell":
        cmd = args.get("cmd")
        if not cmd:
            return []
        return [cmd]

    # adjust_command：调整后的命令
    if name == "adjust_command":
        cmd = args.get("modified_command") or args.get("cmd")
        if not cmd:
            return []
        return [cmd]

    # 未知原子类型，不生成命令
    return []


def render_entry_commands(entry: XpuEntry) -> List[str]:
    """将条目的所有原子操作渲染为扁平的 bash 命令列表

    Args:
        entry: XPU 经验条目

    Returns:
        所有原子操作渲染后的 bash 命令列表
    """
    commands: List[str] = []
    for atom in entry.atoms:
        commands.extend(render_atom_to_commands(atom))  # 逐个渲染并追加
    return commands


def render_candidates_block(entries: Sequence[XpuEntry]) -> str:
    """渲染候选修复方案文本块，用于注入 LLM prompt

    将 XPU 候选列表格式化为可读的文本块，包含每条 XPU 的建议和可执行命令。

    Args:
        entries: 候选 XPU 条目列表

    Returns:
        格式化的文本块字符串
    """
    if not entries:
        return ""

    lines: List[str] = []
    lines.append("Candidate Fixes from XPU (choose only what you need):")
    for e in entries:
        lines.append(f"- Fix (id={e.id}):")
        # 添加自然语言建议
        if e.advice_nl:
            lines.append("  Advice:")
            for adv in e.advice_nl:
                lines.append(f"    - {adv}")
        # 添加可执行 bash 命令
        cmds = render_entry_commands(e)
        if cmds:
            lines.append("  Bash snippet:")
            for c in cmds:
                lines.append(f"    {c}")
    return "\n".join(lines)
