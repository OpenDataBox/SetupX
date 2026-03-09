"""XPU 数据结构定义与评分检索逻辑。"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class XpuAtom:
    """XPU 原子操作（如 pip_install, set_env 等）。"""
    name: str
    args: Dict[str, Any]


@dataclass
class XpuEntry:
    """XPU 经验条目。"""
    id: str
    context: Dict[str, Any]
    signals: Dict[str, Any]
    advice_nl: List[str]
    atoms: List[XpuAtom] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=lambda: {"hits": 0, "successes": 0, "failures": 0})

@dataclass
class XpuContext:
    """XPU 查询上下文（语言、操作系统、Python 版本、工具链）。"""
    lang: Optional[str] = None
    os: Optional[str] = None
    python: Optional[str] = None
    tools: Sequence[str] = tuple()


def _parse_xpu_line(obj: Dict[str, Any]) -> XpuEntry:
    """解析单行 JSON 为 XpuEntry。"""
    atoms_raw = obj.get("atoms") or []
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
    """从 JSONL 文件加载所有 XPU 条目。"""
    entries: List[XpuEntry] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(_parse_xpu_line(obj))
    return entries


def _match_regex(log_snippet: str, patterns: Iterable[str]) -> bool:
    """检测日志片段是否匹配任一正则表达式。"""
    for p in patterns:
        try:
            if re.search(p, log_snippet):
                return True
        except re.error:
            continue
    return False


def _keyword_score(log_snippet: str, keywords: Iterable[str]) -> int:
    """简单关键词重叠评分：统计日志片段中出现了多少个关键词。"""
    text = log_snippet.lower()
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw.lower() in text:
            score += 1
    return score


def _context_match_score(entry: XpuEntry, ctx: XpuContext) -> int:
    """计算上下文匹配分数（语言、工具、Python 版本、操作系统）。"""
    score = 0
    ectx = entry.context

    # 语言匹配
    if ctx.lang and ectx.get("lang") == ctx.lang:
        score += 2

    # 工具交集
    tools_entry = set(ectx.get("tools") or [])
    tools_ctx = set(ctx.tools or [])
    if tools_entry and tools_ctx and tools_entry & tools_ctx:
        score += 2

    # Python 版本前缀匹配
    if ctx.python:
        py_list = ectx.get("python") or []
        for py in py_list:
            if str(py).startswith(str(ctx.python)):
                score += 1
                break

    # 操作系统匹配
    if ctx.os:
        os_list = ectx.get("os") or []
        if ctx.os in os_list:
            score += 1

    return score


def score_xpu(entry: XpuEntry, log_snippet: str, ctx: XpuContext) -> float:
    """计算单条 XPU 条目的综合评分（正则 + 关键词 + 上下文）。"""
    signals = entry.signals or {}
    regexes = signals.get("regex") or []
    keywords = signals.get("keywords") or []

    score = 0.0

    # 正则匹配加大分
    if regexes and _match_regex(log_snippet, regexes):
        score += 10.0

    # 关键词重叠
    score += 1.0 * _keyword_score(log_snippet, keywords)

    # 上下文匹配
    score += 1.5 * _context_match_score(entry, ctx)

    # 有原子操作的条目给小加分
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
    """选取 Top-K XPU 条目。

    评分基于正则 + 关键词 + 上下文；可选优先返回包含原子操作的条目。
    """
    if not entries:
        return []

    # 计算基础分数
    scored: List[tuple[float, XpuEntry]] = []
    for e in entries:
        s = score_xpu(e, log_snippet=log_snippet, ctx=ctx)
        scored.append((s, e))

    if not scored:
        return []

    # 优先返回有原子操作的条目
    if prefer_atoms:
        with_atoms = [(s, e) for s, e in scored if e.atoms]
        without_atoms = [(s, e) for s, e in scored if not e.atoms]

        with_atoms.sort(key=lambda x: x[0], reverse=True)
        without_atoms.sort(key=lambda x: x[0], reverse=True)

        result: List[XpuEntry] = [e for _, e in with_atoms[:k]]
        if len(result) < k:
            result.extend(e for _, e in without_atoms[: k - len(result)])
        return result

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


def render_atom_to_commands(atom: XpuAtom) -> List[str]:
    """将单个原子操作渲染为一条或多条 bash 命令。"""
    name = atom.name
    args = atom.args or {}

    if name == "pip_pin":
        pkg = args.get("name")
        spec = args.get("spec", "")
        if pkg is None:
            return []
        return [f"pip install '{pkg}{spec}'"]

    if name == "pip_install":
        pkg = args.get("name") or args.get("package")
        spec = args.get("spec", "")
        flags = args.get("flags", [])
        if pkg is None:
            return []
        flag_str = " ".join(flags) + " " if flags else ""
        return [f"pip install {flag_str}'{pkg}{spec}'"]

    if name == "set_pytest_flag":
        flag_name = args.get("name")
        value = args.get("value")
        if not flag_name or value is None:
            return []
        return [f"pytest {flag_name}={value}"]

    if name == "set_env":
        key = args.get("key") or args.get("var")
        value = args.get("value")
        if not key or value is None:
            return []
        return [f"export {key}={value}"]

    if name == "set_umask":
        value = args.get("value")
        if value is None:
            return []
        return [f"umask {value}"]

    if name == "set_django_setting":
        key = args.get("key")
        value = args.get("value")
        if not key:
            return []
        return [
            "python - <<'PY'",
            "from django.conf import settings",
            f"settings.{key} = {repr(value)}",
            "PY",
        ]

    if name == "or_upgrade_pkg":
        pkg_manager = args.get("package_manager", "pip")
        if pkg_manager == "apt":
            pkg = args.get("package_name") or args.get("name")
            if not pkg:
                return []
            use_sudo = args.get("use_sudo", False)
            sudo = "sudo " if use_sudo else ""
            return [f"{sudo}apt-get update", f"{sudo}apt-get install -y {pkg}"]
        else:
            pkg = args.get("name") or args.get("package_name")
            min_version = args.get("min_version")
            if not pkg or not min_version:
                return []
            return [f"pip install '{pkg}>={min_version}'"]

    if name == "apt_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]
        if not packages:
            return []
        return ["apt-get update", f"apt-get install -y {' '.join(packages)}"]

    if name == "conda_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]
        if not packages:
            return []
        return [f"conda install -y {' '.join(packages)}"]

    if name == "npm_install":
        packages = args.get("packages") or []
        if isinstance(packages, str):
            packages = [packages]
        if not packages:
            return ["npm install"]
        return [f"npm install {' '.join(packages)}"]

    if name == "shell":
        cmd = args.get("cmd")
        if not cmd:
            return []
        return [cmd]

    if name == "adjust_command":
        cmd = args.get("modified_command") or args.get("cmd")
        if not cmd:
            return []
        return [cmd]

    # 未知原子类型，不生成命令
    return []


def render_entry_commands(entry: XpuEntry) -> List[str]:
    """将条目的所有原子操作渲染为扁平的 bash 命令列表。"""
    commands: List[str] = []
    for atom in entry.atoms:
        commands.extend(render_atom_to_commands(atom))
    return commands


def render_candidates_block(entries: Sequence[XpuEntry]) -> str:
    """渲染候选修复方案文本块，用于注入 LLM prompt。"""
    if not entries:
        return ""

    lines: List[str] = []
    lines.append("Candidate Fixes from XPU (choose only what you need):")
    for e in entries:
        lines.append(f"- Fix (id={e.id}):")
        if e.advice_nl:
            lines.append("  Advice:")
            for adv in e.advice_nl:
                lines.append(f"    - {adv}")
        cmds = render_entry_commands(e)
        if cmds:
            lines.append("  Bash snippet:")
            for c in cmds:
                lines.append(f"    {c}")
    return "\n".join(lines)
