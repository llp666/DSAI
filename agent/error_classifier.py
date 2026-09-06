"""agent/error_classifier.py：错误三分类（阶段 3-2 轨道 A 核心）。

把工具链产生的真实错误文本分类，决定修复策略：
- dialect（方言错）：Parser Error / syntax error / BETWEEN 边界 → 确定性映射表（try_repair），零 LLM
- reference（引用错）：未知维度/未知指标/未知表/BinderError → 回灌命中表 Schema + 可选维度列表，LLM 重写语义查询
- logic（逻辑错）：Conversion / Could not convert（脏枚举/类型不匹配）→ 回灌指标表达式 + 关系定义，LLM 重写

分类器纯函数、无外部依赖，可单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal["dialect", "reference", "logic", "unknown"]

# 引用错特征词（编译器/执行器抛出的实体缺失类错误）
REFERENCE_PATTERNS = [
    re.compile(r"未知维度[：:]\s*([\w]+)"),
    re.compile(r"未知指标[：:]\s*([\w]+)"),
    re.compile(r"未知表"),
    re.compile(r"Binder Error", re.I),
    re.compile(r"Table.*does not exist", re.I),
    re.compile(r"Column.*not found", re.I),
    re.compile(r"referenced column", re.I),
]

# 逻辑错特征词（数据值/类型语义类错误）
LOGIC_PATTERNS = [
    re.compile(r"Could not convert string", re.I),
    re.compile(r"Conversion Error", re.I),
    re.compile(r"out of range", re.I),
    re.compile(r"Invalid Input", re.I),
    re.compile(r"invalid.*literal", re.I),
    re.compile(r"cast from", re.I),
]

# 方言错特征词（语法/方言差异类）
DIALECT_PATTERNS = [
    re.compile(r"Parser Error", re.I),
    re.compile(r"syntax error", re.I),
    re.compile(r"missing.*clause", re.I),
    re.compile(r"unexpected token", re.I),
]


@dataclass
class Classification:
    category: ErrorCategory
    entity: str | None = None  # 引用错：被引用的未知实体名（维度/指标/表）
    reason: str = ""            # 判定依据（命中的模式描述）


def classify_error(error: str, sql: str = "") -> Classification:
    """把工具错误文本分类为 dialect / reference / logic / unknown。

    error：真实错误文本（ToolMessage「Error: …」的 content）；
    sql：出错时的 SQL（方言错判定依赖 BETWEEN 等特征）。
    """
    text = error or ""
    # 1) 方言错：语法类错误优先（BETWEEN 边界是编译器产出，先于枚举转换判定）
    if any(p.search(text) for p in DIALECT_PATTERNS) or "BETWEEN" in text and (
            "conversion" in text.lower() or "syntax" in text.lower()):
        return Classification("dialect", reason="语法/方言错")
    # 2) 引用错：实体缺失类
    for pat in REFERENCE_PATTERNS:
        m = pat.search(text)
        if m:
            return Classification("reference", entity=m.group(1) or "",
                                  reason=f"引用错（{pat.pattern[:30]}）")
    # 3) 逻辑错：数据值/类型语义类
    for pat in LOGIC_PATTERNS:
        if pat.search(text):
            return Classification("logic", reason=f"逻辑错（{pat.pattern[:30]}）")
    return Classification("unknown", reason="未归类")


def format_available_dimensions(layer) -> str:
    """构造可用维度清单（引用错回灌时告诉 LLM 哪些维度可选）。"""
    lines = []
    for name, d in layer.dimensions.items():
        vals = " / ".join(str(v) for v in d["values"])
        lines.append(f"- {name}（{d['display_name']}，表 {d['table']}.{d['column']}）：可选值 {vals}")
    return "\n".join(lines) if lines else "（无可用维度）"


def format_metric_expressions(layer, metric: str) -> str:
    """构造指标表达式与关系定义（逻辑错回灌时让 LLM 理解口径）。"""
    if metric not in layer.metrics:
        return f"（指标 {metric} 不存在）"
    m = layer.metrics[metric]
    lines = [f"指标 {metric}（{m['display_name']}）"]
    parts = m.get("parts", {})
    for name, part in parts.items():
        lines.append(f"  part {name}: base={part.get('base')} "
                     f"expr={part.get('expression')} "
                     f"filters={part.get('filters', [])} "
                     f"time={part.get('time_column')}")
    if "combine" in m:
        lines.append(f"  combine: {m['combine']}")
    rel_lines = []
    for rname, rel in layer.relationships.items():
        rel_lines.append(f"  {rname}: {rel['left']} ↔ {rel['right']} "
                         f"ON {' AND '.join(f'{a}={b}' for a, b in rel['on'])}")
    if rel_lines:
        lines.append("关系：")
        lines.extend(rel_lines)
    return "\n".join(lines)


def build_repair_prompt(question: str, metric: str, category: ErrorCategory,
                        error: str, layer, feedback: str) -> tuple[str, str]:
    """构造分类专用修复 prompt。返回 (system, user)。

    引用错：注入可用维度清单，让 LLM 用合法维度重写语义查询；
    逻辑错：注入指标表达式与关系，让 LLM 修正过滤值/口径后重写。
    """
    if category == "reference":
        dims = format_available_dimensions(layer)
        system = (
            "你是电商数据分析 Agent 的纠错器。上一轮语义查询编译失败（引用了不存在的维度/指标/表）。"
            "请根据下方「可用维度」清单，重新输出一个**合法**的语义查询 JSON。\n\n"
            f"可用维度：\n{dims}\n"
            "只输出合法 JSON（metric/window/dimensions/filters），不要解释。"
        )
        user = f"问题：{question}\n失败错误：{error}\n{feedback}"
    elif category == "logic":
        exprs = format_metric_expressions(layer, metric)
        system = (
            "你是电商数据分析 Agent 的纠错器。上一轮语义查询执行时数据转换/逻辑错误（如脏枚举、类型不匹配）。"
            "请根据下方指标表达式理解正确口径，重新输出一个**合法**的语义查询 JSON。\n\n"
            f"指标表达式与关系：\n{exprs}\n"
            "只输出合法 JSON（metric/window/dimensions/filters），不要解释。"
        )
        user = f"问题：{question}\n失败错误：{error}\n{feedback}"
    elif category == "dialect":
        system = (
            "你是电商数据分析 Agent 的纠错器。上一轮 SQL 语法/方言错误（如日期边界、方言写法）。"
            "请修正语义查询（例如调整时间窗口表达、过滤值写法），重新输出一个**合法**的语义查询 JSON"
            "（metric/window/dimensions/filters），不要解释。"
        )
        user = f"问题：{question}\n错误：{error}\n{feedback}"
    else:
        system = (
            "你是电商数据分析 Agent 的纠错器。上一轮语义查询→SQL 链路失败，原因未明。"
            "请分析后重新输出一个**合法**的语义查询 JSON（metric/window/dimensions/filters），不要解释。"
        )
        user = f"问题：{question}\n错误：{error}\n{feedback}"
    return system, user
