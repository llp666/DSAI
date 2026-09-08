"""orchestration/error_classifier.py：error three-way classification (stage-3-2 track A).

Classifies real tool errors to pick a repair strategy:
- dialect  → Parser/syntax/BETWEEN boundary errors → deterministic rule table (try_repair), zero LLM
- reference → unknown dimension/metric/table/BinderError → feed hit-table schema + dim list, LLM rewrites query
- logic    → Conversion errors (dirty enum / type mismatch) → feed metric expression + relationships, LLM rewrites

Pure functions, no external deps, unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal["dialect", "reference", "logic", "unknown"]

# reference-error signatures (missing-entity errors thrown by compiler/executor)
REFERENCE_PATTERNS = [
    re.compile(r"未知维度[：:]\s*([\w]+)"),
    re.compile(r"未知指标[：:]\s*([\w]+)"),
    re.compile(r"未知表"),
    re.compile(r"Binder Error", re.I),
    re.compile(r"Table.*does not exist", re.I),
    re.compile(r"Column.*not found", re.I),
    re.compile(r"referenced column", re.I),
]

# logic-error signatures (value/type semantic errors)
LOGIC_PATTERNS = [
    re.compile(r"Could not convert string", re.I),
    re.compile(r"Conversion Error", re.I),
    re.compile(r"out of range", re.I),
    re.compile(r"Invalid Input", re.I),
    re.compile(r"invalid.*literal", re.I),
    re.compile(r"cast from", re.I),
]

# dialect-error signatures (syntax / dialect differences)
DIALECT_PATTERNS = [
    re.compile(r"Parser Error", re.I),
    re.compile(r"syntax error", re.I),
    re.compile(r"missing.*clause", re.I),
    re.compile(r"unexpected token", re.I),
]


@dataclass
class Classification:
    category: ErrorCategory
    entity: str | None = None  # reference error: the unknown entity name (dim/metric/table)
    reason: str = ""            # matched-pattern description


def classify_error(error: str, sql: str = "") -> Classification:
    """Classify a tool error as dialect / reference / logic / unknown."""
    text = error or ""
    # 1) dialect: syntax errors first (BETWEEN boundary is compiler output, before enum conversion)
    if any(p.search(text) for p in DIALECT_PATTERNS) or "BETWEEN" in text and (
            "conversion" in text.lower() or "syntax" in text.lower()):
        return Classification("dialect", reason="语法/方言错")
    # 2) reference: missing entity
    for pat in REFERENCE_PATTERNS:
        m = pat.search(text)
        if m:
            return Classification("reference", entity=m.group(1) or "",
                                  reason=f"引用错（{pat.pattern[:30]}）")
    # 3) logic: data value / type semantic error
    for pat in LOGIC_PATTERNS:
        if pat.search(text):
            return Classification("logic", reason=f"逻辑错（{pat.pattern[:30]}）")
    return Classification("unknown", reason="未归类")


def format_available_dimensions(layer) -> str:
    """Render the available-dimension list for reference-error repair prompts."""
    return layer.format_dimensions(with_location=True) or "（无可用维度）"


def format_metric_expressions(layer, metric: str) -> str:
    """Render a metric's expression + relationships for logic-error repair prompts."""
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
    """Build a category-specific repair prompt. Returns (system, user)."""
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