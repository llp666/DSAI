"""agent/repair.py：确定性修复规则表 v0。

原则：能用确定性规则修复的错误（方言映射、日期边界、SQL 提取）不走 LLM 重试，
贯彻方案「确定性优先」。每条规则：匹配错误 → 输出修复后的 SQL（或 None 表示不适用）。
v0 落 3 条规则，阶段三纠错轨直接复用本表。

RULE_TABLE: list[(name, match_fn(error_text, sql)->bool, fix_fn(sql)->str)]
"""

from __future__ import annotations

import re

# ---------- 规则匹配与修复 ----------


def _rule_sql_extraction(error: str, sql: str) -> bool:
    """错误含「syntax error / Parser Error」且 SQL 混入 markdown 围栏/前导文本。"""
    return "Parser Error" in error or "syntax error" in error


def _fix_sql_extraction(sql: str) -> str:
    """提取 SQL：去 markdown 围栏、保留从 SELECT/WITH 起的语句。"""
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S)
    if m:
        sql = m.group(1)
    lines = [ln for ln in sql.splitlines() if ln.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith(("SELECT", "WITH")):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def _rule_enum_drift(error: str, sql: str) -> bool:
    """错误含「Could not convert string '已完成'」→ 脏枚举值混入状态比较。"""
    return "Could not convert string" in error and "已完成" in error


def _fix_enum_drift(sql: str) -> str:
    """把对枚举脏值'已完成'的比较映射回官方枚举'4'（清洗后口径与 ground_truth 一致）。"""
    return re.sub(r"['\"](已完成)['\"]", "'4'", sql)


def _rule_date_boundary(error: str, sql: str) -> bool:
    """错误含「Conversion/Binder」且疑似 BETWEEN 日期边界问题。"""
    return ("conversion error" in error.lower()
            or "could not convert" in error.lower()) and "BETWEEN" in sql


def _fix_date_boundary(sql: str) -> str:
    """重写 BETWEEN 'A' AND 'B' 为「>= 'A' AND < 'B'」（闭开区间，避免月末边界漏/重）。"""
    pattern = re.compile(
        r"BETWEEN\s+([^\s]+)\s+AND\s+([^\s]+)", re.I
    )
    m = pattern.search(sql)
    if not m:
        return sql
    start, end = m.group(1), m.group(2)
    return pattern.sub(lambda mo: f">= {start} AND < {end}", sql, count=1)


# ---------- 规则表 ----------

RULE_TABLE: list[tuple[str, object, object]] = [
    ("sql_extraction", _rule_sql_extraction, _fix_sql_extraction),
    ("enum_drift", _rule_enum_drift, _fix_enum_drift),
    ("date_boundary", _rule_date_boundary, _fix_date_boundary),
]


def try_repair(error: str, sql: str) -> str | None:
    """按规则表依次尝试确定性修复；返回修复后的 SQL，或 None（无规则适用）。"""
    for name, match_fn, fix_fn in RULE_TABLE:
        if match_fn(error, sql):
            repaired = fix_fn(sql)
            if repaired and repaired != sql:
                return repaired
    return None
