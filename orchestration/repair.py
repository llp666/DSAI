"""orchestration/repair.py：deterministic repair rule table v0.

Principle: errors fixable by a deterministic rule (dialect mapping, date boundary, SQL extraction)
skip LLM retries — "deterministic first". Each rule: match the error → output fixed SQL (or None).

RULE_TABLE: list[(name, match_fn(error_text, sql) -> bool, fix_fn(sql) -> str)]
"""

from __future__ import annotations

import re


def _rule_sql_extraction(error: str, sql: str) -> bool:
    """Error contains syntax/parser error and SQL mixes in markdown fences / leading text."""
    return "Parser Error" in error or "syntax error" in error


def _fix_sql_extraction(sql: str) -> str:
    """Extract SQL: strip markdown fences, keep from the first SELECT/WITH line."""
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
    """Error contains "Could not convert string '已完成'" → dirty enum leaked into status compare."""
    return "Could not convert string" in error and "已完成" in error


def _fix_enum_drift(sql: str) -> str:
    """Map the dirty enum '已完成' back to the official '4' (matches clean ground-truth caliber)."""
    return re.sub(r"['\"](已完成)['\"]", "'4'", sql)


def _rule_date_boundary(error: str, sql: str) -> bool:
    """Error looks like conversion/binder with a BETWEEN date boundary."""
    return ("conversion error" in error.lower()
            or "could not convert" in error.lower()) and "BETWEEN" in sql


def _fix_date_boundary(sql: str) -> str:
    """Rewrite BETWEEN 'A' AND 'B' → ">= 'A' AND < 'B'" (half-open, avoids month-end dup/drop)."""
    pattern = re.compile(
        r"BETWEEN\s+([^\s]+)\s+AND\s+([^\s]+)", re.I
    )
    m = pattern.search(sql)
    if not m:
        return sql
    start, end = m.group(1), m.group(2)
    return pattern.sub(lambda mo: f">= {start} AND < {end}", sql, count=1)


RULE_TABLE: list[tuple[str, object, object]] = [
    ("sql_extraction", _rule_sql_extraction, _fix_sql_extraction),
    ("enum_drift", _rule_enum_drift, _fix_enum_drift),
    ("date_boundary", _rule_date_boundary, _fix_date_boundary),
]


def try_repair(error: str, sql: str) -> str | None:
    """Try rules in order; return fixed SQL, or None if no rule applies."""
    for name, match_fn, fix_fn in RULE_TABLE:
        if match_fn(error, sql):
            repaired = fix_fn(sql)
            if repaired and repaired != sql:
                return repaired
    return None