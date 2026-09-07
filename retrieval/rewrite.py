"""retrieval/rewrite.py：问题改写（相对时间 → 具体日期，规则优先，LLM 兜底接口预留）。

规则用 dateutil.relativedelta + 正则把「上个月/今年Q1/最近30天」等绝对化，
使检索与语义查询都拿到可对齐的时间表达。改不动时返回原文（LLM 兜底在 pipeline 层接入）。
"""

from __future__ import annotations

import re
from datetime import date

from dateutil.relativedelta import relativedelta

_LAST_N = re.compile(r"最近\s*(\d+)\s*(天|周|个月|月)")
_QUARTER = re.compile(r"(20\d{2})年?\s*[Qq]([1-4])|([1-4])季度\s*(20\d{2})年?")


def _month_str(d: date) -> str:
    return f"{d.year}年{d.month}月"


def _day_str(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


def rewrite(question: str, ref_date: date) -> str:
    """把问题中的相对时间绝对化；无相对时间则原样返回。"""
    out = question
    # 昨天 / 今天 / 前天 → 具体日期（day 窗口）
    m = re.search(r"昨天|今天|前天", out)
    if m:
        token = m.group(0)
        if token == "昨天":
            target = ref_date - relativedelta(days=1)
        elif token == "前天":
            target = ref_date - relativedelta(days=2)
        else:
            target = ref_date
        out = out.replace(token, _day_str(target))
        return out
    # 上上个月 / 上个月 / 这个月 / 本月
    m = re.search(r"上上个月|上个月|这个月|本月", out)
    if m:
        token = m.group(0)
        if token == "上上个月":
            target = ref_date + relativedelta(months=-2)
        elif token == "上个月":
            target = ref_date + relativedelta(months=-1)
        else:
            target = ref_date
        out = out.replace(token, _month_str(target))
        return out
    # 最近 N 天/周/月
    m = _LAST_N.search(out)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"天": n, "周": 7 * n, "个月": 30 * n, "月": 30 * n}[unit]
        start = ref_date - relativedelta(days=days)
        out = out.replace(m.group(0),
                          f"{_month_str(start)}至{_month_str(ref_date)}")
        return out
    # 今年 / 去年
    m = re.search(r"(今年|去年)", out)
    if m:
        year = ref_date.year - (1 if m.group(1) == "去年" else 0)
        out = out.replace(m.group(1), f"{year}年")
        return out
    # 季度：2026Q1 / 1季度2026年
    m = _QUARTER.search(out)
    if m:
        if m.group(1):
            year, q = int(m.group(1)), int(m.group(2))
        else:
            year, q = int(m.group(4)), int(m.group(3))
        months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[q]
        out = out.replace(m.group(0),
                          f"{year}年{months[0]}月至{year}年{months[1]}月")
        return out
    return out
