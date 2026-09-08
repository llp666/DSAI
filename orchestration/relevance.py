"""orchestration/relevance.py：relevance / intent check (stage 3-2, blocks the LLM fabrication path).

Problem: for questions beyond the DSL, the LLM doesn't honestly degrade but fabricates:
1. hallucination: question mentions DSL-unsupported entities (order id o_, status, logistics) and the LLM
   fabricates an unrelated aggregate (order-status question → channel CPS order count). → block and degrade.
2. granularity: question asks for grouping/filtering (per-province GMV, per-category, city X) but the LLM
   ignores the qualifier and outputs a total aggregate. → block, feed back to repair (add dim/filter).

Pure, deterministic, zero LLM. Judges on the question's intent words vs the generated query structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MismatchKind = Literal["hallucination", "granularity", "none"]

# DSL-unsupported entity words (order id / status / logistics → hallucination, unfixable)
OUT_OF_DSL_PATTERNS = [
    re.compile(r"\bo_\d{8}\b"),          # order id like o_00041423
    re.compile(r"订单状态|支付状态|物流|是否完成|已退款|已发货|已支付|已完成"),
    re.compile(r"状态分布|按状态|状态是|状态的"),
    re.compile(r"订单明细|商品明细|每笔|逐笔|清单"),
]

# dimension words absent from the semantic layer (reflect can't fix → degrade directly)
# note: exact two+ char words only, so single-char 市/省/区/县 don't false-hit 城市 (supported dim)
UNSUPPORTED_DIM_WORDS = [
    "省份", "性别", "价格带", "年龄段", "支付方式", "支付渠道", "尺码",
    "品牌", "仓库", "快递", "优惠券", "会员等级", "门店", "店铺", "销售员", "平台",
]

# granularity mismatch: question asks grouping/filtering but the query has no dimension/filter
GRANULARITY_PATTERNS = [
    re.compile(r"各[省份省市县城区品牌年龄段性别价格带仓库尺码品类频道渠道城类型]"),  # 各省/各品牌/各渠道类型
    re.compile(r"按[^，。？?]{1,8}(维度|分组|分类|统计|计算|看)"),
    re.compile(r"按(品类|渠道|渠道类型|城市|城市线级)"),
    re.compile(r"每[天周月人单渠道品类城市]"),
    re.compile(r"[所在属于]的[品类城市线级渠道]"),
    re.compile(r"哪些|分别|对比|top\s*\d+|最高|最低"),
]

# DSL-supported aggregate metrics
SUPPORTED_METRICS = {
    "gmv", "net_sales_amount", "orders_count", "avg_order_value",
    "refund_amount", "refund_rate", "repurchase_rate", "marketing_roi",
    "net_profit", "gross_margin", "top_sku_by_net_sales",
    "stockout_skus_count",
}


@dataclass
class RelevanceVerdict:
    kind: MismatchKind
    reason: str
    matched_pattern: str | None = None


def _first_match(patterns: list, text: str) -> tuple[re.Pattern | None, str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return pat, m.group(0)
    return None, ""


def check_relevance(question: str, metric: str | None,
                    dimensions: list | None = None,
                    filters: list | None = None) -> RelevanceVerdict:
    """Check whether the question and the generated query are consistent.

    Priority: DSL-unsupported entity → hallucination; grouping/filtering intent without matching
    dimensions/filters → granularity.
    """
    pat, matched = _first_match(OUT_OF_DSL_PATTERNS, question)
    if pat is not None:
        if metric in SUPPORTED_METRICS:
            return RelevanceVerdict(
                "hallucination",
                f"问题含 DSL 外实体「{matched}」但生成了聚合指标 {metric}（幻觉编造）",
                pat.pattern)
        return RelevanceVerdict("none", "指标非聚合指标，交由编译错误处理")

    # dimension word absent from the semantic layer → unfixable, degrade (skip futile reflect)
    for w in UNSUPPORTED_DIM_WORDS:
        if w in question:
            return RelevanceVerdict(
                "hallucination",
                f"问题要求不存在的维度「{w}」（语义层不支持，不可修复）")

    # granularity mismatch: grouping/filtering intent but no dimensions/filters in the query
    gpat, gmatched = _first_match(GRANULARITY_PATTERNS, question)
    has_dim = bool(dimensions) or bool(filters)
    if gpat is not None and not has_dim and metric in SUPPORTED_METRICS:
        return RelevanceVerdict(
            "granularity",
            f"问题要求「{gmatched}」粒度/过滤，但生成的查询无任何维度或过滤（粒度错位）",
            gpat.pattern)
    return RelevanceVerdict("none", "相关问题，无意图错位")