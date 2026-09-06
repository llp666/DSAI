"""agent/relevance.py：相关性/意图校验层（阶段 3-2，堵 LLM 编造出口）。

问题：LLM 对超出 DSL 能力的问题不诚实降级，而可能编造：
1. **hallucination**（幻觉编造）：问题含 DSL 不支持的实体（订单号 o_、状态、物流明细），
   LLM 编造无关聚合查询（如订单状态题 → 渠道 CPS 订单数）。→ 拦截降级，不给错答案。
2. **granularity**（粒度错位）：问题要求分组/过滤（各省份 GMV、按品类、XX 城市），
   LLM 忽略限定词输出总聚合（各省份 GMV → 总 GMV）。→ 拦截后回灌修复（补维度/过滤）。

本层纯函数、确定性、零 LLM。判断依据是「问题意图词」与「生成的语义查询结构」的对照。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MismatchKind = Literal["hallucination", "granularity", "none"]

# DSL 不支持的实体词（订单号/状态/物流等 → 幻觉编造，不可修）
OUT_OF_DSL_PATTERNS = [
    re.compile(r"\bo_\d{8}\b"),          # 订单号 o_00041423
    re.compile(r"订单状态|支付状态|物流|是否完成|已退款|已发货|已支付|已完成"),
    re.compile(r"状态分布|按状态|状态是|状态的"),
    re.compile(r"订单明细|商品明细|每笔|逐笔|清单"),
]

# 语义层不存在的维度词（reflect 也修不动 → 直接降级，省无效 LLM 调用）
# 注意：只用精确双字词，避免单字「市/省/区/县」误伤「城市」（city_tier 是支持维度）
UNSUPPORTED_DIM_WORDS = [
    "省份", "性别", "价格带", "年龄段", "支付方式", "支付渠道", "尺码",
    "品牌", "仓库", "快递", "优惠券", "会员等级", "门店", "店铺", "销售员", "平台",
]

# 粒度错位：问题要求分组/过滤，但生成的查询无维度/过滤
GRANULARITY_PATTERNS = [
    re.compile(r"各[省份省市县城区品牌年龄段性别价格带仓库尺码品类频道渠道城类型]"),  # 各省/各品牌/各渠道类型
    re.compile(r"按[^，。？?]{1,8}(维度|分组|分类|统计|计算|看)"),
    re.compile(r"按(品类|渠道|渠道类型|城市|城市线级)"),
    re.compile(r"每[天周月人单渠道品类城市]"),
    re.compile(r"[所在属于]的[品类城市线级渠道]"),
    re.compile(r"哪些|分别|对比|top\s*\d+|最高|最低"),
]

# DSL 支持的聚合指标
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
    """校验问题与语义查询是否相关。

    question：改写后的问题；metric：指标名；dimensions/filters：查询的维度/过滤。
    优先级：DSL 外实体 → hallucination；分组/过滤意图但查询无对应 → granularity。
    """
    pat, matched = _first_match(OUT_OF_DSL_PATTERNS, question)
    if pat is not None:
        if metric in SUPPORTED_METRICS:
            return RelevanceVerdict(
                "hallucination",
                f"问题含 DSL 外实体「{matched}」但生成了聚合指标 {metric}（幻觉编造）",
                pat.pattern)
        return RelevanceVerdict("none", "指标非聚合指标，交由编译错误处理")

    # 语义层不存在的维度词 → 不可修，直接降级（省 reflect 无效调用）
    for w in UNSUPPORTED_DIM_WORDS:
        if w in question:
            return RelevanceVerdict(
                "hallucination",
                f"问题要求不存在的维度「{w}」（语义层不支持，不可修复）")

    # 粒度错位：有分组/过滤意图但查询无任何维度/过滤
    gpat, gmatched = _first_match(GRANULARITY_PATTERNS, question)
    has_dim = bool(dimensions) or bool(filters)
    if gpat is not None and not has_dim and metric in SUPPORTED_METRICS:
        return RelevanceVerdict(
            "granularity",
            f"问题要求「{gmatched}」粒度/过滤，但生成的查询无任何维度或过滤（粒度错位）",
            gpat.pattern)
    return RelevanceVerdict("none", "相关问题，无意图错位")
