"""orchestration/dialogue.py：business-query vs casual-chat routing.

A deterministic gate: business data questions → the semantic-query state machine;
greetings / small talk → a plain LLM conversation (no SQL / deterministic pipeline).
"""

from __future__ import annotations

import re

# obvious casual / greeting hints (checked first, so 你好/天气 never hit the business markers)
_CASUAL_RE = re.compile(
    r"(你好|您好|哈喽|嗨|早上好|下午好|晚上好|谢谢|再见|拜拜|你是谁|你能做|介绍一下|帮助|天气|笑话|你好吗|在吗|在不在|晚安)"
)

# words that signal a data query (metrics, dims, quantifiers, colloquial metric phrases)
_BUSINESS_MARKERS = (
    "gmv", "销售额", "净销", "销量", "订单", "成交", "收入", "利润", "净利", "毛利",
    "复购", "回购", "客单", "退款", "退货", "roi", "动销", "周转", "库存", "断货",
    "呆滞", "补货", "转化", "花费", "广告", "投放", "成本", "业绩", "单量", "金额",
    "卖了", "赚了", "买了", "排行", "最好", "最高", "最低", "top",
    "多少", "几个", "几单", "占比", "比例",
    "品类", "渠道", "城市", "线级", "sku", "供应商", "仓库",
)


def route_dialogue(question: str) -> str:
    """Return "business" for a data query, "casual" for small talk / general chat."""
    q = question.strip()
    if _CASUAL_RE.search(q):
        return "casual"
    if any(m in q.lower() for m in _BUSINESS_MARKERS):
        return "business"
    return "casual"