"""orchestration/dialogue.py：business-query vs casual-chat vs real-time-search routing.

A deterministic gate:
- business: data questions (metrics/dims/quantifiers) → the semantic-query state machine;
- search: real-time / factual questions (天气/新闻/现在几点/汇率…) → web-search route;
- casual: greetings / small talk → a plain LLM conversation (no SQL / deterministic pipeline).
"""

from __future__ import annotations

import re

# obvious casual / greeting hints (checked first, so 你好/谢谢 never hit the other markers)
_CASUAL_RE = re.compile(
    r"(你好|您好|哈喽|嗨|早上好|下午好|晚上好|谢谢|再见|拜拜|你是谁|你能做|介绍一下|帮助|笑话|你好吗|在吗|在不在|晚安)"
)

# real-time / factual hints → web-search route (keenable): weather, news, current time,
# stock/currency prices, hot topics. Guarded against strong business domain words below,
# so 「最新GMV数据」 still routes to the data chain, not to web search.
_SEARCH_RE = re.compile(
    r"(天气|天气预报|新闻|资讯|实时|最新消息|最新资讯|现在几点|几点钟|几点啦|星期几|"
    r"温度|气温|台风|降雨|下雪|股市|股票|股价|汇率|金价|油价|热搜|热点|时政|世界杯|奥运会)"
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

# strong business domain words only (metrics/dims, no generic quantifiers): a real-time
# search hit is overridden by one of these (e.g. 「今天天气对GMV的影响」 is still data),
# but not by generic quantifiers like 多少 (「人民币汇率是多少」 must reach web search).
_BUSINESS_STRONG = (
    "gmv", "销售额", "净销", "销量", "订单", "成交", "收入", "利润", "净利", "毛利",
    "复购", "回购", "客单", "退款", "退货", "roi", "动销", "周转", "库存", "断货",
    "呆滞", "补货", "转化", "花费", "广告", "投放", "成本", "业绩", "单量", "金额",
    "卖了", "赚了", "买了", "排行", "最好", "最高", "最低", "top",
    "品类", "渠道", "城市", "线级", "sku", "供应商", "仓库",
)


def route_dialogue(question: str) -> str:
    """Return "business" (data query) / "search" (real-time/factual) / "casual" (small talk)."""
    q = question.strip()
    if _CASUAL_RE.search(q):
        return "casual"
    if _SEARCH_RE.search(q) and not any(m in q.lower() for m in _BUSINESS_STRONG):
        return "search"
    if any(m in q.lower() for m in _BUSINESS_MARKERS):
        return "business"
    return "casual"
