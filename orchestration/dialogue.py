"""orchestration/dialogue.py：business-query vs casual-chat routing.

A deterministic gate for the *data chain* only:
- business: data questions (metrics/dims/quantifiers) → the semantic-query state machine;
- casual: everything else (greetings / small talk / real-time questions) → plain LLM chat.

Whether a real-time question (天气/新闻/现在几点/汇率…) needs a web search is the LLM's own
call now: casual chat runs a tool round armed with search_tool, so no keyword router is needed
to reach the search API (stage 3-5).
"""

from __future__ import annotations

import re

# obvious casual / greeting hints (checked first, so 你好/谢谢 never hit the other markers)
_CASUAL_RE = re.compile(
    r"(你好|您好|哈喽|嗨|早上好|下午好|晚上好|谢谢|再见|拜拜|你是谁|你能做|介绍一下|帮助|笑话|你好吗|在吗|在不在|晚安)"
)

# concept / definition questions: 「什么是 XX」「XX 是什么意思」「解释一下 XX」「XX 的定义」…
# 这些是名词解释（知识问答），不是结构化取数——即便带着 GMV/ROI 这类业务词也绝不能进
# 数仓链路（曾因此被强推 SQL 生成、模型自补 2026-09 窗口、超分区被拦截、65 秒才报错）。
# 必须在 business markers 之前判定：命中即 casual（LLM 背景/语义层 meta 回答，必要时
# 由 LLM 自主调 search_tool 联网补充）。
_CONCEPT_RE = re.compile(
    r"(什么是|是什么意思|什么意思|解释一下|解释下|解释一下|定义是什么|的定义|"
    r"指的是什么|指什么|怎么理解|如何理解|帮我解释|讲一下|讲讲|科普一下|介绍一下)"
)


def _is_concept_question(q: str) -> bool:
    """概念/常识类提问判定（解释/定义/含义意图，而非取数意图）。"""
    return bool(_CONCEPT_RE.search(q))


# words that signal a data query — strong metrics/dims + colloquial phrases only. Generic
# quantifiers (多少/几个) are deliberately NOT here: 「人民币汇率是多少」 is a real-time question
# for the LLM to route via search_tool, not a data query. Real data questions always carry a
# strong word (销量/库存/占比/卖了…), so nothing regresses.
_BUSINESS_MARKERS = (
    "gmv", "销售额", "净销", "销量", "订单", "成交", "收入", "利润", "净利", "毛利",
    "复购", "回购", "客单", "退款", "退货", "roi", "动销", "周转", "库存", "断货",
    "呆滞", "补货", "转化", "花费", "广告", "投放", "成本", "业绩", "单量", "金额",
    "卖了", "赚了", "买了", "排行", "最好", "最高", "最低", "top",
    "占比", "比例",
    "品类", "渠道", "城市", "线级", "sku", "供应商", "仓库",
)


def route_dialogue(question: str) -> str:
    """Return "business" (data query) / "casual" (small talk / anything else).

    Concept questions (什么是GMV/解释一下ROI) are casual even when they carry a
    business marker word — a definition needs knowledge, not SQL.
    """
    q = question.strip()
    if _CASUAL_RE.search(q):
        return "casual"
    if _is_concept_question(q):
        return "casual"
    if any(m in q.lower() for m in _BUSINESS_MARKERS):
        return "business"
    return "casual"
