"""orchestration/tool_router.py：diagnostic-tool intent routing (stage 3-4).

Keyword rules first + LLM fallback:
- inventory keywords (断货/呆滞/可售天数/补货/DOI/库存预警/周转/动销/库存体检) → inventory_tool
- marketing keywords (漏斗/转化链路/曝光到支付/ROI拆解/投放效果拆解) → marketing_tool
- no hit → None (normal semantic-query path)

A hit injects a routing result; the LLM fallback is wired in pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

# aggregate-count hints (a single number → semantic query, not the SKU-level diagnostic tool)
_COUNTING_HINT = (
    "数是多少", "数量是多少", "有多少", "多少个", "数有几个", "数量是", "一共", "总数",
)

TOOL_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("inventory_diagnostic",
     ("哪些SKU断货", "哪个SKU断货", "SKU断货", "呆滞", "可售天数", "还够卖",
      "能卖几天", "补货建议", "库存体检", "周转", "动销", "doi", "库存诊断",
      "库存健康", "哪些库存", "补货")),
    ("marketing_funnel",
     ("漏斗", "转化链路", "曝光到支付", "点击到支付", "加购到支付", "roi拆解",
      "投放效果拆解", "转化路径")),
]

# tool descriptions for the LLM fallback prompt
TOOL_DESCRIPTIONS = {
    "inventory_diagnostic": (
        "库存诊断工具：输入 sku_id（可选 category），计算可售天数（库存/近30天日均）、"
        "DOI（可售天数超90=呆滞）、近30天销量环比；输出逐 SKU 洞察 + DOI×环比散点象限图。"
        "用于『某 SKU 还够卖几天』『哪些 SKU 呆滞』『补货建议』类问题。"),
    "marketing_funnel": (
        "营销ROI漏斗工具：拆解曝光→点击→加购→支付各环节转化（按渠道类型）。"
        "用于『投放漏斗』『转化链路』类问题。"),
}


@dataclass
class ToolIntent:
    tool: str | None        # hit tool name (None = normal path)
    rule: str | None        # hit keyword (trace)
    needs_llm: bool = False  # no keyword hit, but worth an LLM fallback judgment

    @property
    def hit(self) -> bool:
        return self.tool is not None


def route_tool_intent(question: str) -> ToolIntent:
    """Keyword routing: hit → tool name; miss → None (LLM fallback optional)."""
    q = question.lower()
    # aggregate-count questions → not the SKU-level tool; let the tool-round LLM judge → SQL
    if any(w in q for w in _COUNTING_HINT):
        sku_level = any(w in q for w in ("还够卖", "能卖几天", "可售天数", "呆滞", "库存预警"))
        if not sku_level:
            return ToolIntent(tool=None, rule=None, needs_llm=True)
    for tool, words in TOOL_INTENT_RULES:
        for w in words:
            if w.lower() in q:
                return ToolIntent(tool=tool, rule=w)
    # domain words without an exact diagnostic intent → worth an LLM fallback
    domain_hint = any(k in q for k in ("库存", "sku", "存货", "仓库", "采购", "供应商")) \
        or any(k in q for k in ("roi", "广告", "投放", "转化"))
    return ToolIntent(tool=None, rule=None, needs_llm=domain_hint)


def llm_fallback_prompt(question: str, intent: ToolIntent) -> tuple[str, str] | None:
    """LLM fallback prompt to judge whether a diagnostic tool applies. None if not worth it."""
    if not intent.needs_llm:
        return None
    desc_lines = "\n".join(f"- {k}：{v}" for k, v in TOOL_DESCRIPTIONS.items())
    system = (
        "你是数据分析 Agent 的工具路由判定器。判断用户问题是否应调用诊断工具（而非普通指标查询）。\n"
        "若问题需要 SKU 级库存体检/可售天数/呆滞判断/补货建议，或需要营销漏斗拆解，返回工具名。\n"
        f"可用工具：\n{desc_lines}\n"
        "只输出一个 JSON：{\"tool\": \"工具名或null\"}，不要解释。"
    )
    user = f"问题：{question}"
    return system, user