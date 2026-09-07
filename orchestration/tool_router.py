"""orchestration/tool_router.py：诊断工具意图路由（阶段 3-4）。

关键词规则优先 + LLM 兜底：
- 库存诊断意图词（断货/呆滞/可售天数/补货/DOI/库存预警/周转/动销/库存体检）
  → inventory_diagnostic_tool
- 营销漏斗意图词（漏斗/转化链路/曝光到支付/ROI拆解/投放效果拆解）
  → marketing_roi_funnel_tool
- 未命中 → None（走常规语义查询链路）

命中诊断意图时，路由结果注入 state（tool_intent），generate_node 提示 LLM
优先调用诊断工具；未命中则完全走原语义查询链路。LLM 兜底在 pipeline 层接入。
"""

from __future__ import annotations

from dataclasses import dataclass

# 意图词 → 工具（顺序匹配，先命中先得）
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

# LLM 兜底判定用的工具描述（供 prompt）
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
    tool: str | None        # 命中的工具名（None=走常规链路）
    rule: str | None        # 命中的意图词（trace 用）
    needs_llm: bool = False  # 关键词未命中，是否值得 LLM 兜底判定

    @property
    def hit(self) -> bool:
        return self.tool is not None


def route_tool_intent(question: str) -> ToolIntent:
    """关键词规则路由：命中返回工具名；未命中返回 None（可 LLM 兜底）。"""
    q = question.lower()
    if any(w in q for w in _COUNTING_HINT):
        sku_level = any(w in q for w in ("还够卖", "能卖几天", "可售天数", "呆滞", "库存预警"))
        if not sku_level:
            return ToolIntent(tool=None, rule=None, needs_llm=True)
    for tool, words in TOOL_INTENT_RULES:
        for w in words:
            if w.lower() in q:
                return ToolIntent(tool=tool, rule=w)
    # 含库存/营销领域词但未精确命中诊断意图 → 值得 LLM 兜底
    domain_hint = any(k in q for k in ("库存", "sku", "存货", "仓库", "采购", "供应商")) \
        or any(k in q for k in ("roi", "广告", "投放", "转化"))
    return ToolIntent(tool=None, rule=None, needs_llm=domain_hint)


def llm_fallback_prompt(question: str, intent: ToolIntent) -> tuple[str, str] | None:
    """LLM 兜底 prompt：判定问题是否该用诊断工具。未命中领域词返回 None（不调 LLM）。"""
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
