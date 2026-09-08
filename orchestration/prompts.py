"""orchestration/prompts.py：semantic-query generation prompt.

Maps a natural-language question to a constrained semantic-query JSON (metric + window +
optional dimensions/filters), citing semantic-layer metrics and value dictionaries; never emits SQL.
"""

from __future__ import annotations

import json

from retrieval.retriever import Retriever
from data.semantic import SemanticLayer


def _metric_block(layer: SemanticLayer) -> str:
    lines = []
    for name, m in layer.metrics.items():
        extra = f"，{m.get('window', '')}窗口" if "window" in m else ""
        lines.append(f"- {name}（{m['display_name']}）：{m['description']}{extra}")
    return "\n".join(lines)


def _dsl_schema(today: str) -> str:
    return (
        '{"metric": "<指标名>", "window": {"type": "month", "value": "YYYY-MM"},'
        ' "dimensions": ["<维度名>"], "filters": [{"dim": "<维度名>", "op": "=", "value": "<值>"}]}\n'
        '# 日窗口变体（问题问某一天/昨天/今天的指标时用）：\n'
        '#   {"type": "day", "value": "YYYY-MM-DD"}，如 "2026-07-15"\n'
        f'# 今天：{today}；「昨天」→ 今天减一天（如 {today} 前一天）'
    )


def _yesterday(today: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(today) - timedelta(days=1)).isoformat()


def render_history(history: list[dict] | None, limit: int = 6) -> str:
    """Render recent Q&A as a conversation-context block (for referent resolution)."""
    if not history:
        return ""
    lines = []
    for h in history[-limit:]:
        q = h.get("question", "") if isinstance(h, dict) else ""
        a = h.get("answer", "") if isinstance(h, dict) else ""
        if q:
            lines.append(f"用户：{q}")
        if a:
            lines.append(f"助手：{a[:400]}")
    return "\n".join(lines)


FEW_SHOT: list[tuple[str, dict]] = [
    (
        "2026年7月GMV是多少？",
        {"metric": "gmv", "window": {"type": "month", "value": "2026-07"},
         "dimensions": [], "filters": []},
    ),
    (
        "2026年7月服饰品类净销售额是多少？",
        {"metric": "net_sales_amount", "window": {"type": "month", "value": "2026-07"},
         "dimensions": ["category_type"],
         "filters": [{"dim": "category_type", "op": "=", "value": "服饰"}]},
    ),
    (
        "上个月一线城市的GMV是多少？",
        {"metric": "gmv", "window": {"type": "month", "value": "2026-08"},
         "dimensions": ["city_tier"],
         "filters": [{"dim": "city_tier", "op": "=", "value": "一线"}]},
    ),
    (
        "2025年11月复购率怎么样？",
        {"metric": "repurchase_rate", "window": {"type": "month", "value": "2025-11"},
         "dimensions": [], "filters": []},
    ),
    (
        "昨天GMV是多少？",
        {"metric": "gmv", "window": {"type": "day", "value": "__YESTERDAY__"},
         "dimensions": [], "filters": []},
    ),
]


def build_system_prompt(layer: SemanticLayer, schema_text: str, today: str) -> str:
    # dynamically fill the "yesterday" few-shot date from the reference day
    yesterday = _yesterday(today)
    few_shot = "\n\n".join(
        f"问：{q}\n答：{json.dumps(sq, ensure_ascii=False).replace('__YESTERDAY__', yesterday)}"
        for q, sq in FEW_SHOT
    )
    return f"""你是电商数据分析 Agent 的语义查询生成器。你的唯一任务：把用户的自然语言问题，映射为一个结构化的语义查询 JSON，引用语义层已定义的指标。你绝不编写原始 SQL。

# 可用指标（语义层 v2 · {len(layer.metrics)} 个，口径已固化）
{_metric_block(layer)}

# 可用维度（含值字典，过滤值必须取自此清单）
{layer.format_dimensions()}

# 相关表结构（已由检索层动态注入，Token 预算裁剪后）
{schema_text}

# 输出格式（只输出合法 JSON，不要输出解释或其他文字）
{_dsl_schema(today)}

# 规则
- 今天的日期：{today}。「上个月」按今天往前推一个月；明确给月份就用该月。
- 问「昨天/今天/某一天」的指标 → 用 day 窗口（window.type="day"，value=YYYY-MM-DD）。「昨天」= {yesterday}。
- 月粒度指标默认 window.type="month"；只有明确问单日才用 day 窗口（复购率等滚动窗口指标不支持 day，只用 month）。
- 无维度下钻时，dimensions 与 filters 输出空数组 []。
- 维度值必须来自值字典；「品类」→ category_type，「城市/线级」→ city_tier，「渠道」→ channel，「渠道类型」→ channel_type。
- 净销售额与 GMV 是不同指标：净销售额扣退款，GMV 不扣。问「销售额/卖了多少钱」默认 net_sales_amount；明确说「GMV/成交额」→ gmv。
- 问「复购率/回购率」→ repurchase_rate；「毛利/毛利率」→ gross_margin；「净利」→ net_profit；「ROI」→ marketing_roi。
- 营销ROI 只支持 channel / channel_type 维度，不要给营销ROI 加品类维度。

# few-shot 示例
{few_shot}
"""