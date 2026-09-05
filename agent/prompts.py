"""agent/prompts.py：语义查询生成的系统提示词 + 重试提示。

提示词职责：把自然语言问题映射为受约束的语义查询 JSON（而非 SQL），
并披露指标口径。few-shot 保证同义问题输出稳定。
"""

from __future__ import annotations

from .retriever import Retriever
from .types import SemanticQuery

SCHEMA_EXAMPLE = SemanticQuery(
    metric="gmv", window={"type": "month", "value": "2026-07"}
).model_dump_json()

FEW_SHOT = [
    (
        "2026年7月GMV是多少？",
        {"metric": "gmv", "window": {"type": "month", "value": "2026-07"}},
    ),
    (
        "上个月（扣掉退款后）净销售额是多少？",
        {"metric": "net_sales_amount", "window": {"type": "month", "value": "2026-08"}},
    ),
    (
        "2025年11月用户复购率怎么样？",
        {"metric": "repurchase_rate", "window": {"type": "month", "value": "2025-11"}},
    ),
]


def build_system_prompt(
    metrics: dict,
    tables: list[dict],
    today: str,
) -> str:
    """构建系统提示词：指标定义 + 注入的表结构 + DSL schema + few-shot。"""
    metric_lines = []
    for name, m in metrics.items():
        metric_lines.append(
            f"- {name}（{m['display_name']}）：{m['description']}"
        )
    metrics_text = "\n".join(metric_lines)
    tables_text = Retriever.render(tables)

    few_shot_lines = []
    for q, sq in FEW_SHOT:
        # 用今天的参考日期校正 few-shot 中的相对时间（避免陈旧月份误导）
        few_shot_lines.append(f'问：{q}\n答：{sq}')
    few_shot_text = "\n\n".join(few_shot_lines)

    return f"""你是电商数据分析 Agent 的语义查询生成器。你的唯一任务：把用户的自然语言问题，映射为一个结构化的语义查询 JSON，引用语义层已定义的指标。你绝不编写原始 SQL。

# 可用指标（语义层 v0，口径已固化）
{metrics_text}

# 相关表结构（已由检索层注入 Top-3）
{tables_text}

# 输出格式（必须只输出合法 JSON，不要输出解释或其他文字）
{{"metric": "<指标名>", "window": {{"type": "month", "value": "YYYY-MM"}}}}

# 时间解析规则
- 今天的日期：{today}。
- 用户说「2026年7月」→ value "2026-07"；「上个月」→ 按今天往前推一个月；「Q1」→ 不适用，若用户提到季度请按最近完整季度回答。
- 只支持月粒度，value 必须是 "YYYY-MM" 格式。

# 口径注意
- 净销售额与 GMV 是不同指标：净销售额扣退款，GMV 不扣。用户问「卖了多少钱」「销售额」「净销售额」→ net_sales_amount；问「GMV」「成交额（不扣退款）」→ gmv。
- 问「复购率」「回购率」→ repurchase_rate。

# few-shot 示例
{few_shot_text}
"""


RETRY_SYSTEM_HINT = "（上一次输出的 JSON 解析失败，请严格按照输出格式重新输出一个合法 JSON，不要输出任何其他内容）"
