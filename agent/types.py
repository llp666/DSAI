"""agent/types.py：语义查询 DSL 与 AgentState（按方案文档 6.2 裁剪）。"""

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

MetricName = Literal["gmv", "net_sales_amount", "repurchase_rate"]


class Window(BaseModel):
    type: Literal["month"] = "month"
    value: str = Field(description="月份，格式 YYYY-MM，如 2026-07")


class SemanticQuery(BaseModel):
    """LLM 输出的受约束语义查询（阶段一只含指标 + 月窗口）。"""

    metric: MetricName
    window: Window


class AgentState(TypedDict):
    question: str
    retrieved_tables: list[str]
    semantic_query: Optional[SemanticQuery]
    compiled_sql: Optional[str]
    execution_result: Any
    execution_error: Optional[str]
    retry_count: int
    answer: str
    trace_id: str
