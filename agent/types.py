"""agent/types.py：语义查询 DSL（阶段二）与 AgentState。"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field


class Window(BaseModel):
    type: Literal["month"] = "month"
    value: str = Field(description="月份，格式 YYYY-MM，如 2026-07")


class DimensionFilter(BaseModel):
    dim: str = Field(description="维度名（语义层 dimensions 键）")
    op: Literal["=", "in"] = "="
    value: Union[str, list[str]] = Field(description="过滤值；'=' 用标量，'in' 用数组")


class SemanticQuery(BaseModel):
    """受约束语义查询（阶段二：指标 + 月窗口 + 可选维度/过滤）。"""

    model_config = ConfigDict(validate_assignment=True)

    metric: str = Field(description="语义层指标名")
    window: Window
    dimensions: list[str] = Field(default_factory=list,
                                  description="下钻维度名列表")
    filters: list[DimensionFilter] = Field(
        default_factory=list, description="维度过滤条件（与 dimensions 配套）")


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
