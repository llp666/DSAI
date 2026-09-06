"""agent/types.py：语义查询 DSL（阶段二）与 AgentState（阶段三状态机）。"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

try:
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages
except ImportError:  # langgraph 未装（CI 等）时降级：messages 字段仅作普通列表
    AnyMessage = Any
    def add_messages(left: Any, right: Any) -> Any:  # type: ignore[misc]
        return list(left) + list(right)


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
    intent: str                    # 意图/域分类标签（营销/订单/供应链/…，judge/repair 可据此取规则）
    stage: str                     # judge 判定结果（answer/repair/degrade，trace 用）
    schema_text: str               # Token 预算裁剪后的相关表结构文本
    retrieved_tables: list[str]
    semantic_query: Optional[SemanticQuery]
    compiled_sql: Optional[str]
    execution_result: Any
    execution_error: Optional[str]
    errors: list[str]              # 累积错误文本（真实错误，回灌给 generate 重试）
    error_feedback: str            # 回灌上下文（repair/reflect 产出）
    retry_count: int
    max_retries: int
    messages: Annotated[list[AnyMessage], add_messages]  # 工具调用消息链（ToolNode 回灌）
    answer: str
    trace_id: str
