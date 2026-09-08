"""compile/types.py：semantic-query DSL + AgentState (LangGraph state machine)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

try:
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages
except ImportError:  # langgraph not installed (CI etc.): fall back to plain list
    AnyMessage = Any
    def add_messages(left: Any, right: Any) -> Any:  # type: ignore[misc]
        return list(left) + list(right)


class Window(BaseModel):
    type: Literal["month", "day"] = "month"
    value: str = Field(description="window value: month = YYYY-MM (e.g. 2026-07), day = YYYY-MM-DD")


class DimensionFilter(BaseModel):
    dim: str = Field(description="dimension name (semantic-layer dimensions key)")
    op: Literal["=", "in"] = "="
    value: Union[str, list[str]] = Field(description="filter value; '=' scalar, 'in' array")


class SemanticQuery(BaseModel):
    """Constrained semantic query (metric + window + optional dimensions/filters)."""

    model_config = ConfigDict(validate_assignment=True)

    metric: str = Field(description="semantic-layer metric name")
    window: Window
    dimensions: list[str] = Field(default_factory=list, description="drill-down dimension names")
    filters: list[DimensionFilter] = Field(
        default_factory=list, description="dimension filters (paired with dimensions)")


class AgentState(TypedDict):
    question: str
    intent: str                    # intent/domain label (marketing/orders/supply-chain/...)
    stage: str                     # judge result (answer/repair/degrade/hallucination, trace)
    hallucination: bool            # relevance check: LLM fabricated an unrelated valid query
    intent_mismatch: bool          # relevance check: granularity mismatch (grouping/filter asked, query has none)
    preflight_kind: str            # preflight result (pass/date/enum/granularity/cutoff)
    preflight_reason: str          # preflight failure reason / fallback suggestion
    empty_result: bool             # execution returned []/None → reflect_empty branch
    _relaxed: bool                 # reflect_empty already relaxed window (guard against loops)
    relax_attempts: int            # relax count (cap to prevent infinite loop)
    schema_text: str               # token-budgeted relevant table schemas
    retrieved_tables: list[str]
    _retrieve_degraded: bool       # retrieval degraded flag (keyword fallback on embedding failure)
    semantic_query: Optional[SemanticQuery]
    compiled_sql: Optional[str]
    execution_result: Any
    execution_error: Optional[str]
    errors: list[str]              # accumulated real errors (fed back to generate for retry)
    error_categories: list[str]    # per-round error three-way categories (dialect/reference/logic/unknown)
    error_classifications: list[dict]  # full classification records (category+entity+reason, to Langfuse)
    error_feedback: str            # feedback context (from repair/reflect)
    _reflect_fixed: bool           # reflect rewrote the query (skip generate, recompile directly)
    retry_count: int
    max_retries: int
    tool_answer: Optional[str]     # tool-path insight answer (from diagnostic tool, skips semantic query)
    tool_used: Optional[str]       # diagnostic tool hit (inventory_diagnostic / marketing_funnel, trace)
    _tool_chart: Optional[str]     # tool-returned Plotly figure JSON (app renders; state holds JSON string only)
    messages: Annotated[list[AnyMessage], add_messages]  # tool-call message chain (ToolNode feedback)
    history: list[dict]                                   # conversation Q&A history (app-owned, injected for context)
    answer: str
    trace_id: str