"""tools/marketing.py：marketing ROI funnel tool skeleton (stage 3-4D).

Full contract triple (args JSON Schema + ToolResult + ToolError), but the funnel breakdown
(曝光→点击→加购→支付 conversion per stage) is not implemented yet: any call raises
ToolError(not_implemented), wrapped by ToolNode handle_tool_errors=True → ToolMessage → shared
repair track, telling the user to use the supported caliber (marketing_roi metric).

Same "contract-driven tool" as tools/inventory.py: depends only on Executor (DuckDB read-only),
not LangGraph. Later stages fill in the funnel logic without touching the contract.
"""

from __future__ import annotations

from tools.contract import (
    ToolError,
    ToolResult,
    validate_args,
)

# args JSON Schema (jsonschema-validated on LLM calls)
MARKETING_ARGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "start_date": {"type": "string",
                       "description": "起始日期 YYYY-MM-DD（缺省近 30 天窗口起点）"},
        "end_date": {"type": "string",
                     "description": "结束日期 YYYY-MM-DD（缺省最近数据日）"},
        "channel_type": {"type": "string",
                         "description": "渠道类型过滤：feeds/search/cps/sms（缺省全渠道）"},
    },
}


def marketing_roi_funnel_tool(executor, args: dict) -> ToolResult:
    """Marketing ROI funnel tool skeleton.

    Current version: contract + arg validation complete, but the funnel breakdown is unimplemented.
    """
    validate_args(MARKETING_ARGS_SCHEMA, args or {})
    raise ToolError(
        "not_implemented",
        "营销 ROI 漏斗拆解工具尚未实现（曝光→点击→加购→支付各环节转化）",
        "请改用语义查询链路的 marketing_roi 指标（按渠道类型聚合 GMV/花费/ROI），"
        "或先回答渠道整体 ROI，再做维度下钻。",
    )