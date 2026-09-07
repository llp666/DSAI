"""agent/tools_marketing.py：营销 ROI 漏斗工具骨架（阶段 3-4D）。

契约三件套完整（入参 JSON Schema + ToolResult 出参 + ToolError），但漏斗拆解
（曝光→点击→加购→支付各环节转化）内部逻辑尚未实现，调用即抛结构化
ToolError(not_implemented)，经 ToolNode handle_tool_errors=True 包装为
ToolMessage 回灌统一纠错轨，由 error_classifier 分类后走 repair/reflect，
提示用户改用支持的口径（marketing_roi 指标）。

与 tools_inventory.py 同为「契约驱动工具」：只依赖 Executor（DuckDB 只读），
不感知 LangGraph。后续阶段补漏斗拆解逻辑时，保持本文件契约不变，仅填充实现。
"""

from __future__ import annotations

from .tool_contract import (
    ToolError,
    ToolResult,
    validate_args,
)

# 入参 JSON Schema（LLM 调用时 jsonschema 校验）
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
    """营销 ROI 漏斗工具骨架。

    当前版本：契约与入参校验完整，但漏斗拆解逻辑未实现。
    调用即抛 ToolError(not_implemented)，结构化错误回灌纠错轨——
    由 LLM 收到「未实现，请改用 marketing_roi 指标」的修复提示后，
    走语义查询链路用 SUPPORTED_METRICS 里的 marketing 指标回答。
    """
    validate_args(MARKETING_ARGS_SCHEMA, args or {})
    raise ToolError(
        "not_implemented",
        "营销 ROI 漏斗拆解工具尚未实现（曝光→点击→加购→支付各环节转化）",
        "请改用语义查询链路的 marketing_roi 指标（按渠道类型聚合 GMV/花费/ROI），"
        "或先回答渠道整体 ROI，再做维度下钻。",
    )
