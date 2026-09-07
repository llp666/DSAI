"""tools/contract.py：诊断工具契约三件套（阶段 3-4）。

1. 入参 JSON Schema：每个工具声明 args JSON Schema，LLM 调用时 pydantic 校验；
2. 出参结构：data（结构化数据）+ insights（文本洞察列表）+ chart（Plotly 图 JSON，可选）；
3. 结构化错误：ToolError（code/message/repair_hint），经 ToolMessage 回灌统一纠错轨。

工具实现只依赖本契约 + Executor（DuckDB 只读），不感知 LangGraph。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# 出参里 chart 的 Plotly figure JSON 结构标记（避免与纯 data 混淆）
CHART_KIND = "plotly_figure_json"


class ToolError(RuntimeError):
    """结构化工具错误：code 分类 + message 人类可读 + repair_hint 给纠错轨。

    错误进统一纠错轨：graph 的 ToolNode handle_tool_errors=True 会把 ToolError
    包装为 ToolMessage「Error: …」，由 error_classifier 分类后走 repair/reflect。
    """

    def __init__(self, code: str, message: str, repair_hint: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code          # not_implemented / invalid_params / no_data / query_failed
        self.message = message
        self.repair_hint = repair_hint  # 给 LLM 的修复建议（回灌）

    def as_text(self) -> str:
        if self.repair_hint:
            return f"{self.message}（修复建议：{self.repair_hint}）"
        return self.message


@dataclass
class ToolResult:
    """工具出参统一结构：data + insights + chart。"""

    data: dict
    insights: list[str] = field(default_factory=list)
    chart: Optional[dict] = None  # {"kind": CHART_KIND, "figure_json": {...}}

    def to_dict(self) -> dict:
        return {"data": self.data, "insights": self.insights, "chart": self.chart}


def render_insights(insights: list[str]) -> str:
    """把洞察列表渲染为多行文本（供 answer 拼接/Streamlit 展示）。"""
    if not insights:
        return ""
    return "\n".join(f"• {i}" for i in insights)


# ---- 入参 JSON Schema 校验（pydantic） ----

def validate_args(schema: dict, args: dict) -> dict:
    """按 JSON Schema 校验工具入参，非法抛 ToolError(invalid_params)。

    用 jsonschema 校验（项目已有依赖）；args 缺字段/类型错 → 结构化错误，
    经 ToolMessage 回灌纠错轨，LLM 可据此重调。
    """
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError as JsValidationError

        validate(instance=args, schema=schema)
    except JsValidationError as e:
        loc = ".".join(str(x) for x in e.absolute_path) or "<root>"
        raise ToolError(
            "invalid_params",
            f"工具入参校验失败 [{loc}]: {e.message}",
            "请按工具 JSON Schema 补全/修正参数后重试（如 sku_id、as_of_date、category 等）。",
        ) from e
    except Exception as e:  # jsonschema 本身异常兜底
        raise ToolError("invalid_params", f"工具入参校验异常: {e}") from e
    return args


def serialize_result(result: ToolResult) -> str:
    """把 ToolResult 序列化为 JSON 文本（ToolNode 返回 content）。"""
    return json.dumps(result.to_dict(), ensure_ascii=False, default=str)


def parse_result(text: str) -> dict:
    """把工具返回 JSON 解析回 dict（供 answer/Streamlit 展示）。"""
    if isinstance(text, dict):
        return text
    return json.loads(text)
