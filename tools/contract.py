"""tools/contract.py：diagnostic-tool contract triple (stage 3-4).

1. args JSON Schema: each tool declares an args schema, validated on LLM calls;
2. result shape: data (structured) + insights (text list) + chart (Plotly JSON, optional);
3. structured error: ToolError (code/message/repair_hint), fed back through ToolMessage to the shared repair track.

Tool implementations depend only on this contract + Executor (DuckDB read-only), not LangGraph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# chart's Plotly figure JSON structure marker (distinguishes it from plain data)
CHART_KIND = "plotly_figure_json"


class ToolError(RuntimeError):
    """Structured tool error: code + human message + repair_hint for the repair track."""

    def __init__(self, code: str, message: str, repair_hint: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code          # not_implemented / invalid_params / no_data / query_failed
        self.message = message
        self.repair_hint = repair_hint  # repair suggestion fed back to the LLM

    def as_text(self) -> str:
        if self.repair_hint:
            return f"{self.message}（修复建议：{self.repair_hint}）"
        return self.message


@dataclass
class ToolResult:
    """Unified tool return shape: data + insights + chart."""

    data: dict
    insights: list[str] = field(default_factory=list)
    chart: Optional[dict] = None  # {"kind": CHART_KIND, "figure_json": {...}}

    def to_dict(self) -> dict:
        return {"data": self.data, "insights": self.insights, "chart": self.chart}


def render_insights(insights: list[str]) -> str:
    """Render an insights list as multi-line bullet text."""
    if not insights:
        return ""
    return "\n".join(f"• {i}" for i in insights)


def validate_args(schema: dict, args: dict) -> dict:
    """Validate tool args against a JSON Schema; raise ToolError(invalid_params) on failure."""
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
    except Exception as e:  # jsonschema internal failure fallback
        raise ToolError("invalid_params", f"工具入参校验异常: {e}") from e
    return args


def serialize_result(result: ToolResult) -> str:
    """Serialize a ToolResult to JSON text (ToolNode return content)."""
    return json.dumps(result.to_dict(), ensure_ascii=False, default=str)


def parse_result(text: str) -> dict:
    """Parse a tool-return JSON back to a dict."""
    if isinstance(text, dict):
        return text
    return json.loads(text)