"""agent/compiler.py：确定性编译器（纯函数）。

将语义查询 SemanticQuery 展开为精确 SQL，口径与 meta/ground_truth.json 严格一致：
- GMV：按订单创建月归集，排除 order_status='1'（待支付），不扣退款；
- 净销售额：当月 GMV − 当月 refund_date 归集的退款（跨月扣减）；
- 复购率：窗口 [月末−89天, 月末] 有单用户中「下单≥2个不同日期」占比。
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path

import yaml

from .types import SemanticQuery

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


class CompileError(RuntimeError):
    pass


def load_metrics(yaml_path: Path) -> dict:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return data["metrics"]


def _month_bounds(ym: str) -> tuple[str, str]:
    """由 YYYY-MM 返回 [月初, 月末]（DuckDB date 字符串，闭区间）。"""
    if not MONTH_RE.match(ym):
        raise CompileError(f"非法月份格式：{ym!r}（应为 YYYY-MM）")
    year, month = int(ym[:4]), int(ym[5:7])
    if not 1 <= month <= 12:
        raise CompileError(f"非法月份：{ym!r}")
    last = calendar.monthrange(year, month)[1]
    return f"{ym}-01", f"{ym}-{last:02d}"


def compile_query(sq: SemanticQuery, metrics: dict) -> str:
    if sq.metric not in metrics:
        raise CompileError(f"未知指标：{sq.metric}")
    meta = metrics[sq.metric]
    ym = sq.window.value
    start, end = _month_bounds(ym)

    # 复购率：90 天窗口（窗口锚月末），与 ground_truth 口径一致
    if "window" in meta:
        year, month = int(ym[:4]), int(ym[5:7])
        last = calendar.monthrange(year, month)[1]
        from datetime import date, timedelta

        w_end = date(year, month, last)
        w_start = w_end - timedelta(days=89)
        base = meta["base"]
        return f"""
WITH w AS (
  SELECT user_id, count(DISTINCT created_at::DATE) AS n_days
  FROM {base}
  WHERE created_at::DATE BETWEEN DATE '{w_start.isoformat()}' AND DATE '{w_end.isoformat()}'
  GROUP BY user_id
)
SELECT round(
  (SELECT count(*)::DOUBLE FROM w WHERE n_days >= 2)
  / NULLIF((SELECT count(*)::DOUBLE FROM w), 0)
, 4) AS result
""".strip()

    base, expr = meta["base"], meta["expression"]
    filters = " AND ".join(meta.get("filters", []))
    time_col = meta.get("time_column", "created_at")

    where = f"{time_col}::DATE BETWEEN DATE '{start}' AND DATE '{end}'"
    if filters:
        where += f" AND {filters}"

    # 净销售额：当月 GMV − 当月退款（退款按 refund_date 归集）
    if "deduction" in meta:
        ded = meta["deduction"]
        ded_where = (
            f"{ded['time_column']} BETWEEN DATE '{start}' AND DATE '{end}'"
        )
        return f"""
WITH orders_m AS (
  SELECT round({expr}::DOUBLE, 2) AS gmv
  FROM {base}
  WHERE {where}
), refunds_m AS (
  SELECT round({ded['expression']}::DOUBLE, 2) AS refund
  FROM {ded['base']}
  WHERE {ded_where}
)
SELECT round(o.gmv - COALESCE(r.refund, 0), 2) AS result
FROM orders_m o LEFT JOIN refunds_m r ON TRUE
""".strip()

    # 普通聚合（GMV）
    return f"""
SELECT round({expr}::DOUBLE, 2) AS result
FROM {base}
WHERE {where}
""".strip()
