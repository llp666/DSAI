"""tools/inventory.py：库存诊断工具（阶段 3-4C）。

inventory_diagnostic_tool 计算：
- 可售天数 = 期末 on_hand_qty / 近30天日均 outbound 销量（销量来自 stock_moves）
- DOI = 可售天数，超过阈值（默认 90）判为呆滞
- 销量环比 = 近30天 outbound vs 前30天
- 结论规则：可售 ≤0 → 已断货，建议立即补货；≤3 → 建议 48 小时补货；>90 → 呆滞建议清库存
- 出参走契约三件套：data + insights + chart（Plotly DOI×环比散点象限图 JSON）

入参：
- sku_id：可选，指定单个 SKU（否则全量，可 top_n 限制）
- category：可选，品类过滤（category_type：服饰/电子/家居/汽配/其他）
- as_of_date：可选，快照日期（默认最近快照日）；无当日快照取 ≤ 该日的最近快照
- doi_threshold：呆滞阈值（默认 90）
- top_n：返回条数上限（默认 20，按可售天数升序=最危险在前）
"""

from __future__ import annotations

from typing import Any

from tools.contract import (
    ToolError,
    ToolResult,
    validate_args,
)

# 入参 JSON Schema（LLM 调用时 pydantic/jsonschema 校验）
INVENTORY_ARGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sku_id": {"type": "string", "description": "指定 SKU（如 SKU-14167349）；缺省查全量"},
        "category": {"type": "string",
                     "description": "品类过滤：服饰/电子/家居/汽配/其他（缺省全品类）"},
        "as_of_date": {"type": "string", "description": "快照日期 YYYY-MM-DD（缺省最近快照日）"},
        "doi_threshold": {"type": "integer", "minimum": 1, "description": "呆滞 DOI 阈值（默认 90）"},
        "top_n": {"type": "integer", "minimum": 1, "maximum": 100, "description": "返回条数（默认 20）"},
    },
}

DOI_THRESHOLD_DEFAULT = 90
SELLOUT_DAYS = 30  # 销量窗口（近30天日均）
URGENT_DAYS = 3    # 剩余可售 ≤3 天 → 48 小时补货


def _recent_snapshot_date(con, as_of: str) -> str:
    """取 ≤ as_of 的最近快照日（快照周粒度，as_of 非快照日时回退）。"""
    row = con.execute(
        "SELECT max(snapshot_date::DATE) FROM ods.inventory_snapshot "
        "WHERE snapshot_date::DATE <= CAST(? AS DATE)", [as_of]).fetchone()
    if row[0] is None:
        raise ToolError("no_data", f"as_of_date {as_of} 之前无库存快照",
                        "请把 as_of_date 调整到 2025-01-05 之后")
    return str(row[0])


def _daily_avg(con, sku_ids: list[str], end: str, days: int) -> dict[str, float]:
    """近 days 天日均 outbound 销量（stock_moves）。"""
    if not sku_ids:
        return {}
    placeholders = ",".join(f"'{s}'" for s in sku_ids)
    # 日均与环比当前窗口统一为半开区间 [end-days, end)。
    rows = con.execute(
        f"SELECT sku_id, sum(qty)::DOUBLE / {days} FROM ods.stock_moves "
        f"WHERE sku_id IN ({placeholders}) AND move_type='outbound' "
        f"AND moved_at >= CAST(? AS DATE) - INTERVAL {days} DAY "
        f"AND moved_at < CAST(? AS DATE) "
        f"GROUP BY sku_id", [end, end]).fetchall()
    return {r[0]: r[1] for r in rows}


def _period_sales(con, sku_ids: list[str], end: str, start: str) -> dict[str, float]:
    """[start, end) 区间 outbound 销量（环比用）。start 是已展开的日期表达式。"""
    if not sku_ids:
        return {}
    placeholders = ",".join(f"'{s}'" for s in sku_ids)
    rows = con.execute(
        f"SELECT sku_id, sum(qty)::DOUBLE FROM ods.stock_moves "
        f"WHERE sku_id IN ({placeholders}) AND move_type='outbound' "
        f"AND moved_at >= {start} AND moved_at < CAST(? AS DATE) "
        f"GROUP BY sku_id", [end]).fetchall()
    return {r[0]: r[1] for r in rows}


def _category_of(con, sku_ids: list[str]) -> dict[str, str]:
    if not sku_ids:
        return {}
    placeholders = ",".join(f"'{s}'" for s in sku_ids)
    rows = con.execute(
        f"SELECT p.sku_id, c.category_type FROM dim.products p "
        f"JOIN dim.categories c ON p.category_id=c.category_id "
        f"WHERE p.sku_id IN ({placeholders})").fetchall()
    return {r[0]: r[1] for r in rows}


def _advice(days: float, mom: float | None, doi_threshold: int) -> str:
    """结论规则：按可售天数给出补货建议。"""
    if days <= 0:
        return "已断货（库存耗尽），建议立即补货"
    if days <= URGENT_DAYS:
        return f"剩余可售 {days:.1f} 天，建议 48 小时补货"
    if days > doi_threshold:
        return f"剩余可售 {days:.1f} 天，超过呆滞阈值（{doi_threshold} 天），建议促销清库存"
    if mom is not None and mom < -0.3:
        return f"剩余可售 {days:.1f} 天，近30天销量环比下降 {abs(mom)*100:.0f}%，建议关注"
    return f"剩余可售 {days:.1f} 天，库存健康"


def _make_scatter_chart(rows: list[dict], doi_threshold: int) -> dict:
    """Plotly DOI×环比散点象限图（figure JSON）。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    # 环比缺失或 DOI 无意义（有库存但近30天零销量 → doi=None）的 SKU 不画点；
    # 否则 marker 颜色比较 None 会 TypeError。
    pts = [(r["doi_days"], r["sales_mom"], r["sku_id"], r["advice"])
           for r in rows if r["sales_mom"] is not None and r["doi_days"] is not None]
    if pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="markers+text", text=[p[2] for p in pts], textposition="top center",
            marker={"color": ["red" if p[0] <= URGENT_DAYS else
                              "orange" if p[0] > doi_threshold else "blue" for p in pts],
                    "size": 10},
            name="SKU",
        ))
    # 阈值参考线：DOI=阈值（呆滞分界）；环比=0
    fig.add_vline(x=doi_threshold, line_dash="dash", line_color="orange")
    fig.add_vline(x=URGENT_DAYS, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="库存健康象限：DOI(可售天数) × 近30天销量环比",
        xaxis_title=f"DOI 可售天数（> {doi_threshold} 呆滞，≤ {URGENT_DAYS} 紧急）",
        yaxis_title="近30天销量环比",
        height=500,
    )
    return {"kind": "plotly_figure_json", "figure_json": fig.to_json()}


def inventory_diagnostic_tool(executor, args: dict) -> ToolResult:
    """库存诊断工具实现（纯函数，不感知 LangGraph）。

    executor：agent.executor.Executor（DuckDB 只读连接）。
    args：JSON Schema 校验后的入参。
    """
    args = validate_args(INVENTORY_ARGS_SCHEMA, args or {})
    con = executor._con
    as_of = args.get("as_of_date")
    if not as_of:
        as_of = str(con.execute(
            "SELECT max(snapshot_date::DATE) FROM ods.inventory_snapshot").fetchone()[0])
    snap = _recent_snapshot_date(con, as_of)

    # 基础快照（可选 sku/category 过滤）
    where = ["snapshot_date::DATE = CAST(? AS DATE)"]
    params: list[str] = [snap]
    if args.get("sku_id"):
        where.append("sku_id = ?")
        params.append(args["sku_id"])
    if args.get("category"):
        where.append("sku_id IN (SELECT p.sku_id FROM dim.products p "
                     "JOIN dim.categories c ON p.category_id=c.category_id "
                     "WHERE c.category_type = ?)")
        params.append(args["category"])
    rows = con.execute(
        f"SELECT sku_id, on_hand_qty::DOUBLE FROM ods.inventory_snapshot "
        f"WHERE {' AND '.join(where)}", params).fetchall()
    if not rows:
        raise ToolError("no_data", f"{snap} 快照无匹配 SKU",
                        "请检查 sku_id / category 参数是否正确")

    sku_ids = [r[0] for r in rows]
    on_hand = {r[0]: r[1] for r in rows}
    daily = _daily_avg(con, sku_ids, snap, SELLOUT_DAYS)
    cat = _category_of(con, sku_ids)
    doi_threshold = args.get("doi_threshold") or DOI_THRESHOLD_DEFAULT

    # 环比：本期近30天销量 vs 上期（前30天）销量
    cur_start = f"DATE '{snap}' - INTERVAL {SELLOUT_DAYS} DAY"
    prev_start = f"DATE '{snap}' - INTERVAL {2 * SELLOUT_DAYS} DAY"
    cur_sales = _period_sales(con, sku_ids, snap, cur_start)
    prev_sales = _period_sales(con, sku_ids, snap, prev_start)

    records = []
    for sid in sku_ids:
        d = daily.get(sid, 0.0)
        days = (on_hand[sid] / d) if d > 0 else (0.0 if on_hand[sid] <= 0 else float("inf"))
        cur_v = cur_sales.get(sid, 0.0)
        prev_v = prev_sales.get(sid, 0.0)
        if prev_v > 0:
            mom = (cur_v - prev_v) / prev_v
        else:
            mom = None  # 上期无销量，环比无意义
        advice = _advice(days, mom, doi_threshold)
        records.append({
            "sku_id": sid,
            "category": cat.get(sid, ""),
            "on_hand": on_hand[sid],
            "daily_avg_30d": round(d, 3),
            "doi_days": round(days, 1) if days != float("inf") else None,
            "sales_mom": round(mom, 3) if mom is not None else None,
            "advice": advice,
        })
    records.sort(key=lambda r: (r["doi_days"] is None, r["doi_days"] if r["doi_days"] is not None else 1e9))
    if args.get("top_n"):
        records = records[: args["top_n"]]

    # insights：逐 SKU 结论（前 top_n）+ 汇总
    insights = []
    for r in records:
        mom_txt = (f"，环比 {r['sales_mom']*100:+.0f}%" if r["sales_mom"] is not None else "")
        insights.append(
            f"{r['sku_id']}（{r['category'] or '未知品类'}）：{r['advice']}{mom_txt}")
    n_stockout = sum(1 for r in records if r["doi_days"] == 0 or r["on_hand"] <= 0)
    if n_stockout:
        insights.insert(0, f"共 {n_stockout} 个 SKU 已断货/库存耗尽（快照日 {snap}）")

    chart = _make_scatter_chart(records, doi_threshold)
    return ToolResult(
        data={"as_of_date": snap, "doi_threshold": doi_threshold, "records": records},
        insights=insights,
        chart=chart,
    )
