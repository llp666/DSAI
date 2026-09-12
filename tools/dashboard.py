"""tools/dashboard.py：看板（库存告警 / 经营概览 / 营销建议）确定性卡片装配与结果图表。

- build_cards(layer, executor)：三张看板卡片的确定性数据装配——compile_query + Executor 取数，
  库存卡复用 inventory_diagnostic_tool。零 LLM 延迟；每张卡独立 try/except，单卡失败返回 None 不崩页。
- answer_chart(result, layer, executor)：看板问答结果的图表决策：
  工具图 > 分组明细柱状（统计类）> 标量窗口指标趋势线（近 N 期）。

纯逻辑模块：不 import Streamlit / Pipeline / config，可独立单测。app.py 经模块属性调用
（dashboard.build_cards / dashboard.answer_chart），AppTest 可 patch 本模块替换为罐头数据。
"""

from __future__ import annotations

from datetime import date, timedelta

from compile.compiler import compile_query
from compile.types import SemanticQuery, Window
from tools.inventory import inventory_diagnostic_tool

# 经营概览 stat tiles：指标 → 展示口径（context.yaml 的 default_calibers）
_OVERVIEW_TILES = (
    "gmv",
    "net_sales_amount",
    "orders_count",
    "avg_order_value",
    "refund_rate",
    "gross_margin",
    "net_profit",
)
_RATE_METRICS = {"refund_rate", "gross_margin", "repurchase_rate"}
_INT_METRICS = {"orders_count", "stockout_skus_count"}

# 呆滞兜底 COUNT：inventory_diagnostic_tool 按 at-risk 升序截断 top_n，高 DOI（呆滞）排最后被切掉。
# DOI>90 ⟺ on_hand / (近30日出库/30) > 90 ⟺ on_hand > 3×近30日出库；另有库存但近30日零出库（DOI=∞）同样呆滞。
_DEAD_STOCK_SQL = """
WITH snap AS (
  SELECT sku_id, on_hand_qty::DOUBLE AS oh FROM ods.inventory_snapshot
  WHERE snapshot_date::DATE = (SELECT max(snapshot_date::DATE) FROM ods.inventory_snapshot)),
sales AS (
  SELECT sku_id, sum(qty)::DOUBLE AS s FROM ods.stock_moves
  WHERE move_type='outbound'
    AND moved_at >= (SELECT max(snapshot_date::DATE) FROM ods.inventory_snapshot)::DATE - INTERVAL 30 DAY
    AND moved_at <  (SELECT max(snapshot_date::DATE) FROM ods.inventory_snapshot)::DATE
  GROUP BY sku_id)
SELECT count(*) FROM snap a LEFT JOIN sales b USING (sku_id)
WHERE a.oh > 0 AND (b.s IS NULL OR b.s = 0 OR a.oh > 3 * b.s)
""".strip()

# 看板实时轮询判变的核心表（库存/订单/营销各取最后时间戳 + 行数）
_FINGERPRINT_SQL = """
SELECT
  (SELECT max(snapshot_date::DATE)::TEXT FROM ods.inventory_snapshot),
  (SELECT count(*)::TEXT FROM ods.inventory_snapshot),
  (SELECT max(moved_at)::TEXT FROM ods.stock_moves),
  (SELECT count(*)::TEXT FROM ods.stock_moves),
  (SELECT max(created_at)::TEXT FROM ods.orders),
  (SELECT count(*)::TEXT FROM ods.orders),
  (SELECT max(stat_date::DATE)::TEXT FROM ods.ads_daily_stats),
  (SELECT count(*)::TEXT FROM ods.ads_daily_stats)
""".strip()


def _warehouse_fingerprint(executor) -> str | None:
    """数仓数据指纹：看板每 30s 轮询比较，有变动 → 卡片自动重建（数据实时跟随）。

    数仓为静态快照时指纹恒定；datagen 重跑 / 增量写入后任一核心表的时间戳或行数
    变化 → 指纹变化 → 看板刷新。任一查询失败返回 None（保守：视为未变）。
    """
    try:
        rows = executor.execute(_FINGERPRINT_SQL)
        return "|".join(str(v) for v in rows[0]) if rows else None
    except Exception:
        return None


# ---------- 图表助手（返回 Plotly figure JSON 字符串） ----------

def _bar_chart(labels: list, values: list[float], title: str, y_title: str) -> str:
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        text=[f"{v:,.0f}" for v in values], textposition="outside",
        marker_color="#0a84ff",
    ))
    fig.update_layout(
        title=title, yaxis_title=y_title, showlegend=False, height=420,
        margin=dict(t=56, b=24, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
    )
    return fig.to_json()


def _line_chart(labels: list, values: list[float], title: str, y_title: str) -> str:
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=list(labels), y=list(values), mode="lines+markers",
        line=dict(width=3, color="#0071e3"), marker=dict(size=7, color="#0a84ff"),
    ))
    fig.update_layout(
        title=title, yaxis_title=y_title, showlegend=False, height=360,
        margin=dict(t=56, b=24, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
    )
    return fig.to_json()


def _sparkline_chart(values: list[float]) -> str:
    """Hero 卡迷你走势线：36px 高、隐藏坐标轴、平滑曲线 + 渐变面积填充。

    刻意压低（36px）——Hero 卡里环比环 + 大数字 + 副行已占主视觉，迷你图只做点缀，
    过高会把卡片撑到溢出/切边。
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=list(range(len(values))), y=list(values), mode="lines",
        line=dict(width=2.2, color="#0071e3", shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,113,227,.14)", hoverinfo="skip",
    ))
    fig.update_layout(
        height=36, showlegend=False, margin=dict(t=2, b=0, l=0, r=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_json()


# ---------- 动态图表（帧动画：▶ 播放 / 滑动条） ----------

def _animation_controls(n: int, labels: list[str]) -> dict:
    """▶ 播放 / ⏸ 暂停 按钮 + 帧滑动条的 plotly layout 片段（供动态图表复用）。"""
    return {
        "updatemenus": [{
            "type": "buttons", "showactive": False, "direction": "left",
            "x": 1.0, "xanchor": "right", "y": 1.15, "yanchor": "top",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": 450, "redraw": True},
                                 "fromcurrent": True, "mode": "immediate"}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
        }],
        "sliders": [{
            "active": 0, "len": 0.95, "x": 0.02, "y": 0,
            "pad": {"b": 6, "t": 18}, "currentvalue": {"visible": False},
            "steps": [
                {"method": "animate", "label": labels[i],
                 "args": [[str(i)], {"mode": "immediate",
                                     "frame": {"duration": 300, "redraw": True}}]}
                for i in range(n)
            ],
        }],
    }


def _animated_line_chart(labels: list, values: list[float], title: str, y_title: str) -> str:
    """动态折线：逐帧加入数据点（▶ 播放），Y 轴固定全量范围避免跳动。"""
    import plotly.graph_objects as go

    n = len(labels)
    trace = dict(
        mode="lines+markers",
        line=dict(width=3, color="#0071e3"), marker=dict(size=8, color="#0a84ff"),
    )
    fig = go.Figure(go.Scatter(x=[labels[0]], y=[values[0]], **trace))
    fig.frames = [
        go.Frame(data=[go.Scatter(x=list(labels[: i + 1]), y=list(values[: i + 1]), **trace)],
                 name=str(i))
        for i in range(n)
    ]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 1.0
    fig.update_layout(
        title=title, yaxis_title=y_title, showlegend=False, height=300,
        yaxis=dict(range=[lo - pad, hi + pad]),
        margin=dict(t=56, b=48, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
        **_animation_controls(n, list(labels)),
    )
    return fig.to_json()


def _animated_bar_chart(labels: list, values: list[float], title: str, y_title: str) -> str:
    """动态柱状：逐帧加入柱体（▶ 播放），Y 轴固定全量范围避免跳动。"""
    import plotly.graph_objects as go

    n = len(labels)
    bar = dict(textposition="outside", marker_color="#0a84ff")
    fig = go.Figure(go.Bar(x=[labels[0]], y=[values[0]],
                           text=[f"{values[0]:,.0f}"], **bar))
    fig.frames = [
        go.Frame(data=[go.Bar(x=list(labels[: i + 1]), y=list(values[: i + 1]),
                              text=[f"{v:,.0f}" for v in values[: i + 1]], **bar)],
                 name=str(i))
        for i in range(n)
    ]
    fig.update_layout(
        title=title, yaxis_title=y_title, showlegend=False, height=300,
        yaxis=dict(range=[0, max(values) * 1.15]),
        margin=dict(t=56, b=48, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
        **_animation_controls(n, list(labels)),
    )
    return fig.to_json()


def _progress_chart(kind: str, labels: list, values: list[float], title: str,
                    y_title: str, idx: int, colors: list | None = None,
                    hline: float | None = None, area: bool = False,
                    height: int = 300, decimals: int = 0) -> str:
    """轮播第 idx 帧的图 JSON：只画前 idx+1 个点，Y 轴固定全量范围（逐帧推进不跳动）。

    与 _animated_* 同款视觉，但无帧动画控件——轮播由前端 @st.fragment(run_every=1) 每拍
    推进 idx 驱动；数仓指纹有变动时整页重建，新数据从下一拍起轮播。idx 超界钳到全量。

    Bento 换肤参数（均有默认值，不传则与旧版输出一致）：
    - colors：逐柱颜色（bar）；不足则回落纯蓝。
    - hline：水平基准线（ROI 盈亏平衡线）。
    - area：line 改平滑曲线 + 渐变面积填充（趋势卡）。
    """
    import plotly.graph_objects as go

    idx = min(idx, len(labels) - 1)
    sub_l = list(labels[: idx + 1])
    sub_v = list(values[: idx + 1])
    if kind == "bar":
        bar_kw = dict(marker_color=list(colors)[: len(sub_v)]) if colors else {"marker_color": "#0a84ff"}
        fig = go.Figure(go.Bar(
            x=sub_l, y=sub_v, text=[f"{v:,.{decimals}f}" for v in sub_v],
            textposition="outside", **bar_kw))
        yaxis = dict(range=[0, max(values) * 1.15])
    else:
        line = dict(width=3, color="#0071e3", shape="spline" if area else "linear")
        trace = go.Scatter(
            x=sub_l, y=sub_v, mode="lines+markers", line=line,
            marker=dict(size=8, color="#0a84ff"))
        if area:
            trace.fill = "tozeroy"
            trace.fillcolor = "rgba(0,113,227,.13)"
        fig = go.Figure(trace)
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.12 or 1.0
        yaxis = dict(range=[lo - pad, hi + pad])
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_width=1.4,
                      line_color="rgba(0,0,0,.32)",
                      annotation_text=f"基准 {hline:g}", annotation_position="top left",
                      annotation_font=dict(size=10, color="rgba(0,0,0,.45)"))
    fig.update_layout(
        title=title, yaxis_title=y_title, showlegend=False, height=height,
        # ⚠ x 轴强制 category：月份标签（"2026-08"）会被 plotly 当日期解析，
        # 轴刻度裂成 "23:59:59.999 Feb 28, 2026" 这类时间戳（Bento 截图实测）。
        xaxis=dict(type="category"),
        yaxis=yaxis, margin=dict(t=56, b=24, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
    )
    return fig.to_json()


# ---------- 窗口与趋势 ----------

def _window_sequence(sq: SemanticQuery, n: int) -> list[Window]:
    """最近 n 个同类型窗口（oldest→newest）。月窗口日历回推（跨年）；日窗口 timedelta 回推。"""
    win = sq.window
    if win.type == "day":
        end = date.fromisoformat(win.value)
        start = end - timedelta(days=n - 1)
        return [Window(type="day", value=(start + timedelta(days=i)).isoformat())
                for i in range(n)]
    y, m = int(win.value[:4]), int(win.value[5:7])
    seq: list[str] = []
    for _ in range(n):
        seq.append(f"{y}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return [Window(type="month", value=v) for v in reversed(seq)]


def _window_label(w: Window) -> str:
    return w.value[5:] if w.type == "day" else w.value


def _metric_trend(sq: SemanticQuery, layer, executor, n: int = 6):
    """把窗口指标重跑最近 n 个同类型窗口 → (labels, values) | None。

    保留原查询的 filters（如「电子品类 GMV」的趋势仍限电子），清空 dimensions
    （趋势按时间展开，不按维度分组）。逐窗口容错：窗口指标（repurchase_rate）
    仅支持月窗口、个别月份无数据 → 该窗口跳过，不中断整体趋势。
    """
    labels: list[str] = []
    values: list[float] = []
    for w in _window_sequence(sq, n):
        q = sq.model_copy(update={"window": w, "dimensions": []})
        try:
            sql = compile_query(q, layer)
            rows = executor.execute(sql)
            val = rows[0][0] if rows else None
        except Exception:
            continue
        labels.append(_window_label(w))
        values.append(float(val) if val is not None else 0.0)
    if not labels:
        return None
    return labels, values


# ---------- 命名与格式化 ----------

def _metric_name(layer, metric: str) -> str:
    meta = (layer.metrics or {}).get(metric)
    return meta.get("display_name", metric) if isinstance(meta, dict) else metric


def _format_tile(metric: str, val) -> str:
    """看板 stat tile 格式化：率 ×100% / 整数千分位 / 金额（≥1万 → 万单位）。"""
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if metric in _RATE_METRICS:
        return f"{v * 100:.2f}%"
    if metric in _INT_METRICS:
        return f"{v:,.0f}"
    if abs(v) >= 1e4:
        return f"{v / 1e4:,.2f}万"
    return f"{v:,.2f}"


def _format_mom(metric: str, mom: float | None) -> str | None:
    """环比标签：率指标显示百分点（+0.40pp），其余显示相对变化率（+12.4%）。

    mom 语义（见 _build_overview_card）：率指标为两期差值（pp/100），其余为相对变化率。
    """
    if mom is None:
        return None
    if metric in _RATE_METRICS:
        return f"{mom * 100:+.2f}pp"
    return f"{mom * 100:+.1f}%"


# ---------- 卡片装配 ----------

def _bucket_of(record: dict) -> str | None:
    """单条库存记录的告警分桶（与工具 advice 语义一致），不属任何桶 → None。

    断货 = on_hand<=0 或 doi==0；低库存 = 0<doi<=3；呆滞 = doi>90 或（有库存但近30日
    零出库，doi=None）。渲染层的 Tag 过滤与卡片计数共用此函数，避免两处语义漂移。
    """
    d = record.get("doi_days")
    if record.get("on_hand", 0) <= 0 or d == 0:
        return "断货"
    if d is not None and d <= 3:
        return "低库存"
    if d is None or d > 90:
        return "呆滞"
    return None

def _build_inventory_card(executor):
    """库存告警卡：断货/低库存/呆滞 counts + Top 风险 SKU + DOI×MoM 散点图 + insights。

    分桶语义与工具 advice 一致：断货 = on_hand<=0 或 doi==0；低库存 = 0<doi<=3；
    呆滞 = doi>90 或（有库存但近30日零出库，doi=None）。
    断货/低库存按风险升序排在 top-100 内，计数精确；呆滞数用精确 COUNT 覆盖
    （top-100 按风险升序会切掉高 DOI，样本里≈0）。
    """
    try:
        res = inventory_diagnostic_tool(executor, {"top_n": 100})
        records = res.data.get("records", [])
        counts = {"断货": 0, "低库存": 0, "呆滞": 0}
        for r in records:
            bucket = _bucket_of(r)
            if bucket:
                counts[bucket] += 1
        dead_stock_total: int | None = None
        try:
            rows = executor.execute(_DEAD_STOCK_SQL)
            dead_stock_total = int(rows[0][0]) if rows else None
        except Exception:
            dead_stock_total = None
        if dead_stock_total is not None:
            counts["呆滞"] = dead_stock_total
        chart_json = res.chart["figure_json"] if res.chart else None
        return {
            "as_of_date": res.data.get("as_of_date"),
            "counts": counts,
            "dead_stock_total": dead_stock_total,
            "top": records[:20],
            "insights": res.insights[:5],
            "chart_json": chart_json,
        }
    except Exception:
        return None


def _build_overview_card(layer, executor, month: str):
    """经营概览卡：stat tiles（当月 7 指标 + 环比）+ GMV 近 6 月趋势 + 各品类 GMV。

    Bento 重构新增（向后兼容，旧字段原样保留）：
    - tiles[m]["mom"]/["mom_text"]：环比上月（率指标差值为百分点）；
    - tiles["gmv"/"net_sales_amount"]["spark"]：迷你走势序列（Hero 卡 Sparkline）；
    - "category_series"：{月: {labels, values}}——趋势↔品类下钻的数据源。
    """
    tiles: dict[str, dict] = {}
    for metric in _OVERVIEW_TILES:
        sq = SemanticQuery(metric=metric, window=Window(type="month", value=month))
        val = None
        mom = None
        # 取最近两窗口：末值即当月指标值，两值之比/差即环比（省一次独立查询）
        try:
            t = _metric_trend(sq, layer, executor, n=2)
        except Exception:
            t = None
        if t:
            _labels, values = t
            if values:
                val = values[-1]
            if len(values) >= 2 and values[-2]:
                prev, cur = values[-2], values[-1]
                # 率指标用百分点差；其余用相对变化率
                mom = (cur - prev) if metric in _RATE_METRICS else (cur - prev) / abs(prev)
        tiles[metric] = {"display": _metric_name(layer, metric), "value": val,
                         "text": _format_tile(metric, val),
                         "mom": mom, "mom_text": _format_mom(metric, mom)}

    trend = None
    try:
        sq = SemanticQuery(metric="gmv", window=Window(type="month", value=month))
        t = _metric_trend(sq, layer, executor, n=6)
        if t:
            labels, values = t
            name = _metric_name(layer, "gmv")
            trend = {"labels": labels, "values": values,
                     "title": f"{name} 近 6 月趋势", "y_title": name,
                     "chart_json": _animated_line_chart(labels, values, f"{name} 近 6 月趋势", name)}
            tiles["gmv"]["spark"] = list(values)   # Hero Sparkline 免额外查询
    except Exception:
        trend = None

    # 净销售额 Sparkline（Hero 副指标）：逐窗口重跑取近 6 月序列
    try:
        sq = SemanticQuery(metric="net_sales_amount", window=Window(type="month", value=month))
        t = _metric_trend(sq, layer, executor, n=6)
        if t:
            tiles["net_sales_amount"]["spark"] = list(t[1])
    except Exception:
        pass

    category = None
    try:
        sq = SemanticQuery(metric="gmv", window=Window(type="month", value=month),
                           dimensions=["category_type"])
        rows = executor.execute(compile_query(sq, layer))
        if rows:
            labels = [str(r[0]) for r in rows]
            values = [float(r[1]) for r in rows]
            name = _metric_name(layer, "gmv")
            category = {"labels": labels, "values": values,
                        "title": f"各品类 {name}（{month}）", "y_title": name,
                        "chart_json": _animated_bar_chart(labels, values,
                                                          f"各品类 {name}（{month}）", name)}
    except Exception:
        category = None

    # 月度品类序列：趋势卡每个月份一条品类切片（点击月份 → 品类图下钻）
    category_series: dict[str, dict] = {}
    if trend:
        for m in trend["labels"]:
            try:
                sq = SemanticQuery(metric="gmv", window=Window(type="month", value=m),
                                   dimensions=["category_type"])
                rows = executor.execute(compile_query(sq, layer))
                if rows:
                    category_series[m] = {"labels": [str(r[0]) for r in rows],
                                          "values": [float(r[1]) for r in rows]}
            except Exception:
                continue

    return {"month": month, "tiles": tiles, "trend": trend, "category": category,
            "category_series": category_series}


def _build_marketing_card(layer, executor, month: str):
    """营销建议卡：各渠道类型 ROI 柱状 + 确定性洞察（最高/最低渠道 + 建议）。"""
    try:
        sq = SemanticQuery(metric="marketing_roi", window=Window(type="month", value=month),
                           dimensions=["channel_type"])
        rows = executor.execute(compile_query(sq, layer))
        if not rows:
            return None
        roi = [(str(r[0]), float(r[1])) for r in rows]
        best = max(roi, key=lambda x: x[1])
        worst = min(roi, key=lambda x: x[1])
        insight = (
            f"{best[0]} 渠道 ROI 最高（{best[1]:.2f}），{worst[0]} 最低（{worst[1]:.2f}）。"
            f"建议复盘 {worst[0]} 的定向与素材，保持 {best[0]} 的投放力度。"
        )
        return {
            "rows": roi,
            "title": f"各渠道类型 ROI（{month}）", "y_title": "ROI",
            "chart_json": _animated_bar_chart(
                [c for c, _ in roi], [v for _, v in roi],
                f"各渠道类型 ROI（{month}）", "ROI"),
            "insight": insight,
        }
    except Exception:
        return None


def build_cards(layer, executor, month: str | None = None) -> dict:
    """三张看板卡片（库存告警 / 经营概览 / 营销建议）的确定性数据装配。

    month 缺省取语义层 visible_data_cutoff 的当月（钉死数据截至月，避免日历月落到无数据窗口，
    如 cutoff=2026-08-31 时日历 2026-09 全为 0）。每张卡独立 try/except → None。
    """
    cutoff = (layer.context or {}).get("visible_data_cutoff")
    month = month or str(cutoff)[:7]
    return {
        "as_of": month,
        "cutoff": str(cutoff) if cutoff else month,
        "inventory": _build_inventory_card(executor),
        "overview": _build_overview_card(layer, executor, month),
        "marketing": _build_marketing_card(layer, executor, month),
    }


def answer_chart(result: dict, layer, executor) -> str | None:
    """看板问答结果 → 图表 figure JSON 字符串或 None。

    优先级：
    1. _tool_chart（工具轨散点/图）已存在 → 直接返回；
    2. execution_result 为分组行列表（统计/分布类）→ 柱状图（row[0]=标签, row[-1]=数值）；
    3. 标量（int/float）且带语义查询 → 趋势折线图（近 6 期）；rank 指标返回字符串 SKU → None；
    4. 空 / None → None。
    整函数兜底 try/except → None（任一异常不崩页面）。
    """
    try:
        tool_chart = result.get("_tool_chart")
        if tool_chart:
            return tool_chart
        sq = result.get("semantic_query")
        res = result.get("execution_result")
        if sq is None:
            return None
        name = _metric_name(layer, sq.metric)
        if isinstance(res, list) and res:
            labels = [str(r[0]) for r in res]
            values = [float(r[-1]) for r in res]
            return _bar_chart(labels, values, f"{name} 分布", name)
        if isinstance(res, (int, float)):
            trend = _metric_trend(sq, layer, executor)
            if trend:
                labels, values = trend
                return _line_chart(labels, values,
                                   f"{name} 趋势（近 {len(labels)} 期）", name)
        return None
    except Exception:
        return None
