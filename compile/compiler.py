"""compile/compiler.py：deterministic compiler v2 (pure functions).

Expands a SemanticQuery into exact SQL:
- metrics driven by the semantic-layer parts/combine structure;
- dimension drill-downs join via entity-graph BFS shortest paths;
- item_grain dimensions switch order-level calibers to item-level (avoid 1:N fan-out double count);
- combine parts are COALESCE(...,0)-wrapped so empty parts don't corrupt arithmetic;
- output SQL is gated by a sqlglot parse (dialect + syntax gate).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

import sqlglot

from data.semantic import (
    SemanticLayer,
    SemanticLayerError,
    build_join_chain,
)
from compile.types import DimensionFilter, SemanticQuery

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ALIAS_RE = re.compile(r"\b([a-z_][a-z_0-9]*)\.", re.ASCII)


class CompileError(RuntimeError):
    pass


def _month_bounds(ym: str) -> tuple[str, str]:
    if not MONTH_RE.match(ym):
        raise CompileError(f"非法月份格式：{ym!r}（应为 YYYY-MM）")
    year, month = int(ym[:4]), int(ym[5:7])
    if not 1 <= month <= 12:
        raise CompileError(f"非法月份：{ym!r}")
    last = calendar.monthrange(year, month)[1]
    return f"{ym}-01", f"{ym}-{last:02d}"


def _day_bounds(d: str) -> tuple[str, str]:
    if not DAY_RE.match(d):
        raise CompileError(f"非法日期格式：{d!r}（应为 YYYY-MM-DD）")
    try:
        date.fromisoformat(d)
    except ValueError as e:
        raise CompileError(f"非法日期：{d!r}") from e
    return d, d


def _bounds_for(win_type: str, value: str) -> tuple[str, str]:
    if win_type == "month":
        return _month_bounds(value)
    if win_type == "day":
        return _day_bounds(value)
    raise CompileError(f"不支持的窗口类型：{win_type!r}（应为 month 或 day）")


def _window_bounds(ym: str) -> tuple[str, str]:
    year, month = int(ym[:4]), int(ym[5:7])
    last = calendar.monthrange(year, month)[1]
    w_end = date(year, month, last)
    return (w_end - timedelta(days=89)).isoformat(), w_end.isoformat()


def _collect_aliases(*texts: str) -> set[str]:
    aliases: set[str] = set()
    for t in texts:
        aliases.update(ALIAS_RE.findall(t or ""))
    return aliases


def _build_from(layer: SemanticLayer, base: str, chain: list[tuple[str, str]]) -> str:
    parts = [f"{layer.resolve_table(base)} {base}"]
    for entity, rel_name in chain:
        rel = layer.relationships[rel_name]
        on_conds = " AND ".join(f"{a} = {b}" for a, b in rel["on"])
        parts.append(f"JOIN {layer.resolve_table(entity)} {entity} ON {on_conds}")
    return " ".join(parts)


def _dim_cond(dim_def: dict, f: DimensionFilter) -> str:
    col = f"{dim_def['table']}.{dim_def['column']}"
    if f.op == "=":
        value = f.value if isinstance(f.value, str) else f.value[0]
        return f"{col} = '{value}'"
    vals = ", ".join(f"'{v}'" for v in f.value)
    return f"{col} IN ({vals})"


def _validate_sql(sql: str) -> None:
    """sqlglot parse gate: syntax-invalid SQL fails compilation."""
    try:
        sqlglot.parse_one(sql, read="duckdb")
    except Exception as e:
        raise CompileError(f"编译器产出 SQL 未通过 sqlglot 语法校验: {e}")


def _qualify_combine(combine: str, part_names: list[str]) -> str:
    """Replace part names in the combine formula with COALESCE(CTE col, 0) to avoid empty-part pollution."""
    out = combine
    for name in sorted(part_names, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", f"COALESCE(p_{name}.{name}, 0)", out)
    return out


def _resolve_dims(layer: SemanticLayer, sq: SemanticQuery) -> list[tuple[str, dict]]:
    names = list(sq.dimensions) + [f.dim for f in sq.filters]
    seen: set[str] = set()
    result: list[tuple[str, dict]] = []
    for name in names:
        if name in seen:
            continue
        if name not in layer.dimensions:
            raise CompileError(f"未知维度：{name}")
        seen.add(name)
        result.append((name, layer.dimensions[name]))
    return result


def _part_base(layer: SemanticLayer, part: dict, fan_out: bool) -> str:
    return "order_items" if (fan_out and part.get("item_expression")) else part["base"]


def _part_expression(part: dict, fan_out: bool) -> str:
    return part.get("item_expression", part["expression"]) if fan_out else part["expression"]


def _build_part_cte(layer: SemanticLayer, name: str, part: dict,
                    dims: list[tuple[str, dict]], filters: list[DimensionFilter],
                    start: str, end: str, fan_out: bool) -> str:
    expr = _part_expression(part, fan_out)
    base = _part_base(layer, part, fan_out)
    group_by = part.get("group_by")
    dim_by_name = dict(dims)

    aliases = _collect_aliases(expr, part.get("time_column", ""),
                               *part.get("filters", []), group_by or "")
    required = set(aliases) | {base} | {d["table"] for _, d in dims}
    required.discard(base)
    chain = build_join_chain(layer.graph, base, required)
    from_clause = _build_from(layer, base, chain)

    conds = [f"({part['time_column']}::DATE BETWEEN DATE '{start}' AND DATE '{end}')"]
    conds += [f"({filt})" for filt in part.get("filters", [])]
    conds += [_dim_cond(dim_by_name[f.dim], f) for f in filters if f.dim in dim_by_name]
    where = " AND ".join(conds)

    if group_by:
        gk = group_by.split(".")[-1]
        select = f"{group_by} AS {gk}, {expr} AS {name}"
        return (f"p_{name} AS (SELECT {select} FROM {from_clause} "
                f"WHERE {where} GROUP BY {group_by})")
    return f"p_{name} AS (SELECT {expr} AS {name} FROM {from_clause} WHERE {where})"


def _compile_parts(layer: SemanticLayer, sq: SemanticQuery, meta: dict,
                   dims: list[tuple[str, dict]], fan_out: bool) -> str:
    start, end = _bounds_for(sq.window.type, sq.window.value)
    part_names = list(meta["parts"].keys())
    ctes = [_build_part_cte(layer, n, meta["parts"][n], dims, sq.filters,
                            start, end, fan_out) for n in part_names]
    qualified = _qualify_combine(meta["combine"], part_names)
    rd = meta.get("round_digits", 2)
    froms = ", ".join(f"p_{n}" for n in part_names)
    return (
        f"WITH {', '.join(ctes)}\n"
        f"SELECT COALESCE(round(({qualified})::DOUBLE, {rd}), 0) AS result\n"
        f"FROM {froms}"
    )


def _compile_window(layer: SemanticLayer, sq: SemanticQuery, meta: dict,
                    dims: list[tuple[str, dict]]) -> str:
    # window metrics (repurchase rate etc.) anchor a 90-day rolling window to month-end; day windows are refused
    if sq.window.type != "month":
        raise CompileError(f"窗口指标 {sq.metric} 仅支持月窗口，不支持 {sq.window.type} 窗口")
    w_start, w_end = _window_bounds(sq.window.value)
    base = "orders"
    required = {d["table"] for _, d in dims} - {base}
    chain = build_join_chain(layer.graph, base, required)
    from_clause = _build_from(layer, base, chain)
    conds = [
        f"({base}.created_at::DATE BETWEEN DATE '{w_start}' AND DATE '{w_end}')"
    ]
    conds += [_dim_cond(d, f) for f in sq.filters if (d := dict(dims)[f.dim])]
    where = " AND ".join(conds)
    rd = meta.get("round_digits", 4)
    return f"""
WITH w AS (
  SELECT {base}.user_id AS user_id, count(DISTINCT {base}.created_at::DATE) AS n_days
  FROM {from_clause}
  WHERE {where}
  GROUP BY {base}.user_id
)
SELECT COALESCE(round(
  (SELECT count(*)::DOUBLE FROM w WHERE n_days >= 2)
  / NULLIF((SELECT count(*)::DOUBLE FROM w), 0)
, {rd}), 0) AS result
""".strip()


def _compile_rank(layer: SemanticLayer, sq: SemanticQuery, meta: dict,
                  dims: list[tuple[str, dict]], fan_out: bool) -> str:
    start, end = _bounds_for(sq.window.type, sq.window.value)
    gk = meta["rank"]["group_key"]
    part_names = list(meta["parts"].keys())
    ctes = [_build_part_cte(layer, n, meta["parts"][n], dims, sq.filters,
                            start, end, fan_out) for n in part_names]
    qualified = _qualify_combine(meta["combine"], part_names)
    order = meta["rank"]["order"]
    limit = meta["rank"]["limit"]
    # align group parts on group_key (LEFT JOIN keeps no-refund/no-sales SKUs)
    joins = f"p_{part_names[0]}"
    for n in part_names[1:]:
        joins += f" LEFT JOIN p_{n} ON p_{part_names[0]}.{gk} = p_{n}.{gk}"
    return f"""
WITH {', '.join(ctes)}
SELECT p_{part_names[0]}.{gk} AS result
FROM {joins}
ORDER BY ({qualified}) {order}
LIMIT {limit}
""".strip()


def _compile_snapshot(layer: SemanticLayer, sq: SemanticQuery, meta: dict,
                      dims: list[tuple[str, dict]]) -> str:
    """kind=snapshot metric: point-in-time lookup (e.g. stockout SKU count at period end).

    meta must carry a snapshot config:
      entity:       snapshot entity name (e.g. inventory_snapshot)
      when:         value ('end'=last snapshot of window / 'latest'=latest snapshot in window)
      cond:         WHERE condition (refers to entity alias, e.g. "inventory_snapshot.on_hand_qty = 0")
      count_col / count_distinct: count mode (default count(*))
    Supports day windows or a single month (snapshot is point-in-time).
    """
    snap = meta.get("snapshot", {})
    entity = snap.get("entity")
    if not entity:
        raise CompileError(f"快照指标 {sq.metric} 缺少 snapshot.entity 配置")
    cond = snap.get("cond")
    if not cond:
        raise CompileError(f"快照指标 {sq.metric} 缺少 snapshot.cond 配置")

    # dimension reachability: snapshot entity must join to the dimension tables
    dim_tables = {d["table"] for _, d in dims}
    try:
        chain = build_join_chain(layer.graph, entity, dim_tables)
    except SemanticLayerError as e:
        raise CompileError(f"快照指标 {sq.metric} 不支持维度：{e}")
    from_clause = _build_from(layer, entity, chain)

    # time-point resolution: day = that day's snapshot; month = latest snapshot <= window end
    start, end = _bounds_for(sq.window.type, sq.window.value)
    snap_table = layer.resolve_table(entity)
    if sq.window.type == "day":
        date_cond = f"({entity}.snapshot_date = DATE '{start}')"
        # snapshot existence: no snapshot that day → HAVING filters group → empty (triggers reflect_empty)
        exist_cond = f"(SELECT count(*) FROM {snap_table} WHERE snapshot_date = DATE '{start}')"
    else:
        # month: latest snapshot day <= window end (weekly snapshots; avoids multi-snapshot double count)
        date_cond = (f"({entity}.snapshot_date = ("
                     f"SELECT max(snapshot_date) FROM {snap_table} "
                     f"WHERE snapshot_date::DATE <= DATE '{end}'))")
        exist_cond = f"(SELECT count(*) FROM {snap_table} WHERE snapshot_date::DATE <= DATE '{end}')"
    conds = [date_cond]
    conds.append(f"({cond})")
    conds += [_dim_cond(d, f) for f in sq.filters if (d := dict(dims)[f.dim])]
    where = " AND ".join(conds)

    inner = f"count(DISTINCT {entity}.{snap.get('count_col', 'sku_id')})"
    rd = meta.get("round_digits", 0)
    # HAVING snapshot-existence: no snapshot → filtered → empty result (honest, no silent 0)
    return f"""
SELECT COALESCE(round({inner}::DOUBLE, {rd}), 0) AS result
FROM {from_clause}
WHERE {where}
HAVING {exist_cond} > 0
""".strip()


def _compile_query(sq: SemanticQuery, layer: SemanticLayer) -> str:
    if sq.metric not in layer.metrics:
        raise CompileError(f"未知指标：{sq.metric}")
    meta = layer.metrics[sq.metric]
    dims = _resolve_dims(layer, sq)
    fan_out = any(d["item_grain"] for _, d in dims)

    # dimension reachability precheck: every part must reach all dimension tables, else illegal combo
    for name, part in meta.get("parts", {}).items():
        base = _part_base(layer, part, fan_out)
        dim_tables = {d["table"] for _, d in dims}
        for t in dim_tables:
            try:
                build_join_chain(layer.graph, base, {t})
            except SemanticLayerError as e:
                raise CompileError(f"指标 {sq.metric} 的 part「{name}」不支持维度：{e}")

    if "window" in meta:
        sql = _compile_window(layer, sq, meta, dims)
    elif "rank" in meta:
        sql = _compile_rank(layer, sq, meta, dims, fan_out)
    elif meta.get("kind") == "snapshot":
        sql = _compile_snapshot(layer, sq, meta, dims)
    else:
        sql = _compile_parts(layer, sq, meta, dims, fan_out)

    _validate_sql(sql)
    return sql


def compile_query(sq: SemanticQuery, layer: SemanticLayer) -> str:  # noqa: F811
    try:
        return _compile_query(sq, layer)
    except SemanticLayerError as e:
        raise CompileError(str(e)) from e