"""agent/compiler.py：确定性编译器 v2（纯函数）。

把语义查询 SemanticQuery 展开为精确 SQL：
- 11 指标由语义层 metrics 的 parts/combine 结构驱动；
- 维度下钻经关系图 BFS 自动补全 JOIN 路径；
- 命中 item_grain 维度时订单级口径自动切明细级（避免 1:N 扇出重复求和）；
- 组合指标 parts 值经 COALESCE(...,0) 保证空部分不污染算术；
- 产出 SQL 用 sqlglot 做 parse 校验（方言适配 + 语法门禁）。
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from pathlib import Path

import sqlglot

from .semantic_layer import (
    SemanticLayer,
    SemanticLayerError,
    build_join_chain,
)
from .types import DimensionFilter, SemanticQuery

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
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
    """sqlglot parse 门禁：语法非法即编译失败。"""
    try:
        sqlglot.parse_one(sql, read="duckdb")
    except Exception as e:
        raise CompileError(f"编译器产出 SQL 未通过 sqlglot 语法校验: {e}")


def _qualify_combine(combine: str, part_names: list[str]) -> str:
    """把 combine 公式中的 part 名替换为 COALESCE(CTE 列, 0)，保证空部分不污染算术。"""
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
    start, end = _month_bounds(sq.window.value)
    part_names = list(meta["parts"].keys())
    ctes = [_build_part_cte(layer, n, meta["parts"][n], dims, sq.filters,
                            start, end, fan_out) for n in part_names]
    qualified = _qualify_combine(meta["combine"], part_names)
    rd = meta.get("round_digits", 2)
    froms = ", ".join(f"p_{n}" for n in part_names)
    sql = (
        f"WITH {', '.join(ctes)}\n"
        f"SELECT COALESCE(round(({qualified})::DOUBLE, {rd}), 0) AS result\n"
        f"FROM {froms}"
    )
    return sql


def _compile_window(layer: SemanticLayer, sq: SemanticQuery, meta: dict,
                    dims: list[tuple[str, dict]]) -> str:
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
    start, end = _month_bounds(sq.window.value)
    gk = meta["rank"]["group_key"]
    part_names = list(meta["parts"].keys())
    ctes = [_build_part_cte(layer, n, meta["parts"][n], dims, sq.filters,
                            start, end, fan_out) for n in part_names]
    qualified = _qualify_combine(meta["combine"], part_names)
    order = meta["rank"]["order"]
    limit = meta["rank"]["limit"]
    # 按 group_key 对齐各分组 part（LEFT JOIN 保证无退款/无销量的 sku 不丢）
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


def _compile_query(sq: SemanticQuery, layer: SemanticLayer) -> str:
    if sq.metric not in layer.metrics:
        raise CompileError(f"未知指标：{sq.metric}")
    meta = layer.metrics[sq.metric]
    dims = _resolve_dims(layer, sq)
    fan_out = any(d["item_grain"] for _, d in dims)

    # 维度可达性预检：每个 part 必须能到达所有维度表，否则该组合非法
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
    else:
        sql = _compile_parts(layer, sq, meta, dims, fan_out)

    _validate_sql(sql)
    return sql


def compile_query(sq: SemanticQuery, layer: SemanticLayer) -> str:  # noqa: F811
    try:
        return _compile_query(sq, layer)
    except SemanticLayerError as e:
        raise CompileError(str(e)) from e
