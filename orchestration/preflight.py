"""orchestration/preflight.py：semantic-query four-layer preflight (stage 3-3).

Runs deterministic checks after generate and before compile, blocking foreseeable failures and
sparing futile LLM retries / invalid compiles:

1. date boundary: window format / date validity;
2. enum dictionary: filter values must come from the semantic-layer value dictionary;
3. granularity fallback: dimension/metric combination legality (graph reachability), with a
   readable fallback suggestion (not a direct LLM fix);
4. cutoff: window past visible_data_cutoff → block + degrade (don't silently query future data).

Pure, zero LLM, unit-testable. Returns PreflightVerdict: pass or a blocking reason.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Literal

from compile.compiler import DAY_RE, MONTH_RE
from data.semantic import SemanticLayer, build_join_chain
from compile.types import SemanticQuery

VerdictKind = Literal["pass", "date", "enum", "granularity", "cutoff"]


@dataclass
class PreflightVerdict:
    kind: VerdictKind
    reason: str = ""
    suggestion: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == "pass"


def _month_end(ym: str) -> str:
    year, month = int(ym[:4]), int(ym[5:7])
    last = calendar.monthrange(year, month)[1]
    return f"{ym}-{last:02d}"


def _window_end(sq: SemanticQuery) -> str:
    if sq.window.type == "month":
        return _month_end(sq.window.value)
    return sq.window.value


def _check_date(sq: SemanticQuery) -> PreflightVerdict:
    v = sq.window.value
    if sq.window.type == "month":
        if not MONTH_RE.match(v):
            return PreflightVerdict("date", f"非法月份格式：{v!r}（应为 YYYY-MM）")
        year, month = int(v[:4]), int(v[5:7])
        if not 1 <= month <= 12:
            return PreflightVerdict("date", f"非法月份：{v!r}")
    else:
        if not DAY_RE.match(v):
            return PreflightVerdict("date", f"非法日期格式：{v!r}（应为 YYYY-MM-DD）")
        try:
            date.fromisoformat(v)
        except ValueError:
            return PreflightVerdict("date", f"非法日期：{v!r}")
    return PreflightVerdict("pass")


def _check_enum(layer: SemanticLayer, sq: SemanticQuery) -> PreflightVerdict:
    if layer is None:
        return PreflightVerdict("pass")  # skip when no layer
    for f in sq.filters:
        if f.dim not in layer.dimensions:
            # dimension itself absent → handled by granularity / compile; here only the value dict is checked
            continue
        allowed = set(str(x) for x in layer.dimensions[f.dim]["values"])
        if f.op == "in":
            vals = set(f.value)
            bad = vals - allowed
            if bad:
                return PreflightVerdict(
                    "enum",
                    f"维度 {f.dim} 的过滤值 {sorted(bad)} 不在值字典内",
                    f"可选值：{' / '.join(sorted(allowed))}")
        else:
            if str(f.value) not in allowed:
                return PreflightVerdict(
                    "enum",
                    f"维度 {f.dim} 的过滤值 {f.value!r} 不在值字典内",
                    f"可选值：{' / '.join(sorted(allowed))}")
    return PreflightVerdict("pass")


def _dim_tables(layer: SemanticLayer, sq: SemanticQuery) -> set[str]:
    """Dimension tables referenced by the query (dimensions + filters)."""
    tables = {layer.dimensions[d]["table"] for d in sq.dimensions}
    tables |= {layer.dimensions[f.dim]["table"] for f in sq.filters
               if f.dim in layer.dimensions}
    return tables


def _check_granularity(layer: SemanticLayer, sq: SemanticQuery) -> PreflightVerdict:
    """Dimension reachability: every part's base must join to all dimension tables.

    Same source as the compiler (build_join_chain), but pre-checks before compile and gives a
    readable fallback (switch/drop a dimension).
    """
    if layer is None:
        return PreflightVerdict("pass")
    unknown = [d for d in sq.dimensions if d not in layer.dimensions]
    unknown += [f.dim for f in sq.filters if f.dim not in layer.dimensions]
    if unknown:
        return PreflightVerdict(
            "granularity",
            f"存在语义层不支持的维度：{sorted(set(unknown))}",
            "请使用语义层定义的维度：品类 category_type / 城市线级 city_tier / "
            "渠道 channel / 渠道类型 channel_type")
    meta = layer.metrics.get(sq.metric)
    if meta is None:
        return PreflightVerdict("pass")  # unknown metric → compiler reports it
    if meta.get("kind") == "snapshot":
        # snapshot metric: check the snapshot entity can reach the dimension tables
        entity = meta.get("snapshot", {}).get("entity")
        if not entity:
            return PreflightVerdict("granularity",
                                    f"快照指标 {sq.metric} 缺少 snapshot.entity",
                                    "请联系数仓管理员补全快照指标配置")
        try:
            build_join_chain(layer.graph, entity, _dim_tables(layer, sq))
        except Exception as e:
            return PreflightVerdict(
                "granularity",
                f"指标 {sq.metric} 的快照实体 {entity} 不支持维度：{e}",
                "请去掉该维度或换用支持该维度的指标")
        return PreflightVerdict("pass")
    if "parts" not in meta:
        return PreflightVerdict("pass")  # window/rank metrics guarded by compiler
    for name, part in meta["parts"].items():
        base = part["base"]
        try:
            build_join_chain(layer.graph, base, _dim_tables(layer, sq))
        except Exception as e:
            return PreflightVerdict(
                "granularity",
                f"指标 {sq.metric} 的 part「{name}」不支持维度：{e}",
                "请去掉该维度或换用支持该维度的指标")
    return PreflightVerdict("pass")


def _check_cutoff(sq: SemanticQuery, cutoff: str | None) -> PreflightVerdict:
    """Window end past visible_data_cutoff → block (data not yet available, don't silently query empty)."""
    if not cutoff:
        return PreflightVerdict("pass")
    end = _window_end(sq)
    try:
        if date.fromisoformat(end) > date.fromisoformat(cutoff):
            return PreflightVerdict(
                "cutoff",
                f"查询窗口末日 {end} 超出可见数据截止 {cutoff}",
                f"请把窗口调整到 {cutoff} 或之前（数据截止日前）")
    except ValueError:
        return PreflightVerdict("date", f"窗口末日解析失败：{end!r}")
    return PreflightVerdict("pass")


def preflight(sq: SemanticQuery, layer: SemanticLayer,
              cutoff: str | None = None) -> PreflightVerdict:
    """Four-layer preflight, in order, first failure wins."""
    for v in (_check_date(sq),
              _check_enum(layer, sq),
              _check_granularity(layer, sq),
              _check_cutoff(sq, cutoff)):
        if not v.ok:
            return v
    return PreflightVerdict("pass")