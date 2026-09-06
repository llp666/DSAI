"""agent/preflight.py：语义查询四层预检（阶段 3-3）。

在 generate 产出语义查询后、编译前做确定性校验，拦截可预见的失败，
避免无效 LLM 重试与无效编译：

1. 日期边界（date）：window 格式/日期合法性；
2. 枚举字典（enum）：filter 维度值必须来自语义层值字典；
3. 粒度回退（granularity）：维度/指标组合合法性（关系图可达性），
   非法时给出回退建议（不直接走 LLM 修复）；
4. 时效 cutoff（cutoff）：窗口超出 visible_data_cutoff → 拦截降级，
   不静默查「数据还没到」的窗口。

纯函数、零 LLM、可单测。返回 PreflightVerdict：pass 或各类拦截原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .compiler import DAY_RE, MONTH_RE
from .semantic_layer import SemanticLayer, build_join_chain
from .types import SemanticQuery

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
    import calendar

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
        return PreflightVerdict("pass")  # 无语义层时跳过
    for f in sq.filters:
        if f.dim not in layer.dimensions:
            # 维度本身不存在 → 由 granularity 层处理（或 compile 报错），此处只查值字典
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


def _check_granularity(layer: SemanticLayer, sq: SemanticQuery) -> PreflightVerdict:
    """维度可达性预检：每个 part 的 base 必须能 JOIN 到所有维度表。

    与编译器同源（build_join_chain），但独立于编译时机提前拦截，
    并给出可读的回退建议（换维度/去维度）。
    """
    if layer is None:
        return PreflightVerdict("pass")  # 无语义层时跳过（日期/枚举/cutoff 不依赖 layer）
    # 未知维度：编译器会报「未知维度」引用错（repair 轨可修），预检提前拦截给回退建议
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
        return PreflightVerdict("pass")  # 指标未知由编译器报错
    if meta.get("kind") == "snapshot":
        # 快照指标：检查快照实体到维度表可达
        entity = meta.get("snapshot", {}).get("entity")
        if not entity:
            return PreflightVerdict("granularity",
                                    f"快照指标 {sq.metric} 缺少 snapshot.entity",
                                    "请联系数仓管理员补全快照指标配置")
        dim_tables = {layer.dimensions[d]["table"] for d in sq.dimensions}
        dim_tables |= {layer.dimensions[f.dim]["table"] for f in sq.filters
                       if f.dim in layer.dimensions}
        try:
            build_join_chain(layer.graph, entity, dim_tables)
        except Exception as e:
            return PreflightVerdict(
                "granularity",
                f"指标 {sq.metric} 的快照实体 {entity} 不支持维度：{e}",
                "请去掉该维度或换用支持该维度的指标")
        return PreflightVerdict("pass")
    if "parts" not in meta:
        return PreflightVerdict("pass")  # window/rank 指标由编译守卫处理
    dim_tables = {layer.dimensions[d]["table"] for d in sq.dimensions}
    dim_tables |= {layer.dimensions[f.dim]["table"] for f in sq.filters
                   if f.dim in layer.dimensions}
    for name, part in meta["parts"].items():
        base = part["base"]
        try:
            build_join_chain(layer.graph, base, dim_tables)
        except Exception as e:
            return PreflightVerdict(
                "granularity",
                f"指标 {sq.metric} 的 part「{name}」不支持维度：{e}",
                "请去掉该维度或换用支持该维度的指标")
    return PreflightVerdict("pass")


def _check_cutoff(sq: SemanticQuery, cutoff: str | None) -> PreflightVerdict:
    """窗口末日超出 visible_data_cutoff → 拦截（数据未到/不可见，不静默查空）。"""
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
    """四层预检，按顺序拦截，返回第一个失败。"""
    for v in (_check_date(sq),
              _check_enum(layer, sq),
              _check_granularity(layer, sq),
              _check_cutoff(sq, cutoff)):
        if not v.ok:
            return v
    return PreflightVerdict("pass")
