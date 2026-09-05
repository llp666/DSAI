"""配置加载：YAML → dataclass（SPEC §4.2）。

每个行业锚点（退货率、转化率、DOI 区间）都是显式参数——数据即文档论证的一部分。
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# 随机流模块注册表：每个生成函数以 make_rng(seed, "<name>") 起手，
# 模块名必须登记在此，保证随机流派生点可枚举、可审计。
MODULES = [
    "dim_date",
    "categories",
    "products",
    "users",
    "channels",
    "orders",
    "order_items",
    "refunds",
    "order_payments",
    "status_log",
    "coupons",
    "user_profiles",
    "user_level_log",
    "ads",
    "ad_conversions",
    "traffic_events",
    "search",
    "supply",
    "inventory",
    "warehouses",
    "dirty",
]


@dataclass
class DateRange:
    start: str
    end: str


@dataclass
class UsersCfg:
    count: int


@dataclass
class ProductsCfg:
    sku: int
    categories: int


@dataclass
class OrdersCfg:
    count: int
    avg_items: float
    paid_share: float


@dataclass
class Seasonality:
    dow: list[float]
    noise_sigma: float


@dataclass
class PromoDay:
    date: str
    factor: float


@dataclass
class SkuZipf:
    exponent: float


@dataclass
class RepeatMix:
    heavy_share: float
    heavy_weight: float
    light_weight: float


@dataclass
class RefundCfg:
    default: float
    apparel: float
    electronics: float
    delay_lognormal: list[float]
    partial_range: list[float]


@dataclass
class AdsCfg:
    channels: int
    cpm: list[float]
    ctr: list[float]
    cvr: list[float]
    attribution: float


@dataclass
class SupplyCfg:
    lead_days: list[int]
    reorder_coverage: int


@dataclass
class InventoryCfg:
    snapshot_grain: str


@dataclass
class DemoCases:
    stockout_skus: list[str]
    dead_stock_skus: list[str]
    negative_roi_channels: list[str]


@dataclass
class EnumDrift:
    rate: float
    partition: str
    mapping: dict[str, str]


@dataclass
class TzShift:
    partition: str
    hours: int


@dataclass
class LateArrival:
    rate: float
    tail_days: int
    tail_keep: float


@dataclass
class DirtyCfg:
    null_rate: float
    enum_drift: EnumDrift
    tz_shift: TzShift
    late_arrival: LateArrival


@dataclass
class OptionalCfg:
    traffic_events: bool
    search_logs: bool


@dataclass
class Config:
    seed: int
    date_range: DateRange
    users: UsersCfg
    products: ProductsCfg
    orders: OrdersCfg
    seasonality: Seasonality
    promo_days: list[PromoDay]
    sku_zipf: SkuZipf
    repeat_mix: RepeatMix
    refund: RefundCfg
    ads: AdsCfg
    supply: SupplyCfg
    inventory: InventoryCfg
    demo_cases: DemoCases
    dirty: DirtyCfg
    optional: OptionalCfg


# YAML 键 → dataclass 构造器。新增配置段时在此注册，
# 未注册的段直接报错而不是静默忽略（防拼写错误吞配置）。
_BUILDERS = {
    "date_range": DateRange,
    "users": UsersCfg,
    "products": ProductsCfg,
    "orders": OrdersCfg,
    "seasonality": Seasonality,
    "sku_zipf": SkuZipf,
    "repeat_mix": RepeatMix,
    "refund": RefundCfg,
    "ads": AdsCfg,
    "supply": SupplyCfg,
    "inventory": InventoryCfg,
    "demo_cases": DemoCases,
    "optional": OptionalCfg,
}


def _build_dirty(d: dict) -> DirtyCfg:
    return DirtyCfg(
        null_rate=d["null_rate"],
        enum_drift=EnumDrift(**d["enum_drift"]),
        tz_shift=TzShift(**d["tz_shift"]),
        late_arrival=LateArrival(**d["late_arrival"]),
    )


def load_config(path: str | Path) -> Config:
    """加载并校验 YAML 配置。未知字段/缺失字段抛 ValueError。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    known = set(_BUILDERS) | {"seed", "promo_days", "dirty"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"配置文件存在未识别字段: {sorted(unknown)}")

    kwargs = {"seed": int(raw["seed"])}
    for key, builder in _BUILDERS.items():
        if key not in raw:
            raise ValueError(f"配置缺少必填字段: {key}")
        kwargs[key] = builder(**raw[key])
    kwargs["promo_days"] = [PromoDay(**p) for p in raw["promo_days"]]
    kwargs["dirty"] = _build_dirty(raw["dirty"])
    return Config(**kwargs)


def dataclasses_asdict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
