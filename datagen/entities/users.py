"""users 用户维度（SPEC §5.1 + 槽位2 已确认参数）。

- register_date：指数分布均值 240 天（截断到数据期内），无下单记录用户保持原样；
  有单用户的 register_date 在槽位2 末尾被钳到首单日前 U(1,45) 天（防未注册先下单）；
- city_tier：一线18/新一线27/二线25/三线18/四线及以下12，城市从对应档池均匀抽；
- age_band：22/38/24/11/5，与 city_tier 独立采样。
"""

import numpy as np
import pandas as pd

from datagen.entities.catalog_data import (
    AGE_BAND_WEIGHTS,
    CITY_POOLS,
    CITY_TIER_WEIGHTS,
)
from datagen.rng import make_rng

CITY_TIERS = ["一线", "新一线", "二线", "三线", "四线及以下"]


def gen_users(cfg, rng=None) -> pd.DataFrame:
    rng = rng or make_rng(cfg.seed, "users")
    n = cfg.users.count
    end = pd.Timestamp(cfg.date_range.end)
    span = (pd.Timestamp(cfg.date_range.start), end)

    # register_date：距数据期末指数分布均值 240 天，截断到数据期内
    age_days = rng.exponential(240.0, n).astype(int)
    reg_offsets = np.clip(age_days, 0, (end - span[0]).days)
    reg_dates = pd.Series(end - pd.to_timedelta(reg_offsets, unit="D"))

    # 大促注册尖峰：以 0.15 概率把注册日移到某个促销日 ±1 天
    if cfg.promo_days and rng.random() < 1.0:
        promo_mask = rng.random(n) < 0.15
        cand = []
        for p in cfg.promo_days:
            d = pd.Timestamp(p.date)
            for off in (-1, 0, 1):
                t = d + pd.Timedelta(days=off)
                if span[0] <= t <= end:
                    cand.append(t)
        if cand and promo_mask.any():
            cand_arr = pd.DatetimeIndex(cand)
            choice = rng.integers(0, len(cand_arr), int(promo_mask.sum()))
            reg_dates.loc[promo_mask] = cand_arr[choice].values

    # city_tier 加权抽样
    tiers = rng.choice(CITY_TIERS, n, p=[CITY_TIER_WEIGHTS[t] for t in CITY_TIERS])
    cities = np.empty(n, dtype=object)
    for t in CITY_TIERS:
        mask = tiers == t
        cities[mask] = rng.choice(CITY_POOLS[t], mask.sum())

    age_bands = rng.choice(list(AGE_BAND_WEIGHTS), n, p=list(AGE_BAND_WEIGHTS.values()))

    user_ids = np.array([f"u_{i:08d}" for i in range(1, n + 1)])
    return pd.DataFrame({
        "user_id": user_ids,
        "register_date": reg_dates.dt.date,
        "city_tier": tiers,
        "city": cities,
        "age_band": age_bands,
    })


def clamp_register_for_orderers(users: pd.DataFrame, first_order_dates: pd.Series, rng=None) -> None:
    """有单用户 register_date 钳到 首单日 − U(1,45) 天（就地修改）。

    first_order_dates 为 DataFrame 行号 → 首单日期 的 Series；无单用户不动。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mask = first_order_dates.notna()
    if not mask.any():
        return
    back = rng.integers(1, 46, mask.sum())
    new_reg = pd.to_datetime(first_order_dates[mask]) - pd.to_timedelta(back, unit="D")
    users.loc[mask, "register_date"] = new_reg.dt.date
