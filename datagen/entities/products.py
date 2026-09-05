"""categories + products 商品维度（SPEC §5.1）。

- categories：50 叶子品类，大类分配 17/10/10/5/8；
- products：Dirichlet 分摊 SKU 到品类（约 60–160 个/品类，总量 5000），
  base_price 对数正态截断到子区间，cost_price = base × U(0.45, 0.65)，
  listing_status 96% 在售，listing_date 分布在数据期前 20 个月。
"""

import numpy as np
import pandas as pd

from datagen.entities.catalog_data import (
    CATEGORIES,
    ELECTRONICS_CATS,
    PRICE_RANGE,
    SKU_ATTRS,
    SKU_NOUNS,
    SUBCAT_PRICE,
)
from datagen.rng import make_rng


def _flatten_categories(n_cat: int) -> list[tuple[str, str, int]]:
    """[(category_name, category_type, 序号)]，CAT-01 起，截断到 n_cat 个。

    小规模配置品类少于常量表时按大类比例截断（逐大类均匀取前 k 个）。
    """
    rows = []
    if n_cat >= sum(len(v) for v in CATEGORIES.values()):
        i = 0
        for ctype, names in CATEGORIES.items():
            for name in names:
                i += 1
                rows.append((name, ctype, i))
        return rows
    # 按各大类品类数比例分配 n_cat
    total = sum(len(v) for v in CATEGORIES.values())
    alloc = {ct: max(1, round(n_cat * len(names) / total)) for ct, names in CATEGORIES.items()}
    # 修正到恰好 n_cat
    while sum(alloc.values()) > n_cat:
        biggest = max(alloc, key=alloc.get)
        if alloc[biggest] > 1:
            alloc[biggest] -= 1
        else:
            break
    extra = n_cat - sum(alloc.values())
    order = sorted(alloc, key=lambda ct: -len(CATEGORIES[ct]))
    for i in range(extra):
        alloc[order[i % len(order)]] += 1
    i = 0
    for ctype, names in CATEGORIES.items():
        for name in names[: alloc[ctype]]:
            i += 1
            rows.append((name, ctype, i))
    return rows


def gen_categories(cfg, rng=None) -> pd.DataFrame:
    rng = rng or make_rng(cfg.seed, "categories")
    rows = _flatten_categories(cfg.products.categories)
    return pd.DataFrame(
        [(f"CAT-{i:02d}", name, ctype) for name, ctype, i in rows],
        columns=["category_id", "category_name", "category_type"],
    )


def _sku_name(ctype: str, rng) -> str:
    attr = rng.choice(SKU_ATTRS[ctype])
    noun = rng.choice(SKU_NOUNS[ctype])
    return f"{attr}{noun}"


def gen_products(cfg, cats_df: pd.DataFrame, rng=None) -> pd.DataFrame:
    rng = rng or make_rng(cfg.seed, "products")
    cats = cats_df.to_dict("records")
    n_sku, n_cat = cfg.products.sku, cfg.products.categories

    # Dirichlet 分摊（确定性）：alpha=15 时品类件数 std≈25，落点约 60–160 个/品类
    alpha = np.full(n_cat, 15.0)
    shares = rng.dirichlet(alpha) * n_sku
    counts = np.maximum(np.floor(shares).astype(int), 1)
    # 差额调整（优先给末品类补齐，保证恰好 n_sku）
    diff = n_sku - counts.sum()
    if diff > 0:
        counts[np.argsort(-(shares - counts))[:diff]] += 1
    elif diff < 0:
        # 从份额最大的品类里减，避免负值
        idx = np.argsort(-(shares - counts))
        for i in idx:
            if counts[i] > 2:
                counts[i] -= 1
                diff += 1
                if diff == 0:
                    break

    records = []
    listing_start = pd.Timestamp(cfg.date_range.end) - pd.DateOffset(months=20)
    for ci, cat in enumerate(cats):
        n = counts[ci]
        lo, hi = SUBCAT_PRICE.get(cat["category_name"], PRICE_RANGE[cat["category_type"]])
        # 对数正态取到分位后线性插到 [lo, hi]（确定性）
        u = rng.random(n)
        logn = np.exp(rng.normal(0, 0.5, n))
        lq, hq = np.quantile(logn, [0.05, 0.95])
        span = hq - lq if hq > lq else 1.0
        price = lo + (hi - lo) * (logn - lq) / span
        price = np.clip(price, lo, hi)
        cost = price * rng.uniform(0.45, 0.65, n)
        listing = rng.random(n) < 0.96
        days = rng.integers(0, 610, n)  # 上架距今 0–20 个月
        reg = rng.integers(0, 10**8, n)
        for j in range(n):
            sku_id = f"SKU-{reg[j]:04d}"
            records.append((
                sku_id,
                _sku_name(cat["category_type"], rng),
                cat["category_id"],
                round(float(cost[j]), 2),
                round(float(price[j]), 2),
                "on_sale" if listing[j] else "off_sale",
                (listing_start + pd.Timedelta(days=int(days[j]))).date(),
            ))
    df = pd.DataFrame(records, columns=[
        "sku_id", "sku_name", "category_id", "cost_price",
        "base_price", "listing_status", "listing_date",
    ])
    return df.sort_values("sku_id").reset_index(drop=True)
