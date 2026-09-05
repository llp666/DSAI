"""订单域生成（SPEC §5.2 + 槽位2 已确认口径）。

口径（已确认）：
- 订单状态按终态快照独立采样：4已完成78 / 5已退款10 / 2已支付7 / 3已发货4 / 1待支付1；
- paid_share=0.30 是广告归因订单占比（营销域参数），不影响本模块状态分布；
- 有单用户 register_date 由流水线在 orders 产出后钳制（users.clamp_register_for_orderers）；
- order_payments.paid_at 与 status_log 的 (1→2) 行使用同一支付延迟，保证勾稽一致；
- refunds 订单级判定（命中订单退全部明细），partial 全额30%/部分U(lo,1.0)，
  refund_date 钳制在数据期内（净销/毛销 ≈ 0.85，SPEC [0.75,0.88]）。

全程 numpy 向量化；唯一允许的循环是日级小循环（残差对齐 <300 次）。
"""

import numpy as np
import pandas as pd

from datagen.rng import make_rng

# 终态快照状态分布（与 paid_share 无关，第3问定稿）
STATUS_CHOICES = np.array(["1", "2", "3", "4", "5"])
STATUS_PROBS = np.array([0.01, 0.07, 0.04, 0.78, 0.10])

PAY_CHANNELS = ["alipay", "wechat_pay", "union_pay", "credit_card"]
PAY_CHANNEL_PROBS = [0.42, 0.38, 0.12, 0.08]

# 退款原因按品类大类分布（服饰以尺码不符为主，SPEC §5.2）
REASONS = {
    "服饰": (["尺码不符", "质量问题", "七天无理由", "不想要了", "色差/与描述不符", "发错货"],
             [0.35, 0.15, 0.20, 0.15, 0.10, 0.05]),
    "电子": (["质量问题", "七天无理由", "与描述不符", "不想要了", "物流损坏", "发错货"],
             [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]),
    "其他": (["七天无理由", "不想要了", "质量问题", "与描述不符", "发错货"],
             [0.30, 0.28, 0.20, 0.15, 0.07]),
}


def _daily_counts(cfg, rng) -> tuple[np.ndarray, np.ndarray]:
    """日单量 = base × dow系数 × 促销系数 × lognormal 噪声。

    返回 (counts, promo_mask)；末 tail_days 天 × tail_keep（迟到分区）；
    最终把总量对齐到 cfg.orders.count（残差摊到非促销日）。
    """
    dates = pd.date_range(cfg.date_range.start, cfg.date_range.end, freq="D")
    base = cfg.orders.count / len(dates)
    dow_f = np.array(cfg.seasonality.dow)[dates.dayofweek]

    promo_f = np.ones(len(dates))
    promo_mask = np.zeros(len(dates), dtype=bool)
    for p in cfg.promo_days:
        d = pd.Timestamp(p.date)
        for off in (-1, 0, 1):
            idx = np.where(dates == d + pd.Timedelta(days=off))[0]
            promo_f[idx] = np.maximum(promo_f[idx], p.factor ** (0.5 if off else 1.0))
            promo_mask[idx] = True

    noise = rng.lognormal(0, cfg.seasonality.noise_sigma, len(dates))
    counts = np.round(base * dow_f * promo_f * noise).astype(int)

    tail = cfg.dirty.late_arrival.tail_days
    counts[-tail:] = (counts[-tail:] * cfg.dirty.late_arrival.tail_keep).astype(int)

    # 总量守恒：非促销日按比例缩放，残差逐个微调（残差 ≤ 数百）
    diff = cfg.orders.count - counts.sum()
    if diff != 0:
        non_promo = np.where(~promo_mask)[0]
        cur = counts[non_promo].sum()
        tgt = cur + diff
        if cur > 0 and tgt > len(non_promo):
            counts[non_promo] = np.maximum(
                np.round(counts[non_promo] * tgt / cur).astype(int), 1
            )
        resid = cfg.orders.count - counts.sum()
        order_idx = non_promo[np.argsort(-counts[non_promo])]
        k = 0
        while resid != 0:
            i = order_idx[k % len(order_idx)]
            if resid > 0:
                counts[i] += 1
                resid -= 1
            elif counts[i] > 1:
                counts[i] -= 1
                resid += 1
            k += 1
    return counts, promo_mask


def _gen_orders_items(cfg, users: pd.DataFrame, products: pd.DataFrame):
    """orders + order_items 主体。返回 (orders, items)。"""
    rng_o = make_rng(cfg.seed, "orders")
    rng_i = make_rng(cfg.seed, "order_items")

    dates = pd.date_range(cfg.date_range.start, cfg.date_range.end, freq="D")
    counts, promo_mask = _daily_counts(cfg, rng_o)
    n_orders = int(counts.sum())

    # 下单用户：heavy/light 混合权重（65/35 划分，heavy 2.5 / light 0.3）
    n_users = len(users)
    heavy = rng_o.random(n_users) < cfg.repeat_mix.heavy_share
    w = np.where(heavy, cfg.repeat_mix.heavy_weight, cfg.repeat_mix.light_weight)
    w /= w.sum()
    user_idx = rng_o.choice(n_users, size=n_orders, p=w)

    order_ids = np.array([f"o_{i:08d}" for i in range(1, n_orders + 1)])
    order_day = np.repeat(np.arange(len(dates)), counts)
    order_dates = dates[order_day]

    # 下单时刻：日内小时非均匀（晚高峰 19–21 时）
    hour_w = np.array([0.5, 0.8, 1.2, 1.5, 1.3, 1.2, 1.1, 1.0, 1.2, 1.5, 1.8, 2.0,
                       2.2, 2.0, 1.8, 1.6, 1.8, 2.2, 2.6, 2.8, 2.5, 2.0, 1.2, 0.6])
    hour_w /= hour_w.sum()
    hours = rng_o.choice(24, n_orders, p=hour_w)
    minutes = rng_o.integers(0, 60, n_orders)
    seconds = rng_o.integers(0, 60, n_orders)
    created_at = (order_dates.values.astype("datetime64[D]")
                  + hours.astype("timedelta64[h]")
                  + minutes.astype("timedelta64[m]")
                  + seconds.astype("timedelta64[s]"))

    # 每单行数：Poisson(avg_items) 截断 [1,10]，促销日 ×1.3
    promo_rep = promo_mask[order_day].astype(float)
    lam = np.where(promo_rep > 0, cfg.orders.avg_items * 1.3, cfg.orders.avg_items)
    n_items = np.clip(rng_i.poisson(lam), 1, 10)

    # SKU 选择：Zipf 权重（头部 SKU 占半数销量）
    sku_w = 1.0 / np.arange(1, len(products) + 1, dtype=float) ** cfg.sku_zipf.exponent
    sku_w /= sku_w.sum()
    total_items = int(n_items.sum())
    sku_idx = rng_i.choice(len(products), size=total_items, p=sku_w)

    base_price = products["base_price"].to_numpy(dtype=float)
    noise = rng_i.lognormal(0, 0.08, total_items)
    promo_disc = np.where(
        np.repeat(promo_rep, n_items) > 0,
        rng_i.uniform(0.85, 0.95, total_items),
        1.0,
    )
    sale_price = np.round(base_price[sku_idx] * noise * promo_disc, 2)
    qty = np.clip(rng_i.poisson(1.3, total_items), 1, 10)

    item_order_rep = np.repeat(np.arange(n_orders), n_items)
    items = pd.DataFrame({
        "order_id": order_ids[item_order_rep],
        "sku_id": products["sku_id"].to_numpy()[sku_idx],
        "qty": qty,
        "sale_price": sale_price,
    })

    # pay_amount = sum(sale_price × qty) 按单汇总（§13 坑6：round(2) 后入库）
    item_amt = sale_price * qty
    pay_amount = np.zeros(n_orders)
    np.add.at(pay_amount, item_order_rep, item_amt)
    pay_amount = np.round(pay_amount, 2)

    status = rng_o.choice(STATUS_CHOICES, n_orders, p=STATUS_PROBS)

    orders = pd.DataFrame({
        "order_id": order_ids,
        "user_id": users["user_id"].to_numpy()[user_idx],
        "order_status": status,
        "pay_amount": pay_amount,
        "created_at": created_at,
    })
    return orders, items


def _build_status_log(order_ids, status, created_at, pay_delay_sec, rng_s, end_ns=None):
    """向量化状态流转日志。行设计：
    每单先有一行创建（from=NULL, to=1，对应 DDL 注释"首行为空"），
    已支付补 (1->2)，已发货补 (2->3)，完成/退款补 (3->4|5)。
    pay_delay_sec：已支付单的支付延迟秒数（与 order_payments.paid_at 同源）。
    end_ns：数据期末时间戳（np.datetime64），所有事件时间钳制到其内。
    """
    n = len(order_ids)
    created = created_at.astype("datetime64[s]")
    paid = status != "1"
    shipped = np.isin(status, ["3", "4", "5"])
    finished = np.isin(status, ["4", "5"])

    pay_sec = np.where(paid, pay_delay_sec, 0)
    t_pay = created + pay_sec.astype("timedelta64[s]")
    t_ship = t_pay + (rng_s.uniform(0.5, 2.0, n) * 86400).astype("timedelta64[s]")
    t_fin = t_ship + (rng_s.uniform(1.0, 5.0, n) * 86400).astype("timedelta64[s]")
    if end_ns is not None:
        t_pay = np.minimum(t_pay, end_ns)
        t_ship = np.minimum(t_ship, end_ns)
        t_fin = np.minimum(t_fin, end_ns)

    segs = [
        (np.ones(n, bool), None, "1", created),
        (paid, "1", "2", t_pay),
        (shipped, "2", "3", t_ship),
        (finished, "3", np.where(status == "5", "5", "4"), t_fin),
    ]

    froms, tos, tss, oids = [], [], [], []
    for mask, f, t, ts in segs:
        m = mask
        oids.append(order_ids[m])
        froms.append(np.full(m.sum(), f, dtype=object) if f is not None
                     else np.full(m.sum(), None, dtype=object))
        tos.append(t[m] if isinstance(t, np.ndarray) else np.full(m.sum(), t, dtype=object))
        tss.append(ts[m])

    log_df = pd.DataFrame({
        "order_id": np.concatenate(oids),
        "from_status": np.concatenate(froms),
        "to_status": np.concatenate(tos),
        "changed_at": np.concatenate(tss),
    })
    log_df = log_df.sort_values(["order_id", "changed_at"], kind="stable").reset_index(drop=True)
    log_df["log_id"] = [f"sl_{i:08d}" for i in range(1, len(log_df) + 1)]
    return log_df


def gen_orders_full(cfg, users: pd.DataFrame, products: pd.DataFrame, cats_df: pd.DataFrame):
    """订单域一站式生成。返回 (orders, items, refunds, payments, status_log)。"""
    rng_r = make_rng(cfg.seed, "refunds")
    rng_p = make_rng(cfg.seed, "order_payments")
    rng_s = make_rng(cfg.seed, "status_log")

    orders, items = _gen_orders_items(cfg, users, products)
    n_orders = len(orders)

    status_arr = orders["order_status"].to_numpy()
    created_arr = orders["created_at"].to_numpy().astype("datetime64[s]")
    end_ns = np.datetime64(pd.Timestamp(cfg.date_range.end).normalize()
                           + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), "s")

    # 支付延迟 3–15 分钟（同一延迟喂 payments.paid_at 与 status_log (1->2)）
    pay_delay = rng_p.integers(180, 900, n_orders).astype(np.int64)

    status_log = _build_status_log(orders["order_id"].to_numpy(), status_arr,
                                   created_arr, pay_delay, rng_s, end_ns)

    # updated_at = 最后一次状态变更时间，直接取 status_log 每单末行（保证勾稽一致）
    last_ts = status_log.groupby("order_id")["changed_at"].max()
    orders["updated_at"] = last_ts.reindex(orders["order_id"]).to_numpy()

    # ---- order_payments：已支付单 1:1，paid_at 钳制到数据期末 ----
    paid = status_arr != "1"
    pay_channel = rng_p.choice(PAY_CHANNELS, n_orders, p=PAY_CHANNEL_PROBS)
    paid_at = np.minimum(created_arr[paid] + pay_delay[paid].astype("timedelta64[s]"),
                         end_ns)
    payments = pd.DataFrame({
        "payment_id": np.array([f"p_{i:08d}" for i in range(1, int(paid.sum()) + 1)]),
        "order_id": orders["order_id"].to_numpy()[paid],
        "pay_channel": pay_channel[paid],
        "pay_amount": orders["pay_amount"].to_numpy()[paid],
        "paid_at": paid_at,
    })

    # ---- refunds：仅已完成/已退款单，订单级判定 → 命中订单退其全部明细 ----
    # 口径（槽位2 已确认 + 槽位3 修正）：行级随机判定导致退款金额占比 ~8%，
    # 净销/毛销 0.92 超出 SPEC [0.75, 0.88]。改为订单级判定（概率按首行品类），
    # 命中订单退全部行，partial 全额退 60% / 部分退 U(lo,1.0) → 退款率 ~13%、净销 ~0.87。
    sku2cat = products.set_index("sku_id")["category_id"]
    cat2type = cats_df.set_index("category_id")["category_type"]
    items_cat_type = items["sku_id"].map(sku2cat).map(cat2type).to_numpy()

    item_order_rep = pd.factorize(items["order_id"], sort=True)[0]
    status_rep = status_arr[item_order_rep]

    eligible = np.isin(status_rep, ["4", "5"])
    p_arr = np.full(len(items), cfg.refund.default, dtype=float)
    p_arr[items_cat_type == "服饰"] = cfg.refund.apparel
    p_arr[items_cat_type == "电子"] = cfg.refund.electronics
    p_arr[~eligible] = 0.0

    # 订单级命中：每单概率取首行品类，rng 逐单判定
    first_line = pd.Series(np.arange(len(items))).groupby(item_order_rep).first().to_numpy()
    order_p = p_arr[first_line]
    n_orders_rep = order_p.size
    hit_order = rng_r.random(n_orders_rep) < order_p
    hit = hit_order[item_order_rep] & eligible  # 命中订单的所有 eligible 行

    # 退款延迟：2 + lognormal(mu-2, sigma) 天（SPEC: delay_lognormal=[2.0, 0.8]），
    # 钳制到数据期内（refund_date 不得越出 date_range.end）
    mu, sigma = cfg.refund.delay_lognormal
    delay = 2 + rng_r.lognormal(mu - 2, sigma, len(items))
    created_rep = created_arr[item_order_rep]
    refund_ts = pd.Series(pd.to_datetime(created_rep)) + pd.to_timedelta(np.round(delay), unit="D")
    d_end = pd.Timestamp(cfg.date_range.end).normalize()
    refund_ts = refund_ts.clip(upper=d_end)

    # partial：60% 全额退、40% 部分退 U(lo, 1.0)，lo 取配置 partial_range 下限
    lo, hi = cfg.refund.partial_range
    is_full = rng_r.random(len(items)) < 0.60
    partial = np.where(
        is_full, hi,
        rng_r.uniform(lo, hi, len(items)),
    )
    item_amt = items["sale_price"].to_numpy() * items["qty"].to_numpy()
    refund_amount = np.round(item_amt * partial, 2)

    reasons = np.full(len(items), "七天无理由", dtype=object)
    for rtype, (names, probs) in REASONS.items():
        m = items_cat_type == rtype
        if m.any():
            reasons[m] = rng_r.choice(names, int(m.sum()), p=probs)

    refunds = pd.DataFrame({
        "refund_id": np.array([f"r_{i:08d}" for i in range(1, int(hit.sum()) + 1)]),
        "order_id": items["order_id"].to_numpy()[hit],
        "sku_id": items["sku_id"].to_numpy()[hit],
        "refund_amount": refund_amount[hit],
        "refund_reason": reasons[hit],
        "refund_date": refund_ts[hit].dt.date,
    }).reset_index(drop=True)

    return orders, items, refunds, payments, status_log
