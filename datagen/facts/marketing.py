"""营销域生成（SPEC §5.3 + 槽位3 已确认参数）。

- 渠道强度分层：search 1.0 / feeds 0.7 / cps 0.4 / sms 0.25；organic 不参与广告统计；
- 归因订单：从状态≠待支付的订单中抽 cfg.orders.paid_share（≈600k×30%=180k 行），
  converted_at = 支付时间（归因到支付日，保证月度 channel_roi 可与订单 GMV 对账）；
- attributed_gmv = 订单实付 × cfg.ads.attribution × U(0.9,1.1)；
  ads_daily_stats 由 ad_conversions 按 campaign×日精确聚合（conversions / attributed_gmv），
  曝光/点击/花费按 campaign 漏斗参数（CTR/CVR/CPM）自转化数反推；
- 负 ROI 渠道 channel_cps_03：高 CPM(45–50) + 低 CTR/CVR → ROI < 1（demo 埋点）；
- 全局校准：若总 ROI 超出 [2.5,3.5]，对 spend 施加统一标量缩放使总 ROI≈3.0（落在 SPEC [1.5,4]）。
"""

import numpy as np
import pandas as pd

from datagen.rng import make_rng

# 渠道强度（按类型），与 DDL comment 的渠道类型值一致
CHANNEL_STRENGTH = {"feeds": 0.7, "search": 1.0, "cps": 0.4, "sms": 0.25}

# 各渠道类型的漏斗参数采样区间（均落在配置范围 cpm[5,50]/ctr[0.008,0.05]/cvr[0.015,0.04] 内）
FUNNEL = {
    "feeds":  dict(cpm=(12, 28), ctr=(0.015, 0.030), cvr=(0.020, 0.030)),
    "search": dict(cpm=(12, 24), ctr=(0.025, 0.045), cvr=(0.028, 0.040)),
    "cps":    dict(cpm=(8, 18),  ctr=(0.010, 0.018), cvr=(0.015, 0.022)),
    "sms":    dict(cpm=(5, 12),  ctr=(0.008, 0.015), cvr=(0.015, 0.020)),
}
NEG_FUNNEL = dict(cpm=(45, 50), ctr=(0.008, 0.008), cvr=(0.015, 0.015))

TARGET_ROI = 3.0  # 校准目标；可接受区间 [2.5, 3.5] ⊂ SPEC [1.5, 4]
NEG_SPEND_MULT = 1.8  # 负 ROI 渠道 spend = gmv × 1.8 → ROI ≈ 0.56（demo 埋点）

# 全站行为漏斗（SPEC §5.5 锚点：全站转化率 2–3%）
# 曝光→点击 0.15、点击→加购 0.20、加购→支付 0.85 → pay/impression ≈ 2.55%
TRAFFIC_FUNNEL = dict(ctr=0.15, cvr=0.20, pcr=0.85)
TRAFFIC_COVER = 0.105       # 覆盖订单比例 → full 约 300 万事件行（SPEC §3.2）
TRAFFIC_GUEST_SHARE = 0.20  # 曝光事件中游客（NULL user_id）占比（DDL 注释约定）

# 大促活动维度（与 promo_days 配置对齐的主要节点 + 补充购物节叙事）
PROMOTIONS = [
    ("promo_01", "618年中大促", "2025-06-01", "2025-06-20", "年中核心大促，流量与转化显著抬升"),
    ("promo_02", "818购物节", "2025-08-01", "2025-08-20", "平台级购物节，服饰/家居品类促销"),
    ("promo_03", "双11狂欢节", "2025-10-20", "2025-11-11", "年度最大促，全品类爆发"),
    ("promo_04", "黑五", "2025-11-24", "2025-11-28", "进口/电子品类促销"),
    ("promo_05", "双12", "2025-12-01", "2025-12-12", "年末大促，日用品补量"),
    ("promo_06", "年货节", "2025-12-20", "2026-01-20", "春节前置采购季"),
    ("promo_07", "新春大促", "2026-01-25", "2026-02-15", "新春囤货季"),
    ("promo_08", "女神节", "2026-03-01", "2026-03-08", "美妆/服饰品类促销"),
    ("promo_09", "618年中大促", "2026-06-01", "2026-06-20", "年中核心大促"),
    ("promo_10", "夏季清仓", "2026-07-15", "2026-07-31", "换季清仓，折扣加深"),
    ("promo_11", "818购物节", "2026-08-01", "2026-08-20", "平台级购物节"),
    ("promo_12", "开学季", "2026-08-20", "2026-09-05", "学生装备/文具品类促销"),
]


def gen_promotions(cfg) -> pd.DataFrame:
    return pd.DataFrame(
        [(pid, name, s, e, desc) for pid, name, s, e, desc in PROMOTIONS],
        columns=["promo_id", "promo_name", "start_date", "end_date", "description"],
    )


def _campaign_dates(cfg, rng, n_long: int, n_short: int):
    """长期 campaign 覆盖数据期全程，短期对齐大促窗口。返回 (starts, ends, promo_ids, windows)。

    活动窗口可能部分落在数据期外（如 small 数据期止于 6/30，年货节窗口到次年），
    一律钳制到 [cfg.start, cfg.end]，保证 campaign 日期不越出数据期。
    windows 为钳制后的 (start, end, promo_id) 列表，供名称匹配复用。
    """
    start = pd.Timestamp(cfg.date_range.start)
    end = pd.Timestamp(cfg.date_range.end)
    windows = []
    for pid_, name_, s, e, _ in PROMOTIONS:
        s_ = pd.Timestamp(s)
        e_ = pd.Timestamp(e)
        if e_ < start or s_ > end:
            continue
        windows.append((max(s_, start), min(e_, end), pid_, name_))
    if not windows:
        windows = [(start, end, None, None)]

    starts, ends = [], []
    for _ in range(n_long):
        starts.append(start)
        ends.append(end)
    for _ in range(n_short):
        s_, e_, _, _ = windows[rng.integers(0, len(windows))]
        w = (e_ - s_).days
        if w > 4:
            off = int(rng.integers(0, w - 3))
            starts.append(s_ + pd.Timedelta(days=off))
            ends.append(starts[-1] + pd.Timedelta(days=int(rng.integers(3, 8))))
        else:
            starts.append(s_)
            ends.append(e_)
        ends[-1] = min(ends[-1], end)
    return pd.Series(pd.to_datetime(starts)), pd.Series(pd.to_datetime(ends)), windows


def gen_campaigns(cfg, channels_df, rng) -> pd.DataFrame:
    """生成 dim.ads_campaigns（含内部列 _strength，写出前由 gen_marketing_full 裁剪）。"""
    paid = channels_df[channels_df["channel_type"] != "organic"].reset_index(drop=True)
    type_map = dict(zip(channels_df["channel_id"], channels_df["channel_type"]))

    n_camp = 10 * len(paid)
    n_short = max(3, n_camp // 30)
    n_long = n_camp - n_short

    # 渠道分配：按渠道类型强度加权，强渠道开更多 campaign
    weights = paid["channel_type"].map(CHANNEL_STRENGTH).to_numpy(float)
    ch_idx = rng.choice(len(paid), size=n_camp, p=weights / weights.sum())
    camp_channel = paid["channel_id"].to_numpy()[ch_idx]

    starts, ends, windows = _campaign_dates(cfg, rng, n_long, n_short)
    name_by_channel = dict(zip(channels_df["channel_id"], channels_df["channel_name"]))
    # 短期 campaign 用活动名，长期用"日常投放"；windows 为钳制后 (start, end, promo_id)
    campaign_names, promo_ids = [], []
    for i in range(n_camp):
        ch_name = name_by_channel[camp_channel[i]]
        if i >= n_long:  # 短期 campaign：找其窗口所属活动
            w_start = starts.iloc[i]
            pid = next((pid_ for (s_, e_, pid_, _) in windows
                        if s_ <= w_start <= e_), None)
            promo_name = next((name_ for (s_, e_, pid_, name_) in windows
                               if pid_ == pid), None)
            campaign_names.append(f"{promo_name if pid else '大促'}-{ch_name}")
            promo_ids.append(pid if pid else None)
        else:
            campaign_names.append(f"日常投放-{ch_name}")
            promo_ids.append(None)

    strength = np.array([CHANNEL_STRENGTH[type_map[c]] * rng.uniform(0.5, 1.5)
                         for c in camp_channel])
    return pd.DataFrame({
        "campaign_id": [f"camp_{i + 1:04d}" for i in range(n_camp)],
        "campaign_name": campaign_names,
        "channel_id": camp_channel,
        "start_date": starts,
        "end_date": ends,
        "budget": 0.0,
        "_strength": strength,
        "_promo_id": promo_ids,
    })


def gen_marketing_full(cfg, orders, payments, channels_df) -> dict[str, pd.DataFrame]:
    rng_a = make_rng(cfg.seed, "ads")
    rng_c = make_rng(cfg.seed, "ad_conversions")
    type_map = dict(zip(channels_df["channel_id"], channels_df["channel_type"]))
    neg_channels = set(cfg.demo_cases.negative_roi_channels)

    promos = gen_promotions(cfg)
    campaigns = gen_campaigns(cfg, channels_df, rng_a)

    # ---- ad_conversions：抽 paid_share 比例的已支付订单 ----
    paid = orders[orders["order_status"] != "1"].reset_index(drop=True)
    n_attr = int(round(cfg.orders.count * cfg.orders.paid_share))
    idx = rng_c.choice(len(paid), size=n_attr, replace=False)
    attr = paid.iloc[idx].reset_index(drop=True)

    paid_at = payments.set_index("order_id")["paid_at"]
    converted_at = pd.Series(pd.to_datetime(paid_at.reindex(attr["order_id"]).to_numpy()),
                             name="converted_at")
    # 数据期末日下单 + 支付延迟会把 paid_at 推到数据期外（small 期末 6/30 23:59 → 7/1），
    # 钳制到数据期末，保证归因日落在 campaign 网格内（ads_daily_stats 精确聚合勾稽）。
    d_end = pd.Timestamp(cfg.date_range.end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    converted_at = converted_at.clip(upper=d_end)

    # 逐日把转化单分摊到当日活跃 campaign（强度归一化 → 多项分布）
    camp_id = campaigns["campaign_id"].to_numpy()
    camp_start = campaigns["start_date"].to_numpy()
    camp_end = campaigns["end_date"].to_numpy()
    camp_strength = campaigns["_strength"].to_numpy()

    day = converted_at.dt.normalize()
    uniq_days = pd.DatetimeIndex(sorted(day.unique()))
    assign = np.empty(n_attr, dtype=object)
    for d in uniq_days:
        m = (day == d).to_numpy()
        n_day = int(m.sum())
        if n_day == 0:
            continue
        d64 = np.datetime64(d, "ns")
        active = (camp_start <= d64) & (d64 <= camp_end)
        if not active.any():
            active = camp_start <= d64
        s = camp_strength[active]
        counts = rng_c.multinomial(n_day, s / s.sum())
        ids = np.repeat(camp_id[active], counts)
        rng_c.shuffle(ids)
        assign[m] = ids

    noise = rng_c.uniform(0.9, 1.1, n_attr)
    gmv = np.round(attr["pay_amount"].to_numpy().astype(float) * cfg.ads.attribution * noise, 2)
    conversions = pd.DataFrame({
        "conversion_id": [f"ac_{i:08d}" for i in range(1, n_attr + 1)],
        "campaign_id": assign,
        "order_id": attr["order_id"].to_numpy(),
        "attributed_gmv": gmv,
        "converted_at": converted_at,
    })

    # ---- ads_daily_stats：campaign×日 展开，转化数/GMV 精确聚合，漏斗反推曝光/点击/花费 ----
    n_camp = len(campaigns)
    cpm = np.empty(n_camp)
    ctr = np.empty(n_camp)
    cvr = np.empty(n_camp)
    for i, cid in enumerate(campaigns["channel_id"].to_numpy()):
        if cid in neg_channels:
            cpm[i] = rng_a.uniform(*NEG_FUNNEL["cpm"])
            ctr[i] = NEG_FUNNEL["ctr"][0]
            cvr[i] = NEG_FUNNEL["cvr"][0]
        else:
            fl = FUNNEL[type_map[cid]]
            cpm[i] = rng_a.uniform(*fl["cpm"])
            ctr[i] = rng_a.uniform(*fl["ctr"])
            cvr[i] = rng_a.uniform(*fl["cvr"])

    frames = []
    for i in range(n_camp):
        days = pd.date_range(campaigns["start_date"].iloc[i],
                             campaigns["end_date"].iloc[i], freq="D")
        frames.append(pd.DataFrame({
            "campaign_id": camp_id[i],
            "stat_date": days,
            "_cpm": cpm[i], "_ctr": ctr[i], "_cvr": cvr[i],
        }))
    grid = pd.concat(frames, ignore_index=True)

    conv_agg = (conversions.assign(_d=day)
                .groupby(["campaign_id", "_d"], as_index=False)
                .agg(n_conv=("conversion_id", "size"),
                     sum_gmv=("attributed_gmv", "sum")))
    stats = grid.merge(conv_agg, left_on=["campaign_id", "stat_date"],
                       right_on=["campaign_id", "_d"], how="left")
    stats["n_conv"] = stats["n_conv"].fillna(0).astype(int)
    stats["sum_gmv"] = stats["sum_gmv"].fillna(0.0)

    clicks = np.ceil(stats["n_conv"] / stats["_cvr"]).astype(int)
    impressions = np.ceil(clicks / stats["_ctr"]).astype(int)
    spend = np.round(impressions * stats["_cpm"] / 1000.0, 2)
    stats["clicks"] = clicks
    stats["impressions"] = impressions
    stats["spend"] = spend

    # 负 ROI 渠道：spend 直接锚定 gmv 的 NEG_SPEND_MULT 倍（ROI ≈ 1/NEG_SPEND_MULT < 1），
    # impressions/clicks 反推保持 cpm/ctr 一致性——不受"低 ctr → impressions 爆炸"的漏斗放大约束。
    neg_stats = stats["campaign_id"].isin(campaigns.loc[
        campaigns["channel_id"].isin(neg_channels), "campaign_id"])
    neg_spend = np.round(stats.loc[neg_stats, "sum_gmv"] * NEG_SPEND_MULT, 2)
    stats.loc[neg_stats, "spend"] = neg_spend
    stats.loc[neg_stats, "impressions"] = np.ceil(
        neg_spend / (stats.loc[neg_stats, "_cpm"] / 1000.0)).astype(int)
    stats.loc[neg_stats, "clicks"] = np.ceil(
        stats.loc[neg_stats, "impressions"] * stats.loc[neg_stats, "_ctr"]).astype(int)
    stats["attributed_gmv"] = np.round(stats["sum_gmv"], 2)

    # ---- 渠道 ROI 校准：只对非负 ROI 渠道缩放 spend 使 ROI≈3.0 ----
    # 负 ROI 渠道保持反推的 spend（真实低效烧钱），不参与校准——
    # 否则它会吃掉 spend 结构，导致正常渠道 spend 被压扁、ROI 虚高（如 search 12）。
    neg_camp = campaigns["channel_id"].isin(neg_channels)
    pos_mask = ~np.isin(stats["campaign_id"], campaigns.loc[neg_camp, "campaign_id"])
    pos_gmv = stats.loc[pos_mask, "attributed_gmv"].sum()
    pos_spend = stats.loc[pos_mask, "spend"].sum()
    roi = pos_gmv / pos_spend if pos_spend else 0.0
    if 0 < roi and not (2.5 <= roi <= 3.5):
        scale = pos_gmv / (TARGET_ROI * pos_spend)
        stats.loc[pos_mask, "spend"] = np.round(stats.loc[pos_mask, "spend"] * scale, 2)

    # ---- budget：各 campaign 实际花费上浮 15%–50%（预算 ≥ 花费） ----
    spend_by_camp = stats.groupby("campaign_id")["spend"].sum()
    budgets = [float(max(2000.0, spend_by_camp.get(cid, 0.0)
                         * rng_a.uniform(1.15, 1.5)))
               for cid in camp_id]
    campaigns["budget"] = np.round(budgets, 2)

    daily = stats[["campaign_id", "stat_date", "impressions", "clicks",
                   "spend", "attributed_gmv"]]
    camp_out = campaigns[["campaign_id", "campaign_name", "channel_id",
                          "start_date", "end_date", "budget"]]
    return {
        "promotions": promos,
        "ads_campaigns": camp_out,
        "ad_conversions": conversions,
        "ads_daily_stats": daily,
    }


def gen_traffic_events(cfg, orders, rng) -> pd.DataFrame:
    """反推漏斗生成用户行为事件流（SPEC §3.2 traffic_events，可选生成）。

    pay 事件绑定抽样订单；add_to_cart/click/impression 按转化系数反推，
    ts 在抽样订单支付时间基础上前置随机延迟（钳制到数据期起点）。
    转化率 pay/impression = 0.15×0.20×0.85 ≈ 2.55% ∈ [2%, 3%] 行业基准带。
    """
    ctr, cvr, pcr = TRAFFIC_FUNNEL.values()
    paid = orders[orders["order_status"] != "1"].reset_index(drop=True)
    n_pay = max(1, min(int(round(cfg.orders.count * TRAFFIC_COVER)), len(paid)))
    idx = rng.choice(len(paid), size=n_pay, replace=False)
    src = paid.iloc[idx]
    t_pay = pd.to_datetime(src["created_at"]).to_numpy()
    pay_user = src["user_id"].to_numpy()

    n_imp = int(np.ceil(n_pay / (ctr * cvr * pcr)))   # pay / 0.0255
    n_click = int(np.ceil(n_pay / (cvr * pcr)))
    n_add = int(np.ceil(n_pay / pcr))
    start = np.datetime64(pd.Timestamp(cfg.date_range.start).normalize())

    def _fwd_ts(n: int, min_: int, max_: int) -> np.ndarray:
        u = rng.integers(0, n_pay, size=n)
        delay = rng.integers(min_, max_, size=n).astype("timedelta64[m]")
        return np.maximum((t_pay[u] - delay).astype("datetime64[ns]"), start)

    ts_imp, ts_click, ts_add = _fwd_ts(n_imp, 120, 2880), _fwd_ts(n_click, 30, 360), _fwd_ts(n_add, 10, 120)

    # 用户：impression 20% 游客 NULL，其余复用抽样订单用户
    u_imp = pay_user[rng.integers(0, n_pay, size=n_imp)].copy()
    u_imp[rng.random(n_imp) < TRAFFIC_GUEST_SHARE] = None
    u_click = pay_user[rng.integers(0, n_pay, size=n_click)]
    u_add = pay_user[rng.integers(0, n_pay, size=n_add)]

    n_total = n_imp + n_click + n_add + n_pay
    event_id = np.char.add(
        np.full(n_total, "ev_", dtype="U3"),
        np.char.zfill((np.arange(1, n_total + 1)).astype("U10"), 10))
    return pd.DataFrame({
        "event_id": event_id,
        "user_id": np.concatenate([u_imp, u_click, u_add, pay_user]),
        "event_type": np.concatenate([
            np.full(n_imp, "impression"), np.full(n_click, "click"),
            np.full(n_add, "add_to_cart"), np.full(n_pay, "pay")]),
        "ts": np.concatenate([ts_imp, ts_click, ts_add, t_pay]),
    })
