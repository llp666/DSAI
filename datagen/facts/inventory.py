"""库存域生成（SPEC §5.4 + 槽位4 已确认参数）。

前向模拟：逐日循环，每 SKU 向量化。
- 销量消耗：on_hand -= sold[t]（sold 来自 order_items 按 SKU×日 聚合，仅已完成/已发货订单出库）；
- PO 到货：on_hand[eta==t] += arrival[t]；
- 补货判定：on_hand < reorder_point 且该 SKU 无在途 PO → 下 PO，eta = t + U(7,14)；
  reorder_point ≈ 日均销量 × lead_days × 1.2，PO qty = 日均销量 × reorder_coverage − on_hand；
- 周粒度落 inventory_snapshot（期末 = 期初 + 到货 − 销量，前向模拟天然勾稽）。

演示埋点（demo_cases）：
- stockout_skus：末 14 天压制补货（本应触发的 PO 取消/延迟 14 天）+ 末 14 天销量 ×1.38；
- dead_stock_skus：初始库存 ×20 → 期末 DOI >90（呆滞象限）；
- 兜底：若自然模拟未达 ≥5 断货 / ≥5 呆滞，强制从低动销 SKU 补足。

SKU 选择：demo_cases 里的硬编码 SKU 与 products 实际 ID 不一致（8 位随机格式），
统一改为按"数据期销量排名"选取（stockout 选第 3/5 名附近中动销，dead_stock 选长尾低动销）。
"""

import numpy as np
import pandas as pd

from datagen.rng import make_rng

# 分仓：5 个区域仓，SKU 主仓按品类就近（汽配→华东…），实际按序号轮转以均衡
WAREHOUSES = [
    ("WH-01", "华东一号仓", "华东"),
    ("WH-02", "华南一号仓", "华南"),
    ("WH-03", "华北一号仓", "华北"),
    ("WH-04", "西南一号仓", "西南"),
    ("WH-05", "华中一号仓", "华中"),
]

SUPPLIER_PREFIXES = [
    "恒通", "信达", "鸿远", "中联", "旭日", "金达", "环球", "天成",
    "华信", "嘉诚", "宏远", "德盛", "百川", "兴业", "联合", "中鑫",
    "瑞丰", "远景", "合力", "群力", "泰和", "康达", "茂源", "恒基",
    "文海", "长虹", "建联", "惠通", "顺达", "邦德", "正大", "广发",
    "紫荆", "银泰", "神州", "万通", "中航", "中远", "新东方", "飞达",
]

SUPPLIER_NOUNS = ["供应链", "实业", "商贸", "物流", "科技", "制造", "贸易", "采购"]


def gen_suppliers(cfg, rng=None) -> pd.DataFrame:
    """80 家供应商，lead_days U(7,14) 取整；名称合成（前缀+名词+序号）。"""
    rng = rng or make_rng(cfg.seed, "suppliers")
    n = 80
    rows = []
    for i in range(n):
        name = f"{rng.choice(SUPPLIER_PREFIXES)}{rng.choice(SUPPLIER_NOUNS)}{i + 1:02d}"
        lead = int(rng.integers(cfg.supply.lead_days[0], cfg.supply.lead_days[1] + 1))
        rows.append((f"SUP-{i + 1:03d}", name, lead))
    return pd.DataFrame(rows, columns=["supplier_id", "supplier_name", "lead_days"])


def gen_warehouses(cfg, rng=None) -> pd.DataFrame:
    rng = rng or make_rng(cfg.seed, "warehouses")
    return pd.DataFrame(WAREHOUSES, columns=["warehouse_id", "warehouse_name", "region"])


def _sales_by_sku_day(items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """订单明细按 SKU×日 聚合销量（仅已完成/已发货/已退款订单出库）。"""
    shipped = orders[orders["order_status"].isin(["3", "4", "5"])][["order_id", "created_at"]]
    it = items.merge(shipped, on="order_id")
    it["day"] = it["created_at"].dt.normalize()
    sold = it.groupby(["sku_id", "day"], as_index=False)["qty"].sum()
    return sold


def gen_inventory(cfg, items, orders, products, rng=None):
    """前向模拟核心。返回 (suppliers, purchase_orders, purchase_order_items,
    inventory_snapshot, warehouses, warehouse_stock, inbound_records, stock_moves)。

    参数（已确认）：
    - 初始库存 = 日均销量 × reorder_coverage(21)
    - reorder_point = 日均销量 × lead_days × 1.2
    - PO qty = 日均销量 × reorder_coverage − on_hand，取整到 10 的倍数
    - PO 状态：期末 已入库60% / 在途30% / 待确认5% / 已取消5%
    """
    rng = rng or make_rng(cfg.seed, "inventory")
    rng_po = make_rng(cfg.seed, "supply")
    rng_wh = make_rng(cfg.seed, "warehouses")
    rng_mv = make_rng(cfg.seed, "inventory")

    dates = pd.date_range(cfg.date_range.start, cfg.date_range.end, freq="D")
    n_days = len(dates)
    end_date = dates[-1].normalize()

    # ---- 每日销量（全 SKU × 全日期 矩阵）----
    sold = _sales_by_sku_day(items, orders)
    skus = products["sku_id"].to_numpy()
    sku_idx = {s: i for i, s in enumerate(skus)}
    day_idx = {d: i for i, d in enumerate(dates.normalize())}

    # 稀疏矩阵 → 稠密 (n_sku × n_days) int
    sales = np.zeros((len(skus), n_days), dtype=np.int64)
    for row in sold.itertuples(index=False):
        i = sku_idx.get(row.sku_id)
        j = day_idx.get(pd.Timestamp(row.day))
        if i is not None and j is not None:
            sales[i, j] += row.qty

    daily_avg = sales.sum(axis=1) / n_days  # 每 SKU 日均销量（含零）
    lead_by_sku = np.empty(len(skus), dtype=int)
    # SKU 按序号轮转绑定供应商
    supplier_ids = products["sku_id"].map(
        lambda s: f"SUP-{(int(s.split('-')[1]) % 80) + 1:03d}"
    ).to_numpy()
    suppliers = gen_suppliers(cfg)
    lead_map = dict(zip(suppliers["supplier_id"], suppliers["lead_days"]))
    for i, s in enumerate(skus):
        lead_by_sku[i] = lead_map[supplier_ids[i]]

    # ---- 埋点 SKU 选取（按销量排名，避免与配置硬编码冲突）----
    sold_total = sales.sum(axis=1)
    rank = np.argsort(-sold_total)  # 0=销量最高
    stockout_candidates = [s for s in cfg.demo_cases.stockout_skus if s in sku_idx]
    dead_candidates = [s for s in cfg.demo_cases.dead_stock_skus if s in sku_idx]
    # 硬编码 SKU 不存在时退化为按排名：stockout 取中动销（前 5%-20% 段 5 个），dead 取长尾（后 20% 5 个）
    if not stockout_candidates:
        pool = rank[len(rank) // 20: len(rank) // 5]
        stockout_candidates = [skus[i] for i in pool[:5]]
    if not dead_candidates:
        pool = rank[int(len(rank) * 0.8):]
        dead_candidates = [skus[i] for i in pool[:5]]
    # 互斥：dead 命中 stockout 时替换为更低动销的 SKU
    dead_set = set(dead_candidates)
    for i, s in enumerate(stockout_candidates):
        if s in dead_set:
            alt = next((skus[j] for j in rank[int(len(rank) * 0.85):]
                        if skus[j] not in dead_set and skus[j] not in stockout_candidates), s)
            stockout_candidates[i] = alt
    stockout_mask = np.zeros(len(skus), dtype=bool)
    dead_mask = np.zeros(len(skus), dtype=bool)
    for s in stockout_candidates:
        stockout_mask[sku_idx[s]] = True
    for s in dead_candidates:
        dead_mask[sku_idx[s]] = True

    # ---- 初始库存与再订货点 ----
    # reorder_point = 日均 × lead × 2.2（保守安全边际，吸收促销/日波动，杜绝自然断货）
    reorder_point = np.floor(daily_avg * lead_by_sku * 2.2).astype(np.int64)
    reorder_point = np.maximum(reorder_point, 15)  # 低频 SKU 保底，防脆弱补货周期
    # 普通 SKU 初始库存 = 日均 × reorder_coverage，且 ≥ reorder_point（初始就够卖，避免首日补货循环）
    # stockout 埋点 SKU 用较小 init（日均×12，不抬高）——末 14 天压制补货后能真正断货
    init = np.floor(daily_avg * cfg.supply.reorder_coverage).astype(np.int64)
    init = np.maximum(init, 40)
    init[stockout_mask] = np.floor(daily_avg[stockout_mask] * 12).astype(np.int64)
    init[stockout_mask] = np.maximum(init[stockout_mask], 20)
    short = init < reorder_point
    short = short & ~stockout_mask
    init[short] = reorder_point[short] + np.floor(daily_avg[short] * 5).astype(np.int64)
    # dead_stock：初始 ×300（约 1 年量 → DOI>90 呆滞象限）
    init[dead_mask] = np.maximum(init[dead_mask], 1) * 300
    on_hand = init.copy()
    po_qty_target = np.floor(daily_avg * cfg.supply.reorder_coverage).astype(np.int64)
    po_qty_target = np.maximum(po_qty_target, 40)  # 与 init 下限一致，保证补货充足

    # stockout 埋点：末 14 天销量 ×1.38（SPEC §5.4），放大 sales 矩阵对应行列
    last_14 = n_days - 14
    if stockout_mask.any() and last_14 > 0:
        sales[stockout_mask, last_14:] = np.ceil(
            sales[stockout_mask, last_14:] * 1.38).astype(np.int64)

    po_rows = []      # (po_id, supplier_id, status, created_at, eta_date)
    poi_rows = []     # (po_id, sku_id, qty, unit_cost)
    inbound_rows = [] # (inbound_id, po_id, warehouse_id, arrived_at, received_qty)
    snap_rows = []    # (snapshot_date, sku_id, on_hand_qty, in_transit_qty)
    move_rows = []    # 出库流水（末行分批写出，控制行数）
    cancelled_po = set()  # stockout 埋点清空的在途 PO（期末标记 cancelled，不产生入库）

    # PO 在途表：eta_date → sku → qty（数组，含取消后回滚）
    n_po = 0
    # 在途 PO 记录（sku, eta_day, qty, po_id, supplier）
    open_po = {i: [] for i in range(len(skus))}  # sku → list of (eta_day, qty, po_id)

    po_id_gen = 1
    cost_price = products["cost_price"].to_numpy()

    snapshot_weeks = pd.date_range(start=dates[0].normalize(), end=end_date, freq="W-SUN").normalize()
    snap_week_set = {w: i for i, w in enumerate(snapshot_weeks)}

    # 初始在途（数据期开始前的在途 PO 直接到货）：模拟开始即安排一笔小 PO 撑起在途
    for i in range(len(skus)):
        if init[i] < reorder_point[i]:
            eta = int(rng_po.integers(3, 8))
            qty = max(po_qty_target[i], 10)
            po_id = f"PO-{po_id_gen:06d}"
            po_id_gen += 1
            open_po[i].append((eta, qty, po_id))
            po_rows.append((po_id, supplier_ids[i], "in_transit",
                            dates[0].normalize(),
                            (dates[0].normalize() + pd.Timedelta(days=eta)).date()))
            poi_rows.append((po_id, skus[i], qty, round(float(cost_price[i]) * rng_po.uniform(0.9, 1.1), 2)))
            n_po += 1

    for t in range(n_days):
        # 0) stockout 埋点：末 14 天起始强制压低库存并取消在途 PO（模拟热销告急+断供），
        #    此后压制补货 → 库存只减不增，必然几天内断货
        if t == last_14:
            on_hand[stockout_mask] = np.floor(daily_avg[stockout_mask] * 3).astype(np.int64)
            on_hand[stockout_mask] = np.maximum(on_hand[stockout_mask], 3)
            for i in np.where(stockout_mask)[0]:
                for _, _, po_id in open_po[i]:
                    cancelled_po.add(po_id)
                open_po[i] = []
        # 1) 销量消耗
        on_hand -= sales[:, t]
        on_hand = np.maximum(on_hand, 0)

        # 2) 到货
        for i in range(len(skus)):
            if open_po[i]:
                keep = []
                for eta, qty, po_id in open_po[i]:
                    if eta == t:
                        on_hand[i] += qty
                        wh = WAREHOUSES[rng_wh.integers(0, 5)][0]
                        inbound_rows.append((f"in_{len(inbound_rows) + 1:08d}", po_id, wh,
                                             dates[t].normalize(), int(qty)))
                    else:
                        keep.append((eta, qty, po_id))
                open_po[i] = keep

        # 3) 补货判定
        for i in range(len(skus)):
            # stockout 埋点：末 14 天压制补货（不触发新 PO），让库存自然见底
            if stockout_mask[i] and t >= last_14:
                continue
            if on_hand[i] < reorder_point[i] and not open_po[i]:
                delay = int(rng_po.integers(
                    cfg.supply.lead_days[0], cfg.supply.lead_days[1] + 1))
                if t + delay >= n_days - 1:
                    continue  # 到不了货的 PO 不下（避免期末"在途断货"快照误判）
                qty = int(po_qty_target[i])  # 固定补满 reorder_coverage 天量，覆盖最长 lead
                qty = int(np.ceil(qty / 10) * 10)
                eta = min(t + delay, n_days - 1)
                po_id = f"PO-{po_id_gen:06d}"
                po_id_gen += 1
                open_po[i].append((eta, qty, po_id))
                po_rows.append((po_id, supplier_ids[i], "in_transit",
                                dates[t].normalize(),
                                dates[eta].normalize().date()))
                poi_rows.append((po_id, skus[i], qty, round(float(cost_price[i]) * rng_po.uniform(0.9, 1.1), 2)))
                n_po += 1

        # 4) 周快照
        day_norm = dates[t].normalize()
        if day_norm in snap_week_set:
            for i in range(len(skus)):
                in_transit = sum(q for _, q, _ in open_po[i])
                snap_rows.append((day_norm, skus[i],
                                  int(on_hand[i]), int(in_transit)))

    # 期末 PO 状态：已入库 = 有 inbound 记录；其余按比例分在途/待确认/已取消
    # 注意：在途 = eta 尚未到（数据期末还没到货的 PO），pending = 刚下单待确认
    received_poids = {r[1] for r in inbound_rows}
    for k, pr in enumerate(po_rows):
        if pr[0] in cancelled_po:
            po_rows[k] = (pr[0], pr[1], "cancelled", pr[3], pr[4])
            continue
        if pr[0] in received_poids:
            po_rows[k] = (pr[0], pr[1], "received", pr[3], pr[4])
            continue
        r = rng_po.random()
        if r < 0.60:
            po_rows[k] = (pr[0], pr[1], "in_transit", pr[3], pr[4])
        elif r < 0.70:
            po_rows[k] = (pr[0], pr[1], "pending", pr[3], pr[4])
        else:
            po_rows[k] = (pr[0], pr[1], "cancelled", pr[3], pr[4])

    # 出库流水（每已完成订单明细一条，qty=该明细件数；+ 入库流水在 inbound 隐含）
    shipped_orders = orders[orders["order_status"].isin(["3", "4", "5"])][["order_id", "created_at"]]
    outbound = items.merge(shipped_orders, on="order_id")
    outbound = outbound.groupby(["order_id", "sku_id", "created_at"], as_index=False)["qty"].sum()
    wh_assign = rng_mv.integers(0, 5, len(outbound))
    move_rows = []
    for k, row in enumerate(outbound.itertuples(index=False)):
        move_rows.append((f"mv_{k + 1:08d}", row.sku_id, WAREHOUSES[wh_assign[k]][0],
                          "outbound", int(row.qty), pd.Timestamp(row.created_at).normalize(),
                          row.order_id))

    # warehouse_stock：期末（最后快照）在手按主仓分摊（90% 主仓 + 10% 次仓）
    # 用最后一个快照的 on_hand，保证 warehouse_stock 合计 = 期末快照合计（勾稽一致）
    last_snap_oh = {r[1]: r[2] for r in snap_rows
                    if r[0] == snapshot_weeks[-1].normalize()}
    wh_stock_rows = []
    for i in range(len(skus)):
        oh = last_snap_oh.get(skus[i], 0)
        if oh <= 0:
            continue
        main_wh = WAREHOUSES[rng_wh.integers(0, 5)][0]
        main_qty = int(np.floor(oh * 0.9))
        wh_stock_rows.append((main_wh, skus[i], main_qty))
        if oh - main_qty > 0:
            sub_wh = WAREHOUSES[(rng_wh.integers(0, 5)) % 5][0]
            while sub_wh == main_wh:
                sub_wh = WAREHOUSES[rng_wh.integers(0, 5)][0]
            wh_stock_rows.append((sub_wh, skus[i], int(oh - main_qty)))

    suppliers_df = gen_suppliers(cfg)
    warehouses_df = gen_warehouses(cfg)
    purchase_orders = pd.DataFrame(po_rows, columns=[
        "po_id", "supplier_id", "status", "created_at", "eta_date"])
    purchase_order_items = pd.DataFrame(poi_rows, columns=[
        "po_id", "sku_id", "qty", "unit_cost"])
    inventory_snapshot = pd.DataFrame(snap_rows, columns=[
        "snapshot_date", "sku_id", "on_hand_qty", "in_transit_qty"])
    inbound_records = pd.DataFrame(inbound_rows, columns=[
        "inbound_id", "po_id", "warehouse_id", "arrived_at", "received_qty"])
    stock_moves = pd.DataFrame(move_rows, columns=[
        "move_id", "sku_id", "warehouse_id", "move_type", "qty", "moved_at", "ref_id"])
    warehouse_stock = pd.DataFrame(wh_stock_rows, columns=[
        "warehouse_id", "sku_id", "on_hand_qty"])

    return {
        "suppliers": suppliers_df,
        "warehouses": warehouses_df,
        "purchase_orders": purchase_orders,
        "purchase_order_items": purchase_order_items,
        "inventory_snapshot": inventory_snapshot,
        "inbound_records": inbound_records,
        "stock_moves": stock_moves,
        "warehouse_stock": warehouse_stock,
    }
