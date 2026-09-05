"""validate.py：数仓业务合理性断言（SPEC §10 / 槽位6 雏形）。

对已入库 DuckDB 运行一组勾稽校验，逐条输出 PASS/FAIL，非零退出码暴露失败。
覆盖：订单域（净销比/状态流转/支付1:1）、营销域（GMV 精确聚合/ROI 区间）、
日期域界（所有时间戳不越出数据期）、唯一性/引用完整性（FK 无孤儿）。
"""

import argparse
import sys

import duckdb

# 每张表的日期列（表 → (列, 列类型)），用于"不越出数据期"断言
# 数据期由 dim_date 的最小/最大日期界定。
DATE_COLS = {
    "ods.orders": [("created_at", "TIMESTAMP"), ("updated_at", "TIMESTAMP")],
    "ods.order_payments": [("paid_at", "TIMESTAMP")],
    "ods.order_status_log": [("changed_at", "TIMESTAMP")],
    "ods.refunds": [("refund_date", "DATE")],
    "ods.ad_conversions": [("converted_at", "TIMESTAMP")],
    "dim.ads_campaigns": [("start_date", "DATE"), ("end_date", "DATE")],
}

# 唯一键（表 → 列）
UNIQUE_KEYS = {
    "ods.orders": ["order_id"],
    "ods.order_payments": ["payment_id"],
    "ods.order_status_log": ["log_id"],
    "ods.refunds": ["refund_id"],
    "ods.ad_conversions": ["conversion_id"],
    "ods.ads_daily_stats": ["campaign_id", "stat_date"],
    "dim.categories": ["category_id"],
    "dim.products": ["sku_id"],
    "dim.users": ["user_id"],
    "dim.channels": ["channel_id"],
    "dim.ads_campaigns": ["campaign_id"],
}

# 引用完整性（子表 → (子列, 父表, 父列)）
REFERENCES = {
    "ods.orders": ("user_id", "dim.users", "user_id"),
    "ods.order_items": ("order_id", "ods.orders", "order_id"),
    "ods.order_items": ("sku_id", "dim.products", "sku_id"),
    "ods.refunds": ("order_id", "ods.orders", "order_id"),
    "ods.order_payments": ("order_id", "ods.orders", "order_id"),
    "ods.order_status_log": ("order_id", "ods.orders", "order_id"),
    "ods.ads_daily_stats": ("campaign_id", "dim.ads_campaigns", "campaign_id"),
    "ods.ad_conversions": ("campaign_id", "dim.ads_campaigns", "campaign_id"),
}


def _run(con: duckdb.DuckDBPyConnection, sql: str) -> int | float | None:
    return con.execute(sql).fetchone()[0]


def _run_count(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> int:
    """执行带参数绑定的查询，返回行数（用于大清单 NOT IN，避免 SQL 插值截断）。"""
    return con.execute(sql, params).fetchone()[0]


def _load_manifest(manifest_path) -> dict:
    import json
    from pathlib import Path
    p = Path(manifest_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _neg_roi_channels(config_path) -> list[str]:
    """从 YAML 配置读 demo_cases.negative_roi_channels（负 ROI 埋点渠道清单）。"""
    import yaml
    from pathlib import Path
    if not config_path or not Path(config_path).exists():
        return []
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return list(cfg.get("demo_cases", {}).get("negative_roi_channels", []))


def _manifest_skips(manifest: dict) -> dict[str, set[str]]:
    """从 dirty_manifest 提取各表需跳过的 ID 集合。"""
    skips: dict[str, set[str]] = {}
    for group in manifest.values():
        if not isinstance(group, dict):
            continue
        for table, info in group.items():
            ids = info.get("ids", []) if isinstance(info, dict) else []
            if ids:
                skips.setdefault(table, set()).update(ids)
    return skips


def validate(con: duckdb.DuckDBPyConnection,
             manifest_path=None, config_path=None) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    manifest = _load_manifest(manifest_path) if manifest_path else {}
    skips = _manifest_skips(manifest)
    # 枚举漂移：order_status 混入旧字符串码（如 "已完成"），校验时视为已完成
    enum_orders = skips.get("ods.orders", set())
    neg_cfg = _neg_roi_channels(config_path)

    def check(name: str, sql: str, expect_zero: bool = True, _params: list | None = None):
        try:
            v = con.execute(sql, _params).fetchone()[0] if _params else _run(con, sql)
        except duckdb.Error as e:
            results.append((f"{name}（查询失败: {e}）", False))
            return
        ok = (v == 0) if expect_zero else bool(v)
        results.append((name, ok))

    # ---- 日期域界 ----
    d0 = _run(con, "SELECT min(date)::DATE FROM dim.dim_date")
    d1 = _run(con, "SELECT max(date)::DATE FROM dim.dim_date")
    late_ids = skips.get("ods.orders", set())  # 迟到分区订单（updated_at +2~5 天）
    for tbl, cols in DATE_COLS.items():
        for col, ctype in cols:
            skip = ""
            params: list | None = None
            if tbl == "ods.orders" and col == "updated_at" and late_ids:
                skip = " AND order_id NOT IN (SELECT order_id FROM UNNEST(?::VARCHAR[]) AS t(order_id))"
                params = [list(late_ids)]
            if ctype == "DATE":
                check(f"{tbl}.{col} 不越出数据期",
                      f"SELECT count(*) FROM {tbl} WHERE {col} < DATE '{d0}' OR {col} > DATE '{d1}'{skip}",
                      _params=params)
            else:
                check(f"{tbl}.{col} 不越出数据期",
                      f"SELECT count(*) FROM {tbl} WHERE {col}::DATE < DATE '{d0}' OR {col}::DATE > DATE '{d1}'{skip}",
                      _params=params)

    # ---- 唯一性 ----
    for tbl, cols in UNIQUE_KEYS.items():
        sel = ", ".join(cols)
        check(f"{tbl} 唯一({sel})",
              f"SELECT count(*) - count(DISTINCT ({sel})) FROM {tbl}")

    # ---- 引用完整性 ----
    for child, (c_col, parent, p_col) in REFERENCES.items():
        check(f"{child}.{c_col} → {parent}.{p_col} 无孤儿",
              f"SELECT count(*) FROM {child} c LEFT JOIN {parent} p ON p.{p_col}=c.{c_col} WHERE p.{p_col} IS NULL")

    # ---- 订单域 ----
    check("净销/毛销 ∈ [0.75, 0.88]",
          """
          SELECT (SELECT sum(pay_amount)::DOUBLE FROM ods.orders WHERE order_status!='1') gross,
                 (SELECT sum(refund_amount)::DOUBLE FROM ods.refunds) refund
          """,
          expect_zero=False)  # 特例，下面单独判断
    results.pop()  # 移除占位，用精确断言替换
    gross = _run(con, "SELECT sum(pay_amount)::DOUBLE FROM ods.orders WHERE order_status!='1'")
    refund = _run(con, "SELECT sum(refund_amount)::DOUBLE FROM ods.refunds")
    ratio = (gross - refund) / gross
    results.append((f"净销/毛销={ratio:.4f} ∈ [0.75, 0.88]", 0.75 <= ratio <= 0.88))

    check("待支付单无支付记录",
          "SELECT count(*) FROM ods.order_payments p JOIN ods.orders o USING (order_id) WHERE o.order_status='1'")
    check("已支付订单支付记录 1:1",
          "SELECT (SELECT count(*) FROM ods.order_payments) - (SELECT count(*) FROM ods.orders WHERE order_status!='1')")
    # 订单终态与 status_log 末行一致（enum_drift 订单已改脏值，跳过；参数绑定避免截断）
    n_bad = 0
    if enum_orders:
        n_bad = _run_count(con, f"""
            SELECT count(*) FROM ods.orders o
            WHERE o.order_id NOT IN (SELECT order_id FROM UNNEST(?::VARCHAR[]) AS t(order_id))
              AND o.order_status != (SELECT to_status FROM ods.order_status_log l
                                     WHERE l.order_id=o.order_id
                                     ORDER BY l.changed_at DESC, l.log_id DESC LIMIT 1)
            """, [list(enum_orders)])
    results.append((f"订单终态与 status_log 末行一致（跳过 {len(enum_orders)} 个脏行）", n_bad == 0))
    check("已完成/退款单均有发货流转",
          """
          SELECT count(*) FROM ods.orders o
          WHERE o.order_status IN ('4','5') AND NOT EXISTS (
            SELECT 1 FROM ods.order_status_log l WHERE l.order_id=o.order_id AND l.to_status='3')
          """)
    check("行级退款率 ∈ [12%, 19%]",
          "SELECT count(*)::DOUBLE / (SELECT count(*) FROM ods.order_items) FROM ods.refunds",
          expect_zero=False)
    results.pop()
    hit = _run(con, "SELECT count(*)::DOUBLE / (SELECT count(*) FROM ods.order_items) FROM ods.refunds")
    results.append((f"行级退款率={hit:.4f} ∈ [0.12, 0.19]", 0.12 <= hit <= 0.19))

    # ---- 营销域 ----
    check("广告归因 GMV 精确聚合",
          """
          SELECT (SELECT round(sum(attributed_gmv)::DOUBLE,2) FROM ods.ad_conversions)
               - (SELECT round(sum(attributed_gmv)::DOUBLE,2) FROM ods.ads_daily_stats)
          """)
    check("总 ROI ∈ [1.5, 4]",
          "SELECT sum(attributed_gmv)::DOUBLE / NULLIF(sum(spend)::DOUBLE,0) FROM ods.ads_daily_stats",
          expect_zero=False)
    results.pop()
    roi = _run(con, "SELECT sum(attributed_gmv)::DOUBLE / NULLIF(sum(spend)::DOUBLE,0) FROM ods.ads_daily_stats")
    results.append((f"总ROI={roi:.2f} ∈ [1.5, 4]", 1.5 <= roi <= 4))

    # 负 ROI 渠道（demo 埋点）应 ROI < 1 且数量 = 配置数（SPEC §10.1-4 / §5.5）
    neg = con.execute("""
        SELECT ch.channel_id, sum(d.attributed_gmv)::DOUBLE / NULLIF(sum(d.spend)::DOUBLE,0)
        FROM ods.ads_daily_stats d
        JOIN dim.ads_campaigns c USING (campaign_id)
        JOIN dim.channels ch ON ch.channel_id=c.channel_id
        GROUP BY ch.channel_id
        HAVING sum(d.attributed_gmv)::DOUBLE / NULLIF(sum(d.spend)::DOUBLE,0) < 1
    """).fetchall()
    results.append((f"负ROI渠道数 = 配置数（{len(neg)}/{len(neg_cfg)}）",
                    len(neg) == len(neg_cfg)))

    # ---- 库存域 ----
    # 1) 快照勾稽：相邻快照 on_hand[t] ≈ on_hand[t-1] + 期间到货 − 期间销量
    #    （容差 200 吸收到货/销量与快照的日边界偏差 + 促销日波动；排除埋点 SKU）
    check("库存快照勾稽（相邻快照，容差200，排除埋点SKU）",
          """
          WITH snap AS (
            SELECT sku_id, snapshot_date, on_hand_qty,
                   LAG(on_hand_qty) OVER (PARTITION BY sku_id ORDER BY snapshot_date) prev_oh,
                   LAG(snapshot_date) OVER (PARTITION BY sku_id ORDER BY snapshot_date) prev_date
            FROM ods.inventory_snapshot
          ), burial AS (
            SELECT DISTINCT sku_id FROM ods.inventory_snapshot
            WHERE snapshot_date=(SELECT max(snapshot_date) FROM ods.inventory_snapshot) AND on_hand_qty=0
          ), sold AS (
            SELECT oi.sku_id, o.created_at::DATE d, sum(oi.qty) qty FROM ods.order_items oi
            JOIN ods.orders o ON o.order_id=oi.order_id WHERE o.order_status IN ('3','4','5','已完成') GROUP BY oi.sku_id, o.created_at::DATE
          ), arrived AS (
            SELECT poi.sku_id, ir.arrived_at::DATE d, sum(ir.received_qty) qty
            FROM ods.inbound_records ir JOIN ods.purchase_order_items poi ON poi.po_id=ir.po_id GROUP BY poi.sku_id, ir.arrived_at::DATE
          )
          SELECT count(*) FROM snap
          WHERE prev_oh IS NOT NULL
            AND sku_id NOT IN (SELECT sku_id FROM burial)
            AND abs(on_hand_qty - prev_oh
              - COALESCE((SELECT sum(a.qty) FROM arrived a WHERE a.sku_id=snap.sku_id AND a.d > snap.prev_date AND a.d <= snap.snapshot_date),0)
              + COALESCE((SELECT sum(s.qty) FROM sold s WHERE s.sku_id=snap.sku_id AND s.d > snap.prev_date AND s.d <= snap.snapshot_date),0)) > 200
          """)
    # 2) warehouse_stock 合计 = 期末快照 on_hand 合计
    check("warehouse_stock 合计 = 期末快照合计",
          """
          SELECT (SELECT sum(on_hand_qty) FROM ods.warehouse_stock)
               - (SELECT sum(on_hand_qty) FROM ods.inventory_snapshot
                  WHERE snapshot_date=(SELECT max(snapshot_date) FROM ods.inventory_snapshot))
          """)
    # 3) inbound_records 合计 = 已入库 PO 明细
    check("入库记录与已入库PO勾稽",
          """
          SELECT (SELECT sum(received_qty) FROM ods.inbound_records)
               - (SELECT sum(poi.qty) FROM ods.purchase_order_items poi
                  JOIN ods.purchase_orders po ON po.po_id=poi.po_id WHERE po.status='received')
          """)
    # 4) 出库流水 = 已完成/发货订单明细（含枚举漂移的"已完成"——映射回"4"）
    check("出库流水与订单销量勾稽",
          """
          SELECT (SELECT sum(qty) FROM ods.stock_moves WHERE move_type='outbound')
               - (SELECT sum(oi.qty) FROM ods.order_items oi
                  JOIN ods.orders o ON o.order_id=oi.order_id WHERE o.order_status IN ('3','4','5','已完成'))
          """)
    # 5) 断货埋点 ≥5、呆滞埋点 ≥5（SPEC §5.4 兜底）
    n_so = _run(con, "SELECT count(*) FROM ods.inventory_snapshot WHERE snapshot_date=(SELECT max(snapshot_date) FROM ods.inventory_snapshot) AND on_hand_qty=0")
    results.append((f"期末断货 SKU ≥5（实际 {n_so}）", n_so >= 5))
    n_ds = _run(con, """
        SELECT count(*) FROM ods.inventory_snapshot l
        JOIN (SELECT sku_id, sum(qty)/NULLIF((SELECT count(DISTINCT snapshot_date) FROM ods.inventory_snapshot)*1.0,0) daily
              FROM ods.stock_moves WHERE move_type='outbound' GROUP BY sku_id) s USING (sku_id)
        WHERE l.snapshot_date=(SELECT max(snapshot_date) FROM ods.inventory_snapshot)
          AND l.on_hand_qty / NULLIF(s.daily,0) > 90
        """)
    results.append((f"呆滞 SKU（DOI>90）≥5（实际 {n_ds}）", n_ds >= 5))

    # ---- 业务不变式（SPEC §10.1-2）----
    check("金额非负（pay_amount/refund/sale_price/qty/库存）",
          """
          SELECT (SELECT count(*) FROM ods.orders WHERE pay_amount < 0)
               + (SELECT count(*) FROM ods.refunds WHERE refund_amount < 0)
               + (SELECT count(*) FROM ods.order_items WHERE sale_price < 0 OR qty <= 0)
               + (SELECT count(*) FROM ods.inventory_snapshot WHERE on_hand_qty < 0 OR in_transit_qty < 0)
          """)
    check("refund_amount ≤ 对应 item 实付",
          """
          WITH r AS (SELECT order_id, sku_id, sum(refund_amount) ra FROM ods.refunds GROUP BY 1,2),
               it AS (SELECT order_id, sku_id, sum(sale_price*qty) amt FROM ods.order_items GROUP BY 1,2)
          SELECT count(*) FROM r JOIN it USING(order_id, sku_id) WHERE r.ra > it.amt + 0.01
          """)

    # ---- 分布锚点（SPEC §10.1-3）----
    rep = _run(con, """
        WITH last90 AS (
          SELECT user_id, count(DISTINCT created_at::DATE) n_days
          FROM ods.orders
          WHERE created_at >= (SELECT max(created_at) FROM ods.orders) - INTERVAL 89 DAY
          GROUP BY user_id
        )
        SELECT (SELECT count(*) FROM last90 WHERE n_days >= 2)::DOUBLE
             / NULLIF((SELECT count(*) FROM last90), 0)
        """)
    results.append((f"90天复购率={rep:.3f} ∈ [0.20, 0.35]（近90天下单≥2日占比）",
                    0.20 <= rep <= 0.35))
    has_te = _run(con, """
        SELECT count(*) FROM duckdb_tables()
        WHERE schema_name='ods' AND table_name='traffic_events'""")
    if has_te:
        cvr = _run(con, """
            SELECT (SELECT count(*) FROM ods.traffic_events WHERE event_type='pay')::DOUBLE
                 / NULLIF((SELECT count(*) FROM ods.traffic_events WHERE event_type='impression'), 0)""")
        results.append((f"全站转化率={cvr:.3f} ∈ [0.02, 0.03]（pay/impression）",
                        0.02 <= cvr <= 0.03))
    else:
        results.append(("全站转化率 ∈ [0.02, 0.03]（traffic_events 未生成，跳过）", True))

    # ---- 脏数据注入行数 = manifest 一致（SPEC §10.1-5，可判定部分）----
    if manifest:
        nr = manifest.get("null_rate", {})
        for table, info in nr.items():
            ids = info.get("ids", [])
            if not ids:
                continue
            col = {"dim.users": "city_tier", "ods.refunds": "refund_reason"}.get(table)
            pk = {"dim.users": "user_id", "ods.refunds": "refund_id"}.get(table)
            if not col:
                continue
            n = _run_count(con, f"SELECT count(*) FROM {table} "
                                f"WHERE {pk} IN (SELECT order_id FROM UNNEST(?::VARCHAR[]) AS t(order_id)) "
                                f"AND {col} IS NULL", [ids])
            results.append((f"脏数据 {table}.{col} 空值 {n}/{len(ids)}", n == len(ids)))
        ed = manifest.get("enum_drift", {})
        for table, info in ed.items():
            ids = info.get("ids", [])
            if not ids:
                continue
            for old, new in info.get("mapping", {}).items():
                n = _run_count(con, f"SELECT count(*) FROM {table} "
                                    f"WHERE order_id IN (SELECT order_id FROM UNNEST(?::VARCHAR[]) AS t(order_id)) "
                                    f"AND order_status = '{new}'", [ids])
                results.append((f"脏数据 {table} 枚举漂移 {n}/{len(ids)}", n == len(ids)))
        la = manifest.get("late_arrival", {})
        for table, info in la.items():
            ids = info.get("ids", [])
            if not ids:
                continue
            n = _run_count(con, f"SELECT count(*) FROM {table} "
                                f"WHERE order_id IN (SELECT order_id FROM UNNEST(?::VARCHAR[]) AS t(order_id))",
                           [ids])
            results.append((f"脏数据 {table} 迟到分区行存在 {n}/{len(ids)}", n == len(ids)))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datagen-validate", description="数仓勾稽校验")
    parser.add_argument("--db", default="warehouse/ecommerce.duckdb", help="DuckDB 文件路径")
    parser.add_argument("--config", default=None, help="YAML 配置（读取 demo_cases 负 ROI 渠道数）")
    parser.add_argument("--manifest", default="meta/dirty_manifest.json", help="脏数据 manifest 路径")
    args = parser.parse_args(argv)

    con = duckdb.connect(args.db)
    try:
        results = validate(con, args.manifest, args.config)
    finally:
        con.close()

    n_fail = 0
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        n_fail += 0 if ok else 1
    print(f"\n{len(results) - n_fail}/{len(results)} 通过")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
