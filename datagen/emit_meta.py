"""emit_meta.py：meta 三件套输出（SPEC §7）。

- meta/table_docs.json：W4 RAG 语料来源，DDL COMMENT 自动抽取 + 配置补 description/sample_questions；
- meta/ground_truth.json：评测锚点（注脏前计算！月度 GMV/净销/订单/复购率/top SKU/渠道 ROI）；
- warehouse/manifest.json：seed / config 哈希 / 行数 / 耗时（幂等凭证）。
"""

import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

# 表 → 领域分组（与 DDL 文件对应）
DOMAIN_MAP = {
    "订单域": ["orders", "order_items", "refunds", "order_payments", "order_status_log",
               "coupons", "order_coupons"],
    "用户域": ["users", "user_profiles", "user_level_log"],
    "商品域": ["categories", "products", "product_price_log"],
    "营销域": ["channels", "ads_campaigns", "ads_daily_stats", "ad_conversions",
               "promotions", "traffic_events"],
    "供应链与库存域": ["suppliers", "purchase_orders", "purchase_order_items",
                      "inventory_snapshot", "warehouses", "warehouse_stock",
                      "inbound_records", "stock_moves"],
    "日期域": ["dim_date", "festival_calendar"],
    "其他": ["after_sales", "search_logs"],
}

# 表 → sample_questions（评测/检索用）
SAMPLE_QUESTIONS = {
    "ods.orders": ["月度订单量与 GMV 趋势", "扣退款后的净销售额"],
    "ods.refunds": ["退款原因分布", "按品类的退款率"],
    "ods.order_payments": ["支付渠道占比", "支付成功率"],
    "ods.order_status_log": ["订单状态流转时延", "平均履约周期"],
    "dim.users": ["注册用户城市分布", "新老用户占比"],
    "dim.products": ["品类销售额结构", "SKU 价格带分布"],
    "ods.ads_daily_stats": ["渠道 ROI 对比", "广告花费趋势"],
    "ods.ad_conversions": ["归因转化明细", "渠道转化率"],
    "ods.inventory_snapshot": ["库存健康度（断货/呆滞）", "库存周转天数"],
    "ods.purchase_orders": ["供应商补货周期", "在途库存量"],
    "ods.stock_moves": ["出入库流水", "SKU 动销率"],
}


def gen_table_docs(ddl_root: Path) -> dict:
    """从 DDL COMMENT 抽取表/列描述，组装 table_docs.json。"""
    tables: dict[str, dict] = {}
    for f in sorted(ddl_root.glob("*.sql")):
        text = f.read_text(encoding="utf-8")
        # 表 COMMENT
        for m in re.finditer(r"COMMENT ON TABLE\s+(\S+)\s+IS\s+'([^']*)'", text):
            tbl = m.group(1)
            tables[tbl] = {
                "table": tbl,
                "schema": tbl.split(".")[0],
                "domain": _domain_of(tbl),
                "description": m.group(2),
                "columns": [],
                "pk": [], "fks": [], "update_freq": "T+1",
                "sample_questions": SAMPLE_QUESTIONS.get(tbl, []),
            }
        # 列 COMMENT
        for m in re.finditer(
                r"COMMENT ON COLUMN\s+(\S+)\.(\S+)\s+IS\s+'([^']*)'", text):
            tbl, col, comment = m.group(1), m.group(2), m.group(3)
            if tbl in tables:
                tables[tbl]["columns"].append({"name": col, "comment": comment})
    return tables


def _domain_of(table: str) -> str:
    name = table.split(".")[-1]
    for domain, names in DOMAIN_MAP.items():
        if name in names:
            return domain
    return "其他"


def gen_ground_truth(con: duckdb.DuckDBPyConnection, cfg) -> dict:
    """基于干净数据（注脏前）计算评测锚点。月度粒度，±1% 容差。"""
    start, end = cfg.date_range.start, cfg.date_range.end

    # 月度 GMV / 净销 / 订单数（订单与退款分别聚合，按月份合并）
    monthly = con.execute(f"""
        WITH orders_m AS (
          SELECT strftime(created_at, '%Y-%m') ym,
                 round(sum(CASE WHEN order_status != '1' THEN pay_amount ELSE 0 END)::DOUBLE, 2) gmv,
                 count(*) orders
          FROM ods.orders
          WHERE created_at::DATE BETWEEN DATE '{start}' AND DATE '{end}'
          GROUP BY 1
        ), refunds_m AS (
          SELECT strftime(refund_date, '%Y-%m') ym,
                 round(sum(refund_amount)::DOUBLE, 2) refund
          FROM ods.refunds
          WHERE refund_date BETWEEN DATE '{start}' AND DATE '{end}'
          GROUP BY 1
        )
        SELECT o.ym, o.gmv, round(o.gmv - COALESCE(r.refund, 0), 2) net_sales, o.orders
        FROM orders_m o LEFT JOIN refunds_m r USING (ym)
        ORDER BY 1
    """).fetchall()

    # 月度 top SKU（按净销 = 销售额 − 退款，CTE 聚合避免相关子查询）
    top_sku = con.execute(f"""
        WITH items AS (
          SELECT strftime(o.created_at, '%Y-%m') ym, oi.order_id, oi.sku_id,
                 sum(oi.sale_price * oi.qty)::DOUBLE gross
          FROM ods.order_items oi JOIN ods.orders o ON o.order_id = oi.order_id
          WHERE o.order_status != '1' AND o.created_at::DATE BETWEEN DATE '{start}' AND DATE '{end}'
          GROUP BY 1, oi.order_id, oi.sku_id
        ), refunds_s AS (
          SELECT r.order_id, r.sku_id, sum(r.refund_amount)::DOUBLE refund
          FROM ods.refunds r GROUP BY 1, 2
        ), net AS (
          SELECT i.ym, i.sku_id, sum(i.gross - COALESCE(r.refund, 0))::DOUBLE net_sales
          FROM items i LEFT JOIN refunds_s r ON r.order_id = i.order_id AND r.sku_id = i.sku_id
          GROUP BY 1, 2
        )
        SELECT ym, sku_id FROM (
          SELECT ym, sku_id, ROW_NUMBER() OVER (PARTITION BY ym ORDER BY net_sales DESC) rn
          FROM net
        ) WHERE rn = 1 ORDER BY ym
    """).fetchall()

    # 月度渠道 ROI
    channel_roi = con.execute(f"""
        SELECT strftime(d.stat_date, '%Y-%m') ym, ch.channel_id,
               round(sum(d.attributed_gmv)::DOUBLE / NULLIF(sum(d.spend)::DOUBLE, 0), 2) roi
        FROM ods.ads_daily_stats d
        JOIN dim.ads_campaigns c USING (campaign_id)
        JOIN dim.channels ch ON ch.channel_id = c.channel_id
        WHERE d.stat_date BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()

    # 90 天复购率（月度）：对每个观察月，窗口 = [月末−89天, 月末]，
    # 窗口内有单用户中下单≥2 个不同日期的用户占比（与 validate.py 口径一致）。
    repeat = {}
    for ym, *_ in monthly:
        w_end = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        w_start = (w_end - pd.Timedelta(days=89)).date()
        r = con.execute("""
            WITH w AS (
              SELECT user_id, count(DISTINCT created_at::DATE) n_days
              FROM ods.orders
              WHERE created_at::DATE BETWEEN ? AND ?
              GROUP BY user_id
            )
            SELECT (SELECT count(*) FROM w WHERE n_days >= 2)::DOUBLE
                 / NULLIF((SELECT count(*) FROM w), 0)
            """, [str(w_start), str(w_end.date())]).fetchone()[0]
        repeat[ym] = round(float(r or 0.0), 4)

    gt = {
        "monthly": {
            m[0]: {"gmv": m[1], "net_sales": m[2], "orders": m[3]} for m in monthly
        },
        "top_sku_by_net_sales": {m[0]: m[1] for m in top_sku},
        "channel_roi": {},
        "repeat_rate_90d": repeat,
        "visible_data_cutoff": end,
    }
    for ym, ch, roi in channel_roi:
        gt["channel_roi"].setdefault(ym, {})[ch] = roi
    return gt


def gen_manifest(cfg, con: duckdb.DuckDBPyConnection, counts: dict,
                 elapsed_sec: float, config_path: Path) -> dict:
    """warehouse/manifest.json：seed / config 哈希 / 行数 / 耗时。"""
    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    rows = {k: v for k, v in counts.items() if not k.startswith("_")}
    return {
        "seed": cfg.seed,
        "config_file": str(config_path),
        "config_hash": cfg_hash,
        "date_range": [cfg.date_range.start, cfg.date_range.end],
        "row_counts": rows,
        "elapsed_sec": round(elapsed_sec, 1),
        "generated_at": pd.Timestamp.now().isoformat(),
    }


def emit_all(con: duckdb.DuckDBPyConnection, cfg, counts: dict, elapsed_sec: float,
             ddl_root: Path, config_path: Path, meta_root: Path, manifest_path: Path) -> dict:
    """槽位5 出口：输出 table_docs / ground_truth / manifest。"""
    meta_root.mkdir(parents=True, exist_ok=True)

    table_docs = gen_table_docs(ddl_root)
    (meta_root / "table_docs.json").write_text(
        json.dumps(table_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    ground_truth = gen_ground_truth(con, cfg)
    (meta_root / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = gen_manifest(cfg, con, counts, elapsed_sec, config_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"table_docs": len(table_docs), "ground_truth": len(ground_truth.get("monthly", {})),
            "manifest": len(manifest.get("row_counts", {}))}
