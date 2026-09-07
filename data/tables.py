"""data/tables.py：真实数据表白名单（has_data 判定依据）。

真实表 = 有数据的表（datagen 生成，31 张）；catalog 表 = 目录级空表（0 行）。
检索层据此给语料打 has_data 元数据标签，替代硬编码后缀/白名单散落判断。
"""

from __future__ import annotations

REAL_TABLES = {
    "dim.dim_date", "dim.categories", "dim.products", "dim.users",
    "dim.channels", "dim.ads_campaigns", "dim.promotions", "dim.suppliers",
    "dim.warehouses", "dim.festival_calendar",
    "ods.orders", "ods.order_items", "ods.refunds", "ods.order_payments",
    "ods.order_status_log", "ods.ads_daily_stats", "ods.ad_conversions",
    "ods.purchase_orders", "ods.purchase_order_items", "ods.inventory_snapshot",
    "ods.warehouse_stock", "ods.inbound_records", "ods.stock_moves",
    "ods.traffic_events", "ods.product_price_log", "ods.coupons",
    "ods.order_coupons", "ods.user_profiles", "ods.user_level_log",
    "ods.after_sales", "ods.search_logs",
}


def has_data(table: str) -> bool:
    return table in REAL_TABLES
