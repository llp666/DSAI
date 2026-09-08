"""data/tables.py：real-data table whitelist (has_data).

Real tables = tables with data (datagen-generated, 31); catalog tables = directory-level empty tables (0 rows).
The retrieval layer tags corpus entries with has_data from here, replacing scattered suffix/whitelist checks.
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