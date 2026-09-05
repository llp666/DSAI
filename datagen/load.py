"""parquet 中间产物写出 + DuckDB 加载（SPEC §2/§9）。

- 写出：duckdb COPY（无需 pyarrow），一表一文件：warehouse/parquet/<schema>/<table>.parquet；
- 加载：INSERT INTO <schema>.<table> SELECT ... FROM read_parquet(...)，
  日期/金额列在 SQL 层 CAST（金额 DECIMAL(18,2)，§13 坑6）；
- 未生成的表（optional 关闭）保持 DDL 建好的空表。
"""

from pathlib import Path

import duckdb
import pandas as pd

# 各表的日期列（parquet 里统一为 TIMESTAMP/DATETIME，加载时 CAST AS DATE）
DATE_COLS: dict[str, list[str]] = {
    "dim_date": ["date"],
    "products": ["listing_date"],
    "users": ["register_date"],
    "festival_calendar": ["festival_date"],
    "refunds": ["refund_date"],
    "ads_campaigns": ["start_date", "end_date"],
    "ads_daily_stats": ["stat_date"],
    "promotions": ["start_date", "end_date"],
    "purchase_orders": ["eta_date"],
    "inventory_snapshot": ["snapshot_date"],
}

# 各表的金额列（加载时 CAST AS DECIMAL(18,2)）
DECIMAL_COLS: dict[str, list[str]] = {
    "products": ["cost_price", "base_price"],
    "orders": ["pay_amount"],
    "order_items": ["sale_price"],
    "refunds": ["refund_amount"],
    "order_payments": ["pay_amount"],
    "ads_campaigns": ["budget"],
    "ads_daily_stats": ["spend", "attributed_gmv"],
    "ad_conversions": ["attributed_gmv"],
    "purchase_order_items": ["unit_cost"],
}

TABLE_SCHEMA: dict[str, str] = {
    "dim_date": "dim", "categories": "dim", "products": "dim", "users": "dim",
    "festival_calendar": "dim", "channels": "dim", "suppliers": "dim",
    "warehouses": "dim", "ads_campaigns": "dim", "promotions": "dim",
    "orders": "ods", "order_items": "ods", "refunds": "ods",
    "order_payments": "ods", "order_status_log": "ods", "coupons": "ods",
    "order_coupons": "ods", "user_profiles": "ods", "user_level_log": "ods",
    "traffic_events": "ods", "ad_conversions": "ods", "product_price_log": "ods",
    "warehouse_stock": "ods", "inbound_records": "ods", "stock_moves": "ods",
    "after_sales": "ods", "search_logs": "ods", "ads_daily_stats": "ods",
    "purchase_orders": "ods", "purchase_order_items": "ods",
    "inventory_snapshot": "ods",
}


def normalize_df(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """写出前归一化：日期列转 datetime64（parquet 无原生 DATE 对象列问题）。"""
    df = df.copy()
    for col in DATE_COLS.get(table, []):
        df[col] = pd.to_datetime(df[col])
    return df


def table_dir(parquet_root: Path, table: str) -> Path:
    return parquet_root / TABLE_SCHEMA[table]


def write_parquet(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame,
                  parquet_root: Path) -> Path:
    """注册 DataFrame 后 COPY 为 parquet（覆盖旧文件）。"""
    out = table_dir(parquet_root, table)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{table}.parquet"
    view = f"__df_{table}"
    con.register(view, normalize_df(table, df))
    try:
        tmp = path.with_suffix(".parquet.tmp")
        con.execute(f"COPY {view} TO '{path.as_posix()}' (FORMAT PARQUET)")
        if tmp.exists():
            tmp.unlink()
    finally:
        con.unregister(view)
    return path


def load_table(con: duckdb.DuckDBPyConnection, table: str, parquet_root: Path) -> int:
    """INSERT INTO ... SELECT ... CAST。返回加载行数。"""
    schema = TABLE_SCHEMA[table]
    path = (table_dir(parquet_root, table) / f"{table}.parquet").as_posix()
    cols = [r[0] for r in con.execute(
        f"SELECT column_name FROM duckdb_columns() WHERE schema_name='{schema}' "
        f"AND table_name='{table}' ORDER BY column_index"
    ).fetchall()]

    exprs = []
    for c in cols:
        e = f'"{c}"'
        if c in DATE_COLS.get(table, []):
            e = f'CAST("{c}" AS DATE)'
        elif c in DECIMAL_COLS.get(table, []):
            e = f'CAST("{c}" AS DECIMAL(18,2))'
        exprs.append(e)
    select = ", ".join(exprs)
    con.execute(f"INSERT INTO {schema}.{table} SELECT {select} FROM read_parquet('{path}')")
    n = con.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone()[0]
    return n
