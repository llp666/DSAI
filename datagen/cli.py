"""cli.py：参数解析与编排入口（SPEC §11）。

槽位2 流水线：维度域 → 订单域 → 用户首单钳制 → parquet 写出 → DuckDB 加载。
营销/库存/脏数据/meta 在后续槽位按 SPEC §4.4 DAG 顺序接入。
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb

from .config import load_config

DDL_PACKAGE = "datagen.ddl"


def ddl_files() -> list[Path]:
    """按文件名序返回 DDL 文件（01_xxx.sql ... 05_xxx.sql）。"""
    root = Path(__file__).parent / "ddl"
    files = sorted(root.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"未找到 DDL 文件: {root}")
    return files


def apply_ddl(db_path: Path) -> int:
    """在 DuckDB 中执行全部 DDL，返回建表数量。幂等：重复执行前先删库文件。"""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        for f in ddl_files():
            con.execute(f.read_text(encoding="utf-8"))
        n_tables = con.execute(
            """
            SELECT count(*) FROM duckdb_tables()
            WHERE schema_name IN ('dim', 'ods', 'dwd', 'dws', 'ads')
            """
        ).fetchone()[0]
    finally:
        con.execute("CHECKPOINT")
        con.close()
    return n_tables


def run_pipeline(cfg, db_path: Path, parquet_root: Path, meta_root: Path,
                 dirty_manifest_path: Path | None = None,
                 config_path: Path | None = None) -> dict[str, int]:
    """按 DAG 顺序生成全部已启用表并入库，返回各表行数。

    ⑬ ground_truth（干净数据）→ ⑭ dirty 注入 → ⑮ emit_meta。
    """
    from datagen.entities import (
        gen_categories,
        gen_channels,
        gen_dim_date,
        gen_festival_calendar,
        gen_products,
        gen_users,
    )
    from datagen.entities.users import clamp_register_for_orderers
    from datagen.facts import gen_inventory, gen_marketing_full, gen_orders_full
    from datagen.load import load_table, write_parquet
    from datagen.rng import make_rng

    t0 = time.perf_counter()
    con = duckdb.connect(str(db_path))

    # ①-④ 维度域
    cats = gen_categories(cfg)
    products = gen_products(cfg, cats)
    users = gen_users(cfg)
    dims = {
        "dim_date": gen_dim_date(cfg),
        "festival_calendar": gen_festival_calendar(cfg),
        "categories": cats,
        "products": products,
        "users": users,
        "channels": gen_channels(cfg, cfg.demo_cases.negative_roi_channels),
    }

    # ⑤-⑦ 订单域
    orders, items, refunds, payments, status_log = gen_orders_full(cfg, users, products, cats)

    # 有单用户 register_date 钳到首单日前 U(1,45) 天（防未注册先下单）
    first_order = orders.groupby("user_id")["created_at"].min()
    first_by_user = first_order.reindex(users["user_id"]).reset_index(drop=True)
    clamp_register_for_orderers(users, first_by_user, rng=None)

    # ⑧-⑨ 营销域（依赖订单/支付/渠道）
    marketing = gen_marketing_full(cfg, orders, payments, dims["channels"])
    # 全站行为漏斗（可选，SPEC §3.2 P1 可关）
    from datagen.facts.marketing import gen_traffic_events
    if cfg.optional.traffic_events:
        marketing["traffic_events"] = gen_traffic_events(
            cfg, orders, make_rng(cfg.seed, "traffic_events"))

    # ⑩-⑫ 库存域（依赖订单明细/商品/供应商）
    inventory = gen_inventory(cfg, items, orders, products)

    facts = {
        "orders": orders,
        "order_items": items,
        "refunds": refunds,
        "order_payments": payments,
        "order_status_log": status_log,
    }
    facts.update(marketing)
    facts.update(inventory)

    # ⑮ 写出 + 加载（parquet → DuckDB）
    counts: dict[str, int] = {}
    for name, df in {**dims, **facts}.items():
        write_parquet(con, name, df, parquet_root)
        counts[name] = load_table(con, name, parquet_root)

    # ⑬ ground_truth：基于干净数据计算（必须注脏前！）
    from datagen.emit_meta import emit_all, gen_ground_truth
    ground_truth = gen_ground_truth(con, cfg)
    counts["_ground_truth_months"] = len(ground_truth.get("monthly", {}))

    # ⑭ dirty 注入（改库，manifest 记录注入清单）
    if dirty_manifest_path is not None:
        from datagen.dirty import inject_dirty
        inject_dirty(cfg, con, dirty_manifest_path)

    # ⑮ emit_meta：table_docs / ground_truth / manifest 写盘
    emit_all(con, cfg, counts, time.perf_counter() - t0,
             Path(__file__).parent / "ddl", config_path or Path("config.yml"),
             meta_root, db_path.parent / "manifest.json")

    con.execute("CHECKPOINT")
    con.close()

    counts["_elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datagen", description="电商模拟数仓数据生成器")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的 seed")
    parser.add_argument(
        "--db", default="warehouse/ecommerce.duckdb", help="DuckDB 文件路径"
    )
    parser.add_argument(
        "--parquet-root", default="warehouse/parquet", help="parquet 中间产物目录"
    )
    parser.add_argument("--meta-root", default="meta", help="meta 三件套输出目录")
    parser.add_argument("--catalog-only", type=int, default=0,
                        help="只建 catalog 目录级空表（N 张），跳过数据生成（SPEC §8）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    t0 = time.perf_counter()
    if args.catalog_only:
        from datagen.catalog import gen_catalog_ddl, gen_catalog_table_docs, gen_table_names
        db_path = Path(args.db)
        # catalog-only 不重建已有库（否则清掉全量数据）：库不存在才 apply_ddl
        if not db_path.exists():
            n_tables = apply_ddl(db_path)
        names = gen_table_names(args.catalog_only)
        con = duckdb.connect(str(db_path))
        try:
            con.execute(gen_catalog_ddl(names))
            n = con.execute("SELECT count(*) FROM duckdb_tables() "
                            "WHERE schema_name IN ('dwd','dws','ads','ods')").fetchone()[0]
            # catalog 表的 table_docs 并入 meta/table_docs.json（RAG 干扰项）
            docs_path = Path(args.meta_root) / "table_docs.json"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            cat_docs = gen_catalog_table_docs(names)
            if docs_path.exists():
                import json
                existing = json.loads(docs_path.read_text(encoding="utf-8"))
                existing.update(cat_docs)
                docs_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
            else:
                docs_path.write_text(json.dumps(cat_docs, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        finally:
            con.execute("CHECKPOINT")
            con.close()
        print(f"[datagen] catalog-only：建 {len(names)} 张空表（全库 {n} 张），耗时 "
              f"{time.perf_counter() - t0:.1f}s，table_docs 已并入 {docs_path}")
        return 0

    n_tables = apply_ddl(Path(args.db))
    counts = run_pipeline(
        cfg, Path(args.db), Path(args.parquet_root), Path(args.meta_root),
        dirty_manifest_path=Path(args.meta_root) / "dirty_manifest.json",
        config_path=Path(args.config),
    )
    elapsed = time.perf_counter() - t0

    print(f"[datagen] 完成：{len(counts) - 1} 张表有数据，耗时 {elapsed:.1f}s，seed={cfg.seed}")
    for name, n in sorted(counts.items()):
        if not name.startswith("_"):
            print(f"  {name:20s} {n:>9,}")
    print(f"[datagen] meta 输出：{args.meta_root}/，dirty_manifest 已写盘")
    return 0


if __name__ == "__main__":
    sys.exit(main())
