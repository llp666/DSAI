# datagen · 电商模拟数仓数据生成器

从 YAML 配置出发，程序化生成五大数据域、约 30 张表、行业锚点对齐的电商数据，
写入 DuckDB，并产出 meta 三件套（table_docs / ground_truth / dirty_manifest）。

开发文档：`docs/DATAGEN_SPEC.md`

## 环境

已有 Python 环境（3.10+），核心依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 当前进度（槽位4：库存域完成）

- [x] config 加载器（`datagen/config.py`，YAML → dataclass，全字段）
- [x] 随机流管理（`datagen/rng.py`，模块名派生，含单测）
- [x] 31 表 DDL 全量（`datagen/ddl/01~05_*.sql`，含 COMMENT ON，dim 建 PK、ods 约束-free）
- [x] apply 脚本（`datagen/cli.py`，幂等建库）
- [x] 维度域生成（`datagen/entities/`：dim_date/festival_calendar/categories/products/users/channels）
- [x] 订单域生成（`datagen/facts/orders.py`：orders/order_items/refunds/order_payments/order_status_log，向量化）
- [x] 营销域生成（`datagen/facts/marketing.py`：promotions/ads_campaigns/ad_conversions/ads_daily_stats，反推勾稽）
- [x] 库存域生成（`datagen/facts/inventory.py`：suppliers/purchase_orders/inventory_snapshot/warehouse_stock 等 8 表，前向模拟 + 埋点）
- [x] parquet 写出 + DuckDB 加载（`datagen/load.py`，schema 分目录，金额/日期 SQL 层 CAST）
- [x] cli 流水线编排（生成 → 用户首单钳制 → parquet → 入库）
- [x] validate.py 勾稽校验（`make validate`，41 项断言，small/full 均全过）
- [ ] 脏数据注入 + meta 三件套（槽位 5）
- [ ] 目录级 270 空表（槽位 7–8）

## 使用

```bash
# 建库建表（当前阶段）
python -m datagen.cli --config datagen/config/scale_small.yaml

# 全量数据（槽位4后可用）
python -m datagen.cli --config datagen/config/scale_full.yaml

# 单测
python -m pytest datagen/tests -q
```

Windows 下无 make 时直接运行上述 python 命令；有 make 环境可用
`make data-small` / `make data-full` / `make test` / `make validate`。

## 目录结构

```
datagen/
├── cli.py            # 参数解析与编排
├── rng.py            # 随机流管理（模块名派生）
├── config.py         # YAML → dataclass
├── ddl/              # 01_order_domain.sql ... 05_inventory_supply.sql
├── config/           # scale_full.yaml / scale_small.yaml
├── entities/         # （槽位2）维度生成
├── facts/            # （槽位2-4）事实生成
├── dirty.py          # （槽位5）受控脏数据注入 + manifest
├── emit_meta.py      # （槽位5）table_docs + ground_truth + manifest
├── catalog.py        # （槽位7-8）目录级空表
├── validate.py       # （槽位6）业务合理性断言
└── tests/
```

## 产物

- `warehouse/ecommerce.duckdb` —— 数仓（dim/ods schema，后续 +300 目录表）
- `warehouse/manifest.json` —— seed / 配置哈希 / 行数 / 生成时间（可复现凭证）
- `meta/` —— table_docs.json / ground_truth.json / dirty_manifest.json
