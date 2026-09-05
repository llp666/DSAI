"""datagen：电商模拟数仓数据生成器。

按 docs/DATAGEN_SPEC.md 实现：YAML 配置 → 确定性生成 30 表 → DuckDB，
同时产出 meta 三件套（table_docs / ground_truth / dirty_manifest）。
"""

__version__ = "0.1.0"
