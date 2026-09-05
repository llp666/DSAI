"""catalog.py：目录级空表生成（SPEC §8）。

从模板池组合出大量"像真实表"的表名（ods_/dwd_/dws_/ads_ × 域词 × 后缀），
建到 dwd/dws/ads schema，不填数据。用于：
- 压测（Catalog 规模、ChromaDB 语料量、检索评测的干扰项密度）；
- 近亲表（同域不同前缀/后缀）刻意制造检索歧义，考验语义层/评测脚本的消歧能力。

用法：`python -m datagen.cli --config ... --catalog-only 270`
"""

import re

# ---- 域词池：与真实数仓表名对齐的"业务主题" ----
DOMAIN_WORDS = [
    "order", "user", "product", "traffic", "supply", "inventory",
    "marketing", "pay", "refund", "stock", "channel", "campaign",
]

# ---- 后缀池：贴近真实数仓的分层/衍生表命名 ----
SUFFIXES = [
    "_wide", "_ext", "_diag", "_tmp", "_bak", "_2025",
    "_detail", "_agg", "_d1", "_latest", "_his", "_snapshot",
]

# ---- 近亲组：同域下刻意制造"只差一个词"的表，检索时互成干扰 ----
NEAR_DUPES = [
    ("ads_order_wide", "dwd_order_detail"),
    ("dws_user_daily", "dwd_user_d1"),
    ("ads_product_ext", "dwd_product_ext"),
    ("dws_traffic_agg", "ads_traffic_agg"),
]


def gen_table_names(n: int) -> list[str]:
    """从前缀×域词×后缀模板池组合出 n 个唯一表名（确定性：字母序）。"""
    prefixes = ["ods_", "dwd_", "dws_", "ads_"]
    names: list[str] = []
    # 近亲组优先（保证检索干扰项存在）
    for a, b in NEAR_DUPES:
        names.append(a)
        names.append(b)
    # 再按 前缀×域词×后缀 模板组合填充
    for p in prefixes:
        for domain in DOMAIN_WORDS:
            for suf in SUFFIXES:
                names.append(f"{p}{domain}{suf}")
    # 去重保序，截到 n
    seen: set[str] = set()
    out: list[str] = []
    for nm in names:
        if nm in seen:
            continue
        seen.add(nm)
        out.append(nm)
        if len(out) >= n:
            break
    return out


# ---- 字段模板池：每个域一套真实列，catalog 表拼接 3-5 列 ----
FIELD_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "order": [
        ("order_id", "VARCHAR", "订单号"),
        ("order_status", "VARCHAR", "订单状态枚举"),
        ("pay_amount", "DECIMAL(18,2)", "实付金额（元）"),
        ("created_at", "TIMESTAMP", "下单时间"),
    ],
    "user": [
        ("user_id", "VARCHAR", "用户ID"),
        ("register_date", "DATE", "注册日期"),
        ("city_tier", "VARCHAR", "城市等级"),
        ("is_active", "INTEGER", "是否活跃"),
    ],
    "product": [
        ("sku_id", "VARCHAR", "商品SKU"),
        ("category_id", "VARCHAR", "品类ID"),
        ("base_price", "DECIMAL(18,2)", "基准价（元）"),
        ("listing_status", "VARCHAR", "上架状态"),
    ],
    "traffic": [
        ("event_id", "VARCHAR", "事件ID"),
        ("user_id", "VARCHAR", "触发用户"),
        ("event_type", "VARCHAR", "事件类型"),
        ("ts", "TIMESTAMP", "事件时间"),
    ],
    "supply": [
        ("supplier_id", "VARCHAR", "供应商ID"),
        ("po_id", "VARCHAR", "采购单号"),
        ("qty", "INTEGER", "采购数量"),
        ("eta_date", "DATE", "预计到货"),
    ],
    "inventory": [
        ("sku_id", "VARCHAR", "商品SKU"),
        ("snapshot_date", "DATE", "快照日期"),
        ("on_hand_qty", "INTEGER", "在库数量"),
        ("in_transit_qty", "INTEGER", "在途数量"),
    ],
    "marketing": [
        ("campaign_id", "VARCHAR", "投放计划ID"),
        ("channel_id", "VARCHAR", "渠道ID"),
        ("spend", "DECIMAL(18,2)", "花费（元）"),
        ("impressions", "BIGINT", "曝光次数"),
    ],
    "pay": [
        ("payment_id", "VARCHAR", "支付流水号"),
        ("order_id", "VARCHAR", "订单号"),
        ("pay_channel", "VARCHAR", "支付渠道"),
        ("pay_amount", "DECIMAL(18,2)", "支付金额（元）"),
    ],
    "refund": [
        ("refund_id", "VARCHAR", "退款单号"),
        ("order_id", "VARCHAR", "订单号"),
        ("refund_amount", "DECIMAL(18,2)", "退款金额（元）"),
        ("refund_date", "DATE", "退款日期"),
    ],
    "stock": [
        ("sku_id", "VARCHAR", "商品SKU"),
        ("warehouse_id", "VARCHAR", "仓库ID"),
        ("on_hand_qty", "INTEGER", "在库数量"),
        ("move_type", "VARCHAR", "出入库类型"),
    ],
    "channel": [
        ("channel_id", "VARCHAR", "渠道ID"),
        ("channel_type", "VARCHAR", "渠道类型"),
        ("channel_name", "VARCHAR", "渠道名称"),
    ],
    "campaign": [
        ("campaign_id", "VARCHAR", "投放计划ID"),
        ("start_date", "DATE", "开始日期"),
        ("end_date", "DATE", "结束日期"),
        ("budget", "DECIMAL(18,2)", "预算（元）"),
    ],
}


def _domain_of(name: str) -> str:
    """从表名反推域词（`ads_order_wide` → order）。"""
    m = re.match(r"(?:ods|dwd|dws|ads)_(\w+)", name)
    if not m:
        return "other"
    dom = m.group(1)
    for key in FIELD_TEMPLATES:
        if dom.startswith(key) or key in dom:
            return key
    return "other"


def _table_desc(name: str, domain: str) -> str:
    """模板化 description：带易混提示，制造检索干扰。"""
    layer = name.split("_", 1)[0]
    layer_cn = {"ods": "贴源明细", "dwd": "明细中间层", "dws": "汇总服务层",
                "ads": "应用输出层"}.get(layer, "分层")
    return (f"{layer_cn}表：{name}，业务域为{domain}，"
            f"与同域相关表（如 {_nearby(name)}）字段口径可能不同，使用前请确认口径")


def _nearby(name: str) -> str:
    """返回一个与 name 同域但前缀/后缀不同的近亲表名（干扰提示用）。"""
    for a, b in NEAR_DUPES:
        if name in (a, b):
            return b if name == a else a
    for pair in NEAR_DUPES:
        for nm in pair:
            if _domain_of(name) == _domain_of(nm):
                return nm
    return "对应明细表"


def gen_catalog_ddl(names: list[str]) -> str:
    """为每个表生成 CREATE TABLE + COMMENT 语句（空表，无数据）。"""
    parts: list[str] = ["CREATE SCHEMA IF NOT EXISTS dwd;",
                        "CREATE SCHEMA IF NOT EXISTS dws;",
                        "CREATE SCHEMA IF NOT EXISTS ads;",
                        "CREATE SCHEMA IF NOT EXISTS ods;"]
    for name in names:
        dom = _domain_of(name)
        fields = FIELD_TEMPLATES[dom]
        # 每表拼 3-5 列：取该域模板前 4 列
        cols = fields[:4]
        # 表名缺 schema 前缀时落到 main——显式带上生成的 schema
        if "." not in name:
            layer = name.split("_", 1)[0]
            fq = f"{layer}.{name}"
        else:
            fq = name
        col_lines = ",\n    ".join(f"{c} {t}" for c, t, _ in cols)
        parts.append(f"CREATE TABLE IF NOT EXISTS {fq} (\n    {col_lines}\n);")
        parts.append(f"COMMENT ON TABLE {fq} IS '{_table_desc(name, dom)}';")
        for c, t, comment in cols:
            parts.append(f"COMMENT ON COLUMN {fq}.{c} IS '{comment}';")
    return "\n\n".join(parts)


def gen_catalog_table_docs(names: list[str]) -> dict[str, dict]:
    """catalog 表的 table_docs 条目（并入 RAG 语料，作为检索干扰项）。"""
    docs: dict[str, dict] = {}
    for name in names:
        dom = _domain_of(name)
        cols = FIELD_TEMPLATES[dom][:4]
        docs[name] = {
            "table": name,
            "schema": name.split("_", 1)[0],
            "domain": dom,
            "description": _table_desc(name, dom),
            "columns": [{"name": c, "comment": cmt} for c, _, cmt in cols],
            "pk": [], "fks": [], "update_freq": "T+1",
            "sample_questions": [f"{name} 的数据口径", f"{name} 与近亲表的差异"],
        }
    return docs
