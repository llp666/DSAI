"""agent/retriever.py：检索器（v1 向量 / v2 混合检索）。

- 阶段一：硬编码 Top-3（orders/order_items/refunds）注入；
- 阶段二 v1：向量召回（ChromaDB）Top-k 表文档；
- 阶段二 v2：混合检索——向量召回 Top-N + 表名/关键词精确加权，RRF 融合取 Top-k；
  可选 jina rerank 重排。

v2 动机：catalog 干扰表与表名强信号问题（纯向量对表名不敏感，见 ACCURACY.md）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 阶段一硬编码 Top-3（月度 GMV/净销/复购率评测所需表）
TOP3_KEYS = ["ods.orders", "ods.order_items", "ods.refunds"]

# 中文业务词 → 强关联表名关键词（表名/字段名命中加权）
DOMAIN_TERMS = {
    "订单": ["order"], "退款": ["refund"], "支付": ["pay"], "发货": ["order_status_log"],
    "sku": ["product", "sku"], "商品": ["product"], "品类": ["categor"],
    "用户": ["user"], "复购": ["user", "order"], "会员": ["user"],
    "渠道": ["channel"], "roi": ["ads_daily_stats", "ad_conversions"],
    "广告": ["ads", "ad_"], "花费": ["ads_daily_stats"],
    "库存": ["inventory", "stock"], "断货": ["inventory", "stock_moves"],
    "补货": ["purchase_order"], "在途": ["purchase_order"], "供应商": ["supplier"],
    "入库": ["inbound"], "仓库": ["warehouse"], "动销": ["stock_moves", "order_items"],
    "退货": ["refund"], "净销售额": ["order", "refund"], "毛利": ["order_items", "product"],
    "促销": ["promo"], "大促": ["promo"], "618": ["promo", "order"],
    "时间": ["dim_date"], "季度": ["dim_date", "order"], "城市": ["user"],
}


# 真实数据表白名单（31 张，meta/table_docs.json 中非 catalog 的真实表）
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


def _is_catalog(table: str) -> bool:
    """catalog 表：不在真实表白名单内（datagen 模板池生成，0 行空表）。"""
    return table not in REAL_TABLES


def _keyword_rank(question: str, doc: dict) -> float:
    """基于问题业务词 → 表名/字段/描述 的关键词加权分。

    catalog 干扰表不参与关键词加分（避免 ods_channel_agg 这类相似名被误抬）。
    """
    if _is_catalog(doc.get("table", "")):
        return 0.0
    q = question.lower()
    score = 0.0
    table_name = doc.get("table", "")
    fields = " ".join(c["name"] for c in doc.get("columns", []))
    desc = (doc.get("description", "") + " " + fields).lower()
    for term, keys in DOMAIN_TERMS.items():
        if term.lower() in q:
            # 表名命中加权最高，字段/描述次之；订单/退款核心表权重更高
            for k in keys:
                if k in table_name.lower():
                    score += 3.0 if k in ("order", "refund") else 2.0
                elif k in desc:
                    score += 1.0
    # 表名直接包含问题里的英文词（如 sku_id）加分
    for word in re.findall(r"[a-z_]+", q):
        if word in table_name.lower():
            score += 1.5
    return score


def _rrf_fuse(vector_ranks: dict[str, int], keyword_scores: dict[str, float],
              k: int = 60) -> list[str]:
    """RRF 融合：Σ 1/(k+rank) 按向量排序 + 关键词加权（作为伪 rank）。"""
    fused: dict[str, float] = {}
    for tid, rank in vector_ranks.items():
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (k + rank)
    for tid, score in keyword_scores.items():
        if score > 0:
            # 关键词命中：按分数折算伪 rank（分数越高 rank 越前，加成越大）
            pseudo_rank = max(1, 60 - int(score * 10))
            fused[tid] = fused.get(tid, 0.0) + 1.0 / (k + pseudo_rank)
    return sorted(fused, key=fused.get, reverse=True)


class Retriever:
    def __init__(self, meta_dir: Path):
        docs = json.loads((Path(meta_dir) / "table_docs.json").read_text(encoding="utf-8"))
        self._docs = docs

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """阶段一：忽略 question，固定返回硬编码 Top-3。"""
        return [self._docs[key] for key in TOP3_KEYS[:k]]

    def retrieve_vector(self, question: str, vector_store, *,
                        top_tables: int = 5,
                        where: dict | None = None) -> list[dict]:
        """阶段二 v1：向量召回，过滤出表文档，返回 Top-k。"""
        hits = vector_store.query(question, top_k=top_tables + 5, where=where)
        table_ids = [h["id"] for h in hits if h["doc_type"] == "table"]
        out = []
        for t in table_ids:
            if t in self._docs:
                out.append(self._docs[t])
            if len(out) >= top_tables:
                break
        return out

    def retrieve_hybrid(self, question: str, vector_store, *,
                        top_tables: int = 3, vector_n: int = 10,
                        reranker=None) -> list[dict]:
        """阶段二 v2：混合检索。

        - 向量召回 Top-N（N>K），记 rank；
        - 关键词精确加权（表名/字段/描述命中）；
        - RRF 融合 → Top-K 表文档；
        - 可选 rerank 重排。
        """
        hits = vector_store.query(question, top_k=vector_n)
        # 向量 rank（只算表文档）
        vector_ranks = {}
        for i, h in enumerate(hits):
            if h["doc_type"] == "table":
                vector_ranks[h["id"]] = i  # 0-based

        # 关键词加权（全库真实表计算，向量没召回的强命中也能进候选）
        keyword_scores = {}
        q_lower = question.lower()
        for tid, doc in self._docs.items():
            if _is_catalog(tid):
                continue
            kw = _keyword_rank(question, doc)
            if kw > 0:
                keyword_scores[tid] = kw
            else:
                table_name = doc.get("table", "").lower()
                for word in re.findall(r"[a-z_]+", q_lower):
                    if word in table_name and len(word) >= 4:
                        keyword_scores[tid] = keyword_scores.get(tid, 0.0) + 1.5

        fused = _rrf_fuse(vector_ranks, keyword_scores)
        # 优先真实表：catalog 干扰表（空表）不进注入 Top-K，用后续真实表替补
        ranked_ids = [t for t in fused if t in self._docs]
        out_ids = []
        for t in ranked_ids:
            if len(out_ids) >= top_tables:
                break
            if not _is_catalog(t):
                out_ids.append(t)
        out = [self._docs[t] for t in out_ids]

        if reranker and len(out) > 1:
            docs_txt = [Retriever._table_text(d) for d in out]
            ranked = reranker.rerank(question, docs_txt, top_n=len(out))
            # rerank 返回 index 顺序，重排 out
            ordered = [out[i["index"]] for i in sorted(ranked, key=lambda x: x["score"], reverse=True)]
            out = ordered
        return out

    @staticmethod
    def _table_text(doc: dict) -> str:
        return f"{doc.get('table','')}：{doc.get('description','')} " \
               f"{' '.join(c['name'] for c in doc.get('columns', []))}"

    @staticmethod
    def render(tables: list[dict]) -> str:
        parts = []
        for t in tables:
            cols = "\n".join(
                f"    - {c['name']}: {c.get('type', '')} {c.get('comment', '')}".rstrip()
                for c in t.get("columns", [])
            )
            parts.append(
                f"### 表 {t['table']}\n"
                f"描述：{t.get('description', '')}\n"
                f"字段：\n{cols}"
            )
        return "\n\n".join(parts)
