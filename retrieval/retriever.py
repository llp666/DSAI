"""retrieval/retriever.py：retriever (v1 vector / v2 hybrid).

- stage 1: hardcoded Top-3 (orders/order_items/refunds);
- stage 2 v1: vector recall (ChromaDB) Top-k table docs;
- stage 2 v2: hybrid — vector recall Top-N + table-name/keyword exact weighting, RRF fusion → Top-k;
  optional jina rerank.

v2 motivation: catalog decoy tables and table-name strong signal (pure vectors are insensitive to
table names, see ACCURACY.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from data.tables import has_data

# stage-1 hardcoded Top-3 (tables needed for monthly GMV/net-sales/repurchase eval)
TOP3_KEYS = ["ods.orders", "ods.order_items", "ods.refunds"]

# Chinese business word → strongly-related table-name keywords (table/field name hit weighting)
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
    "促销": ["promo"], "大促": ["promo", "order"], "618": ["promo", "order"],
    "销量": ["order_items", "order"], "成交额": ["order"],
    "时间": ["dim_date"], "季度": ["dim_date", "order"], "城市": ["user"],
}
# question contains these → force-recall the orders table (score +3, pseudo rank 1)
ORDER_FORCE_WORDS = ("复购", "净销售额", "销量", "卖得最好", "大促", "GMV", "成交额", "毛利")


def _is_catalog(table: str) -> bool:
    """catalog table: no data (has_data=False, datagen template pool 0-row empty table)."""
    return not has_data(table)


def _keyword_rank(question: str, doc: dict) -> float:
    """Keyword weighting: business word → table name / field / description.

    catalog decoy tables get no keyword boost (so similar names like ods_channel_agg don't get bumped).
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
            # table-name hit weighted highest, field/description next; order/refund core tables higher
            for k in keys:
                if k in table_name.lower():
                    score += 3.0 if k in ("order", "refund") else 2.0
                elif k in desc:
                    score += 1.0
    # order-strong words: orders table (name contains order) extra +3 → guaranteed Top-1
    if any(w.lower() in q for w in ORDER_FORCE_WORDS) and "order" in table_name.lower():
        score += 3.0
    # table name directly contains an English word from the question (e.g. sku_id)
    for word in re.findall(r"[a-z_]+", q):
        if word in table_name.lower():
            score += 1.5
    return score


def _rrf_fuse(vector_ranks: dict[str, int], keyword_scores: dict[str, float],
              k: int = 60) -> list[str]:
    """RRF fusion: vector ranks + keyword weights (strong keyword hit → low pseudo-rank ≈ Top-1)."""
    fused: dict[str, float] = {}
    for tid, rank in vector_ranks.items():
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (k + rank)
    for tid, score in keyword_scores.items():
        if score > 0:
            # keyword score >= 3 → pseudo-rank 3-5 (strong, near vector Top-1); lower → rank 8+
            pseudo_rank = max(1, 8 - int(score))
            fused[tid] = fused.get(tid, 0.0) + 1.0 / (k + pseudo_rank)
    return sorted(fused, key=fused.get, reverse=True)


class Retriever:
    def __init__(self, meta_dir: Path):
        docs = json.loads((Path(meta_dir) / "table_docs.json").read_text(encoding="utf-8"))
        self._docs = docs

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """stage 1: ignore the question, return the fixed hardcoded Top-3."""
        return [self._docs[key] for key in TOP3_KEYS[:k]]

    def retrieve_vector(self, question: str, vector_store, *,
                        top_tables: int = 5,
                        where: dict | None = None) -> list[dict]:
        """stage 2 v1: vector recall, filter to table docs, return Top-k."""
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
        """stage 2 v2: hybrid retrieval.

        - vector recall Top-N (N > K), record rank;
        - keyword exact weighting (table name / field / description);
        - RRF fusion → Top-K table docs;
        - optional rerank.
        """
        hits = vector_store.query(question, top_k=vector_n)
        # vector rank (table docs only)
        vector_ranks = {}
        for i, h in enumerate(hits):
            if h["doc_type"] == "table":
                vector_ranks[h["id"]] = i  # 0-based

        # keyword weighting (over all real tables; a strong hit not vector-recalled still enters)
        keyword_scores = self._keyword_scores(question, english_fallback=True)

        fused = _rrf_fuse(vector_ranks, keyword_scores)
        # real tables first: catalog decoy tables (empty) never enter Top-K, next real table backs up
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
            ordered = [out[i["index"]] for i in sorted(ranked, key=lambda x: x["score"], reverse=True)]
            out = ordered
        return out

    def retrieve_keyword_only(self, question: str, *, top_tables: int = 3) -> list[dict]:
        """Pure keyword retrieval (embedding failure fallback, no vector_store).

        Sorts real tables by keyword score and takes Top-K. Lower hit rate than hybrid, but usable.
        """
        keyword_scores = self._keyword_scores(question)
        ranked = sorted(keyword_scores, key=keyword_scores.get, reverse=True)
        out = []
        for tid in ranked:
            if len(out) >= top_tables:
                break
            out.append(self._docs[tid])
        return out

    def _keyword_scores(self, question: str, *, english_fallback: bool = False) -> dict[str, float]:
        """Keyword scores over all non-catalog docs."""
        scores: dict[str, float] = {}
        q_lower = question.lower()
        for tid, doc in self._docs.items():
            if _is_catalog(tid):
                continue
            kw = _keyword_rank(question, doc)
            if kw > 0:
                scores[tid] = kw
            elif english_fallback:
                # low-signal fallback: table name directly contains an English word (len >= 4)
                table_name = doc.get("table", "").lower()
                for word in re.findall(r"[a-z_]+", q_lower):
                    if word in table_name and len(word) >= 4:
                        scores[tid] = scores.get(tid, 0.0) + 1.5
        return scores

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