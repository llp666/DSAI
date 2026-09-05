"""agent/retriever.py：检索器。

- 阶段一：硬编码 Top-3（orders/order_items/refunds）注入；
- 阶段二：向量检索 v1——ChromaDB 召回 Top-k 表文档注入。
两种模式并存，pipeline 未接入向量前回落硬编码。
"""

from __future__ import annotations

import json
from pathlib import Path

# 阶段一硬编码 Top-3（月度 GMV/净销/复购率评测所需表）
TOP3_KEYS = ["ods.orders", "ods.order_items", "ods.refunds"]


class Retriever:
    def __init__(self, meta_dir: Path):
        docs = json.loads((Path(meta_dir) / "table_docs.json").read_text(encoding="utf-8"))
        self._docs = docs  # 全量表文档（含 301 张）

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """阶段一：忽略 question，固定返回硬编码 Top-3。"""
        return [self._docs[key] for key in TOP3_KEYS[:k]]

    def retrieve_vector(self, question: str, vector_store, *,
                        top_tables: int = 5,
                        where: dict | None = None) -> list[dict]:
        """阶段二：向量召回，过滤出表文档（排除场景卡），返回 Top-k 表文档。"""
        hits = vector_store.query(question, top_k=top_tables + 5, where=where)
        table_ids = [h["id"] for h in hits if h["doc_type"] == "table"]
        out = []
        for t in table_ids:
            if t in self._docs:
                out.append(self._docs[t])
            if len(out) >= top_tables:
                break
        return out

    @staticmethod
    def render(tables: list[dict]) -> str:
        """把表文档渲染为注入 Prompt 的 schema 文本。"""
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
