"""agent/retriever.py：阶段一硬编码 Top-3 表检索。

从 meta/table_docs.json 取 orders / order_items / refunds 三张核心表的结构，
渲染为注入 Prompt 的受控文本（对应方案 5.3 的动态注入格式，本阶段先硬编码）。
"""

from __future__ import annotations

import json
from pathlib import Path

# 阶段一固定 Top-3（覆盖月度 GMV/净销/复购率评测所需的全部表）
TOP3_KEYS = ["ods.orders", "ods.order_items", "ods.refunds"]


class Retriever:
    def __init__(self, meta_dir: Path):
        docs = json.loads((Path(meta_dir) / "table_docs.json").read_text(encoding="utf-8"))
        self._docs = {k: docs[k] for k in TOP3_KEYS}

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        # 阶段一：忽略 question 语义，固定返回 Top-3（检索层待阶段二接入向量库）
        return [self._docs[key] for key in TOP3_KEYS[:k]]

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
