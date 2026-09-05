"""agent/ingest_corpus.py：检索语料构建 + 入库 CLI。

用法：python -m agent.ingest_corpus
产出：warehouse/chroma/（ChromaDB persistent，含 domain/doc_type 标签）
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from .config import load_config, resolve
from .corpus import build_corpus
from .embedding import Embedder
from .vector_store import VectorStore


def main() -> int:
    cfg = load_config()
    emb_cfg = cfg.embedding or cfg.alt_embedding
    if not emb_cfg:
        print("缺少 EMBEDDING_* 配置（.env），跳过入库")
        return 1
    updated_at = date.today().isoformat()
    entries = build_corpus(
        meta_dir=resolve(cfg.meta_dir),
        scenarios_path=resolve(Path("semantic_layer/scenarios.yaml")),
        schema_path=resolve(Path("semantic_layer/schema/corpus.schema.json")),
        updated_at=updated_at,
    )
    print(f"语料构建完成：{len(entries)} 条（通过 jsonschema 校验）")
    n_table = sum(1 for e in entries if e["doc_type"] == "table")
    n_scn = len(entries) - n_table
    print(f"  表文档 {n_table} 条、场景卡 {n_scn} 张")

    embedder = Embedder(
        provider=emb_cfg.provider, base_url=emb_cfg.base_url,
        api_key=emb_cfg.api_key, model=emb_cfg.model,
        dimensions=emb_cfg.dimensions, instruction=emb_cfg.instruction,
        task=emb_cfg.task,
    )
    store = VectorStore(emb_cfg.chroma_dir, embedder)
    n = store.rebuild(entries)
    print(f"ChromaDB 入库完成：{n} 条（embedding={emb_cfg.provider}/{emb_cfg.model}）"
          f" → {resolve(Path(emb_cfg.chroma_dir))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
