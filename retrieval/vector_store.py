"""retrieval/vector_store.py：ChromaDB persistent retrieval store.

- ingest: corpus entries → embeddings → Chroma collection, tagged domain/doc_type/updated_at;
- query: vector recall Top-k (v1 pure vector; hybrid is in retriever).
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from data.corpus import build_corpus
from retrieval.embedding import Embedder
from config import resolve

COLLECTION = "dsai_corpus"


class VectorStore:
    def __init__(self, chroma_dir, embedder: Embedder):
        self._client = chromadb.PersistentClient(
            path=str(resolve(Path(chroma_dir))),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedder = embedder
        self._col = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"})

    def rebuild(self, entries: list[dict], *, batch: int = 16) -> int:
        """Full corpus rebuild. Returns the number of ingested entries."""
        # batch=16: gitee free embedding tier rejects batch >= 64 with 400
        self._client.delete_collection(COLLECTION)
        self._col = self._client.create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"})

        ids, docs, metas = [], [], []
        for e in entries:
            ids.append(e["id"])
            docs.append(e["document"])
            metas.append({"doc_type": e["doc_type"], "domain": e["domain"],
                          **e["metadata"]})
        for i in range(0, len(ids), batch):
            chunk_ids = ids[i:i + batch]
            chunk_docs = docs[i:i + batch]
            emb = self._embedder.encode(chunk_docs)
            self._col.add(ids=chunk_ids, embeddings=emb, documents=chunk_docs,
                          metadatas=metas[i:i + batch])
        return len(ids)

    def query(self, question: str, *, top_k: int = 5,
              where: dict | None = None) -> list[dict]:
        """Vector recall Top-k. ``where`` filters metadata, e.g. {"domain": "订单域"}."""
        emb = self._embedder.encode_one(question, query=True)
        res = self._col.query(query_embeddings=[emb], n_results=top_k, where=where)
        out = []
        for i, rid in enumerate(res["ids"][0]):
            out.append({
                "id": rid,
                "distance": res["distances"][0][i],
                "doc_type": res["metadatas"][0][i].get("doc_type"),
                "domain": res["metadatas"][0][i].get("domain"),
            })
        return out

    @property
    def count(self) -> int:
        return self._col.count()