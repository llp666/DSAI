"""agent/embedding.py：embedding 适配层（provider 可切换）。

- gitee Qwen3-Embedding（OpenAI 兼容，instruction 查询侧指令）
- jina-embeddings-v5-omni-small（REST API，task=retrieval.query，1024 维）
两实现同维度，可无缝切换（如 gitee 免费额度耗尽 → 切 jina）。
"""

from __future__ import annotations

import json

import requests
from openai import OpenAI


class Embedder:
    def __init__(self, provider: str, base_url: str, api_key: str, model: str,
                 dimensions: int, instruction: str = "",
                 task: str = "retrieval.query"):
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._dims = dimensions
        self._instruction = instruction
        self._task = task
        self._cache: dict[str, list[float]] = {}
        self._jina_client = None
        self._openai_client = None
        if provider == "jina":
            self._jina_client = requests.Session()
        else:  # openai 兼容
            self._openai_client = OpenAI(
                base_url=base_url, api_key=api_key,
                default_headers={"X-Failover-Enabled": "true"}, timeout=120)

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        to_fetch: list[str] = []
        idx: list[int] = []
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        for i, t in enumerate(texts):
            key = self._cache_key(t, query)
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                to_fetch.append(t)
                idx.append(i)
        if to_fetch:
            emb = self._encode_api(to_fetch, query=query)
            for pos, e in zip(idx, emb):
                self._cache[self._cache_key(texts[pos], query)] = e
                out[pos] = e
        return out

    def _encode_api(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if self._provider == "jina":
            payload = {
                "model": self._model,
                "task": "retrieval.query" if query else "retrieval.passage",
                "normalized": True,
                "input": texts,
            }
            r = self._jina_client.post(
                self._base_url, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}"},
                data=json.dumps(payload), timeout=120)
            r.raise_for_status()
            data = r.json()
            return [d["embedding"] for d in data["data"]]
        # openai 兼容（gitee）
        inputs = [
            f"{self._instruction}{t}" if (query and self._instruction) else t
            for t in texts
        ]
        resp = self._openai_client.embeddings.create(
            input=inputs, model=self._model, dimensions=self._dims)
        return [d.embedding for d in resp.data]

    def encode_one(self, text: str, *, query: bool = False) -> list[float]:
        return self.encode([text], query=query)[0]

    def _cache_key(self, text: str, query: bool) -> str:
        return ("q" if query else "d") + self._provider + text


class Reranker:
    """jina rerank：对候选文档按查询相关性重排（混合检索 v2 用）。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._url = base_url
        self._key = api_key
        self._model = model

    def rerank(self, query: str, documents: list[str], *, top_n: int = 5
               ) -> list[dict]:
        r = requests.post(self._url, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "query": query,
                  "documents": documents, "top_n": top_n,
                  "return_documents": False}, timeout=120)
        r.raise_for_status()
        data = r.json()
        out = []
        for res in data["results"]:
            out.append({"index": res["index"], "score": res["relevance_score"]})
        return out
