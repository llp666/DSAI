"""retrieval/embedding.py：embedding adapter (swappable provider).

- gitee Qwen3-Embedding (OpenAI-compatible; instruction on the query side)
- jina-embeddings-v5-omni-small (REST API, task=retrieval.query, 1024 dims)
Both produce the same dimensionality, so they swap seamlessly (e.g. gitee free quota exhausted → jina).
"""

from __future__ import annotations

import json

import requests
from openai import OpenAI

# (connect, read) 秒。连接阶段必须短：api.jina.ai 不可达时 120s 的连接超时会让检索节点
# 整整挂两分钟才掉进关键词兜底（实测撞到过 ConnectTimeout 120s），而检索本该是秒级。
# 读阶段留宽：一次批量嵌入本身要几秒。
HTTP_TIMEOUT = (5, 30)


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
        else:  # openai-compatible
            self._openai_client = OpenAI(
                base_url=base_url, api_key=api_key,
                default_headers={"X-Failover-Enabled": "true"}, timeout=HTTP_TIMEOUT[1])

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
                data=json.dumps(payload), timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return [d["embedding"] for d in data["data"]]
        # openai-compatible (gitee)
        inputs = [
            f"{self._instruction}{t}" if (query and self._instruction) else t
            for t in texts
        ]
        resp = self._openai_client.embeddings.create(
            input=inputs, model=self._model, dimensions=self._dims)
        return [d.embedding for d in resp.data]

    def encode_one(self, text: str, *, query: bool = False) -> list[float]:
        return self.encode([text], query=query)[0]

    def warm_up(self) -> None:
        """Open the HTTP connection (DNS + TLS) ahead of the first real query.

        进程内第一次嵌入实测 ~1970ms，热态 ~320ms — 差值全是握手。app 启动时先热一下，
        把这一秒多从用户的第一个问题里挪走。直连 ``_encode_api`` 以避开缓存（不污染
        缓存键）。尽力而为：失败静默吞掉（嵌入不可达时检索层本就会降级到纯关键词）。

        注意这是**尽力而为**：连接池空闲久了仍会被对端回收，热启动只保证「开 app 立刻提问」
        这条最常见路径不付握手钱。
        """
        try:
            self._encode_api(["预热"], query=True)
        except Exception:
            pass

    def _cache_key(self, text: str, query: bool) -> str:
        return ("q" if query else "d") + self._provider + text


class Reranker:
    """jina rerank: reorders candidates by query relevance (hybrid v2)."""

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
                  "return_documents": False}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out = []
        for res in data["results"]:
            out.append({"index": res["index"], "score": res["relevance_score"]})
        return out