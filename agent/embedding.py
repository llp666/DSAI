"""agent/embedding.py：embedding 适配层（gitee Qwen3-Embedding，OpenAI 兼容）。

- 支持 instruction（查询侧指令，BGE/Qwen 系提升检索质量）；
- 批量编码 + 简单缓存（同文本不重复调用 API）。
"""

from __future__ import annotations

from openai import OpenAI


class Embedder:
    def __init__(self, base_url: str, api_key: str, model: str,
                 dimensions: int, instruction: str = ""):
        self._client = OpenAI(
            base_url=base_url, api_key=api_key,
            default_headers={"X-Failover-Enabled": "true"}, timeout=120)
        self._model = model
        self._dims = dimensions
        self._instruction = instruction
        self._cache: dict[str, list[float]] = {}

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
            # instruction 只加在查询侧，避免文档侧污染
            inputs = [
                f"{self._instruction}{t}" if (query and self._instruction) else t
                for t in to_fetch
            ]
            resp = self._client.embeddings.create(
                input=inputs, model=self._model, dimensions=self._dims)
            for pos, emb in zip(idx, resp.data):
                self._cache[self._cache_key(texts[pos], query)] = emb.embedding
                out[pos] = emb.embedding
        return out

    def encode_one(self, text: str, *, query: bool = False) -> list[float]:
        return self.encode([text], query=query)[0]

    def _cache_key(self, text: str, query: bool) -> str:
        return ("q" if query else "d") + text
