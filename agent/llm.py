"""agent/llm.py：LLM 适配层。

- LLMClient：协议（pipeline 只依赖它，隔离供应商差异）
- OpenAICompatClient：OpenAI 兼容接口实现（agnes / DeepSeek / 通义 等）
- build_llm()：按配置返回供应商实例
"""

from __future__ import annotations

from openai import OpenAI

from .config import LLMConfig, Config


class LLMError(RuntimeError):
    pass


class LLMClient:
    """LLM 适配层协议。供应商实现需可替换，pipeline 不感知具体厂商。"""

    last_usage: dict | None = None  # 最近一次调用的 token 用量（供埋点，可选）

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=120)
        self.last_usage: dict | None = None  # 最近一次调用的 token 用量（供埋点）

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
        except Exception as e:  # 供应商网络/鉴权等错误统一包装
            raise LLMError(f"LLM 调用失败 [{self._cfg.provider}/{self._cfg.model}]: {e}") from e
        u = resp.usage
        self.last_usage = {
            "input": getattr(u, "prompt_tokens", 0),
            "output": getattr(u, "completion_tokens", 0),
            "total": getattr(u, "total_tokens", 0),
        }
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise LLMError(
                f"LLM 返回空内容 [{self._cfg.provider}/{self._cfg.model}]，"
                "可能是 thinking 模式耗尽 max_tokens，请调大 LLM_MAX_TOKENS"
            )
        return content

    @property
    def model(self) -> str:
        return self._cfg.model


def build_llm(cfg: Config, *, use_alt: bool = False) -> LLMClient:
    """按配置构建 LLM 客户端；use_alt=True 时切到备选供应商（适配层隔离演示）。"""
    target = cfg.alt_llm if use_alt else cfg.llm
    if target is None:
        raise LLMError("LLM 配置缺失：请检查 .env 中的 LLM_*（或 ALT_LLM_*）变量")
    return OpenAICompatClient(target)
