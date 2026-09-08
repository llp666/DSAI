"""orchestration/llm.py：LLM adapter layer.

- LLMClient: protocol (pipeline depends only on it, isolating vendor differences)
- OpenAICompatClient: OpenAI-compatible implementation (agnes / DeepSeek / Qwen etc.)
- build_llm(): returns a vendor instance by config
"""

from __future__ import annotations

import json
import time

from openai import OpenAI

from config import LLMConfig, Config


class LLMError(RuntimeError):
    pass


class LLMClient:
    """LLM adapter protocol; implementations are swappable, pipeline is vendor-agnostic."""

    last_usage: dict | None = None  # most recent call's token usage (for metering, optional)

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_with_tools(self, system: str, user: str,
                            tools: list[dict]) -> tuple[str, list | None]:
        """Optional tool call: returns (content, tool_calls or None)."""
        raise NotImplementedError

    def stream_complete(self, system: str, user: str):
        """Stream a completion as text chunks (generator); records last_usage."""
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=120)
        self.last_usage: dict | None = None

    def complete(self, system: str, user: str) -> str:
        content, _ = self._chat(system, user, tools=None)
        return content

    def complete_with_tools(self, system: str, user: str,
                            tools: list[dict]) -> tuple[str, list | None]:
        content, tool_calls = self._chat(system, user, tools=tools)
        return content, tool_calls

    def stream_complete(self, system: str, user: str):
        """Stream a completion as text chunks (generator); records last_usage.

        429 is retried only before streaming starts (mid-stream errors propagate).
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                stream = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    stream=True,
                )
                buf: list[str] = []
                usage = None
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        buf.append(delta.content)
                        yield delta.content
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        usage = u
                full = "".join(buf).strip()
                if not full:
                    raise LLMError(
                        f"LLM 返回空内容 [{self._cfg.provider}/{self._cfg.model}]，"
                        "可能是 thinking 模式耗尽 max_tokens，请调大 LLM_MAX_TOKENS")
                self.last_usage = {
                    "input": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output": getattr(usage, "completion_tokens", 0) if usage else max(1, len(full) // 4),
                    "total": getattr(usage, "total_tokens", 0) if usage else max(1, len(full) // 4),
                }
                return
            except LLMError:
                raise
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    last_exc = e
                    continue
                raise LLMError(
                    f"LLM 调用失败 [{self._cfg.provider}/{self._cfg.model}]: {e}") from e
        raise LLMError(
            f"LLM 调用失败（限流重试3次仍失败）[{self._cfg.provider}/{self._cfg.model}]: {last_exc}")

    def _chat(self, system: str, user: str, tools: list | None):
        """Shared chat.completions call: returns (content, tool_calls or None).

        429 rate-limit backoff retry (max 3); records last_usage for the budget monitor.
        """
        last_exc: Exception | None = None
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    **kwargs,
                )
            except Exception as e:  # wrap vendor network/auth errors uniformly
                if "429" in str(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))  # exponential backoff 3s/6s
                    last_exc = e
                    continue
                raise LLMError(
                    f"LLM 调用失败 [{self._cfg.provider}/{self._cfg.model}]: {e}"
                ) from e
            break
        else:
            raise LLMError(
                f"LLM 调用失败（限流重试3次仍失败）[{self._cfg.provider}/{self._cfg.model}]: {last_exc}"
            )
        u = resp.usage
        self.last_usage = {
            "input": getattr(u, "prompt_tokens", 0),
            "output": getattr(u, "completion_tokens", 0),
            "total": getattr(u, "total_tokens", 0),
        }
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = []
            for tc in msg.tool_calls:
                fn = tc.function
                args = getattr(fn, "arguments", None)
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": fn.name,
                    "arguments": args,
                })
        if not content and not tool_calls:
            raise LLMError(
                f"LLM 返回空内容 [{self._cfg.provider}/{self._cfg.model}]，"
                "可能是 thinking 模式耗尽 max_tokens，请调大 LLM_MAX_TOKENS"
            )
        return content, tool_calls

    @property
    def model(self) -> str:
        return self._cfg.model


def build_llm(cfg: Config, *, use_alt: bool = False) -> LLMClient:
    """Build an LLM client; use_alt=True switches to the alternate vendor."""
    target = cfg.alt_llm if use_alt else cfg.llm
    if target is None:
        raise LLMError("LLM 配置缺失：请检查 .env 中的 LLM_*（或 ALT_LLM_*）变量")
    return OpenAICompatClient(target)