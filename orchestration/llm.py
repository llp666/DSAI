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
    last_reasoning: str | None = None  # most recent call's reasoning/thinking content (optional)
    last_tool_calls: list | None = None  # most recent streamed call's tool_calls (optional)

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_with_tools(self, system: str, user: str,
                            tools: list[dict]) -> tuple[str, list | None]:
        """Optional tool call: returns (content, tool_calls or None)."""
        raise NotImplementedError

    def stream_complete(self, system: str, user: str, *, no_thinking: bool = False):
        """Stream a completion as tagged (kind, text) chunks: kind is 'reasoning' or 'content'.

        ``no_thinking`` asks the provider to disable chain-of-thought (and drops any residual
        reasoning chunks) — used for casual chat, where thinking only adds latency.
        """
        raise NotImplementedError

    def stream_complete_with_tools(self, system: str, user: str, tools: list[dict],
                                   *, no_thinking: bool = False):
        """Stream with tools armed, as tagged (kind, text) chunks.

        The model either answers directly (content streams live; ``last_tool_calls`` stays
        None — the yielded content IS the final reply) or chooses a tool (in practice content
        is empty and the fragments accumulate into ``last_tool_calls`` for the caller to
        execute and follow up with). ``no_thinking`` behaves as in :meth:`stream_complete`.
        """
        raise NotImplementedError

    def complete_stream(self, system: str, user: str, *, on_reasoning=None) -> str:
        """Full completion while forwarding live reasoning chunks to ``on_reasoning``."""
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=120)
        self.last_usage: dict | None = None
        self.last_reasoning: str | None = None
        self.last_tool_calls: list | None = None

    def complete(self, system: str, user: str) -> str:
        content, _ = self._chat(system, user, tools=None)
        return content

    def complete_with_tools(self, system: str, user: str,
                            tools: list[dict]) -> tuple[str, list | None]:
        content, tool_calls = self._chat(system, user, tools=tools)
        return content, tool_calls

    def stream_complete(self, system: str, user: str, *, no_thinking: bool = False):
        """Stream a completion as tagged (kind, text) chunks ('reasoning' then 'content').

        agnes-style models emit chain-of-thought in delta.reasoning_content before the
        answer; chunks yield ("reasoning", t) / ("content", t) so the UI can render a
        Kimi-style collapsible thinking panel. ``no_thinking`` disables the provider's
        thinking mode via chat_template_kwargs (docs/model.md) and skips any residual
        reasoning chunks — used for casual chat, where chain-of-thought would only add
        latency and risk leaking system-prompt echoes. 429 is retried only before streaming
        starts (mid-stream errors propagate).
        """
        last_exc: Exception | None = None
        extra: dict = {}
        if no_thinking:
            # agnes: chat_template_kwargs toggles Thinking in OpenAI-compatible mode
            extra["extra_body"] = {"chat_template_kwargs": {"thinking": {"type": "disabled"}}}
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
                    **extra,
                )
                buf: list[str] = []
                reasoning_parts: list[str] = []
                usage = None
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta:
                        r = getattr(delta, "reasoning_content", None)
                        if r and not no_thinking:
                            reasoning_parts.append(r)
                            yield "reasoning", r
                        if delta.content:
                            buf.append(delta.content)
                            yield "content", delta.content
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        usage = u
                self.last_reasoning = "".join(reasoning_parts).strip() or None
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

    def stream_complete_with_tools(self, system: str, user: str, tools: list[dict],
                                   *, no_thinking: bool = False):
        """Stream with tools armed, as tagged ('reasoning'/'content', text) chunks.

        Lets the model decide for itself: answer directly (content streams live, ``yield``
        chunks ARE the final reply, ``self.last_tool_calls`` stays None) or call a tool
        (content is then empty in practice; per-index ``delta.tool_calls`` fragments are
        accumulated and parsed into ``self.last_tool_calls`` for the caller to execute and
        stream a follow-up). 429 is retried only before streaming starts (mid-stream errors
        propagate).
        """
        last_exc: Exception | None = None
        extra: dict = {}
        if no_thinking:
            # agnes: chat_template_kwargs toggles Thinking in OpenAI-compatible mode
            extra["extra_body"] = {"chat_template_kwargs": {"thinking": {"type": "disabled"}}}
        self.last_tool_calls = None
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
                    tools=tools,
                    tool_choice="auto",
                    **extra,
                )
                buf: list[str] = []
                reasoning_parts: list[str] = []
                tc_acc: dict[int, dict] = {}  # tool_call index → {id, name, arguments}
                usage = None
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta:
                        r = getattr(delta, "reasoning_content", None)
                        if r and not no_thinking:
                            reasoning_parts.append(r)
                            yield "reasoning", r
                        if delta.content:
                            buf.append(delta.content)
                            yield "content", delta.content
                        for tc in (getattr(delta, "tool_calls", None) or []):
                            acc = tc_acc.setdefault(tc.index,
                                                    {"id": "", "name": "", "arguments": ""})
                            if tc.id:
                                acc["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if fn.name:
                                    acc["name"] = fn.name
                                if fn.arguments:
                                    acc["arguments"] += fn.arguments
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        usage = u
                self.last_reasoning = "".join(reasoning_parts).strip() or None
                tool_calls = None
                if tc_acc:
                    tool_calls = []
                    for i in sorted(tc_acc):
                        a = tc_acc[i]
                        try:
                            args = json.loads(a["arguments"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({"id": a["id"], "name": a["name"],
                                           "arguments": args})
                self.last_tool_calls = tool_calls
                full = "".join(buf).strip()
                if not full and not tool_calls:
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

    def complete_stream(self, system: str, user: str, *, on_reasoning=None) -> str:
        """Full completion (aggregated) while forwarding live reasoning to ``on_reasoning``."""
        buf: list[str] = []
        for kind, text in self.stream_complete(system, user):
            if kind == "reasoning":
                if on_reasoning is not None:
                    on_reasoning(text)
            else:
                buf.append(text)
        return "".join(buf)

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
        self.last_reasoning = (getattr(msg, "reasoning_content", None) or "").strip() or None
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