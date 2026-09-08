"""orchestration/monitor.py：per-question three-layer budget (stage 3-3).

Guards a single question against agnes free-tier exhaustion / loops / timeouts:
1. call count cap (primary generation + repair/reflection rounds);
2. token cap (cumulative, from LLM.last_usage);
3. wall-clock timeout.

QuestionMonitor records calls/usage/elapsed and raises BudgetExceeded on any overrun;
MonitoredLLM wraps LLMClient so every complete() writes usage back to the monitor.
"""

from __future__ import annotations

import time


class BudgetExceeded(RuntimeError):
    """Budget overrun: the question was cut off rather than silently truncated."""

    def __init__(self, reason: str, used: int, limit: int):
        super().__init__(f"{reason}（已用 {used}，上限 {limit}）")
        self.reason = reason
        self.used = used
        self.limit = limit


class QuestionMonitor:
    """Per-question budget monitor: call count / token usage / elapsed layers."""

    def __init__(self, max_calls: int = 8,
                 max_tokens: int = 32000,
                 max_seconds: int = 240):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.calls = 0
        self.tokens = 0
        self._start = time.monotonic()

    def record_call(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise BudgetExceeded("LLM 调用轮次超限", self.calls, self.max_calls)
        self._check_time()

    def add_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        total = int(usage.get("total", 0) or 0)
        self.tokens += total
        if self.tokens > self.max_tokens:
            raise BudgetExceeded("LLM 累计 token 超限", self.tokens, self.max_tokens)

    def _check_time(self) -> None:
        elapsed = time.monotonic() - self._start
        if elapsed > self.max_seconds:
            raise BudgetExceeded("单问耗时超限", int(elapsed), self.max_seconds)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def snapshot(self) -> dict:
        return {"calls": self.calls, "tokens": self.tokens,
                "elapsed_s": round(self.elapsed, 2)}


class MonitoredLLM:
    """Wraps LLMClient: every complete() counts into the monitor."""

    def __init__(self, llm, monitor: QuestionMonitor):
        self._llm = llm
        self._monitor = monitor

    def complete(self, system: str, user: str) -> str:
        self._monitor.record_call()
        resp = self._llm.complete(system, user)
        self._monitor.add_usage(self._llm.last_usage)
        return resp

    def complete_with_tools(self, system: str, user: str,
                            tools: list[dict]) -> tuple[str, list | None]:
        self._monitor.record_call()
        content, tool_calls = self._llm.complete_with_tools(system, user, tools)
        self._monitor.add_usage(self._llm.last_usage)
        return content, tool_calls

    def stream_complete(self, system: str, user: str):
        self._monitor.record_call()
        yield from self._llm.stream_complete(system, user)
        self._monitor.add_usage(self._llm.last_usage)

    def complete_stream(self, system: str, user: str, *, on_reasoning=None) -> str:
        self._monitor.record_call()
        content = self._llm.complete_stream(system, user, on_reasoning=on_reasoning)
        self._monitor.add_usage(self._llm.last_usage)
        return content

    @property
    def last_reasoning(self) -> str | None:
        return self._llm.last_reasoning

    @property
    def model(self) -> str:
        return self._llm.model