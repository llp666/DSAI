"""orchestration/monitor.py：单问三层熔断（阶段 3-3）。

对一次问题处理做预算监控，防 agnes 免费版额度耗尽 / 死循环 / 超时：
1. 轮次熔断：LLM 调用次数上限（主生成 + 纠错/反思轮）；
2. Token 熔断：LLM 累计 token 上限（从 LLM.last_usage 累加）；
3. 超时熔断：单问总耗时上限（wall-clock）。

QuestionMonitor 记录调用/用量/耗时，任何预算超限即抛 BudgetExceeded；
MonitoredLLM 包装 LLMClient，每次 complete 后把 usage 回写 monitor。
graph.answer 捕获 BudgetExceeded 降级回答（诚实告知预算耗尽，不静默截断）。
"""

from __future__ import annotations

import time

# 单问预算默认值（agnes 免费版安全边际）
# 轮次语义 = 独立 LLM 交互轮（generate 算 1 轮含内部解析重试、repair/reflect/reflect_empty 各算 1 轮）。
# 完整纠错链最多：主 generate 1 + 2 轮纠错（reflect+generate）×2 + reflect_empty 1 = 8 轮。
# 熔断必须先于反思存在：所有 LLM 调用都经 MonitoredLLM 计预算，反思循环同样受限。
DEFAULT_MAX_CALLS = 8        # LLM 调用轮次上限（完整纠错链上限，Token 预算再兜底）
DEFAULT_MAX_TOKENS = 32000   # 累计 token 上限（主生成 ~2k，余量给纠错/反思）
DEFAULT_MAX_SECONDS = 240    # 单问总耗时上限（wall-clock）


class BudgetExceeded(RuntimeError):
    """预算超限：单问处理被熔断。"""

    def __init__(self, reason: str, used: int, limit: int):
        super().__init__(f"{reason}（已用 {used}，上限 {limit}）")
        self.reason = reason
        self.used = used
        self.limit = limit


class QuestionMonitor:
    """单问预算监控：调用轮次 / token 用量 / 耗时三层熔断。"""

    def __init__(self, max_calls: int = DEFAULT_MAX_CALLS,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_seconds: int = DEFAULT_MAX_SECONDS):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.calls = 0
        self.tokens = 0
        self._start = time.monotonic()

    def record_call(self) -> None:
        """记录一次 LLM 调用（轮次熔断：超限抛 BudgetExceeded）。"""
        self.calls += 1
        if self.calls > self.max_calls:
            raise BudgetExceeded("LLM 调用轮次超限", self.calls, self.max_calls)
        self._check_time()

    def add_usage(self, usage: dict | None) -> None:
        """累加一次调用的 token 用量（Token 熔断）。"""
        if not usage:
            return
        total = int(usage.get("total", 0) or 0)
        self.tokens += total
        if self.tokens > self.max_tokens:
            raise BudgetExceeded("LLM 累计 token 超限", self.tokens, self.max_tokens)

    def _check_time(self) -> None:
        """超时熔断：单问总耗时超限抛 BudgetExceeded。"""
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
    """包装 LLMClient：每次 complete 计入 monitor 预算，超限抛 BudgetExceeded。"""

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

    @property
    def model(self) -> str:
        return self._llm.model
