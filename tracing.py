"""tracing.py：observability adapter.

- NullTracer: no-op fallback when Langfuse is disabled (same interface).
- LangfuseTracer: Langfuse 4.x full tracing.

Interface: ``with tracer.trace(name) as t:`` → ``t.set_trace_io(...)`` / ``t.span(...)`` / ``t.trace_id``.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext


class NullTracer:
    """No-op tracer used when Langfuse is not enabled."""

    def trace(self, name: str):
        return _NullSession()


class _NullSession:
    trace_id = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def set_trace_io(self, **kwargs) -> None:
        pass

    def span(self, name: str, **kwargs):
        return nullcontext()


class LangfuseTracer:
    def __init__(self, host: str, public_key: str, secret_key: str,
                 flush_timeout: float = 2.0):
        from langfuse import Langfuse

        self._lf = Langfuse(public_key=public_key, secret_key=secret_key,
                            host=host, timeout=30)
        self._flush_timeout = flush_timeout
        self._flush_thread: threading.Thread | None = None

    def trace(self, name: str):
        return _LangfuseSession(self, name)

    def flush(self) -> None:
        """Flush queued spans without stalling the answer path beyond ``flush_timeout``.

        Langfuse's ``flush()`` blocks until the batch exporter drains; pointed at a
        self-hosted server that isn't up, that is ~50s of retry backoff **per question**
        — the answer is already computed by then, so the user just watches a finished
        reply hang. The SDK exports from its own background thread regardless, so the
        only thing the wait buys is that short-lived processes (eval scripts) don't exit
        before the last traces ship. So: cap the wait, and never run two flushes at once
        — a stuck flush would otherwise spawn a fresh thread on every question.
        """
        inflight = self._flush_thread
        if inflight is not None and inflight.is_alive():
            return  # already draining; the pending batch rides along with it
        t = threading.Thread(target=self._flush_quietly, daemon=True, name="langfuse-flush")
        self._flush_thread = t
        t.start()
        if self._flush_timeout > 0:
            t.join(self._flush_timeout)
        # timed out → abandon the wait, not the thread: its export continues in background

    def _flush_quietly(self) -> None:
        """Best-effort export — tracing must never break the answer path."""
        try:
            self._lf.flush()
        except Exception:
            pass


class _LangfuseSession:
    def __init__(self, tracer: LangfuseTracer, name: str):
        self._tracer = tracer
        self._lf = tracer._lf
        self._name = name
        self._root = None
        self._root_cm = None

    def __enter__(self):
        self._root_cm = self._lf.start_as_current_observation(
            name=self._name, as_type="span")
        self._root = self._root_cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._root_cm.__exit__(*exc)
        self._tracer.flush()

    @property
    def trace_id(self) -> str:
        return self._root.trace_id if self._root is not None else ""

    def set_trace_io(self, *, input=None, output=None) -> None:
        if self._root is not None:
            self._root.set_trace_io(input=input, output=output)

    def span(self, name: str, *, input=None, output=None, metadata=None):
        return self._lf.start_as_current_observation(
            name=name, as_type="span", input=input, output=output, metadata=metadata)


def build_tracer(cfg):
    """Return a tracer; fall back to NullTracer when Langfuse is off / missing creds."""
    if not cfg.langfuse_enabled or not cfg.langfuse_host:
        return NullTracer()
    return LangfuseTracer(host=cfg.langfuse_host, public_key=cfg.langfuse_public_key,
                          secret_key=cfg.langfuse_secret_key,
                          flush_timeout=getattr(cfg, "langfuse_flush_timeout", 2.0))