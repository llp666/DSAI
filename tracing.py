"""tracing.py：observability adapter.

- NullTracer: no-op fallback when Langfuse is disabled (same interface).
- LangfuseTracer: Langfuse 4.x full tracing.

Interface: ``with tracer.trace(name) as t:`` → ``t.set_trace_io(...)`` / ``t.span(...)`` / ``t.trace_id``.
"""

from __future__ import annotations

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
    def __init__(self, host: str, public_key: str, secret_key: str):
        from langfuse import Langfuse

        self._lf = Langfuse(public_key=public_key, secret_key=secret_key,
                            host=host, timeout=30)

    def trace(self, name: str):
        return _LangfuseSession(self._lf, name)


class _LangfuseSession:
    def __init__(self, lf, name: str):
        self._lf = lf
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
        self._lf.flush()

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
                          secret_key=cfg.langfuse_secret_key)