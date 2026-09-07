"""tracing.py：可观测性适配层。

- LangfuseTracer：Langfuse 4.x SDK（observation 风格）全链路埋点；
- NoopTracer：未启用 Langfuse 时的降级实现，接口一致、不阻塞 pipeline。

统一接口：
    with tracer.trace("question:1") as t:
        t.set_trace_io(input=..., output=...)
        with t.span("retrieve"): ...
        with t.generation("generate", model=..., input=..., output=..., usage=...): ...
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterator


class TraceSession:
    def __enter__(self) -> "TraceSession":
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError

    @property
    def trace_id(self) -> str:
        raise NotImplementedError

    def set_trace_io(self, *, input: Any = None, output: Any = None) -> None:
        raise NotImplementedError

    def span(self, name: str, *, input: Any = None, output: Any = None,
             metadata: Any = None) -> Any:
        raise NotImplementedError

    def generation(self, name: str, *, model: str | None = None,
                   input: Any = None, output: Any = None, usage: dict | None = None,
                   metadata: Any = None) -> Any:
        raise NotImplementedError


class _Noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


class NoopTracer:
    def trace(self, name: str) -> TraceSession:
        return NoopSession(name)


class NoopSession(TraceSession):
    def __init__(self, name: str):
        self._tid = ""

    def __enter__(self) -> "NoopSession":
        return self

    def __exit__(self, *exc) -> None:
        pass

    @property
    def trace_id(self) -> str:
        return self._tid

    def set_trace_io(self, *, input=None, output=None) -> None:
        pass

    def span(self, name, *, input=None, output=None, metadata=None):
        return _Noop()

    def generation(self, name, *, model=None, input=None, output=None,
                   usage=None, metadata=None):
        return _Noop()


class LangfuseTracer:
    def __init__(self, host: str, public_key: str, secret_key: str):
        from langfuse import Langfuse

        self._lf = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            timeout=30,
        )

    def trace(self, name: str) -> TraceSession:
        return LangfuseSession(self._lf, name)


class LangfuseSession(TraceSession):
    def __init__(self, lf, name: str):
        self._lf = lf
        self._name = name
        self._root = None

    def __enter__(self) -> "LangfuseSession":
        cm = self._lf.start_as_current_observation(name=self._name, as_type="span")
        self._root = cm.__enter__()  # 进入后的 observation 对象（有 set_trace_io/update）
        self._root_cm = cm  # 原始 context manager（用于正确退出）
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

    def span(self, name, *, input=None, output=None, metadata=None):
        return self._lf.start_as_current_observation(
            name=name, as_type="span", input=input, output=output, metadata=metadata
        )

    def generation(self, name, *, model=None, input=None, output=None,
                   usage=None, metadata=None):
        return self._lf.start_as_current_observation(
            name=name, as_type="generation", model=model,
            input=input, output=output, usage_details=usage, metadata=metadata,
        )


def build_tracer(cfg) -> Any:
    """按配置返回 Tracer；未启用/缺凭据时降级为 Noop。"""
    if not cfg.langfuse_enabled or not cfg.langfuse_host:
        return NoopTracer()
    return LangfuseTracer(
        host=cfg.langfuse_host,
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
    )
