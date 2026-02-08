"""Camada de observabilidade do agente com suporte opcional a Langfuse."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from rag_app.config import AppSettings


class SpanLike(Protocol):
    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atualiza um span em andamento."""


class TraceLike(Protocol):
    def span(
        self,
        name: str,
        *,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[SpanLike]:
        """Cria um span de rastreamento."""

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atualiza o trace raiz."""


@dataclass
class _NoopSpan:
    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del output, metadata


class NoopTrace:
    @contextmanager
    def span(
        self,
        name: str,
        *,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_NoopSpan]:
        del name, input_data, metadata
        yield _NoopSpan()

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del output, metadata


class _LangfuseSpan:
    def __init__(self, native_span: Any) -> None:
        self._native_span = native_span

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            self._native_span.update(**kwargs)


class _LangfuseTrace:
    def __init__(self, native_trace: Any) -> None:
        self._native_trace = native_trace

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_LangfuseSpan]:
        kwargs: dict[str, Any] = {"name": name}
        if input_data is not None:
            kwargs["input"] = input_data
        if metadata is not None:
            kwargs["metadata"] = metadata

        native_span = self._native_trace.span(**kwargs)
        span = _LangfuseSpan(native_span)
        try:
            yield span
        finally:
            native_span.end()

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            self._native_trace.update(**kwargs)


class AgentTracer:
    """Factory de traces do pipeline com fallback seguro para no-op."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._langfuse_client: Any | None = None

        if not settings.OBSERVABILITY_ENABLED:
            return
        if settings.OBSERVABILITY_PROVIDER != "langfuse":
            return
        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            return

        try:
            from langfuse import Langfuse

            self._langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        except Exception:
            self._langfuse_client = None

    def start_trace(
        self,
        *,
        name: str,
        trace_id: str,
        session_id: str | None,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> TraceLike:
        if self._langfuse_client is None:
            return NoopTrace()

        payload: dict[str, Any] = {
            "name": name,
            "id": trace_id,
            "input": input_data,
        }
        if session_id:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata

        native_trace = self._langfuse_client.trace(**payload)
        return _LangfuseTrace(native_trace)

    def flush(self) -> None:
        if self._langfuse_client is None:
            return
        try:
            self._langfuse_client.flush()
        except Exception:
            pass
