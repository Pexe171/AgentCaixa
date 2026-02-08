"""Camada de observabilidade do agente com suporte a Langfuse e LangSmith."""

from __future__ import annotations

import os
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


class _LangSmithSpan:
    def __init__(self, run_tree: Any) -> None:
        self._run_tree = run_tree

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["outputs"] = {"result": output}
        if metadata is not None:
            kwargs["extra"] = metadata
        if kwargs:
            self._run_tree.patch(**kwargs)


class _LangSmithTrace:
    def __init__(self, root_run: Any) -> None:
        self._root_run = root_run

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_LangSmithSpan]:
        kwargs: dict[str, Any] = {"name": name, "run_type": "chain"}
        if input_data is not None:
            kwargs["inputs"] = {"input": input_data}
        if metadata is not None:
            kwargs["extra"] = metadata

        child_run = self._root_run.create_child(**kwargs)
        child_run.post()
        span = _LangSmithSpan(child_run)
        try:
            yield span
        finally:
            child_run.end()
            child_run.patch()

    def update(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["outputs"] = {"result": output}
        if metadata is not None:
            kwargs["extra"] = metadata
        if kwargs:
            self._root_run.patch(**kwargs)


class AgentTracer:
    """Factory de traces do pipeline com fallback seguro para no-op."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._langfuse_client: Any | None = None
        self._langsmith_run_tree_cls: Any | None = None

        if not settings.OBSERVABILITY_ENABLED:
            return

        if settings.OBSERVABILITY_PROVIDER == "langfuse":
            self._setup_langfuse()
            return

        if settings.OBSERVABILITY_PROVIDER == "langsmith":
            self._setup_langsmith()

    def _setup_langfuse(self) -> None:
        if (
            not self._settings.LANGFUSE_PUBLIC_KEY
            or not self._settings.LANGFUSE_SECRET_KEY
        ):
            return

        try:
            from langfuse import Langfuse

            self._langfuse_client = Langfuse(
                public_key=self._settings.LANGFUSE_PUBLIC_KEY,
                secret_key=self._settings.LANGFUSE_SECRET_KEY,
                host=self._settings.LANGFUSE_HOST,
            )
        except Exception:
            self._langfuse_client = None

    def _setup_langsmith(self) -> None:
        if not self._settings.LANGSMITH_API_KEY:
            return

        os.environ.setdefault("LANGCHAIN_API_KEY", self._settings.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self._settings.LANGSMITH_ENDPOINT)
        os.environ.setdefault("LANGCHAIN_PROJECT", self._settings.LANGSMITH_PROJECT)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

        try:
            from langsmith.run_trees import RunTree

            self._langsmith_run_tree_cls = RunTree
        except Exception:
            self._langsmith_run_tree_cls = None

    def start_trace(
        self,
        *,
        name: str,
        trace_id: str,
        session_id: str | None,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> TraceLike:
        if self._langfuse_client is not None:
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

        if self._langsmith_run_tree_cls is not None:
            payload: dict[str, Any] = {
                "id": trace_id,
                "name": name,
                "run_type": "chain",
                "inputs": {"input": input_data},
            }
            trace_metadata = dict(metadata or {})
            if session_id:
                trace_metadata["session_id"] = session_id
            if trace_metadata:
                payload["extra"] = trace_metadata

            root_run = self._langsmith_run_tree_cls(**payload)
            root_run.post()
            return _LangSmithTrace(root_run)

        return NoopTrace()

    def flush(self) -> None:
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
            except Exception:
                pass
