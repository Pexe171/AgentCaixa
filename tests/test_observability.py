from rag_app.agent.observability import AgentTracer, NoopTrace
from rag_app.config import AppSettings


class _FakeRunTree:
    def __init__(self, **kwargs):
        self.payload = kwargs
        self.post_called = False
        self.ended = False
        self.patch_calls: list[dict] = []
        self.children: list[_FakeRunTree] = []

    def post(self):
        self.post_called = True

    def create_child(self, **kwargs):
        child = _FakeRunTree(**kwargs)
        self.children.append(child)
        return child

    def patch(self, **kwargs):
        self.patch_calls.append(kwargs)

    def end(self):
        self.ended = True


def test_agent_tracer_returns_noop_when_disabled() -> None:
    settings = AppSettings(
        OBSERVABILITY_ENABLED=False,
        OBSERVABILITY_PROVIDER="none",
    )
    tracer = AgentTracer(settings=settings)

    trace = tracer.start_trace(
        name="teste",
        trace_id="trace-1",
        session_id=None,
        input_data={"k": "v"},
    )

    assert isinstance(trace, NoopTrace)


def test_agent_tracer_returns_noop_when_missing_langfuse_keys() -> None:
    settings = AppSettings(
        OBSERVABILITY_ENABLED=True,
        OBSERVABILITY_PROVIDER="langfuse",
        LANGFUSE_PUBLIC_KEY=None,
        LANGFUSE_SECRET_KEY=None,
    )
    tracer = AgentTracer(settings=settings)

    trace = tracer.start_trace(
        name="teste",
        trace_id="trace-2",
        session_id="sessao-1",
        input_data={"pergunta": "oi"},
    )

    assert isinstance(trace, NoopTrace)


def test_agent_tracer_builds_langsmith_trace(monkeypatch) -> None:
    settings = AppSettings(
        OBSERVABILITY_ENABLED=True,
        OBSERVABILITY_PROVIDER="langsmith",
        LANGSMITH_API_KEY="test-key",
        LANGSMITH_PROJECT="agentcaixa-test",
    )

    tracer = AgentTracer(settings=settings)
    monkeypatch.setattr(tracer, "_langsmith_run_tree_cls", _FakeRunTree)

    trace = tracer.start_trace(
        name="fluxo-credito",
        trace_id="trace-3",
        session_id="sessao-3",
        input_data={"pergunta": "como validar renda?"},
        metadata={"canal": "api"},
    )

    assert trace.__class__.__name__ == "_LangSmithTrace"

    with trace.span("recuperacao", input_data={"q": "renda"}) as span:
        span.update(output={"hits": 2}, metadata={"fonte": "qdrant"})

    trace.update(output={"status": "ok"}, metadata={"latency_ms": 80})

    root_run = trace._root_run
    assert root_run.post_called is True
    assert root_run.payload["extra"]["session_id"] == "sessao-3"
    assert root_run.children[0].ended is True
    assert root_run.children[0].patch_calls
    assert root_run.patch_calls
