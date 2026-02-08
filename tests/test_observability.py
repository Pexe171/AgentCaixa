from rag_app.agent.observability import AgentTracer, NoopTrace
from rag_app.config import AppSettings


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
