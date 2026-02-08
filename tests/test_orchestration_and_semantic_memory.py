from pathlib import Path

from rag_app.agent.orchestration import route_to_specialist
from rag_app.agent.schemas import AgentChatRequest
from rag_app.agent.semantic_memory import (
    SQLiteSemanticMemoryStore,
    build_semantic_summary,
)
from rag_app.agent.service import AgentService
from rag_app.config import AppSettings


def test_router_selects_credit_specialist() -> None:
    decision = route_to_specialist("Preciso revisar meu limite de crédito e score.")
    assert decision.specialist == "analista_credito"


def test_router_selects_legal_specialist() -> None:
    decision = route_to_specialist(
        "Tenho dúvidas de compliance e cláusulas de contrato."
    )
    assert decision.specialist == "especialista_juridico"


def test_semantic_memory_store_retrieves_relevant_summary(tmp_path: Path) -> None:
    memory = SQLiteSemanticMemoryStore(database_path=str(tmp_path / "semantic.db"))
    memory.add_summary(
        "sessao-1",
        "Cliente quer aumentar limite de crédito consignado.",
    )
    memory.add_summary(
        "sessao-1",
        "Cliente questionou prazo de assinatura do contrato.",
    )

    recalls = memory.retrieve_relevant(
        session_id="sessao-1",
        query="limite de crédito",
        top_k=1,
    )

    assert recalls
    assert "crédito" in recalls[0]


def test_service_persists_semantic_memory_periodically(tmp_path: Path) -> None:
    settings = AppSettings(
        SESSION_STORE_BACKEND="sqlite",
        SESSION_DB_PATH=str(tmp_path / "session.db"),
        SEMANTIC_MEMORY_BACKEND="sqlite",
        SEMANTIC_MEMORY_DB_PATH=str(tmp_path / "semantic.db"),
        SEMANTIC_MEMORY_SUMMARY_INTERVAL=2,
    )
    service = AgentService(settings=settings)

    service.chat(AgentChatRequest(user_message="Primeira pergunta", session_id="abc"))

    semantic = SQLiteSemanticMemoryStore(database_path=str(tmp_path / "semantic.db"))
    assert semantic.count("abc") >= 1


def test_build_semantic_summary_combines_last_turn() -> None:
    summary = build_semantic_summary(
        [
            ("usuario", "Quero financiamento imobiliário."),
            ("agente", "Solicite comprovantes e análise cadastral."),
        ]
    )

    assert "Pedido do cliente" in summary
    assert "Resposta entregue" in summary
