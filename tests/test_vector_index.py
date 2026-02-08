from rag_app.agent.vector_index import VectorRetriever
from rag_app.config import AppSettings


def test_vector_retriever_disabled_returns_empty() -> None:
    retriever = VectorRetriever(settings=AppSettings(VECTOR_PROVIDER="none"))

    assert retriever.retrieve(query="agente corporativo", top_k=3) == []


def test_vector_retriever_faiss_fallback_returns_ranked_context() -> None:
    retriever = VectorRetriever(settings=AppSettings(VECTOR_PROVIDER="faiss"))

    results = retriever.retrieve(query="OpenAI latência custo", top_k=2)

    assert len(results) == 2
    assert all(item.score >= 0 for item in results)
    assert results[0].source.startswith("vector-")
