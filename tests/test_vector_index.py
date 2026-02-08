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


def test_vector_retriever_uses_small_to_big_strategy() -> None:
    retriever = VectorRetriever(settings=AppSettings(VECTOR_PROVIDER="qdrant"))

    results = retriever.retrieve(query="versionamento de prompts", top_k=1)

    assert len(results) == 1
    assert "OpenAI API" in results[0].content
    assert "versionamento de prompts" in results[0].content


def test_vector_retriever_embedding_cache_reuses_query_embedding() -> None:
    retriever = VectorRetriever(
        settings=AppSettings(
            VECTOR_PROVIDER="qdrant",
            EMBEDDING_CACHE_BACKEND="memory",
        )
    )

    retriever.retrieve(query="latência de produção", top_k=2)
    first_size = len(retriever._embedding_cache._memory_store)  # type: ignore[attr-defined]

    retriever.retrieve(query="latência de produção", top_k=2)
    second_size = len(retriever._embedding_cache._memory_store)  # type: ignore[attr-defined]

    assert first_size == second_size


def test_vector_retriever_pinecone_sem_dependencia_faz_fallback_local() -> None:
    retriever = VectorRetriever(settings=AppSettings(VECTOR_PROVIDER="pinecone"))

    results = retriever.retrieve(query="estratégia de observabilidade", top_k=1)

    assert len(results) == 1
    assert results[0].source.startswith("vector-pinecone")
