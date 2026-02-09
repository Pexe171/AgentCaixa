"""Índice vetorial e recuperação semântica com fallback local."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

from rag_app.agent.schemas import ContextSnippet
from rag_app.config import AppSettings, load_settings


@dataclass(frozen=True)
class _KnowledgeEntry:
    source: str
    content: str


_DEFAULT_KNOWLEDGE_BASE: tuple[_KnowledgeEntry, ...] = (
    _KnowledgeEntry(
        source="vector-01",
        content=(
            "OpenAI API exige monitoramento de latência e custo para manter SLA em produção. "
            "A estratégia recomendada inclui cache de resposta e telemetria de custo por sessão."
        ),
    ),
    _KnowledgeEntry(
        source="vector-02",
        content=(
            "A prática de versionamento de prompts deve usar histórico de alterações e rollback controlado. "
            "OpenAI API e times de plataforma se beneficiam de testes A/B para prompts críticos."
        ),
    ),
    _KnowledgeEntry(
        source="vector-03",
        content=(
            "Observabilidade corporativa combina logs estruturados, traces distribuídos e métricas de negócio."
        ),
    ),
)


class _InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self._memory_store: dict[str, list[float]] = {}

    def get(self, key: str) -> list[float] | None:
        return self._memory_store.get(key)

    def set(self, key: str, embedding: list[float]) -> None:
        self._memory_store[key] = embedding


class _NoopEmbeddingCache:
    def get(self, key: str) -> list[float] | None:
        return None

    def set(self, key: str, embedding: list[float]) -> None:
        return None


def _embed_text(text: str, dimensions: int = 96) -> list[float]:
    """Gera embedding determinístico local sem dependência externa."""

    normalized = " ".join(text.lower().split())
    if not normalized:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in normalized.split(" "):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(dimensions):
            bucket = digest[index % len(digest)]
            vector[index] += (bucket / 255.0) - 0.5

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * dimensions
    return [value / norm for value in vector]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calcula similaridade cosseno entre vetores."""

    if not vector_a or not vector_b:
        return 0.0

    dimensions = min(len(vector_a), len(vector_b))
    numerator = sum(vector_a[i] * vector_b[i] for i in range(dimensions))
    norm_a = math.sqrt(sum(vector_a[i] * vector_a[i] for i in range(dimensions)))
    norm_b = math.sqrt(sum(vector_b[i] * vector_b[i] for i in range(dimensions)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    raw = numerator / (norm_a * norm_b)
    return max(0.0, min(1.0, (raw + 1) / 2))


class VectorRetriever:
    """Recuperador vetorial com suporte a múltiplos providers e fallback local."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._embedding_cache = self._build_cache()
        self._knowledge_base = self._build_knowledge_base()

    def _build_cache(self) -> _InMemoryEmbeddingCache | _NoopEmbeddingCache:
        if self._settings.EMBEDDING_CACHE_BACKEND == "memory":
            return _InMemoryEmbeddingCache()
        return _NoopEmbeddingCache()

    def _build_knowledge_base(self) -> tuple[_KnowledgeEntry, ...]:
        provider = self._settings.VECTOR_PROVIDER
        if provider == "pinecone":
            return tuple(
                _KnowledgeEntry(
                    source=f"vector-pinecone-{index+1}",
                    content=item.content,
                )
                for index, item in enumerate(_DEFAULT_KNOWLEDGE_BASE)
            )
        return _DEFAULT_KNOWLEDGE_BASE

    def _get_query_embedding(self, query: str) -> list[float]:
        cached = self._embedding_cache.get(query)
        if cached is not None:
            return cached
        embedding = _embed_text(query)
        self._embedding_cache.set(query, embedding)
        return embedding

    def retrieve(self, query: str, top_k: int = 5) -> list[ContextSnippet]:
        provider = self._settings.VECTOR_PROVIDER
        if provider == "none":
            return []

        query_embedding = self._get_query_embedding(query)
        ranked: list[tuple[float, _KnowledgeEntry]] = []
        query_tokens = {token.strip(".,:;!?()[]{}\"' ").lower() for token in query.split() if token}
        for item in self._knowledge_base:
            semantic_score = _cosine_similarity(query_embedding, _embed_text(item.content))
            content_tokens = {
                token.strip(".,:;!?()[]{}\"' ").lower()
                for token in item.content.split()
                if token
            }
            overlap = len(query_tokens.intersection(content_tokens))
            lexical_score = overlap / max(1, len(query_tokens))
            score = (semantic_score * 0.45) + (lexical_score * 0.55)
            ranked.append((score, item))

        ranked.sort(key=lambda value: value[0], reverse=True)
        return [
            ContextSnippet(source=item.source, content=item.content, score=score)
            for score, item in ranked[: max(1, top_k)]
        ]


class VectorIndex:
    def __init__(self, settings: AppSettings | None = None):
        from langchain_ollama import OllamaEmbeddings

        self._settings = settings or load_settings()
        self.embeddings = OllamaEmbeddings(model=self._settings.OLLAMA_EMBEDDING_MODEL)
        self.index_path = os.path.join(os.getcwd(), "data", "index")

    def ingest_file(self, file_path: str):
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print(f"📄 Indexando Procedimento: {os.path.basename(file_path)}")

        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", "▪", "•", "."],
        )
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, self.embeddings)

        os.makedirs(self.index_path, exist_ok=True)
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            db = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            db.merge_from(vectorstore)
            db.save_local(self.index_path)
        else:
            vectorstore.save_local(self.index_path)
        print("✅ Memória operacional atualizada!")
