"""Integração para recuperação vetorial com fallback local."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from rag_app.agent.embedding_cache import EmbeddingCache
from rag_app.agent.knowledge import DEFAULT_KNOWLEDGE_BASE
from rag_app.agent.schemas import ContextSnippet
from rag_app.config import AppSettings

VectorProvider = Literal["none", "faiss", "qdrant", "pinecone", "pgvector", "weaviate"]


@dataclass(frozen=True)
class _VectorDoc:
    source: str
    content: str
    parent_content: str
    embedding: list[float]


def _split_sentences(text: str) -> list[str]:
    chunks = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence
    ]
    return chunks or [text]


def _embed_text(text: str, dimensions: int = 192) -> list[float]:
    """Gera embedding denso determinístico sem dependência externa."""

    vector = [0.0] * dimensions
    tokens = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]

    for token in tokens:
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    score = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    return max(0.0, min(1.0, (score + 1) / 2))


class VectorRetriever:
    """Fachada para recuperação vetorial com provedores configuráveis."""

    def __init__(self, settings: AppSettings) -> None:
        self._provider: VectorProvider = settings.VECTOR_PROVIDER
        self._settings = settings
        self._embedding_cache = EmbeddingCache(settings=settings)
        self._documents: list[_VectorDoc] = []
        self._qdrant_client: Any | None = None
        self._pinecone_index: Any | None = None

        for item in DEFAULT_KNOWLEDGE_BASE:
            for sentence in _split_sentences(item.content):
                self._documents.append(
                    _VectorDoc(
                        source=item.source,
                        content=sentence,
                        parent_content=item.content,
                        embedding=self._embed_cached(sentence),
                    )
                )

        if self._provider == "qdrant":
            self._qdrant_client = self._build_qdrant_client()
            if self._qdrant_client is not None:
                self._sync_qdrant_collection()

        if self._provider == "pinecone":
            self._pinecone_index = self._build_pinecone_index()
            if self._pinecone_index is not None:
                self._sync_pinecone_index()

    def _embed_cached(self, text: str) -> list[float]:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        vector = _embed_text(text)
        self._embedding_cache.set(text, vector)
        return vector

    def _retrieve_locally(
        self,
        query: str,
        top_k: int,
        source_prefix: str,
    ) -> list[ContextSnippet]:
        query_vector = self._embed_cached(query)
        ranked = sorted(
            self._documents,
            key=lambda doc: _cosine_similarity(query_vector, doc.embedding),
            reverse=True,
        )

        merged_by_parent: dict[tuple[str, str], ContextSnippet] = {}
        for doc in ranked:
            score = _cosine_similarity(query_vector, doc.embedding)
            key = (doc.source, doc.parent_content)
            existing = merged_by_parent.get(key)
            if existing is None or score > existing.score:
                merged_by_parent[key] = ContextSnippet(
                    source=f"{source_prefix}:{doc.source}",
                    content=doc.parent_content,
                    score=score,
                )

        snippets = sorted(
            merged_by_parent.values(),
            key=lambda snippet: snippet.score,
            reverse=True,
        )
        return [snippet for snippet in snippets[:top_k] if snippet.score > 0.0]

    def _build_qdrant_client(self) -> Any | None:
        if importlib.util.find_spec("qdrant_client") is None:
            return None
        try:
            qdrant_module = importlib.import_module("qdrant_client")
            client_constructor = getattr(qdrant_module, "QdrantClient", None)
            if client_constructor is None:
                return None
            return client_constructor(
                url=self._settings.QDRANT_URL,
                api_key=self._settings.QDRANT_API_KEY,
                timeout=2.0,
            )
        except Exception:
            return None

    def _sync_qdrant_collection(self) -> None:
        if self._qdrant_client is None:
            return

        try:
            models = importlib.import_module("qdrant_client.http.models")
            distance = models.Distance.COSINE
            vector_params = models.VectorParams(size=192, distance=distance)
            collection_name = self._settings.QDRANT_COLLECTION
            self._qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
            )
            points = [
                models.PointStruct(
                    id=index,
                    vector=doc.embedding,
                    payload={
                        "source": doc.source,
                        "content": doc.content,
                        "parent_content": doc.parent_content,
                    },
                )
                for index, doc in enumerate(self._documents)
            ]
            self._qdrant_client.upsert(collection_name=collection_name, points=points)
        except Exception:
            self._qdrant_client = None

    def _retrieve_qdrant(self, query: str, top_k: int) -> list[ContextSnippet]:
        if self._qdrant_client is None:
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-qdrant",
            )

        try:
            query_vector = self._embed_cached(query)
            results = self._qdrant_client.search(
                collection_name=self._settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=max(top_k * 4, top_k),
                with_payload=True,
            )
            merged: dict[tuple[str, str], ContextSnippet] = {}
            for item in results:
                payload = item.payload or {}
                source = str(payload.get("source", "desconhecido"))
                parent_content = str(
                    payload.get("parent_content", payload.get("content", ""))
                )
                score = float(getattr(item, "score", 0.0) or 0.0)
                key = (source, parent_content)
                existing = merged.get(key)
                if existing is None or score > existing.score:
                    merged[key] = ContextSnippet(
                        source=f"vector-qdrant:{source}",
                        content=parent_content,
                        score=score,
                    )
            ranked = sorted(
                merged.values(),
                key=lambda snippet: snippet.score,
                reverse=True,
            )
            return ranked[:top_k]
        except Exception:
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-qdrant",
            )

    def _build_pinecone_index(self) -> Any | None:
        if importlib.util.find_spec("pinecone") is None:
            return None
        if not self._settings.PINECONE_API_KEY:
            return None

        try:
            pinecone_module = importlib.import_module("pinecone")
            pinecone_cls = getattr(pinecone_module, "Pinecone", None)
            if pinecone_cls is None:
                return None
            client = pinecone_cls(api_key=self._settings.PINECONE_API_KEY)
            return client.Index(self._settings.PINECONE_INDEX)
        except Exception:
            return None

    def _sync_pinecone_index(self) -> None:
        if self._pinecone_index is None:
            return

        try:
            vectors = [
                {
                    "id": str(index),
                    "values": doc.embedding,
                    "metadata": {
                        "source": doc.source,
                        "content": doc.content,
                        "parent_content": doc.parent_content,
                    },
                }
                for index, doc in enumerate(self._documents)
            ]
            self._pinecone_index.upsert(
                vectors=vectors,
                namespace=self._settings.PINECONE_NAMESPACE,
            )
        except Exception:
            self._pinecone_index = None

    def _retrieve_pinecone(self, query: str, top_k: int) -> list[ContextSnippet]:
        if self._pinecone_index is None:
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-pinecone",
            )

        try:
            query_vector = self._embed_cached(query)
            response = self._pinecone_index.query(
                vector=query_vector,
                top_k=max(top_k * 4, top_k),
                namespace=self._settings.PINECONE_NAMESPACE,
                include_metadata=True,
            )
            matches = (
                response.get("matches", [])
                if isinstance(response, dict)
                else getattr(response, "matches", [])
            )
            merged: dict[tuple[str, str], ContextSnippet] = {}
            for match in matches:
                metadata = (
                    match.get("metadata", {})
                    if isinstance(match, dict)
                    else getattr(match, "metadata", {})
                )
                score = (
                    match.get("score", 0.0)
                    if isinstance(match, dict)
                    else getattr(match, "score", 0.0)
                )
                source = str(metadata.get("source", "desconhecido"))
                parent_content = str(
                    metadata.get("parent_content", metadata.get("content", ""))
                )
                key = (source, parent_content)
                existing = merged.get(key)
                if existing is None or float(score) > existing.score:
                    merged[key] = ContextSnippet(
                        source=f"vector-pinecone:{source}",
                        content=parent_content,
                        score=float(score),
                    )
            ranked = sorted(
                merged.values(),
                key=lambda snippet: snippet.score,
                reverse=True,
            )
            return ranked[:top_k]
        except Exception:
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-pinecone",
            )

    def retrieve(self, query: str, top_k: int) -> list[ContextSnippet]:
        if self._provider == "none":
            return []

        if self._provider == "faiss":
            if importlib.util.find_spec("faiss") is None:
                return self._retrieve_locally(
                    query=query,
                    top_k=top_k,
                    source_prefix="vector-fallback",
                )
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-faiss",
            )

        if self._provider == "qdrant":
            return self._retrieve_qdrant(query=query, top_k=top_k)

        if self._provider == "pinecone":
            return self._retrieve_pinecone(query=query, top_k=top_k)

        source = f"vector-{self._provider}"
        return self._retrieve_locally(query=query, top_k=top_k, source_prefix=source)
