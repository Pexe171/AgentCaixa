"""Integração para recuperação vetorial com fallback local."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Literal

from rag_app.agent.embedding_cache import EmbeddingCache
from rag_app.agent.knowledge import DEFAULT_KNOWLEDGE_BASE
from rag_app.agent.schemas import ContextSnippet
from rag_app.config import AppSettings

VectorProvider = Literal["none", "faiss", "qdrant", "pgvector", "weaviate"]


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
        self._embedding_cache = EmbeddingCache(settings=settings)
        self._documents: list[_VectorDoc] = []
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
            return self._retrieve_locally(
                query=query,
                top_k=top_k,
                source_prefix="vector-qdrant",
            )

        source = f"vector-{self._provider}"
        return self._retrieve_locally(query=query, top_k=top_k, source_prefix=source)
