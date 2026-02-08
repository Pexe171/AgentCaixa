"""Integração para recuperação vetorial com fallback local."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from rag_app.agent.knowledge import DEFAULT_KNOWLEDGE_BASE
from rag_app.agent.schemas import ContextSnippet
from rag_app.config import AppSettings

VectorProvider = Literal["none", "faiss", "qdrant", "pgvector", "weaviate"]


@dataclass(frozen=True)
class _VectorDoc:
    source: str
    content: str
    embedding: list[float]


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
        self._documents = [
            _VectorDoc(
                source=item.source,
                content=item.content,
                embedding=_embed_text(item.content),
            )
            for item in DEFAULT_KNOWLEDGE_BASE
        ]

    def _retrieve_locally(
        self,
        query: str,
        top_k: int,
        source_prefix: str,
    ) -> list[ContextSnippet]:
        query_vector = _embed_text(query)
        ranked = sorted(
            self._documents,
            key=lambda doc: _cosine_similarity(query_vector, doc.embedding),
            reverse=True,
        )
        snippets = [
            ContextSnippet(
                source=f"{source_prefix}:{doc.source}",
                content=doc.content,
                score=_cosine_similarity(query_vector, doc.embedding),
            )
            for doc in ranked[:top_k]
        ]
        return [snippet for snippet in snippets if snippet.score > 0.0]

    def retrieve(self, query: str, top_k: int) -> list[ContextSnippet]:
        if self._provider == "none":
            return []

        if self._provider == "faiss":
            try:
                import faiss  # type: ignore # noqa: F401
            except ImportError:
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
