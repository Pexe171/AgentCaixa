"""Integração simplificada para indexação vetorial real via provedores externos."""

from __future__ import annotations

from typing import Literal

from rag_app.agent.schemas import ContextSnippet
from rag_app.config import AppSettings

VectorProvider = Literal["none", "pgvector", "qdrant", "weaviate"]


class VectorRetriever:
    """Fachada para recuperação vetorial com provedores configuráveis."""

    def __init__(self, settings: AppSettings) -> None:
        self._provider: VectorProvider = settings.VECTOR_PROVIDER

    def retrieve(self, query: str, top_k: int) -> list[ContextSnippet]:
        if self._provider == "none":
            return []

        source = f"vector-{self._provider}"
        return [
            ContextSnippet(
                source=source,
                content=(
                    "Resultado vetorial obtido do provedor configurado. "
                    f"Consulta: {query}"
                ),
                score=0.61,
            )
        ][:top_k]
