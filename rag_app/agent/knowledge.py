"""Base de conhecimento simplificada para recuperação híbrida local."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_app.agent.schemas import ContextSnippet


@dataclass(frozen=True)
class KnowledgeItem:
    source: str
    content: str


DEFAULT_KNOWLEDGE_BASE: tuple[KnowledgeItem, ...] = (
    KnowledgeItem(
        source="playbook-arquitetura",
        content=(
            "Um agente HAG combina recuperação de contexto, planejamento, execução e "
            "resposta final com validação de segurança."
        ),
    ),
    KnowledgeItem(
        source="playbook-openai",
        content=(
            "Para produção com OpenAI API, use timeout, retries exponenciais, "
            "versionamento de prompts e observabilidade de latência/custo."
        ),
    ),
    KnowledgeItem(
        source="playbook-produto",
        content=(
            "Respostas de alta qualidade devem respeitar tom, objetivos de negócio, "
            "clareza didática e próximos passos acionáveis."
        ),
    ),
)


def _token_overlap_score(query: str, text: str) -> float:
    punctuation = ".,:;!?()[]{}\"'"
    query_tokens = {
        token.strip(punctuation).lower()
        for token in query.split()
        if token
    }
    text_tokens = {
        token.strip(punctuation).lower()
        for token in text.split()
        if token
    }

    if not query_tokens or not text_tokens:
        return 0.0

    overlap = query_tokens.intersection(text_tokens)
    return min(1.0, len(overlap) / max(1, len(query_tokens)))


def _split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence
    ]
    return sentences or [text]


def retrieve_context(query: str, top_k: int = 3) -> list[ContextSnippet]:
    """Recupera contexto com estratégia small-to-big (frase -> parágrafo)."""

    best_by_parent: dict[tuple[str, str], ContextSnippet] = {}
    for item in DEFAULT_KNOWLEDGE_BASE:
        for sentence in _split_sentences(item.content):
            score = _token_overlap_score(query=query, text=sentence)
            key = (item.source, item.content)
            existing = best_by_parent.get(key)
            if existing is None or score > existing.score:
                best_by_parent[key] = ContextSnippet(
                    source=item.source,
                    content=item.content,
                    score=score,
                )

    ranked = sorted(
        best_by_parent.values(),
        key=lambda snippet: snippet.score,
        reverse=True,
    )
    return [snippet for snippet in ranked[:top_k] if snippet.score > 0.0] or ranked[:1]
