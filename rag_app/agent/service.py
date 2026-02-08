"""Orquestração do agente HAG (Hybrid Agentic Generation)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol

from rag_app.agent.knowledge import retrieve_context
from rag_app.agent.llm_gateway import (
    LLMOutput,
    MockLLMGateway,
    OllamaLLMGateway,
    OpenAILLMGateway,
)
from rag_app.agent.scanner import scan_folder
from rag_app.agent.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentDiagnostics,
    AgentScanRequest,
    AgentScanResponse,
)
from rag_app.config import AppSettings


class LLMGateway(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMOutput:
        """Gera resposta textual a partir de prompts."""


def _build_system_prompt(tone: str, depth: str, require_citations: bool) -> str:
    citation_rule = (
        "Inclua citações explícitas às fontes de contexto usadas."
        if require_citations
        else "Não é obrigatório incluir citações."
    )
    return (
        "Você é um agente HAG de altíssima qualidade para uso corporativo. "
        f"Tom exigido: {tone}. Profundidade de raciocínio: {depth}. "
        "Entregue resposta estruturada, prática, fiel ao contexto "
        "e sem inventar fatos. "
        f"{citation_rule}"
    )


def _build_user_prompt(user_message: str, context_blocks: list[str]) -> str:
    context_text = "\n".join(context_blocks)
    return (
        f"Pergunta do usuário:\n{user_message}\n\n"
        f"Contexto recuperado:\n{context_text}\n\n"
        "Responda em português do Brasil com objetividade e completude."
    )


def _resolve_gateway(settings: AppSettings) -> LLMGateway:
    if (
        settings.LLM_PROVIDER == "openai"
        and settings.OPENAI_API_KEY
        and settings.OPENAI_MODEL
    ):
        return OpenAILLMGateway(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            timeout_s=settings.OPENAI_TIMEOUT_SECONDS,
        )

    if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_MODEL:
        return OllamaLLMGateway(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout_s=settings.OLLAMA_TIMEOUT_SECONDS,
        )

    return MockLLMGateway()


class AgentService:
    """Serviço principal de execução do agente."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._gateway = _resolve_gateway(settings)

    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        start = time.perf_counter()

        snippets = retrieve_context(
            query=request.user_message,
            top_k=self._settings.RETRIEVE_TOP_K_DEFAULT,
        )
        context_blocks = [
            "- Fonte: "
            f"{snippet.source} | Relevância: {snippet.score:.2f} "
            f"| Conteúdo: {snippet.content}"
            for snippet in snippets
        ]

        system_prompt = _build_system_prompt(
            tone=request.tone,
            depth=request.reasoning_depth,
            require_citations=request.require_citations,
        )
        user_prompt = _build_user_prompt(
            user_message=request.user_message,
            context_blocks=context_blocks,
        )

        llm_output = self._gateway.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        diagnostics = AgentDiagnostics(
            provider_used=llm_output.provider,
            model=llm_output.model,
            latency_ms=latency_ms,
            retrieved_context_count=len(snippets),
            fallback_used=llm_output.provider == "mock",
        )

        return AgentChatResponse(
            answer=llm_output.text,
            citations=snippets,
            diagnostics=diagnostics,
            timestamp=datetime.now(timezone.utc),
        )

    def scan_codebase(self, request: AgentScanRequest) -> AgentScanResponse:
        """Executa varredura completa em pasta para suporte a debug."""

        return scan_folder(
            folder_path=request.folder_path,
            include_hidden=request.include_hidden,
            max_files=request.max_files,
        )
