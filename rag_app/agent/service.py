"""Orquestração do agente HAG (Hybrid Agentic Generation)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from rag_app.agent.guardrails import append_audit_event, validate_user_message
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
from rag_app.agent.session_memory import SessionMemoryStore, SQLiteSessionMemoryStore
from rag_app.agent.vector_index import VectorRetriever
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


def _build_user_prompt(
    user_message: str,
    context_blocks: list[str],
    memory_blocks: list[str],
) -> str:
    context_text = "\n".join(context_blocks)
    memory_text = "\n".join(memory_blocks) if memory_blocks else "Sem histórico prévio"
    return (
        f"Pergunta do usuário:\n{user_message}\n\n"
        f"Memória da sessão:\n{memory_text}\n\n"
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


def _resolve_session_memory(
    settings: AppSettings,
) -> SessionMemoryStore | SQLiteSessionMemoryStore:
    if settings.SESSION_STORE_BACKEND == "sqlite":
        return SQLiteSessionMemoryStore(database_path=settings.SESSION_DB_PATH)
    return SessionMemoryStore()


class AgentService:
    """Serviço principal de execução do agente."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._gateway = _resolve_gateway(settings)
        self._memory = _resolve_session_memory(settings)
        self._vector_retriever = VectorRetriever(settings=settings)

    def _estimate_cost(self, prompt: str, answer: str) -> float:
        approx_tokens = max(1, (len(prompt) + len(answer)) // 4)
        return round((approx_tokens / 1000) * self._settings.COST_PER_1K_TOKENS_USD, 6)

    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        start = time.perf_counter()
        trace_id = str(uuid4())

        allowed, security_message = validate_user_message(request.user_message)
        if not allowed:
            append_audit_event(
                audit_file=self._settings.AUDIT_LOG_PATH,
                event="chat_blocked_by_guardrail",
                trace_id=trace_id,
            )
            diagnostics = AgentDiagnostics(
                provider_used="guardrail",
                model="policy",
                latency_ms=1,
                retrieved_context_count=0,
                fallback_used=False,
                trace_id=trace_id,
                estimated_cost_usd=0.0,
            )
            return AgentChatResponse(
                answer=security_message or "Mensagem bloqueada por política.",
                citations=[],
                diagnostics=diagnostics,
                timestamp=datetime.now(timezone.utc),
            )

        snippets = retrieve_context(
            query=request.user_message,
            top_k=self._settings.RETRIEVE_TOP_K_DEFAULT,
        )
        vector_snippets = self._vector_retriever.retrieve(
            query=request.user_message,
            top_k=self._settings.RETRIEVE_TOP_K_DEFAULT,
        )
        snippets = snippets + vector_snippets

        context_blocks = [
            "- Fonte: "
            f"{snippet.source} | Relevância: {snippet.score:.2f} "
            f"| Conteúdo: {snippet.content}"
            for snippet in snippets
        ]

        memory_blocks: list[str] = []
        if request.session_id:
            memory_blocks = self._memory.get_recent(request.session_id)

        system_prompt = _build_system_prompt(
            tone=request.tone,
            depth=request.reasoning_depth,
            require_citations=request.require_citations,
        )
        user_prompt = _build_user_prompt(
            user_message=request.user_message,
            context_blocks=context_blocks,
            memory_blocks=memory_blocks,
        )

        llm_output = self._gateway.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if request.session_id:
            self._memory.append(request.session_id, "usuario", request.user_message)
            self._memory.append(request.session_id, "agente", llm_output.text)

        latency_ms = int((time.perf_counter() - start) * 1000)
        diagnostics = AgentDiagnostics(
            provider_used=llm_output.provider,
            model=llm_output.model,
            latency_ms=latency_ms,
            retrieved_context_count=len(snippets),
            fallback_used=llm_output.provider == "mock",
            trace_id=trace_id,
            estimated_cost_usd=self._estimate_cost(user_prompt, llm_output.text),
        )

        append_audit_event(
            audit_file=self._settings.AUDIT_LOG_PATH,
            event="chat_completed",
            trace_id=trace_id,
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
            run_linters=request.run_linters or self._settings.ENABLE_LINTER_SCAN,
        )
