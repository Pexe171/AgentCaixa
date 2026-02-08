"""Orquestração do agente HAG (Hybrid Agentic Generation)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from rag_app.agent.guardrails import append_audit_event, validate_user_message
from rag_app.agent.knowledge import retrieve_context
from rag_app.agent.llm_gateway import (
    LLMOutput,
    MockLLMGateway,
    OllamaLLMGateway,
    OpenAILLMGateway,
    ToolExecutor,
)
from rag_app.agent.orchestration import route_to_specialist, specialist_instruction
from rag_app.agent.scanner import scan_folder
from rag_app.agent.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentDiagnostics,
    AgentScanRequest,
    AgentScanResponse,
    ContextSnippet,
)
from rag_app.agent.semantic_memory import (
    SQLiteSemanticMemoryStore,
    build_semantic_summary,
)
from rag_app.agent.session_memory import SessionMemoryStore, SQLiteSessionMemoryStore
from rag_app.agent.vector_index import VectorRetriever
from rag_app.config import AppSettings


class LLMGateway(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> LLMOutput:
        """Gera resposta textual a partir de prompts e tool calling opcional."""


def _build_planning_prompt(user_message: str, memory_blocks: list[str]) -> str:
    memory_text = "\n".join(memory_blocks) if memory_blocks else "Sem histórico prévio"
    return (
        "Antes da resposta final, elabore um Plano de Execução curto "
        "para resolver o pedido.\n"
        "Formato obrigatório:\n"
        "1) Objetivo\n"
        "2) Contextos necessários\n"
        "3) Ferramentas/ações\n"
        "4) Critérios de qualidade\n\n"
        f"Pergunta do usuário:\n{user_message}\n\n"
        f"Memória da sessão:\n{memory_text}\n"
    )


def _build_system_prompt(
    tone: str,
    depth: str,
    require_citations: bool,
    specialist: str,
) -> str:
    citation_rule = (
        "Inclua citações explícitas às fontes de contexto usadas."
        if require_citations
        else "Não é obrigatório incluir citações."
    )
    return (
        "Você é um agente HAG de altíssima qualidade para uso corporativo. "
        f"Tom exigido: {tone}. Profundidade de raciocínio: {depth}. "
        f"Especialização ativa: {specialist_instruction(specialist)} "
        "Entregue resposta estruturada, prática, fiel ao contexto "
        "e sem inventar fatos. "
        f"{citation_rule}"
    )


def _build_user_prompt(
    user_message: str,
    execution_plan: str,
    context_blocks: list[str],
    memory_blocks: list[str],
) -> str:
    context_text = "\n".join(context_blocks)
    memory_text = "\n".join(memory_blocks) if memory_blocks else "Sem histórico prévio"
    return (
        f"Pergunta do usuário:\n{user_message}\n\n"
        f"Plano de execução aprovado:\n{execution_plan}\n\n"
        f"Memória da sessão:\n{memory_text}\n\n"
        f"Contexto recuperado:\n{context_text}\n\n"
        "Responda em português do Brasil com objetividade e completude."
    )


def _deduplicate_snippets(snippets: list[ContextSnippet]) -> list[ContextSnippet]:
    deduplicated: dict[tuple[str, str], ContextSnippet] = {}
    for snippet in snippets:
        key = (snippet.source, snippet.content)
        existing = deduplicated.get(key)
        if existing is None or snippet.score > existing.score:
            deduplicated[key] = snippet
    return list(deduplicated.values())


def _rank_candidates(
    query: str, snippets: list[ContextSnippet]
) -> list[ContextSnippet]:
    query_tokens = {token.strip(".,:;!?()[]{}\"'").lower() for token in query.split()}
    if not query_tokens:
        return snippets

    def rerank_score(snippet: ContextSnippet) -> float:
        snippet_tokens = {
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in snippet.content.split()
            if token
        }
        overlap = len(query_tokens.intersection(snippet_tokens))
        lexical_boost = overlap / max(1, len(query_tokens))
        return (snippet.score * 0.7) + (lexical_boost * 0.3)

    return sorted(snippets, key=rerank_score, reverse=True)


def _build_rerank_prompt(query: str, snippets: list[ContextSnippet]) -> str:
    serialized = [
        {
            "index": index,
            "source": snippet.source,
            "score": round(snippet.score, 4),
            "content": snippet.content,
        }
        for index, snippet in enumerate(snippets)
    ]
    return (
        "Reordene os trechos abaixo para responder a pergunta com máxima precisão. "
        'Retorne apenas JSON no formato {"selected_indexes":[i1,i2,...]} '
        "com até 5 índices únicos.\n"
        f"Pergunta: {query}\n"
        f"Trechos: {json.dumps(serialized, ensure_ascii=False)}"
    )


def _extract_selected_indexes(raw_text: str, max_index: int) -> list[int]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        payload = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return []

    selected_indexes = payload.get("selected_indexes")
    if not isinstance(selected_indexes, list):
        return []

    valid_indexes: list[int] = []
    for value in selected_indexes:
        if not isinstance(value, int):
            continue
        if value < 0 or value >= max_index:
            continue
        if value in valid_indexes:
            continue
        valid_indexes.append(value)
    return valid_indexes[:5]




def _build_query_translation_prompt(user_message: str) -> str:
    return (
        "Reescreva a pergunta para busca semântica corporativa, "
        "removendo ambiguidades, "
        "expandindo nomes próprios abreviados e preservando intenção. "
        "Retorne apenas uma única frase em português do Brasil.\n\n"
        f"Pergunta original: {user_message}"
    )


def _normalize_translated_query(user_message: str, translated: str) -> str:
    cleaned = translated.strip().replace("\n", " ")
    if not cleaned:
        return user_message
    if len(cleaned) < 12:
        return user_message
    if "[modo mock]" in cleaned.lower():
        return user_message
    return cleaned


def _build_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "consultar_caixa",
            "description": (
                "Consulta dados consolidados do cliente na Caixa a partir "
                "do id_cliente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id_cliente": {
                        "type": "string",
                        "description": "Identificador único do cliente.",
                    }
                },
                "required": ["id_cliente"],
                "additionalProperties": False,
            },
        }
    ]


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


def _build_tool_executor(context_blocks: list[str]) -> ToolExecutor:
    def execute(tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "consultar_caixa":
            return {"erro": f"Ferramenta desconhecida: {tool_name}"}

        client_id = str(arguments.get("id_cliente", "")).strip()
        if not client_id:
            return {"erro": "Parâmetro id_cliente é obrigatório."}

        normalized_id = client_id.lower()
        related_context = [
            block for block in context_blocks if normalized_id in block.lower()
        ][:3]

        return {
            "id_cliente": client_id,
            "status": "consulta_realizada",
            "fontes_relacionadas": related_context,
            "resumo": (
                "Consulta simulada executada. Utilize os dados retornados "
                "para compor a resposta."
            ),
        }

    return execute


def _resolve_session_memory(
    settings: AppSettings,
) -> SessionMemoryStore | SQLiteSessionMemoryStore:
    if settings.SESSION_STORE_BACKEND == "sqlite":
        return SQLiteSessionMemoryStore(database_path=settings.SESSION_DB_PATH)
    return SessionMemoryStore()


def _resolve_semantic_memory(settings: AppSettings) -> SQLiteSemanticMemoryStore | None:
    if settings.SEMANTIC_MEMORY_BACKEND == "sqlite":
        return SQLiteSemanticMemoryStore(database_path=settings.SEMANTIC_MEMORY_DB_PATH)
    return None


class AgentService:
    """Serviço principal de execução do agente."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._gateway = _resolve_gateway(settings)
        self._memory = _resolve_session_memory(settings)
        self._semantic_memory = _resolve_semantic_memory(settings)
        self._vector_retriever = VectorRetriever(settings=settings)

    def _maybe_persist_semantic_memory(self, session_id: str) -> None:
        if self._semantic_memory is None:
            return

        message_count = self._memory.count(session_id)
        if message_count % self._settings.SEMANTIC_MEMORY_SUMMARY_INTERVAL != 0:
            return

        recent_entries = self._memory.get_recent_entries(session_id)
        summary = build_semantic_summary(recent_entries=recent_entries)
        self._semantic_memory.add_summary(session_id=session_id, summary=summary)

    def _estimate_cost(self, prompt: str, answer: str) -> float:
        approx_tokens = max(1, (len(prompt) + len(answer)) // 4)
        return round((approx_tokens / 1000) * self._settings.COST_PER_1K_TOKENS_USD, 6)

    def _build_execution_plan(
        self,
        user_message: str,
        memory_blocks: list[str],
        system_prompt: str,
    ) -> str:
        planning_prompt = _build_planning_prompt(
            user_message=user_message,
            memory_blocks=memory_blocks,
        )
        plan_output = self._gateway.generate(
            system_prompt=(
                f"{system_prompt} "
                "Você está na etapa de planejamento e não deve escrever "
                "a resposta final."
            ),
            user_prompt=planning_prompt,
        )
        return plan_output.text

    def _rewrite_query_for_retrieval(
        self,
        user_message: str,
        system_prompt: str,
    ) -> str:
        try:
            translation_output = self._gateway.generate(
                system_prompt=(
                    f"{system_prompt} "
                    "Sua tarefa agora é apenas traduzir a consulta para recuperação."
                ),
                user_prompt=_build_query_translation_prompt(user_message),
            )
            return _normalize_translated_query(user_message, translation_output.text)
        except Exception:
            return user_message

    def _rerank_snippets(
        self,
        query: str,
        snippets: list[ContextSnippet],
    ) -> list[ContextSnippet]:
        if not snippets:
            return []

        ranked_candidates = _rank_candidates(query=query, snippets=snippets)[:20]
        rerank_prompt = _build_rerank_prompt(query=query, snippets=ranked_candidates)

        try:
            rerank_output = self._gateway.generate(
                system_prompt=(
                    "Você é um reranker especializado em recuperação híbrida. "
                    "Responda estritamente em JSON válido."
                ),
                user_prompt=rerank_prompt,
            )
            selected_indexes = _extract_selected_indexes(
                raw_text=rerank_output.text,
                max_index=len(ranked_candidates),
            )
            if selected_indexes:
                return [ranked_candidates[index] for index in selected_indexes]
        except Exception:
            pass

        return ranked_candidates[:5]

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

        rewritten_query = self._rewrite_query_for_retrieval(
            user_message=request.user_message,
            system_prompt=(
                "Você otimiza consultas para recuperação de conhecimento "
                "em ambientes financeiros e corporativos."
            ),
        )

        lexical_snippets = retrieve_context(
            query=rewritten_query,
            top_k=20,
        )
        vector_snippets = self._vector_retriever.retrieve(
            query=rewritten_query,
            top_k=20,
        )
        snippets = self._rerank_snippets(
            query=rewritten_query,
            snippets=_deduplicate_snippets(lexical_snippets + vector_snippets),
        )

        context_blocks = [
            "- Fonte: "
            f"{snippet.source} | Relevância: {snippet.score:.2f} "
            f"| Conteúdo: {snippet.content}"
            for snippet in snippets
        ]

        memory_blocks: list[str] = []
        if request.session_id:
            memory_blocks = self._memory.get_recent(request.session_id)
            if self._semantic_memory is not None:
                semantic_blocks = self._semantic_memory.retrieve_relevant(
                    session_id=request.session_id,
                    query=rewritten_query,
                    top_k=self._settings.SEMANTIC_MEMORY_RETRIEVE_TOP_K,
                )
                if semantic_blocks:
                    memory_blocks.append("Memória semântica de longo prazo:")
                    memory_blocks.extend(
                        [f"- {semantic_block}" for semantic_block in semantic_blocks]
                    )

        routing = route_to_specialist(request.user_message)

        system_prompt = _build_system_prompt(
            tone=request.tone,
            depth=request.reasoning_depth,
            require_citations=request.require_citations,
            specialist=routing.specialist,
        )
        execution_plan = self._build_execution_plan(
            user_message=request.user_message,
            memory_blocks=memory_blocks,
            system_prompt=system_prompt,
        )
        user_prompt = _build_user_prompt(
            user_message=request.user_message,
            execution_plan=execution_plan,
            context_blocks=context_blocks,
            memory_blocks=memory_blocks,
        )

        llm_output = self._gateway.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=_build_openai_tools(),
            tool_executor=_build_tool_executor(context_blocks=context_blocks),
        )

        if request.session_id:
            self._memory.append(request.session_id, "usuario", request.user_message)
            self._memory.append(request.session_id, "agente", llm_output.text)
            self._maybe_persist_semantic_memory(request.session_id)

        latency_ms = int((time.perf_counter() - start) * 1000)
        diagnostics = AgentDiagnostics(
            provider_used=llm_output.provider,
            model=llm_output.model,
            latency_ms=latency_ms,
            retrieved_context_count=len(snippets),
            fallback_used=llm_output.provider == "mock",
            trace_id=trace_id,
            estimated_cost_usd=self._estimate_cost(user_prompt, llm_output.text),
            routed_specialist=routing.specialist,
            routing_reason=routing.reason,
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

    def chat_stream(
        self,
        request: AgentChatRequest,
        chunk_size: int = 120,
    ) -> Iterator[dict[str, Any]]:
        """Executa chat e emite eventos incrementais para SSE."""

        yield {
            "event": "status",
            "stage": "starting",
            "message": "Iniciando pipeline do agente.",
        }
        response = self.chat(request)
        yield {
            "event": "status",
            "stage": "streaming",
            "message": "Transmitindo resposta em tempo real.",
        }

        answer = response.answer
        for start in range(0, len(answer), chunk_size):
            delta = answer[start : start + chunk_size]
            if not delta:
                continue
            yield {"event": "delta", "delta": delta}

        yield {
            "event": "done",
            "answer": response.answer,
            "citations": [item.model_dump() for item in response.citations],
            "diagnostics": response.diagnostics.model_dump(),
            "timestamp": response.timestamp.isoformat(),
        }

    def scan_codebase(self, request: AgentScanRequest) -> AgentScanResponse:
        """Executa varredura completa em pasta para suporte a debug."""

        return scan_folder(
            folder_path=request.folder_path,
            include_hidden=request.include_hidden,
            max_files=request.max_files,
            run_linters=request.run_linters or self._settings.ENABLE_LINTER_SCAN,
        )
