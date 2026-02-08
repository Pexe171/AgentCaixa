"""Schemas da camada de agente HAG."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ContextSnippet(BaseModel):
    """Trecho de contexto retornado por recuperação híbrida."""

    source: str = Field(description="Fonte do contexto, ex.: FAQ, documento interno")
    content: str = Field(description="Conteúdo textual relevante")
    score: float = Field(ge=0.0, le=1.0, description="Relevância normalizada")


class AgentChatRequest(BaseModel):
    """Entrada principal do agente."""

    user_message: str = Field(min_length=1, description="Mensagem do usuário")
    session_id: str | None = Field(
        default=None,
        description="Identificador de sessão para memória de curto prazo",
    )
    tone: Literal["profissional", "amigavel", "tecnico", "didatico"] = "profissional"
    reasoning_depth: Literal["rapido", "padrao", "profundo"] = "padrao"
    require_citations: bool = True


class AgentDiagnostics(BaseModel):
    """Metadados para observabilidade do pipeline."""

    provider_used: str
    model: str
    latency_ms: int
    retrieved_context_count: int
    fallback_used: bool
    trace_id: str
    estimated_cost_usd: float
    routed_specialist: str = "atendimento_geral"
    routing_reason: str = "Sem roteamento especializado."


class AgentChatResponse(BaseModel):
    """Saída da API do agente."""

    answer: str
    citations: list[ContextSnippet]
    diagnostics: AgentDiagnostics
    timestamp: datetime


class ScanIssue(BaseModel):
    """Achado de análise estática simples."""

    file_path: str
    line_number: int
    severity: Literal["baixa", "media", "alta"]
    category: Literal[
        "possivel-segredo",
        "tratamento-de-erros",
        "debug-residual",
        "codigo-comentado",
        "boas-praticas",
    ]
    message: str
    suggestion: str


class AgentScanRequest(BaseModel):
    """Entrada para varredura completa de pasta."""

    folder_path: str = Field(min_length=1)
    include_hidden: bool = False
    max_files: int = Field(default=400, ge=1, le=5000)
    run_linters: bool = False


class AgentScanResponse(BaseModel):
    """Saída da varredura com diagnóstico consolidado."""

    folder_path: str
    files_scanned: int
    languages_detected: list[str]
    issues: list[ScanIssue]
    linter_findings: list[str] = Field(default_factory=list)
    summary: str
