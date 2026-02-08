"""FastAPI app entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from rag_app.agent.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentScanRequest,
    AgentScanResponse,
)
from rag_app.agent.service import AgentService
from rag_app.config import load_settings

settings = load_settings()
agent_service = AgentService(settings=settings)

app = FastAPI(title=settings.PROJECT_NAME)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


@app.post("/v1/agent/chat", response_model=AgentChatResponse)
def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
    """Executa fluxo completo do agente HAG."""
    return agent_service.chat(payload)


@app.post("/v1/agent/scan", response_model=AgentScanResponse)
def agent_scan(payload: AgentScanRequest) -> AgentScanResponse:
    """Executa varredura de uma pasta inteira para análise e debug."""
    try:
        return agent_service.scan_codebase(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
