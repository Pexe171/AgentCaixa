"""FastAPI app entrypoint."""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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


@app.post("/v1/agent/chat/stream")
def agent_chat_stream(payload: AgentChatRequest) -> StreamingResponse:
    """Executa fluxo de chat com transmissão incremental via SSE."""

    def event_stream() -> str:
        for event in agent_service.chat_stream(payload):
            event_name = str(event.get("event", "message"))
            serialized = json.dumps(event, ensure_ascii=False)
            yield f"event: {event_name}\ndata: {serialized}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/agent/scan", response_model=AgentScanResponse)
def agent_scan(payload: AgentScanRequest) -> AgentScanResponse:
    """Executa varredura de uma pasta inteira para análise e debug."""
    try:
        return agent_service.scan_codebase(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
