"""FastAPI app entrypoint."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from rag_app.agent.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentScanRequest,
    AgentScanResponse,
    ImageDataAnalysisRequest,
    ImageDataAnalysisResponse,
)
from rag_app.agent.service import AgentService
from rag_app.config import load_settings
from rag_app.integrations.whatsapp import (
    EvolutionWhatsAppClient,
    normalize_whatsapp_number,
    parse_evolution_webhook,
)

settings = load_settings()
agent_service = AgentService(settings=settings)

app = FastAPI(title=settings.PROJECT_NAME)
whatsapp_client = EvolutionWhatsAppClient(
    base_url=settings.WHATSAPP_EVOLUTION_BASE_URL,
    api_key=settings.WHATSAPP_EVOLUTION_API_KEY or "",
    instance=settings.WHATSAPP_EVOLUTION_INSTANCE,
)


def _parse_audit_events(audit_file: str) -> list[dict[str, str]]:
    path = Path(audit_file)
    if not path.exists():
        return []

    events: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        item: dict[str, str] = {"timestamp": parts[0]}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            item[key] = value
        events.append(item)
    return events


def _load_judge_results(judge_path: str) -> dict[str, Any]:
    path = Path(judge_path)
    if not path.exists():
        return {"media": 0.0, "resultados": [], "judge_provider": "indisponível"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"media": 0.0, "resultados": [], "judge_provider": "arquivo-inválido"}


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


@app.get("/admin/metrics")
def admin_metrics() -> dict[str, Any]:
    """Métricas para dashboard administrativo."""
    events = _parse_audit_events(settings.AUDIT_LOG_PATH)
    event_counter = Counter(item.get("event", "desconhecido") for item in events)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    recent_events: list[dict[str, str]] = []
    for item in events:
        try:
            event_time = datetime.fromisoformat(item["timestamp"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        if event_time >= cutoff:
            recent_events.append(item)

    judge = _load_judge_results(settings.JUDGE_RESULTS_PATH)
    return {
        "total_eventos": len(events),
        "eventos_ultima_hora": len(recent_events),
        "eventos_por_tipo": dict(event_counter),
        "judge_media": float(judge.get("media", 0.0)),
        "judge_provider": str(judge.get("judge_provider", "indisponível")),
        "judge_resultados": judge.get("resultados", []),
    }


@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard() -> str:
    """Dashboard administrativo simples para auditoria e avaliações do judge."""
    return """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard Administrativo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f8fa; }
    .cards { display: grid; grid-template-columns: repeat(3, minmax(200px,1fr)); gap: 12px; margin-bottom: 18px; }
    .card { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
    .title { color: #444; font-size: 12px; text-transform: uppercase; }
    .value { font-size: 28px; font-weight: bold; }
    canvas { background: white; border-radius: 10px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; background: white; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }
  </style>
</head>
<body>
  <h1>Monitorização do Agente</h1>
  <div class="cards">
    <div class="card"><div class="title">Eventos totais</div><div class="value" id="total">0</div></div>
    <div class="card"><div class="title">Eventos última hora</div><div class="value" id="hour">0</div></div>
    <div class="card"><div class="title">Média Judge</div><div class="value" id="judge">0</div></div>
  </div>
  <canvas id="bar" width="900" height="260"></canvas>
  <h2>Avaliações do Judge</h2>
  <table>
    <thead><tr><th>Caso</th><th>Nota</th><th>Justificativa</th></tr></thead>
    <tbody id="judgeRows"></tbody>
  </table>
<script>
async function loadData() {
  const res = await fetch('/admin/metrics');
  const data = await res.json();
  document.getElementById('total').textContent = data.total_eventos;
  document.getElementById('hour').textContent = data.eventos_ultima_hora;
  document.getElementById('judge').textContent = data.judge_media.toFixed(2);

  const ctx = document.getElementById('bar').getContext('2d');
  const labels = Object.keys(data.eventos_por_tipo);
  const values = Object.values(data.eventos_por_tipo);
  const max = Math.max(1, ...values);
  ctx.clearRect(0, 0, 900, 260);
  labels.forEach((label, i) => {
    const x = 40 + i * 160;
    const h = (values[i] / max) * 160;
    const y = 220 - h;
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(x, y, 80, h);
    ctx.fillStyle = '#111';
    ctx.fillText(String(values[i]), x + 22, y - 8);
    ctx.fillText(label, x, 242);
  });

  const rows = data.judge_resultados || [];
  document.getElementById('judgeRows').innerHTML = rows.map((row) =>
    `<tr><td>${row.caso}</td><td>${row.nota}</td><td>${row.justificativa}</td></tr>`
  ).join('');
}
loadData();
setInterval(loadData, 10000);
</script>
</body>
</html>
"""


@app.post("/v1/agent/chat", response_model=AgentChatResponse)
def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
    """Executa fluxo completo do agente HAG."""
    try:
        return agent_service.chat(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/agent/chat/stream")
def agent_chat_stream(payload: AgentChatRequest) -> StreamingResponse:
    """Executa fluxo de chat com transmissão incremental via SSE."""

    def event_stream() -> str:
        try:
            for event in agent_service.chat_stream(payload):
                event_name = str(event.get("event", "message"))
                serialized = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {serialized}\n\n"
        except ValueError as exc:
            error_payload = {"event": "error", "detail": str(exc)}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/agent/image/analyze", response_model=ImageDataAnalysisResponse)
def agent_image_analyze(payload: ImageDataAnalysisRequest) -> ImageDataAnalysisResponse:
    """Executa análise de dados de imagem com vetores locais (sem API externa)."""
    try:
        return agent_service.analyze_image_data(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/agent/scan", response_model=AgentScanResponse)
def agent_scan(payload: AgentScanRequest) -> AgentScanResponse:
    """Executa varredura de uma pasta inteira para análise e debug."""
    try:
        return agent_service.scan_codebase(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/channels/whatsapp/evolution/webhook")
def whatsapp_evolution_webhook(
    payload: dict[str, Any],
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Recebe eventos do WhatsApp e responde pelo canal configurado."""
    if not settings.WHATSAPP_CHANNEL_ENABLED:
        return {"status": "ignored", "reason": "canal-desabilitado"}

    if settings.WHATSAPP_WEBHOOK_SECRET and (
        x_webhook_secret != settings.WHATSAPP_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=401, detail="Webhook secret inválido")

    incoming = parse_evolution_webhook(payload)
    if incoming is None:
        return {"status": "ignored", "reason": "evento-nao-suportado"}

    if not whatsapp_client.is_enabled:
        raise HTTPException(status_code=500, detail="Cliente WhatsApp não configurado")

    chat_response = agent_service.chat(
        AgentChatRequest(
            user_message=incoming.text,
            session_id=f"whatsapp:{incoming.remote_jid}",
            tone="profissional",
            reasoning_depth="padrao",
            require_citations=False,
        )
    )
    outbound_number = normalize_whatsapp_number(incoming.remote_jid)
    answer_text = (
        "🤖 *Agente Caixa*\n\n"
        f"{chat_response.answer}\n\n"
        "_Se quiser, peça mais detalhes ou um passo a passo._"
    )
    whatsapp_client.send_text(number=outbound_number, text=answer_text)

    return {
        "status": "processed",
        "to": outbound_number,
        "trace_id": chat_response.diagnostics.trace_id,
    }
