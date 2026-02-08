from fastapi.testclient import TestClient

from rag_app.api.main import app
from rag_app.integrations.whatsapp import (
    normalize_whatsapp_number,
    parse_evolution_webhook,
)


def test_parse_evolution_webhook_extracts_text_message() -> None:
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5511999999999@s.whatsapp.net"},
            "pushName": "Analista Caixa",
            "message": {"conversation": "Qual status da proposta 123?"},
        },
    }

    incoming = parse_evolution_webhook(payload)

    assert incoming is not None
    assert incoming.remote_jid == "5511999999999@s.whatsapp.net"
    assert incoming.sender_name == "Analista Caixa"
    assert incoming.text == "Qual status da proposta 123?"


def test_normalize_whatsapp_number_removes_suffix() -> None:
    assert normalize_whatsapp_number("5511999999999@s.whatsapp.net") == "5511999999999"


def test_webhook_processes_message_when_channel_enabled(monkeypatch) -> None:
    client = TestClient(app)

    from rag_app.api import main as api_main

    monkeypatch.setattr(api_main.settings, "WHATSAPP_CHANNEL_ENABLED", True)
    monkeypatch.setattr(api_main.settings, "WHATSAPP_WEBHOOK_SECRET", "segredo-123")

    sent = {}

    def fake_send_text(*, number: str, text: str) -> None:
        sent["number"] = number
        sent["text"] = text

    monkeypatch.setattr(api_main.whatsapp_client, "_api_key", "token")
    monkeypatch.setattr(api_main.whatsapp_client, "_instance", "agentecaixa")
    monkeypatch.setattr(api_main.whatsapp_client, "send_text", fake_send_text)

    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5511988887777@s.whatsapp.net"},
            "pushName": "Analista",
            "message": {"conversation": "Olá, preciso de um resumo."},
        },
    }

    response = client.post(
        "/v1/channels/whatsapp/evolution/webhook",
        json=payload,
        headers={"x-webhook-secret": "segredo-123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processed"
    assert sent["number"] == "5511988887777"
    assert "Agente Caixa" in sent["text"]


def test_webhook_returns_401_with_invalid_secret(monkeypatch) -> None:
    client = TestClient(app)

    from rag_app.api import main as api_main

    monkeypatch.setattr(api_main.settings, "WHATSAPP_CHANNEL_ENABLED", True)
    monkeypatch.setattr(api_main.settings, "WHATSAPP_WEBHOOK_SECRET", "segredo-correto")

    response = client.post(
        "/v1/channels/whatsapp/evolution/webhook",
        json={"event": "messages.upsert", "data": {}},
        headers={"x-webhook-secret": "segredo-errado"},
    )

    assert response.status_code == 401
