"""Integração com WhatsApp via Evolution API."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class WhatsAppIncomingMessage:
    """Mensagem de entrada normalizada a partir do webhook."""

    remote_jid: str
    sender_name: str
    text: str


class EvolutionWhatsAppClient:
    """Cliente simplificado para enviar mensagens via Evolution API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        instance: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._instance = instance
        self._timeout_seconds = timeout_seconds

    @property
    def is_enabled(self) -> bool:
        return bool(self._base_url and self._api_key and self._instance)

    def send_text(self, *, number: str, text: str) -> None:
        """Envia texto para o número/contato de destino."""
        url = f"{self._base_url}/message/sendText/{self._instance}"
        headers = {"apikey": self._api_key, "Content-Type": "application/json"}
        payload = {
            "number": number,
            "text": text,
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()


def parse_evolution_webhook(payload: dict) -> WhatsAppIncomingMessage | None:
    """Extrai dados da mensagem de texto do webhook padrão da Evolution API."""
    if payload.get("event") != "messages.upsert":
        return None

    data = payload.get("data") or {}
    key = data.get("key") or {}
    remote_jid = str(key.get("remoteJid") or "").strip()
    if not remote_jid:
        return None

    message = data.get("message") or {}
    conversation = message.get("conversation")

    extended = message.get("extendedTextMessage") or {}
    text = str(conversation or extended.get("text") or "").strip()
    if not text:
        return None

    sender_name = str(data.get("pushName") or "Usuário WhatsApp").strip()
    return WhatsAppIncomingMessage(
        remote_jid=remote_jid,
        sender_name=sender_name,
        text=text,
    )


def normalize_whatsapp_number(remote_jid: str) -> str:
    """Converte remoteJid para o formato esperado pelo endpoint de envio."""
    return remote_jid.split("@")[0]
