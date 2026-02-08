"""Guardrails e trilha de auditoria para entradas do agente."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "bypass",
    "exfiltrate",
    "desative os guardrails",
    "ignore todas as regras",
)

PII_PATTERNS: dict[str, str] = {
    "cpf": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "cartao_credito": r"\b(?:\d[ -]*?){13,16}\b",
}

SENSITIVE_INTENT_PATTERNS = (
    "lista de clientes",
    "base completa",
    "extraia dados",
    "dados pessoais",
    "credenciais",
)

PERSONA_SHIFT_PATTERNS = (
    "agora você é",
    "finja que você é",
    "mude seu papel para",
    "atue como",
)


def _detect_blocked_pattern(user_message: str) -> str | None:
    lower_message = user_message.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_message:
            return pattern
    return None


def _detect_pii_extraction(user_message: str) -> str | None:
    lower_message = user_message.lower()
    if any(pattern in lower_message for pattern in SENSITIVE_INTENT_PATTERNS):
        return "tentativa de extração massiva de dados sensíveis"

    for pii_name, regex in PII_PATTERNS.items():
        if re.search(regex, user_message):
            return f"possível exposição de {pii_name}"
    return None


def _detect_behavior_shift(user_message: str, previous_messages: list[str]) -> str | None:
    lower_message = user_message.lower()
    if any(pattern in lower_message for pattern in PERSONA_SHIFT_PATTERNS):
        return "tentativa de mudança drástica de comportamento"

    if not previous_messages:
        return None

    historical_text = " ".join(previous_messages).lower()
    mentions_security_before = any(
        keyword in historical_text
        for keyword in ("segurança", "compliance", "política", "lgpd")
    )
    asks_to_ignore_now = any(
        token in lower_message
        for token in ("ignore", "desconsidere", "quebre as regras", "sem restrições")
    )
    if mentions_security_before and asks_to_ignore_now:
        return "mudança inconsistente de intenção detectada"
    return None


def validate_user_message(
    user_message: str,
    previous_messages: list[str] | None = None,
) -> tuple[bool, str | None]:
    blocked_pattern = _detect_blocked_pattern(user_message)
    if blocked_pattern:
        return False, (
            "Mensagem bloqueada por política de segurança: "
            f"padrão '{blocked_pattern}' detectado."
        )

    pii_issue = _detect_pii_extraction(user_message)
    if pii_issue:
        return False, (
            "Mensagem bloqueada por política de privacidade: "
            f"{pii_issue}."
        )

    behavior_issue = _detect_behavior_shift(user_message, previous_messages or [])
    if behavior_issue:
        return False, (
            "Mensagem bloqueada por política comportamental: "
            f"{behavior_issue}."
        )

    return True, None


def append_audit_event(audit_file: str, event: str, trace_id: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} trace_id={trace_id} event={event}\n"
    path = Path(audit_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as audit_stream:
        audit_stream.write(line)
