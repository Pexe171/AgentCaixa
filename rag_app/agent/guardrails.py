"""Guardrails e trilha de auditoria para entradas do agente."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "bypass",
    "exfiltrate",
)


def validate_user_message(user_message: str) -> tuple[bool, str | None]:
    lower_message = user_message.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower_message:
            return False, (
                "Mensagem bloqueada por política de segurança: "
                f"padrão '{pattern}' detectado."
            )
    return True, None


def append_audit_event(audit_file: str, event: str, trace_id: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} trace_id={trace_id} event={event}\n"
    path = Path(audit_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as audit_stream:
        audit_stream.write(line)
