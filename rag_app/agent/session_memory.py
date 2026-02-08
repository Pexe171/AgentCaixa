"""Memória de conversa por sessão."""

from __future__ import annotations

from collections import defaultdict, deque


class SessionMemoryStore:
    """Armazena mensagens recentes por ``session_id`` em memória local."""

    def __init__(self, max_messages: int = 8) -> None:
        self._max_messages = max_messages
        self._messages: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=self._max_messages)
        )

    def append(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append(f"{role}: {content}")

    def get_recent(self, session_id: str) -> list[str]:
        return list(self._messages.get(session_id, []))
