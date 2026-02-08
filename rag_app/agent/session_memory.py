"""Memória de conversa por sessão com backend configurável."""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from pathlib import Path


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

    def get_recent_entries(self, session_id: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for item in self._messages.get(session_id, []):
            if ": " not in item:
                continue
            role, content = item.split(": ", maxsplit=1)
            entries.append((role, content))
        return entries

    def count(self, session_id: str) -> int:
        return len(self._messages.get(session_id, []))


class SQLiteSessionMemoryStore:
    """Memória persistente por sessão usando SQLite."""

    def __init__(self, database_path: str, max_messages: int = 8) -> None:
        self._db_path = Path(database_path)
        self._max_messages = max_messages
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )

    def get_recent(self, session_id: str) -> list[str]:
        rows = self.get_recent_entries(session_id)
        return [f"{role}: {content}" for role, content in rows]

    def get_recent_entries(self, session_id: str) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM session_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, self._max_messages),
            ).fetchall()

        return list(reversed(rows))

    def count(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM session_messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return int(row[0] if row else 0)
