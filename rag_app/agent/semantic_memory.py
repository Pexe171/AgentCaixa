"""Memória semântica de longo prazo baseada em SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag_app.agent.vector_index import _cosine_similarity, _embed_text


class SQLiteSemanticMemoryStore:
    """Armazena resumos semânticos por sessão com recuperação vetorial."""

    def __init__(self, database_path: str) -> None:
        self._db_path = Path(database_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add_summary(self, session_id: str, summary: str) -> None:
        embedding = ",".join(f"{value:.8f}" for value in _embed_text(summary))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_memories (session_id, summary, embedding)
                VALUES (?, ?, ?)
                """,
                (session_id, summary, embedding),
            )

    def retrieve_relevant(
        self,
        session_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        query_embedding = _embed_text(query)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT summary, embedding
                FROM semantic_memories
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (session_id,),
            ).fetchall()

        ranked: list[tuple[float, str]] = []
        for summary, embedding_serialized in rows:
            embedding = [float(item) for item in embedding_serialized.split(",")]
            score = _cosine_similarity(query_embedding, embedding)
            ranked.append((score, summary))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [summary for score, summary in ranked[:top_k] if score > 0.0]

    def count(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM semantic_memories
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return int(row[0] if row else 0)


def build_semantic_summary(recent_entries: list[tuple[str, str]]) -> str:
    """Consolida os fatos mais recentes em resumo curto e persistível."""

    if not recent_entries:
        return "Sem fatos recentes para resumir."

    latest_user = ""
    latest_agent = ""
    for role, content in reversed(recent_entries):
        if role == "usuario" and not latest_user:
            latest_user = content
        if role == "agente" and not latest_agent:
            latest_agent = content
        if latest_user and latest_agent:
            break

    factual_points = []
    if latest_user:
        factual_points.append(f"Pedido do cliente: {latest_user[:220]}")
    if latest_agent:
        factual_points.append(f"Resposta entregue: {latest_agent[:220]}")

    if factual_points:
        return " | ".join(factual_points)
    return "Sem fatos recentes para resumir."
