from rag_app.agent.session_memory import SQLiteSessionMemoryStore


def test_sqlite_session_memory_persists_between_instances(tmp_path) -> None:
    database_path = tmp_path / "session.db"

    first_store = SQLiteSessionMemoryStore(
        database_path=str(database_path),
        max_messages=4,
    )
    first_store.append("sessao-1", "usuario", "oi")
    first_store.append("sessao-1", "agente", "olá")

    second_store = SQLiteSessionMemoryStore(
        database_path=str(database_path),
        max_messages=4,
    )
    history = second_store.get_recent("sessao-1")

    assert history == ["usuario: oi", "agente: olá"]


def test_sqlite_session_memory_limits_recent_messages(tmp_path) -> None:
    database_path = tmp_path / "session_limit.db"
    store = SQLiteSessionMemoryStore(database_path=str(database_path), max_messages=2)

    store.append("sessao-2", "usuario", "m1")
    store.append("sessao-2", "agente", "m2")
    store.append("sessao-2", "usuario", "m3")

    assert store.get_recent("sessao-2") == ["agente: m2", "usuario: m3"]
