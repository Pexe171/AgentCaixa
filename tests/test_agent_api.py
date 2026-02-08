from pathlib import Path

from fastapi.testclient import TestClient

from rag_app.api.main import app


def test_agent_chat_returns_structured_response() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/agent/chat",
        json={
            "user_message": "Quero criar um agente com OpenAI API e monitoramento.",
            "tone": "tecnico",
            "reasoning_depth": "profundo",
            "require_citations": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert "answer" in payload
    assert "citations" in payload
    assert "diagnostics" in payload
    assert payload["diagnostics"]["provider_used"] in {"mock", "openai", "ollama"}
    assert isinstance(payload["citations"], list)


def test_agent_scan_returns_report(tmp_path: Path) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('debug')\nAPI_KEY = '123'\n", encoding="utf-8")

    client = TestClient(app)
    response = client.post(
        "/v1/agent/scan",
        json={
            "folder_path": str(tmp_path),
            "include_hidden": False,
            "max_files": 100,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["files_scanned"] >= 1
    assert payload["issues"]


def test_agent_scan_invalid_folder_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/agent/scan",
        json={
            "folder_path": "/caminho/nao/existe",
            "include_hidden": False,
            "max_files": 100,
        },
    )

    assert response.status_code == 400
