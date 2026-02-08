from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from rag_app.api.main import app


@pytest.fixture(autouse=True)
def _stub_ollama_generate(monkeypatch):
    from rag_app.agent import service as service_module

    def fake_generate(self, system_prompt: str, user_prompt: str, tools=None, tool_executor=None):
        del self, system_prompt, user_prompt, tools, tool_executor
        return service_module.LLMOutput(
            text="Resposta de teste via Ollama",
            provider="ollama",
            model="llama3.2",
        )

    monkeypatch.setattr(service_module.OllamaLLMGateway, "generate", fake_generate)



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
    assert payload["diagnostics"]["provider_used"] in {"openai", "ollama"}
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


def test_agent_scan_can_run_linters(tmp_path: Path) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('debug')\n", encoding="utf-8")

    client = TestClient(app)
    response = client.post(
        "/v1/agent/scan",
        json={
            "folder_path": str(tmp_path),
            "include_hidden": False,
            "max_files": 100,
            "run_linters": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "linter_findings" in payload
    assert "Linters executados: sim" in payload["summary"]


def test_agent_chat_stream_returns_sse_events() -> None:
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/agent/chat/stream",
        json={
            "user_message": "Explique como melhorar monitoramento do agente.",
            "tone": "tecnico",
            "reasoning_depth": "padrao",
            "require_citations": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: status" in body
    assert "event: delta" in body
    assert "event: done" in body


def test_admin_metrics_and_dashboard_endpoints() -> None:
    client = TestClient(app)

    metrics_response = client.get("/admin/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "total_eventos" in metrics
    assert "judge_media" in metrics

    dashboard_response = client.get("/admin/dashboard")
    assert dashboard_response.status_code == 200
    assert "Monitorização do Agente" in dashboard_response.text


def test_agent_image_analysis_returns_vector_metrics(tmp_path: Path) -> None:
    pytest.importorskip("PIL", reason="Pillow não instalado")
    from PIL import Image

    image_path = tmp_path / "image.png"
    reference_path = tmp_path / "reference.png"

    Image.new("RGB", (40, 24), color=(80, 120, 200)).save(image_path)
    Image.new("RGB", (40, 24), color=(82, 121, 199)).save(reference_path)

    client = TestClient(app)
    response = client.post(
        "/v1/agent/image/analyze",
        json={
            "image_path": str(image_path),
            "reference_image_path": str(reference_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["vector_dimensions"] > 100
    assert payload["similarity_score"] is not None
    assert payload["top_palette"]


def test_agent_image_analysis_invalid_path_returns_400() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/agent/image/analyze",
        json={"image_path": "/caminho/invalido/imagem.png"},
    )

    assert response.status_code == 400
