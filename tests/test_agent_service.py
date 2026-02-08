from pathlib import Path

import pytest

from rag_app.agent.schemas import AgentChatRequest, AgentScanRequest
from rag_app.agent.service import AgentService
from rag_app.config import AppSettings


def test_agent_service_uses_mock_when_openai_not_configured() -> None:
    settings = AppSettings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=None,
        OPENAI_MODEL=None,
    )
    service = AgentService(settings=settings)

    response = service.chat(
        AgentChatRequest(user_message="Preciso de um plano para agente.")
    )

    assert response.diagnostics.provider_used == "mock"
    assert response.diagnostics.fallback_used is True
    assert response.citations


def test_agent_service_uses_mock_when_ollama_not_configured() -> None:
    settings = AppSettings(
        LLM_PROVIDER="ollama",
        OLLAMA_MODEL=None,
    )
    service = AgentService(settings=settings)

    response = service.chat(
        AgentChatRequest(user_message="Quero rodar localmente com ollama.")
    )

    assert response.diagnostics.provider_used == "mock"
    assert response.diagnostics.fallback_used is True


def test_agent_scan_service(tmp_path: Path) -> None:
    file_path = tmp_path / "index.js"
    file_path.write_text("console.log('tmp')\n", encoding="utf-8")

    service = AgentService(settings=AppSettings())
    response = service.scan_codebase(
        AgentScanRequest(folder_path=str(tmp_path), max_files=100)
    )

    assert response.files_scanned >= 1
    assert "javascript" in response.languages_detected
    assert response.issues


def test_provider_timeouts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AppSettings(OPENAI_TIMEOUT_SECONDS=0)

    with pytest.raises(ValueError):
        AppSettings(OLLAMA_TIMEOUT_SECONDS=0)
