from rag_app.config import load_settings


def test_load_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("RETRIEVE_TOP_K_DEFAULT", raising=False)

    settings = load_settings()

    assert settings.PROJECT_NAME == "rag_app"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.RETRIEVE_TOP_K_DEFAULT == 6


def test_load_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PROJECT_NAME", "rag_app_local")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RETRIEVE_TOP_K_DEFAULT", "8")

    settings = load_settings()

    assert settings.PROJECT_NAME == "rag_app_local"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.RETRIEVE_TOP_K_DEFAULT == 8
