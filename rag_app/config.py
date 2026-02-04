"""Centralized configuration for rag_app."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator


class AppSettings(BaseSettings):
    PROJECT_NAME: str = "rag_app"
    DATA_DIR: str = "data"
    PROCESSED_DIR: str = "data/processed"
    INDEX_DIR: str = "data/index"
    LOG_LEVEL: str = "INFO"
    DISABLE_LOGS: bool = False
    API_KEY: str | None = None
    MAX_SNIPPET_CHARS: int = 240
    CHUNK_MAX_CHARS: int = 4200
    CHUNK_MIN_CHARS: int = 400
    RETRIEVE_TOP_K_DEFAULT: int = 6
    HYBRID_ALPHA: float = 0.65
    STRICT_MIN_SCORE: float = 0.20
    ENABLE_DECOMPOSITION: bool = True
    LLM_PROVIDER: Literal["mock", "openai"] = "mock"
    OPENAI_MODEL: str | None = None
    OPENAI_API_KEY: str | None = Field(default=None, repr=False)

    @validator("HYBRID_ALPHA")
    def _validate_hybrid_alpha(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("HYBRID_ALPHA must be between 0 and 1.")
        return value

    @validator("STRICT_MIN_SCORE")
    def _validate_min_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("STRICT_MIN_SCORE must be between 0 and 1.")
        return value


def load_settings() -> AppSettings:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return AppSettings()
