"""Centralized configuration for rag_app."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


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
    LLM_PROVIDER: Literal["openai", "ollama"] = "ollama"
    OPENAI_MODEL: str | None = None
    OPENAI_API_KEY: str | None = Field(default=None, repr=False)
    OPENAI_TIMEOUT_SECONDS: float = 20.0
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str | None = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT_SECONDS: float = 30.0
    AGENT_NAME: str = "HAG-PTBR"
    VECTOR_PROVIDER: Literal[
        "none",
        "faiss",
        "qdrant",
        "pinecone",
        "pgvector",
        "weaviate",
    ] = "none"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = Field(default=None, repr=False)
    QDRANT_COLLECTION: str = "rag_app_documents"
    INGEST_WATCH_DIR: str = "data/inbox"
    INGEST_SCANNER_STATE_PATH: str = "data/processed/scanner_state.json"
    INGEST_SCANNER_POLL_SECONDS: float = 3.0
    PINECONE_API_KEY: str | None = Field(default=None, repr=False)
    PINECONE_INDEX: str = "rag-app-index"
    PINECONE_NAMESPACE: str = "default"
    EMBEDDING_CACHE_BACKEND: Literal["none", "memory", "redis"] = "memory"
    EMBEDDING_CACHE_REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_CACHE_KEY_PREFIX: str = "rag_app:embedding"
    EMBEDDING_CACHE_TTL_SECONDS: int = 86400
    RESPONSE_CACHE_BACKEND: Literal["none", "memory", "redis"] = "memory"
    RESPONSE_CACHE_REDIS_URL: str = "redis://localhost:6379/1"
    RESPONSE_CACHE_KEY_PREFIX: str = "rag_app:response"
    RESPONSE_CACHE_TTL_SECONDS: int = 43200
    SESSION_STORE_BACKEND: Literal["memory", "sqlite"] = "memory"
    SESSION_DB_PATH: str = "data/memory/session_memory.db"
    SEMANTIC_MEMORY_BACKEND: Literal["none", "sqlite"] = "sqlite"
    SEMANTIC_MEMORY_DB_PATH: str = "data/memory/semantic_memory.db"
    SEMANTIC_MEMORY_RETRIEVE_TOP_K: int = 3
    SEMANTIC_MEMORY_SUMMARY_INTERVAL: int = 4
    ENABLE_LINTER_SCAN: bool = False
    AUDIT_LOG_PATH: str = "data/audit/agent_audit.log"
    COST_PER_1K_TOKENS_USD: float = 0.002
    JUDGE_RESULTS_PATH: str = "data/evals/judge_results.json"
    OBSERVABILITY_ENABLED: bool = False
    OBSERVABILITY_PROVIDER: Literal["none", "langfuse", "langsmith"] = "none"
    LANGSMITH_API_KEY: str | None = Field(default=None, repr=False)
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "agentcaixa"
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: str | None = Field(default=None, repr=False)
    LANGFUSE_SECRET_KEY: str | None = Field(default=None, repr=False)
    WHATSAPP_CHANNEL_ENABLED: bool = False
    WHATSAPP_PROVIDER: Literal["evolution"] = "evolution"
    WHATSAPP_EVOLUTION_BASE_URL: str = "http://localhost:8080"
    WHATSAPP_EVOLUTION_INSTANCE: str = ""
    WHATSAPP_EVOLUTION_API_KEY: str | None = Field(default=None, repr=False)
    WHATSAPP_WEBHOOK_SECRET: str | None = Field(default=None, repr=False)

    @field_validator("HYBRID_ALPHA")
    def _validate_hybrid_alpha(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("HYBRID_ALPHA must be between 0 and 1.")
        return value

    @field_validator("STRICT_MIN_SCORE")
    def _validate_min_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("STRICT_MIN_SCORE must be between 0 and 1.")
        return value

    @field_validator("COST_PER_1K_TOKENS_USD")
    def _validate_cost(cls, value: float) -> float:
        if value < 0:
            raise ValueError("COST_PER_1K_TOKENS_USD must be greater or equal to 0.")
        return value

    @field_validator("OPENAI_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS")
    def _validate_provider_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Provider timeout must be greater than 0.")
        return value

    @field_validator("INGEST_SCANNER_POLL_SECONDS")
    def _validate_ingest_poll(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("INGEST_SCANNER_POLL_SECONDS must be greater than 0.")
        return value

    @field_validator(
        "SEMANTIC_MEMORY_RETRIEVE_TOP_K",
        "SEMANTIC_MEMORY_SUMMARY_INTERVAL",
    )
    def _validate_semantic_memory_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Semantic memory values must be greater than 0.")
        return value

    @field_validator("EMBEDDING_CACHE_TTL_SECONDS", "RESPONSE_CACHE_TTL_SECONDS")
    def _validate_cache_ttl(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Cache TTL values must be greater than 0.")
        return value


def load_settings() -> AppSettings:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return AppSettings()
