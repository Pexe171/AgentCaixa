"""Camada de cache para embeddings com suporte a memória e Redis."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field

from rag_app.config import AppSettings


@dataclass
class EmbeddingCache:
    """Cache simples de embeddings para reduzir custo e latência."""

    settings: AppSettings
    _memory_store: dict[str, list[float]] = field(default_factory=dict)
    _redis_client: object | None = None

    def __post_init__(self) -> None:
        if self.settings.EMBEDDING_CACHE_BACKEND == "redis":
            self._redis_client = self._build_redis_client()

    def _build_redis_client(self) -> object | None:
        if importlib.util.find_spec("redis") is None:
            return None

        redis_module = importlib.import_module("redis")
        redis_constructor = getattr(redis_module, "Redis", None)
        if redis_constructor is None:
            return None

        return redis_constructor.from_url(self.settings.EMBEDDING_CACHE_REDIS_URL)

    def _build_cache_key(self, text: str) -> str:
        return f"{self.settings.EMBEDDING_CACHE_KEY_PREFIX}:{text}"

    def get(self, text: str) -> list[float] | None:
        if self.settings.EMBEDDING_CACHE_BACKEND == "none":
            return None

        key = self._build_cache_key(text)

        if self.settings.EMBEDDING_CACHE_BACKEND == "memory":
            return self._memory_store.get(key)

        if self._redis_client is None:
            return self._memory_store.get(key)

        cached_value = self._redis_client.get(key)
        if cached_value is None:
            return None

        if isinstance(cached_value, bytes):
            cached_value = cached_value.decode("utf-8")

        try:
            parsed = json.loads(cached_value)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, list) else None

    def set(self, text: str, vector: list[float]) -> None:
        if self.settings.EMBEDDING_CACHE_BACKEND == "none":
            return

        key = self._build_cache_key(text)

        if self.settings.EMBEDDING_CACHE_BACKEND == "memory":
            self._memory_store[key] = vector
            return

        if self._redis_client is None:
            self._memory_store[key] = vector
            return

        payload = json.dumps(vector)
        self._redis_client.setex(
            key,
            self.settings.EMBEDDING_CACHE_TTL_SECONDS,
            payload,
        )
