"""Cache de respostas completas com backend em memória ou Redis."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import dataclass, field

from rag_app.config import AppSettings


@dataclass
class ResponseCache:
    settings: AppSettings
    _memory_store: dict[str, str] = field(default_factory=dict)
    _redis_client: object | None = None

    def __post_init__(self) -> None:
        if self.settings.RESPONSE_CACHE_BACKEND == "redis":
            self._redis_client = self._build_redis_client()

    def _build_redis_client(self) -> object | None:
        if importlib.util.find_spec("redis") is None:
            return None

        redis_module = importlib.import_module("redis")
        redis_constructor = getattr(redis_module, "Redis", None)
        if redis_constructor is None:
            return None

        return redis_constructor.from_url(self.settings.RESPONSE_CACHE_REDIS_URL)

    def _build_cache_key(self, cache_input: dict[str, str | bool]) -> str:
        serialized = json.dumps(cache_input, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{self.settings.RESPONSE_CACHE_KEY_PREFIX}:{digest}"

    def get(self, cache_input: dict[str, str | bool]) -> str | None:
        if self.settings.RESPONSE_CACHE_BACKEND == "none":
            return None

        key = self._build_cache_key(cache_input)

        if self.settings.RESPONSE_CACHE_BACKEND == "memory":
            return self._memory_store.get(key)

        if self._redis_client is None:
            return self._memory_store.get(key)

        cached_value = self._redis_client.get(key)
        if cached_value is None:
            return None

        if isinstance(cached_value, bytes):
            cached_value = cached_value.decode("utf-8")

        return str(cached_value)

    def set(self, cache_input: dict[str, str | bool], answer: str) -> None:
        if self.settings.RESPONSE_CACHE_BACKEND == "none":
            return

        key = self._build_cache_key(cache_input)

        if self.settings.RESPONSE_CACHE_BACKEND == "memory":
            self._memory_store[key] = answer
            return

        if self._redis_client is None:
            self._memory_store[key] = answer
            return

        self._redis_client.setex(
            key,
            self.settings.RESPONSE_CACHE_TTL_SECONDS,
            answer,
        )
