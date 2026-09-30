"""infra/redis/learned_limits.py — learned limits shared between API and workers (Redis hash)."""
from __future__ import annotations

import redis as sync_redis

KEY = "llm_gateway:learned:max_output_tokens"


class RedisLearnedLimits:
    def __init__(self, sync_client: sync_redis.Redis, key: str = KEY) -> None:
        self._r = sync_client
        self._key = key

    def get_max_output_tokens(self, provider: str) -> int | None:
        value = self._r.hget(self._key, provider)
        return int(value) if value is not None else None

    def set_max_output_tokens(self, provider: str, limit: int) -> None:
        self._r.hset(self._key, provider, int(limit))

    def all_max_output_tokens(self) -> dict[str, int]:
        raw = self._r.hgetall(self._key) or {}
        return {str(k): int(v) for k, v in raw.items()}


__all__ = ["KEY", "RedisLearnedLimits"]
