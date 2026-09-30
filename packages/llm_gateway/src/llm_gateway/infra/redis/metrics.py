"""
infra/redis/metrics.py — RedisMetricsSink: persistent worker counters.

Perf migration (CR §2.2): the counters live in ONE Redis hash (`wmetrics`,
HINCRBY) instead of N string keys. The Prometheus scrape does a single HGETALL
instead of dozens of SCAN+GET.
"""

from __future__ import annotations

import json

import redis as sync_redis

WORKER_METRICS_HASH_KEY = "wmetrics"

# Ring buffer of the latest LLM errors (text) — Prometheus only stores
# numbers, so the raw messages live in a capped Redis list,
# read back by the /errors/recent endpoint (cf. api/routes.py).
RECENT_ERRORS_KEY = "llm:recent_errors"
RECENT_ERRORS_MAX = 50


class RedisMetricsSink:
    def __init__(self, sync_client: sync_redis.Redis) -> None:
        self._r = sync_client

    def incr(self, name: str, amount: int = 1) -> None:
        self._r.hincrby(WORKER_METRICS_HASH_KEY, name, amount)

    def get(self, name: str) -> int:
        val = self._r.hget(WORKER_METRICS_HASH_KEY, name)
        return int(val) if val else 0

    def items(self) -> dict[str, int]:
        raw = self._r.hgetall(WORKER_METRICS_HASH_KEY)
        return {
            (k.decode() if isinstance(k, bytes) else k): int(v)
            for k, v in raw.items()
        }

    # ------------------------------------------------------------------
    # Ring buffer of the latest LLM errors (text)
    # ------------------------------------------------------------------

    def push_error(self, entry: dict) -> None:
        """Pushes an LLM error at the head of the list and trims to RECENT_ERRORS_MAX."""
        self._r.lpush(RECENT_ERRORS_KEY, json.dumps(entry, ensure_ascii=False, default=str))
        self._r.ltrim(RECENT_ERRORS_KEY, 0, RECENT_ERRORS_MAX - 1)

    def recent_errors(self, n: int = RECENT_ERRORS_MAX) -> list[dict]:
        """Returns the last n LLM errors, from the most recent to the oldest."""
        raw = self._r.lrange(RECENT_ERRORS_KEY, 0, max(0, n - 1))
        out: list[dict] = []
        for item in raw:
            s = item.decode() if isinstance(item, bytes) else item
            try:
                out.append(json.loads(s))
            except (ValueError, TypeError):
                continue
        return out
