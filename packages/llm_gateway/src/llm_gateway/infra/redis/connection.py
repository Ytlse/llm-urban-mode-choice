"""
infra/redis/connection.py — Redis client factories.

Clients are created explicitly by the composition (create_app, worker
runtime) and injected into the infra implementations — never built as a
side effect of an import.
"""

from __future__ import annotations

import redis as sync_redis
import redis.asyncio as aioredis


def create_sync_redis(redis_url: str) -> sync_redis.Redis:
    return sync_redis.from_url(redis_url, encoding="utf-8", decode_responses=True)


def create_async_redis(redis_url: str) -> aioredis.Redis:
    return aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
