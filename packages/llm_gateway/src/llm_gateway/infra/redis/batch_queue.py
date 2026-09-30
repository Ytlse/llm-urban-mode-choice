"""
infra/redis/batch_queue.py — RedisBatchQueue: micro-batching queues.

Sorted Set `batch:{batch_key}` sorted by priority_score (min departure_time);
SETNX flag `batch_sched:{batch_key}` to deduplicate the deferred dispatch.
"""

from __future__ import annotations

import redis as sync_redis
import redis.asyncio as aioredis

from llm_gateway.core.models import Task
from llm_gateway.infra.redis.task_store import RedisTaskStore

BATCH_QUEUE_PREFIX = "batch:"
BATCH_SCHED_PREFIX = "batch_sched:"
BATCH_QUEUE_TTL_SECONDS = 3600


def batch_queue_key(batch_key: str) -> str:
    return f"{BATCH_QUEUE_PREFIX}{batch_key}"


def _batch_sched_key(batch_key: str) -> str:
    return f"{BATCH_SCHED_PREFIX}{batch_key}"


class RedisBatchQueue:
    def __init__(
        self,
        task_store: RedisTaskStore,
        sync_client: sync_redis.Redis | None = None,
        async_client: aioredis.Redis | None = None,
    ) -> None:
        self._store = task_store
        self._sync = sync_client
        self._async = async_client

    # ------------------------------------------------------------------
    # API side (async)
    # ------------------------------------------------------------------

    async def add(self, batch_key: str, task_id: str, score: float) -> int:
        key = batch_queue_key(batch_key)
        await self._async.zadd(key, {task_id: score})
        await self._async.expire(key, BATCH_QUEUE_TTL_SECONDS)
        return await self._async.zcard(key)

    async def try_mark_scheduled(self, batch_key: str, ttl: int) -> bool:
        """Sets the "deferred dispatch scheduled" flag (SETNX).

        True if the flag was just set (the caller must schedule the
        dispatch), False if it already existed. Removes the race where two
        simultaneous requests each observe queue_size > 1 and neither schedules
        the worker. The TTL is a safety net if the Celery message is lost.
        """
        return bool(await self._async.set(_batch_sched_key(batch_key), "1", nx=True, ex=ttl))

    # ------------------------------------------------------------------
    # Worker side (sync)
    # ------------------------------------------------------------------

    def clear_scheduled(self, batch_key: str) -> None:
        """Releases the dispatch flag — called just before popping, so that
        tasks arriving afterwards can schedule a new dispatch."""
        self._sync.delete(_batch_sched_key(batch_key))

    def pop(self, batch_key: str, max_agents: int) -> list[Task]:
        """Pops the most urgent tasks (min score) up to the agent limit."""
        key = batch_queue_key(batch_key)
        selected_tasks: list[Task] = []
        current_agents_count = 0

        while True:
            result = self._sync.zpopmin(key, count=1)
            if not result:
                break

            task_id = result[0][0]
            if isinstance(task_id, bytes):
                task_id = task_id.decode()

            task = self._store.get_sync(task_id)
            if not task:
                continue

            agents_in_task = len(task.request.agents)
            if current_agents_count + agents_in_task > max_agents and current_agents_count > 0:
                # Over the limit: put the task back with its original score
                self._sync.zadd(key, {task_id: task.priority_score})
                break

            selected_tasks.append(task)
            current_agents_count += agents_in_task

        return selected_tasks

    def requeue(self, batch_key: str, tasks: list[Task]) -> None:
        """On error (retry), puts the tasks back with their original score."""
        if not tasks:
            return
        mapping = {t.task_id: t.priority_score for t in tasks}
        self._sync.zadd(batch_queue_key(batch_key), mapping)

    def size(self, batch_key: str) -> int:
        return self._sync.zcard(batch_queue_key(batch_key))

    def depths(self) -> dict[str, int]:
        """Depth of the non-empty queues — for the llm_task_queue_depth gauge."""
        result: dict[str, int] = {}
        for key in self._sync.scan_iter(f"{BATCH_QUEUE_PREFIX}*"):
            name = key.decode() if isinstance(key, bytes) else key
            if name.startswith(BATCH_SCHED_PREFIX):
                continue
            depth = self._sync.zcard(name)
            if depth > 0:
                result[name] = depth
        return result
