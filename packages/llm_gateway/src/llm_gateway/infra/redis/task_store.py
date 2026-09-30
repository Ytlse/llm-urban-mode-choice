"""
infra/redis/task_store.py — RedisTaskStore: status/result persistence + pub/sub.

Implements the TaskStore (async, API) and SyncTaskStore (sync, worker) ports.
Keys `task:{id}` (TTL 1h); completion notification on the `task_done:{id}` channel
(latency ~10ms vs polling).
"""

from __future__ import annotations

import asyncio
import time

import redis as sync_redis
import redis.asyncio as aioredis
import redis.exceptions as redis_exc

from llm_gateway.core.models import Task, TaskStatus

TASK_KEY_PREFIX = "task:"
TASK_DONE_CHANNEL_PREFIX = "task_done:"
TASK_TTL_SECONDS = 3600


def task_key(task_id: str) -> str:
    return f"{TASK_KEY_PREFIX}{task_id}"


def task_done_channel(task_id: str) -> str:
    return f"{TASK_DONE_CHANNEL_PREFIX}{task_id}"


class RedisTaskStore:
    def __init__(
        self,
        sync_client: sync_redis.Redis | None = None,
        async_client: aioredis.Redis | None = None,
    ) -> None:
        self._sync = sync_client
        self._async = async_client

    # ------------------------------------------------------------------
    # Async (API)
    # ------------------------------------------------------------------

    async def save(self, task: Task) -> None:
        await self._async.set(task_key(task.task_id), task.model_dump_json(), ex=TASK_TTL_SECONDS)

    async def get(self, task_id: str) -> Task | None:
        raw = await self._async.get(task_key(task_id))
        if raw is None:
            return None
        return Task.model_validate_json(raw)

    async def wait_done(self, task_id: str, timeout: float) -> Task | None:
        """
        Blocks until the terminal state (success/failed) via pub/sub, or timeout.

        Reconnects automatically if the pubsub socket is interrupted before
        the end of the time budget (avoids false timeouts on a Redis cut).
        Returns the last known state, None if the task does not exist.
        """
        deadline = time.monotonic() + timeout
        channel = task_done_channel(task_id)

        task = await self.get(task_id)
        if task is None:
            return None
        if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
            return task

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            pubsub = self._async.pubsub()
            await pubsub.subscribe(channel)
            try:
                # Re-check after subscribe: closes the race between the end of the
                # task and the subscription.
                task = await self.get(task_id) or task
                if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                    break

                async def _recv() -> str:
                    async for msg in pubsub.listen():  # noqa: B023 — consumed within the same iteration
                        if msg["type"] == "message":
                            return msg["data"]

                try:
                    raw = await asyncio.wait_for(_recv(), timeout=remaining)
                    task = Task.model_validate_json(raw)
                    break
                except TimeoutError:
                    # Budget exhausted — return the last known state.
                    task = await self.get(task_id) or task
                    break
                except redis_exc.TimeoutError:
                    # Pubsub socket cut before the end; retry if time remains.
                    task = await self.get(task_id) or task
                    if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                        break
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

        return task

    # ------------------------------------------------------------------
    # Sync (Celery worker)
    # ------------------------------------------------------------------

    def save_sync(self, task: Task) -> None:
        self._sync.set(task_key(task.task_id), task.model_dump_json(), ex=TASK_TTL_SECONDS)

    def get_sync(self, task_id: str) -> Task | None:
        raw = self._sync.get(task_key(task_id))
        if raw is None:
            return None
        return Task.model_validate_json(raw)

    def publish_done_sync(self, task: Task) -> None:
        self._sync.publish(task_done_channel(task.task_id), task.model_dump_json())
