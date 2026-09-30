"""
infra/memory/task_store.py — InMemoryTaskStore: TaskStore/SyncTaskStore port
in pure Python, for tests. Zero dependencies.
"""

from __future__ import annotations

import asyncio
import time

from llm_gateway.core.models import Task, TaskStatus


class InMemoryTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    # ── Async (API) ──────────────────────────────────────────────────────

    async def save(self, task: Task) -> None:
        self._tasks[task.task_id] = task.model_copy(deep=True)

    async def get(self, task_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        return task.model_copy(deep=True) if task else None

    async def wait_done(self, task_id: str, timeout: float) -> Task | None:
        deadline = time.monotonic() + timeout
        task = await self.get(task_id)
        if task is None:
            return None
        while time.monotonic() < deadline:
            task = await self.get(task_id) or task
            if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                break
            await asyncio.sleep(0.005)
        return task

    # ── Sync (worker) ────────────────────────────────────────────────────

    def save_sync(self, task: Task) -> None:
        self._tasks[task.task_id] = task.model_copy(deep=True)

    def get_sync(self, task_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        return task.model_copy(deep=True) if task else None

    def publish_done_sync(self, task: Task) -> None:
        # wait_done() watches the store directly — save_sync is enough.
        self._tasks[task.task_id] = task.model_copy(deep=True)
