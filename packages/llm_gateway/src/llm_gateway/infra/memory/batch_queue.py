"""
infra/memory/batch_queue.py — InMemoryBatchQueue: BatchQueue port in pure
Python, for tests.
"""

from __future__ import annotations

import time
from collections import defaultdict

from llm_gateway.core.models import Task
from llm_gateway.infra.memory.task_store import InMemoryTaskStore


class InMemoryBatchQueue:
    def __init__(self, task_store: InMemoryTaskStore) -> None:
        self._store = task_store
        # batch_key → {task_id: score}
        self._queues: dict[str, dict[str, float]] = defaultdict(dict)
        self._scheduled_until: dict[str, float] = {}

    # ── API side (async) ─────────────────────────────────────────────────

    async def add(self, batch_key: str, task_id: str, score: float) -> int:
        self._queues[batch_key][task_id] = score
        return len(self._queues[batch_key])

    async def try_mark_scheduled(self, batch_key: str, ttl: int) -> bool:
        if time.time() < self._scheduled_until.get(batch_key, 0.0):
            return False
        self._scheduled_until[batch_key] = time.time() + ttl
        return True

    # ── Worker side (sync) ───────────────────────────────────────────────

    def clear_scheduled(self, batch_key: str) -> None:
        self._scheduled_until.pop(batch_key, None)

    def pop(self, batch_key: str, max_agents: int) -> list[Task]:
        queue = self._queues[batch_key]
        selected: list[Task] = []
        agents_count = 0

        while queue:
            task_id = min(queue, key=queue.get)
            queue.pop(task_id)

            task = self._store.get_sync(task_id)
            if not task:
                continue

            agents_in_task = len(task.request.agents)
            if agents_count + agents_in_task > max_agents and agents_count > 0:
                queue[task_id] = task.priority_score
                break

            selected.append(task)
            agents_count += agents_in_task

        return selected

    def requeue(self, batch_key: str, tasks: list[Task]) -> None:
        for t in tasks:
            self._queues[batch_key][t.task_id] = t.priority_score

    def size(self, batch_key: str) -> int:
        return len(self._queues[batch_key])

    def depths(self) -> dict[str, int]:
        return {f"batch:{k}": len(q) for k, q in self._queues.items() if q}
