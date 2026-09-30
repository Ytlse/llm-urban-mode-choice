"""
ports/batch_queue.py — Contract of the micro-batching queues.

Tasks are grouped by batch_key (cf. core.batching.compute_batch_key)
and sorted by priority_score (min departure_time). The "scheduled" flag
guarantees that exactly one deferred dispatch is armed per batch cycle.
"""

from __future__ import annotations

from typing import Protocol

from llm_gateway.core.models import Task


class BatchQueue(Protocol):
    # ── API side (async) ─────────────────────────────────────────────────
    async def add(self, batch_key: str, task_id: str, score: float) -> int:
        """Adds a task to the queue and returns the current size."""
        ...

    async def try_mark_scheduled(self, batch_key: str, ttl: int) -> bool:
        """Sets the "deferred dispatch scheduled" flag (SETNX). True if set
        by this call — the caller must then schedule the dispatch."""
        ...

    # ── Worker side (sync) ───────────────────────────────────────────────
    def clear_scheduled(self, batch_key: str) -> None:
        """Releases the dispatch flag — called just before the pop."""
        ...

    def pop(self, batch_key: str, max_agents: int) -> list[Task]:
        """Pops the most urgent tasks up to the agent limit."""
        ...

    def requeue(self, batch_key: str, tasks: list[Task]) -> None:
        """Puts the tasks back (retry) with their original score."""
        ...

    def size(self, batch_key: str) -> int: ...

    def depths(self) -> dict[str, int]:
        """Depth of all the non-empty queues (monitoring)."""
        ...
