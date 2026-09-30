"""
ports/task_store.py — Contract for task persistence (status, result, pub/sub).
"""

from __future__ import annotations

from typing import Protocol

from llm_gateway.core.models import Task


class TaskStore(Protocol):
    """Async variant — used by the FastAPI API."""

    async def save(self, task: Task) -> None: ...

    async def get(self, task_id: str) -> Task | None: ...

    async def wait_done(self, task_id: str, timeout: float) -> Task | None:
        """Blocks until the task's terminal state (pub/sub) or the timeout.
        Returns the task's last known state, None if it does not exist."""
        ...


class SyncTaskStore(Protocol):
    """Sync variant — used by the Celery worker."""

    def save_sync(self, task: Task) -> None: ...

    def get_sync(self, task_id: str) -> Task | None: ...

    def publish_done_sync(self, task: Task) -> None:
        """Notifies the long-polls that the task has reached a terminal state."""
        ...
