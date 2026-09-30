"""
worker/app.py — Celery application factory.

create_celery_app(config) builds the app explicitly from the Settings.
The module-level instance stays in worker/task_worker.py (required by the
Celery CLI: `-A llm_gateway.worker.task_worker.celery_app`).
"""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_shutdown

from llm_gateway.config import Settings
from llm_gateway.telemetry.logger import configure_logging


@worker_shutdown.connect
def _close_shared_http_clients(**_kwargs) -> None:
    """Releases the adapters' shared httpx clients when the worker shuts down."""
    from llm_gateway.adapters.base import close_all_adapters
    close_all_adapters()


def create_celery_app(settings: Settings) -> Celery:
    configure_logging(settings.telemetry)
    app = Celery(
        "llm_worker",
        broker=settings.executor.celery_broker_url,
        backend=settings.executor.celery_result_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        worker_prefetch_multiplier=1,   # One message at a time per Worker (long LLM calls)
        task_acks_late=True,            # Ack AFTER processing (avoids loss on crash)
    )
    return app
