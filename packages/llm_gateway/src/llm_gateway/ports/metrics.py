"""
ports/metrics.py — Contract of the worker metrics counters.

The Celery worker exposes no /metrics: its counters are persisted
(without TTL) and read back by the API's custom Prometheus collector.
Names are composed as "metric:label[:label2]", e.g. "llm_calls_ok_total:groq_llama4".
"""

from __future__ import annotations

from typing import Protocol


class MetricsSink(Protocol):
    def incr(self, name: str, amount: int = 1) -> None: ...

    def get(self, name: str) -> int: ...

    def items(self) -> dict[str, int]:
        """Snapshot of all the counters — a single read per scrape."""
        ...
