"""
infra/memory/metrics.py — InMemoryMetricsSink: MetricsSink port in pure Python.
"""

from __future__ import annotations

from collections import Counter, deque

RECENT_ERRORS_MAX = 50


class InMemoryMetricsSink:
    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._recent_errors: deque[dict] = deque(maxlen=RECENT_ERRORS_MAX)

    def incr(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def get(self, name: str) -> int:
        return self._counters.get(name, 0)

    def items(self) -> dict[str, int]:
        return dict(self._counters)

    def push_error(self, entry: dict) -> None:
        self._recent_errors.appendleft(entry)

    def recent_errors(self, n: int = RECENT_ERRORS_MAX) -> list[dict]:
        return list(self._recent_errors)[:n]
