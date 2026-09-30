"""
telemetry/alarms.py — Prometheus counter of [ALARME] alarms.

Project convention: confirmed anomalies are logged at ERROR with the
[ALARME] prefix (cf. CLAUDE.md, `make error`). This module makes these alarms
visible in Grafana: each emitting site calls `fire_alarme(source)`
right next to its `logger.error("[ALARME] …")`.

Scope per process:
- GAMA controller (llm-agents + llm_gateway.sdk): direct counter, scraped
  on :8002;
- Celery worker: NOT this module — the worker exposes no /metrics, its
  alarms go through RedisMetricsSink (key `alarme:{source}`) and are read back
  by WorkerMetricsCollector (api/metrics.py) under the same family name.
The counter is created on the first `fire_alarme`, never at import: the API process
can therefore import this module (via the SDK) without colliding with the
`alarme_total` family emitted by its Redis collector.
"""

from prometheus_client import Counter

_ALARME_TOTAL: Counter | None = None


def _counter() -> Counter:
    """The counter is registered in the Prometheus registry only on the first call.

    Importing this module must register nothing: the API process already exposes the
    `alarme_total` family via the Redis collector (api/metrics.py), and two declarations of
    the same family make startup fail (DuplicateTimeseries).
    """
    global _ALARME_TOTAL
    if _ALARME_TOTAL is None:
        try:
            _ALARME_TOTAL = Counter(
                'alarme_total',
                '[ALARME] alarms emitted since startup, per source',
                ['source'],
            )
        except ValueError:
            # The family is already served by the API Redis collector in this process:
            # count outside the registry (visible in [ALARME] logs, not doubled at scrape).
            _ALARME_TOTAL = Counter(
                'alarme_total',
                '[ALARME] alarms emitted since startup, per source',
                ['source'],
                registry=None,
            )
    return _ALARME_TOTAL


def fire_alarme(source: str) -> None:
    """Increment the alarm counter; `source` is a short, stable slug
    (e.g. 'backlog', 'event_loop', 'cache_llm') — low cardinality required."""
    _counter().labels(source=source).inc()
