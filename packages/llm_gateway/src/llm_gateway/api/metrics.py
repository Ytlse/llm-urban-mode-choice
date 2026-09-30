"""
api/metrics.py — API-side Prometheus metrics.

The Celery worker exposes no /metrics: its counters are persisted in
the Redis hash `wmetrics` (cf. infra/redis/metrics.py) and read back here by a
custom collector — a single HGETALL per scrape.
"""

from __future__ import annotations

from prometheus_client import REGISTRY, Counter
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily

from llm_gateway.api.deps import GatewayDeps
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)

# Agents received from GAMA (before mini-batching)
AGENTS_RECEIVED = Counter(
    'gama_agents_received_total',
    'Total agents received from GAMA (before mini-batching)',
    ['category'],
)


def _by_prefix(counters: dict[str, int], prefix: str) -> dict[str, int]:
    """Subset of the counters whose name starts with `prefix` (stripped)."""
    return {k[len(prefix):]: v for k, v in counters.items() if k.startswith(prefix)}


def _bundle_family(spec, counters: dict[str, int]):
    """Render a family declared by a bundle from the `<prefix>:<labels…>` counters."""
    fam = CounterMetricFamily(spec.name, spec.help, labels=list(spec.labels))
    if not spec.labels:
        fam.add_metric([], counters.get(spec.redis_prefix, 0))
        yield fam
        return
    n = len(spec.labels)
    for suffix, val in _by_prefix(counters, spec.redis_prefix + ":").items():
        parts = suffix.split(":", n - 1)
        if len(parts) == n:
            fam.add_metric(parts, val)
    yield fam


class WorkerMetricsCollector:
    """
    Prometheus collector that reads the Worker's persistent counters from Redis.
    Exposes:
      - llm_provider_calls_ok_total   {provider}  : successful LLM calls per provider
      - llm_provider_calls_err_total  {provider}  : failed LLM calls per provider
      - llm_prompts_sent_total        {category}  : LLM prompts sent per category
      - llm_agents_batched_total      {category}  : agents batched per category
      … as well as the domain metrics (modes, distances) and the provider state.
    """

    def __init__(self, deps: GatewayDeps | None = None) -> None:
        self.deps = deps

    def collect(self):
        if self.deps is None:
            return
        try:
            yield from self._collect()
        except Exception as exc:
            logger.error(f"WorkerMetricsCollector.collect() failed: {exc}")

    def _collect(self):
        deps = self.deps
        settings = deps.settings
        counters = deps.metrics.items()   # a single HGETALL per scrape

        # ── LLM calls per provider ───────────────────────────────────────────
        ok_fam  = CounterMetricFamily('llm_provider_calls_ok_total',  'Successful LLM calls per provider', labels=['provider'])
        err_fam = CounterMetricFamily('llm_provider_calls_err_total', 'Failed LLM calls per provider',     labels=['provider'])
        ok_by_provider  = _by_prefix(counters, "llm_calls_ok_total:")
        err_by_provider = _by_prefix(counters, "llm_calls_err_total:")
        # Union: a provider may have errors without successes (and vice versa);
        # providers configured but without calls show up at 0.
        seen = set(ok_by_provider) | set(err_by_provider) | set(settings.providers)
        for provider in seen:
            ok_fam.add_metric([provider], ok_by_provider.get(provider, 0))
            err_fam.add_metric([provider], err_by_provider.get(provider, 0))
        yield ok_fam
        yield err_fam

        # ── Prompts sent vs agents batched (mini-batch ratio) ────────────────
        prompts_fam = CounterMetricFamily('llm_prompts_sent_total',   'LLM prompts sent per category', labels=['category'])
        batched_fam = CounterMetricFamily('llm_agents_batched_total', 'Agents batched per category',   labels=['category'])
        batched_by_cat = _by_prefix(counters, "agents_batched_total:")
        for category, count in _by_prefix(counters, "prompts_sent_total:").items():
            prompts_fam.add_metric([category], count)
            batched_fam.add_metric([category], batched_by_cat.get(category, 0))
        yield prompts_fam
        yield batched_fam

        # ── Tokens consumed per provider ──────────────────────────────────────
        tok_in_fam  = CounterMetricFamily('llm_tokens_in_total',  'Input tokens (prompt) consumed per provider', labels=['provider'])
        tok_out_fam = CounterMetricFamily('llm_tokens_out_total', 'Output tokens (completion) consumed per provider', labels=['provider'])
        tokens_out_by_provider = _by_prefix(counters, "tokens_out_total:")
        for provider, val_in in _by_prefix(counters, "tokens_in_total:").items():
            tok_in_fam.add_metric([provider], val_in)
            tok_out_fam.add_metric([provider], tokens_out_by_provider.get(provider, 0))
        yield tok_in_fam
        yield tok_out_fam

        # ── Errors per provider AND error type ───────────────────────────────
        err_type_fam = CounterMetricFamily(
            'llm_provider_errors_by_type_total',
            'Erreurs LLM par provider et type (rate_limit, tpm_exceeded, quota_exceeded, …)',
            labels=['provider', 'error_type'],
        )
        for suffix, val in _by_prefix(counters, "llm_errors_by_type:").items():
            parts = suffix.split(":", 1)
            if len(parts) == 2:
                err_type_fam.add_metric(parts, val)
        yield err_type_fam

        # ── Capacity reroutes (413 avoided) ──────────────────────────────────
        reroute_fam = CounterMetricFamily(
            'llm_capacity_reroute_total',
            'Batches replayed on another provider: rendered prompt exceeds per-request capacity (HTTP 413 avoided)',
            labels=['provider'],
        )
        for provider, val in _by_prefix(counters, "capacity_reroute_total:").items():
            reroute_fam.add_metric([provider], val)
        yield reroute_fam

        # ── Families declared by the category bundles ────────────────────────
        # The gateway does not know what a transport mode is: each bundle declares the
        # families its `observe` hook feeds (MetricFamilySpec); they are rendered here under
        # the name it chose, with its labels, from the worker's Redis counters.
        registry = getattr(deps, "registry", None)
        for bundle in (registry.bundles() if registry is not None else []):
            for spec in bundle.metric_families:
                yield from _bundle_family(spec, counters)
        # ── Worker [ALARME] alarms (the controller exposes its own directly) ─────
        alarme_fam = CounterMetricFamily(
            'alarme_total',
            '[ALARME] alarms emitted since startup, per source',
            labels=['source'],
        )
        for source, val in _by_prefix(counters, "alarme:").items():
            alarme_fam.add_metric([source], val)
        yield alarme_fam

        # ── Provider state ─────────────────────────────────────────────────────
        # 0=sans_cle_api, 1=desactive_temporairement (disable TTL), 2=cooldown, 3=actif
        state_fam = GaugeMetricFamily(
            'llm_provider_state',
            'Provider state: 0=sans_cle_api, 1=desactive_tmp, 2=cooldown, 3=actif',
            labels=['provider'],
        )
        for provider in settings.declared_providers:
            if provider not in settings.providers:
                state_fam.add_metric([provider], 0)
            elif deps.limiter.is_disabled(provider):
                state_fam.add_metric([provider], 1)
            elif deps.limiter.is_in_cooldown(provider):
                state_fam.add_metric([provider], 2)
            else:
                state_fam.add_metric([provider], 3)
        yield state_fam

        # ── TTL remaining before reactivation (temporary disable OR cooldown) ───
        disable_ttl_fam = GaugeMetricFamily(
            'llm_provider_disable_ttl_seconds',
            'Seconds before automatic re-activation of the provider (0 if active)',
            labels=['provider'],
        )
        for provider in settings.providers:
            ttl = max(
                deps.limiter.disabled_ttl(provider),
                deps.limiter.cooldown_ttl(provider),
            )
            disable_ttl_fam.add_metric([provider], ttl)
        yield disable_ttl_fam

        # ── Depth of the Redis batch queues (PENDING tasks per batch_key) ─────
        queue_fam = GaugeMetricFamily(
            'llm_task_queue_depth',
            'Tasks pending in the Redis batch queues (batch:* keys)',
            labels=['batch_key'],
        )
        for batch_key, depth in deps.queue.depths().items():
            queue_fam.add_metric([batch_key], depth)
        yield queue_fam

        # ── Queue composition per prompt category (ticket 010, A4) ────────────
        # The batch_key is `<category>:<md5>` (cf. core/batching.compute_batch_key):
        # aggregate on the prefix to read itinary_multi_agent vs stm_reflection at a
        # glance — the data that enabled the diagnosis of the 2026-08-03 run.
        by_category_fam = GaugeMetricFamily(
            'llm_task_queue_depth_by_category',
            'Tasks pending in the batch queues, aggregated per prompt category',
            labels=['category'],
        )
        by_category: dict[str, int] = {}
        for batch_key, depth in deps.queue.depths().items():
            category = batch_key.split(':', 1)[0]
            by_category[category] = by_category.get(category, 0) + depth
        for category, depth in by_category.items():
            by_category_fam.add_metric([category], depth)
        yield by_category_fam

        # ── Celery worker utilization rate per provider ──────────────────────
        util_fam = GaugeMetricFamily(
            'celery_worker_utilization_ratio',
            'active_workers / concurrency_limit par provider (1.0 = saturation)',
            labels=['provider'],
        )
        for provider, cfg in settings.providers.items():
            active = deps.limiter.active_workers(provider)
            limit = cfg.concurrency_limit
            util_fam.add_metric([provider], active / limit if limit > 0 else 0.0)
        yield util_fam

        # ── Static provider info (model, adapter) ──────────────────────────────
        info_fam = GaugeMetricFamily(
            'llm_provider_info',
            'Static provider information (model, adapter)',
            labels=['provider', 'model', 'adapter'],
        )
        for provider, cfg in settings.providers.items():
            adapter = cfg.adapter or provider
            info_fam.add_metric([provider, cfg.default_model, adapter], 1)
        yield info_fam

        # ── Configured limits (RPM / RPD / TPD) ───────────────────────────────
        rpm_limit_fam = GaugeMetricFamily('llm_provider_rpm_limit', 'Configured requests/minute limit', labels=['provider'])
        rpd_limit_fam = GaugeMetricFamily('llm_provider_rpd_limit', 'Configured requests/day limit (0 = unlimited)', labels=['provider'])
        tpd_limit_fam = GaugeMetricFamily('llm_provider_tpd_limit', 'Configured tokens/day limit (0 = unlimited)', labels=['provider'])
        for provider, cfg in settings.providers.items():
            rpm_limit_fam.add_metric([provider], cfg.rpm_limit)
            rpd_limit_fam.add_metric([provider], cfg.rpd_limit or 0)
            tpd_limit_fam.add_metric([provider], cfg.tpd_limit or 0)
        yield rpm_limit_fam
        yield rpd_limit_fam
        yield tpd_limit_fam

        # ── Today's consumption (UTC) and usage ratio ─────────────────────────
        req_today_fam = GaugeMetricFamily('llm_provider_requests_today', 'Requests consumed today (UTC window)', labels=['provider'])
        tok_today_fam = GaugeMetricFamily('llm_provider_tokens_today', 'Tokens consumed today (UTC window)', labels=['provider'])
        rpd_ratio_fam = GaugeMetricFamily('llm_provider_daily_usage_ratio', 'requests_today / rpd_limit (1.0 = quota jour atteint)', labels=['provider'])
        exhausted_fam = GaugeMetricFamily('llm_provider_quota_exhausted', 'Provider set aside until midnight UTC, daily quota reached (1=exhausted)', labels=['provider'])
        for provider, cfg in settings.providers.items():
            # Ticket 105 — seen from this gateway only; the same key also serves
            # `scripts/synthesis/*` and `prompt_calibration`, so these gauges UNDERESTIMATE
            # actual consumption. Metric names unchanged: Grafana reads them.
            req_today = deps.limiter.daily_requests_local_seulement(provider)
            tok_today = deps.limiter.daily_tokens_local_seulement(provider)
            req_today_fam.add_metric([provider], req_today)
            tok_today_fam.add_metric([provider], tok_today)
            if cfg.rpd_limit:
                rpd_ratio_fam.add_metric([provider], req_today / cfg.rpd_limit)
            exhausted_fam.add_metric([provider], 1 if deps.limiter.is_quota_exhausted(provider) else 0)
        yield req_today_fam
        yield tok_today_fam
        yield rpd_ratio_fam
        yield exhausted_fam


# The prometheus REGISTRY is process-global: the collector is registered only
# once, and its deps are re-pointed if create_app() is called again (tests).
_collector: WorkerMetricsCollector | None = None


def install_collector(deps: GatewayDeps) -> WorkerMetricsCollector:
    global _collector
    if _collector is None:
        _collector = WorkerMetricsCollector(deps)
        REGISTRY.register(_collector)
    else:
        _collector.deps = deps
    return _collector
