"""
worker/task_worker.py — Celery worker with retry and exponential backoff.

The Worker is the core of the processing:
  1. Pops the tasks from the batch queue (BatchQueue)
  2. Builds the prompt via the PromptManager
  3. Selects the provider via the LoadBalancer
  4. Runs the LLM call via the appropriate Adapter
  5. Validates the structured output
  6. Persists the result (TaskStore) and notifies via Pub/Sub

The dependencies (Redis, balancer, metrics) are composed by
worker/runtime.py on the first processing — not at module import.

Retry with exponential backoff on 5xx errors:
  attempt 1 → wait 1s
  attempt 2 → wait 2s
  attempt 3 → wait 4s
  attempt 4 → wait 8s  (max_retries=settings.resilience.max_retries in config)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from llm_gateway.adapters.base import (
    ProviderClientError,
    ProviderParseError,
    ProviderServerError,
    get_adapter,
)
from llm_gateway.balancer.router import RestrictionInstances
from llm_gateway.config import get_settings, learn_provider_max_output_tokens
from llm_gateway.core.inference import resolve_inference
from llm_gateway.core.models import _FALLBACK_PRIORITY_SCORE, InternalRequest, Task, TaskStatus
from llm_gateway.core.quota import DEFAUT_FUSEAU_QUOTA, is_daily_quota_error, next_quota_reset
from llm_gateway.core.rejeu_ab import cle_rejeu, enregistrement, espace_valide, magasin_rejeu
from llm_gateway.telemetry.logger import get_logger, log_llm_call, log_llm_error, log_llm_exchange
from llm_gateway.worker.app import create_celery_app
from llm_gateway.worker.runtime import WorkerRuntime, get_worker_runtime

logger = get_logger(__name__)


class ProviderCapacityError(Exception):
    """Rendered prompt too large for the provider's per-request capacity.

    Raised BEFORE the HTTP call: the 413 "request too large" is avoided, the RPM/TPM
    slot is given back and the batch is replayed on another model (rotation).
    """

    def __init__(self, provider: str, prompt_tokens_est: int, max_tokens_per_request: int):
        self.provider = provider
        self.prompt_tokens_est = prompt_tokens_est
        self.max_tokens_per_request = max_tokens_per_request
        super().__init__(
            f"Prompt ≈{prompt_tokens_est} estimated tokens > per-request capacity "
            f"({max_tokens_per_request} tokens, minimum output included) of {provider}"
        )

# ---------------------------------------------------------------------------
# Celery application — module-level instance required by the CLI:
#   celery -A llm_gateway.worker.task_worker.celery_app worker …
# Building the app does not touch Redis (the broker connection is lazy).
# ---------------------------------------------------------------------------

celery_app = create_celery_app(get_settings())


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="process_batch_task",
    bind=True,
    max_retries=get_settings().resilience.max_retries,
)
def process_batch_task(self, batch_key: str, force_provider: str | None = None,
                       min_tpm_required: int | None = None,
                       min_output_required: int | None = None,
                       instances_admises: list[str] | None = None,
                       debut_attente: float | None = None) -> None:
    """
    Celery entry point for batch processing (micro-batching).
    `bind=True` to access `self.retry()`.

    `debut_attente` (wall clock, seconds) dates the batch's first attempt: a replay may
    change process, and the wait is measured against the client's, not per attempt.
    """
    rt = get_worker_runtime()
    settings = rt.settings
    debut_attente = debut_attente or time.time()
    # Every replay of THIS batch restarts with its constraints AND its first-attempt date.
    # Without `debut_attente`, each replay reset the counter to zero and the worker's wait
    # exceeded the client's without anything noticing.
    rejeu = {
        "force_provider": force_provider,
        "min_tpm_required": min_tpm_required,
        "min_output_required": min_output_required,
        "instances_admises": instances_admises,
        "debut_attente": debut_attente,
    }

    # Wait for a provider slot in a local loop rather than via self.retry().
    # self.retry() creates a new Celery message + Redis round-trips on every attempt;
    # a plain time.sleep() in the same worker is much cheaper.
    res = settings.resilience
    deadline = time.monotonic() + res.provider_wait_seconds
    provider_name: str | None = None
    _p5_provider_wait_start = time.monotonic()
    while True:
        try:
            provider_name = rt.balancer.select_provider(
                force=force_provider,
                min_tpm=min_tpm_required,
                min_output=min_output_required,
                admises=instances_admises,
            )
            break
        except RestrictionInstances as e:
            # Ticket 085, lot A. `RestrictionInstances` is a ValueError: it went through the
            # task BEFORE the `rt.queue.pop` below, leaving the batch QUEUED — every
            # following dispatch picked it up and failed identically. It was not a failure
            # costing 120 s once, it was a failure that re-armed itself, up to the
            # circuit breaker (incident of 2026-09-16, three hours lost).
            #
            # A deterministic refusal is reported ONCE, RIGHT AWAY, and never retried:
            # the contradiction is in the configuration, a replay can only reproduce it.
            #
            # No RPM/TPM slot to give back: selection failed BEFORE any reservation.
            rt.queue.clear_scheduled(batch_key)
            tasks = _vider_file(rt, batch_key)
            motif = (
                f"Restriction d'instances non satisfiable : {e} "
                f"[instances admises déclarées : {sorted(instances_admises) if instances_admises else 'aucune'} ; "
                f"fournisseur épinglé : {force_provider or 'aucun'}] — "
                f"refus déterministe, aucun rejeu n'est planifié : corrigez la configuration."
            )
            rt.metrics.incr("alarme:restriction_instances")
            logger.error(
                f"[ALARME] Unsatisfiable instance restriction — {len(tasks)} task(s) "
                f"refused WITHOUT replay | batch_key={batch_key} "
                f"instances_admises={instances_admises or 'aucune'} "
                f"force_provider={force_provider or 'aucun'} detail={e}"
            )
            for t in tasks:
                # `error_kind` carries the nature of the failure to the client: it is neither
                # a busy gateway nor an exhausted quota, and filing it in either bucket sends
                # people looking for a quota where there is only one YAML line to fix.
                _fail_task(rt, t, motif, error_kind="restriction_instances")
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                # What remains of the CLIENT's wait once the next attempt is over. Negative: the
                # client will have given up before we retry, and the task would no longer serve
                # anyone (cf. `resilience.client_wait_seconds`).
                restant = _attente_client_restante(settings, force_provider, debut_attente) - (
                    res.provider_wait_seconds + res.saturation_retry_seconds
                )
                if self.request.retries < res.saturation_retries and restant > 0:
                    logger.warning(
                        f"Providers saturated for {res.provider_wait_seconds:.0f}s, "
                        f"retry in {res.saturation_retry_seconds:.0f}s | batch_key={batch_key} "
                        f"attempt={self.request.retries + 1}/{res.saturation_retries}"
                    )
                    raise self.retry(countdown=res.saturation_retry_seconds, args=[batch_key], kwargs=rejeu)
                _statuses = rt.balancer.get_status()
                # Only the instances THIS batch can receive count: the pinned one, otherwise the
                # admitted ones, otherwise all. Counting the others made two keys cooling down
                # look like a waiting queue, as long as an excluded model stayed free
                # (2026-09-23: 54 × HTTP 503, the batch waited, the client timed out at 120 s).
                eligibles = _eligibles(_statuses, force_provider, instances_admises)
                verdict = _verdict_saturation(
                    _statuses,
                    eligibles,
                    {n: rt.limiter.cooldown_ttl(n) for n in eligibles if _statuses[n].get("cooldown")},
                    restant,
                    attendre_si_occupe=not res.abandon_when_busy,
                    peut_rejouer=self.request.retries < self.max_retries,
                )
                if verdict.attendre:
                    # Full RPM/TPM window, smoothing or concurrency: this is not an outage, it is
                    # a waiting queue. Giving up here lost decisions at every peak
                    # (run of 2026-09-07: half the requests on an instance forced to
                    # 15 RPM). An instance cooling down that reopens before the client gives up
                    # is waited for likewise. The wait stays bounded by the client's.
                    if verdict.motif == "occupe" and self.request.retries == res.saturation_retries:
                        rt.metrics.incr("alarme:providers_occupes")
                        logger.error(
                            f"[ALARME] Providers occupied (window full, not an outage): the batch waits "
                            f"for the window instead of being dropped | batch_key={batch_key} "
                            f"attempt={self.request.retries + 1}/{self.max_retries} "
                            f"providers={eligibles}"
                        )
                    elif verdict.motif == "refroidissement":
                        logger.warning(
                            f"Eligible instances cooling down, the batch waits for them to reopen "
                            f"(≤ {verdict.reprise_dans_s:.0f}s, remaining client wait "
                            f"{restant:.0f}s) | batch_key={batch_key} providers={eligibles} "
                            f"attempt={self.request.retries + 1}/{self.max_retries}"
                        )
                    raise self.retry(countdown=res.saturation_retry_seconds, args=[batch_key], kwargs=rejeu)
                tasks = rt.queue.pop(batch_key, 100)
                _cooldowns = [n for n in eligibles if _statuses[n].get("cooldown")]
                # The worker exposes no /metrics: the alarm goes through Redis
                # and comes out as alarme_total{source} via WorkerMetricsCollector.
                rt.metrics.incr("alarme:providers_satures")
                logger.error(
                    f"[ALARME] Tous les providers LLM saturés ou indisponibles — "
                    f"{len(tasks)} tâche(s) abandonnée(s) | batch_key={batch_key} "
                    f"eligibles={eligibles or 'aucun'} "
                    f"providers_en_cooldown={_cooldowns or 'aucun'} "
                    f"genre={verdict.genre or 'aucun'} "
                    f"attente_client_restante={restant:.0f}s "
                    f"(quotas RPM/TPM épuisés ? voir /health)"
                )
                reprise = _reprise(settings, verdict, eligibles)
                for t in tasks:
                    _fail_task(
                        rt, t,
                        f"Providers saturés ou indisponibles après {res.provider_wait_seconds:.0f}s "
                        f"({self.request.retries} retries épuisés)"
                        + (f" — {verdict.genre} sur {eligibles}" if verdict.genre else ""),
                        error_kind=verdict.genre,
                        resume_at=reprise,
                    )
                return
            time.sleep(res.saturation_poll_seconds)
    _p5_provider_wait_ms = (time.monotonic() - _p5_provider_wait_start) * 1000

    # ------------------------------------------------------------------
    # Start of provider occupancy tracking
    # ------------------------------------------------------------------
    rt.limiter.incr_active(provider_name)
    try:
        batch_limit = settings.providers[provider_name].batch_max_agents

        # Releases the deferred-dispatch flag BEFORE the pop: any task added
        # afterwards can reschedule its own dispatch (cf. BatchQueue.try_mark_scheduled).
        rt.queue.clear_scheduled(batch_key)
        tasks = rt.queue.pop(batch_key, batch_limit)
        if not tasks:
            # No request will go out: give back the RPM + TPM slot reserved
            # by select_provider (otherwise it stays consumed for 60s for nothing).
            rt.limiter.release_slot(provider_name)
            return

        # Mark as running
        for task in tasks:
            task.status = TaskStatus.RUNNING
            task.updated_at = datetime.now(UTC)
            rt.store.save_sync(task)

        batch_id = f"batch_{tasks[0].task_id[:8]}_{len(tasks)}"

        try:
            _execute_batch(rt, tasks, batch_id, provider_name, _p5_provider_wait_ms, self.request.retries)

        except ProviderCapacityError as e:
            # Detected before the HTTP call (and before the TPM adjustment): the RPM/TPM
            # slot reserved by select_provider was of no use, it is given back.
            rt.limiter.release_slot(provider_name)
            rt.metrics.incr(f"capacity_reroute_total:{provider_name}")
            _switch_provider_or_fail(
                self, rt, tasks, batch_key, e, provider_name,
                reason="Prompt trop volumineux pour la capacité par requête",
                min_tpm_required=min_tpm_required,
                min_output_required=min_output_required,
                force_provider=force_provider,
                instances_admises=instances_admises,
            )

        except ProviderServerError as e:
            # 5xx error → temporary exclusion, exponential backoff and retry
            rt.limiter.cooldown(e.provider, seconds=_COOLDOWN_5XX_S)
            delay = min(settings.resilience.backoff_base_seconds * (2 ** self.request.retries), 30.0)
            # The batch's oldest task dates the client's wait better than the first attempt:
            # it may have waited in the queue before it.
            debut = min([debut_attente] + [t.created_at.timestamp() for t in tasks if getattr(t, "created_at", None)])
            restant = _attente_client_restante(settings, force_provider, debut) - delay
            logger.warning(
                f"Provider server error, retry scheduled | task_id={batch_id} "
                f"provider={e.provider} http_status={e.status_code} "
                f"retry_in={delay:.1f}s attempt={self.request.retries + 1} "
                f"attente_client_restante={restant:.0f}s"
            )
            if self.request.retries < self.max_retries and restant > 0:
                rt.queue.requeue(batch_key, tasks)
                raise self.retry(exc=e, countdown=delay, args=[batch_key], kwargs={**rejeu, "debut_attente": debut})
            # The client gives up before the next attempt: we tell it so, with a kind it
            # knows how to handle (clean run stop) instead of a silent "Timeout expiré" that it
            # files as a fallback — a decision the model did not make.
            reprise = datetime.now(UTC) + timedelta(seconds=_COOLDOWN_5XX_S)
            rt.metrics.incr("alarme:surcharge_fournisseur")
            logger.error(
                f"[ALARME] Provider overload (HTTP {e.status_code}) on {e.provider} — "
                f"{len(tasks)} task(s) returned to the client before it gives up | "
                f"batch_key={batch_key} attempt={self.request.retries + 1}/{self.max_retries} "
                f"attente_client_restante={restant:.0f}s reprise={reprise.isoformat(timespec='seconds')}"
            )
            for t in tasks:
                _fail_task(
                    rt, t,
                    f"Surcharge fournisseur : HTTP {e.status_code} sur {e.provider}, "
                    f"{self.request.retries + 1} essai(s), attente du client épuisée",
                    error_kind="surcharge_fournisseur",
                    resume_at=reprise,
                )

        except ProviderClientError as e:
            if e.status_code == 429 and is_daily_quota_error(str(e)):
                # DAILY quota exhausted. For Google Gemini, its `retryDelay` is useless (0.7s to 57s)
                # and reopening is at midnight (next_quota_reset). For Groq however, the window
                # is sliding (TPD) and its `ratelimit_reset` gives the exact unblocking time.
                # TEMPORARY, TO BE REMOVED (debug ticket 077: support for the Groq sliding window)
                raw_reset = getattr(e, "ratelimit_reset", None)
                if e.provider.startswith("groq") and raw_reset:
                    cooldown = _parse_ratelimit_reset_seconds(raw_reset, default=3600)
                    reprise = datetime.now(UTC) + timedelta(seconds=cooldown)
                    logger.warning(
                        f"[quota] Groq sliding TPD window reached on '{e.provider}' — "
                        f"resume computed at {reprise.isoformat(timespec='seconds')} ({cooldown}s) "
                        f"from ratelimit_reset={raw_reset!r}"
                    )
                else:
                    cfg_p = settings.providers.get(e.provider)
                    tz = getattr(cfg_p, "quota_reset_tz", DEFAUT_FUSEAU_QUOTA) or DEFAUT_FUSEAU_QUOTA
                    reprise = next_quota_reset(tz)
                ttl = rt.limiter.mark_quota_exhausted_until(e.provider, kind="rpd", until=reprise)
                logger.error(
                    f"[ALARME] Daily quota exhausted on '{e.provider}' — instance set aside "
                    f"until {reprise.isoformat(timespec='seconds')} ({ttl}s) | task_id={batch_id}"
                )
                if force_provider is None and self.request.retries < self.max_retries:
                    # Unpinned instance: the batch goes again, the balancer will skip this one and
                    # take the next one in the cascade (other key, other quota bucket).
                    rt.queue.requeue(batch_key, tasks)
                    raise self.retry(exc=e, countdown=1, args=[batch_key], kwargs=rejeu)
                # Pinned instance (an experiment pins its decision-maker): no admissible
                # alternative. We TELL the caller — `quota_journalier` + the resume time —
                # so that it waits for the window instead of retrying every 30 s.
                for t in tasks:
                    _fail_task(
                        rt, t,
                        f"Quota journalier épuisé sur {e.provider}, fenêtre rouverte à "
                        f"{reprise.isoformat(timespec='seconds')}",
                        error_kind="quota_journalier",
                        resume_at=reprise,
                    )
            elif e.status_code == 429:
                # Per-minute/second rate limit → cooldown set on x-ratelimit-reset if available
                cooldown_secs = _parse_ratelimit_reset_seconds(getattr(e, "ratelimit_reset", None))
                rt.limiter.cooldown(e.provider, seconds=cooldown_secs)
                delay = min(settings.resilience.backoff_base_seconds * (2 ** self.request.retries), 30.0)
                logger.warning(
                    f"Rate limit (429) reached, provider in cooldown, retry scheduled | "
                    f"task_id={batch_id} provider={e.provider} "
                    f"cooldown={cooldown_secs}s ratelimit_reset={getattr(e, 'ratelimit_reset', None)} "
                    f"retry_in={delay:.1f}s attempt={self.request.retries + 1}"
                )
                debut = min([debut_attente] + [t.created_at.timestamp() for t in tasks if getattr(t, "created_at", None)])
                restant = _attente_client_restante(settings, force_provider, debut) - delay
                if self.request.retries < self.max_retries and restant > 0:
                    rt.queue.requeue(batch_key, tasks)
                    raise self.retry(exc=e, countdown=delay, args=[batch_key], kwargs={**rejeu, "debut_attente": debut})
                # Same bound as for a 5xx: a per-minute limit is a transient overload,
                # and the client must learn it BEFORE giving up, not suffer it as a fallback.
                reprise = datetime.now(UTC) + timedelta(seconds=cooldown_secs)
                rt.metrics.incr("alarme:surcharge_fournisseur")
                logger.error(
                    f"[ALARME] Provider overload (HTTP 429 per minute) on {e.provider} — "
                    f"{len(tasks)} task(s) returned to the client before it gives up | "
                    f"batch_key={batch_key} attempt={self.request.retries + 1}/{self.max_retries} "
                    f"attente_client_restante={restant:.0f}s reprise={reprise.isoformat(timespec='seconds')}"
                )
                for t in tasks:
                    _fail_task(
                        rt, t,
                        f"Surcharge fournisseur : Rate Limits (429) sur {e.provider}, "
                        f"{self.request.retries + 1} essai(s), attente du client épuisée",
                        error_kind="surcharge_fournisseur",
                        resume_at=reprise,
                    )
            elif e.status_code == 402:
                # Credits exhausted ("payment required"): neither saturation nor an invalid
                # request. The instance will not become servable again by itself within the
                # minute — it is disabled instead of a short cooldown, with an alarm on the
                # rising edge: this is an event that costs a whole campaign.
                _credits_epuises(
                    self, rt, tasks, batch_key, e,
                    force_provider=force_provider,
                    instances_admises=instances_admises,
                    min_tpm_required=min_tpm_required,
                    min_output_required=min_output_required,
                )
            elif (limit := _parse_max_tokens_limit(str(e))) and learn_provider_max_output_tokens(rt.settings, rt.learned, e.provider, limit):
                # 400 "max_tokens must be ≤ N" → learned completion limit
                # (in-memory config + providers.yaml). The batch is replayed: the
                # next attempt caps max_tokens at N (or goes to another
                # provider via the rotation). If the limit was already known
                # (learn → False), we fall back to the final failure below
                # so as not to loop on the same 400.
                logger.warning(
                    f"max_tokens above the model limit, batch replayed with the "
                    f"learned limit | task_id={batch_id} provider={e.provider} "
                    f"max_output_tokens={limit} attempt={self.request.retries + 1}"
                )
                if self.request.retries < self.max_retries:
                    rt.queue.requeue(batch_key, tasks)
                    raise self.retry(exc=e, countdown=1, args=[batch_key], kwargs=rejeu)
                else:
                    for t in tasks:
                        _fail_task(rt, t, f"Max retries dépassé({self.max_retries}) sur {e.provider} : {str(e)}")
            else:
                # Other 4xx error → often tied to the provider/model: we try another
                # model before giving up (final failure if all are exhausted).
                _switch_provider_or_fail(
                    self, rt, tasks, batch_key, e, e.provider,
                    reason="Erreur 4xx non récupérable",
                    min_tpm_required=min_tpm_required,
                    min_output_required=min_output_required,
                    force_provider=force_provider,
                    instances_admises=instances_admises,
                )

        except RuntimeError as e:
            logger.exception(f"Unexpected RuntimeError in the worker | task_id={batch_id}")
            for t in tasks:
                _fail_task(rt, t, f"RuntimeError interne : {str(e)}")

        except ProviderParseError as e:
            # Unreadable/off-schema response: often specific to the model. We try another
            # model before giving up; as a last resort the raw output is passed up.
            logger.warning(
                f"Parsing error on the provider response, switch attempted | "
                f"task_id={batch_id} detail={str(e)} raw_preview={str(e.raw)[:300]!r}"
            )
            _switch_provider_or_fail(
                self, rt, tasks, batch_key, e, e.provider,
                reason="Erreur de parsing de la réponse LLM",
                min_tpm_required=min_tpm_required,
                min_output_required=min_output_required,
                final_error_msg=f"{str(e)}\nRaw LLM response:\n{e.raw}",
                force_provider=force_provider,
                instances_admises=instances_admises,
            )

        except Exception as e:
            # Generic case — logged with the full stacktrace via logger.exception
            logger.exception(f"Unexpected exception in the worker | task_id={batch_id} error={e}")
            for t in tasks:
                _fail_task(rt, t, f"Exception interne : {str(e)}")

        else:
            # Full success: start a worker again if items remain in this queue
            if rt.queue.size(batch_key) > 0:
                process_batch_task.delay(
                    batch_key, force_provider, min_tpm_required, min_output_required,
                    instances_admises,
                )
    finally:
        # Whatever happens (success, failure, retry, timeout...), release the slot!
        rt.limiter.decr_active(provider_name)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def _execute_batch(rt: WorkerRuntime, tasks: list[Task], batch_id: str, provider_name: str, p5_provider_wait_ms: float = 0.0, retries: int = 0) -> None:
    settings = rt.settings
    base_req = tasks[0].request
    merged_agents = []
    for t in tasks:
        merged_agents.extend(t.request.agents)

    # 2. Prompt construction — by the category (registered bundle): it validates the
    # items and owns template and schema. The worker knows no domain.
    handle = rt.registry.get(base_req.category)
    _t_prompt = time.monotonic()
    items = handle.validate_items(merged_agents)
    messages = handle.render(items, base_req.parameters)
    schema = handle.output_schema
    p5_prompt_ms = (time.monotonic() - _t_prompt) * 1000

    # 3. Assembly of the internal request
    # max_tokens sent by the client is a PER-task budget (1 agent): the batch
    # merges N agents and the output grows linearly (stm_reflection ≈ 500-1800
    # tokens/agent). Without scaling, a batch of 10 saturates 4096 and the JSON is
    # truncated midway → JSONDecodeError. Bounded by max_output_tokens
    # (the models' completion limit) then by the provider's capacity.
    provider_cfg = settings.providers.get(provider_name)
    # Cascade request > provider > defaults (core.inference); max_tokens is a PER-task budget.
    inference = resolve_inference(
        base_req.parameters, provider_cfg.inference if provider_cfg else None, settings.inference
    )
    per_task_tokens = inference.max_tokens
    max_tokens = min(per_task_tokens * max(1, len(merged_agents)), settings.batching.max_output_tokens)
    if provider_cfg and provider_cfg.max_output_tokens:
        max_tokens = min(max_tokens, provider_cfg.max_output_tokens)
    # 413 safeguard: the per-request capacity is checked against the ACTUAL size of the
    # rendered prompt, not against assumed_prompt_tokens — stm_reflection prompts are
    # ~2× the static estimate and went doomed to the small-quota providers
    # (38 HTTP 413 on the 2026-07-11 run). If even the minimal output no longer
    # fits in the budget, we replay elsewhere BEFORE burning the call.
    prompt_chars = sum(len(m.content or "") for m in messages)
    prompt_tokens_est = int(prompt_chars / settings.batching.token_chars_ratio)
    max_tokens = _fit_request_budget(
        provider_cfg, prompt_tokens_est, max_tokens, settings.batching.min_output_tokens, provider_name
    )
    internal_req = InternalRequest(
        provider=provider_name,
        messages=messages,
        response_schema=schema,
        temperature=inference.temperature,
        top_p=inference.top_p,
        max_tokens=max_tokens,
        # THINKING budget, distinct from the output budget: it is neither multiplied by the
        # batch size nor bounded by the provider's caps, it applies per call. The adapter
        # decides whether it can pass it on (`applique_reflexion`) and reserves the
        # output accordingly.
        thinking_budget=inference.thinking_budget,
        thinking_level=inference.thinking_level,
    )

    # ── Adjusting the TPM reservation to the actual batch size ──────────────
    # select_provider() reserved the static estimate tpm_estimate_per_request
    # (full-batch flat rate) BEFORE knowing the batch content. Now that the
    # prompt is rendered, we correct the sliding window: a small batch gives back
    # headroom to the other workers, a batch of reflections (~4,500 tokens_in/agent)
    # reserves its true cost instead of the 2,200 flat rate — this undercounting is
    # what pushed small-quota providers over their real TPM (429).
    reserved_tokens: int | None = None
    if provider_cfg and provider_cfg.tpm_limit:
        reserved_tokens = (
            prompt_tokens_est
            + len(merged_agents) * settings.batching.assumed_output_tokens
        )
        rt.limiter.adjust_tokens(
            provider_name,
            reserved_tokens - (provider_cfg.tpm_estimate_per_request or 0),
        )

    # 4. LLM call via the appropriate adapter
    adapter = get_adapter(provider_name)
    start_ts = time.monotonic()

    try:
        llm_output, tokens_in, tokens_out = adapter.call(internal_req)
    except Exception as exc:
        # Gives back the amount actually reserved (adjusted above), not the flat rate.
        rt.limiter.release_slot(provider_name, reserved_tokens)
        logger.error(f"Error with provider | task_id={batch_id} provider={provider_name}")
        rt.metrics.incr(f"llm_calls_err_total:{provider_name}")
        error_type = getattr(exc, "error_type", "unknown")
        http_status = getattr(exc, "status_code", None)
        log_llm_error(
            task_id=batch_id,
            provider=provider_name,
            error_type=error_type,
            error_message=str(exc),
            http_status=http_status,
            ratelimit_reset=getattr(exc, "ratelimit_reset", None),
            telemetry=settings.telemetry,
            origine=base_req.origine,
        )
        rt.metrics.incr(f"llm_errors_by_type:{provider_name}:{error_type}")
        # Text ring buffer read by /errors/recent (Grafana cockpit).
        rt.metrics.push_error({
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider_name,
            "error_type": error_type,
            "http_status": http_status,
            "message": str(exc)[:500],
            "task_id": batch_id,
        })
        consecutive = rt.limiter.record_failure(provider_name)
        if consecutive >= settings.resilience.disable_after_consecutive_errors:
            cfg = settings.providers.get(provider_name)
            timeout = cfg.disable_timeout if cfg else 180
            rt.limiter.disable(provider_name, seconds=timeout)
            logger.error(
                f"Provider disabled for {timeout}s after {consecutive} consecutive errors | "
                f"provider={provider_name}"
            )
        raise

    rt.limiter.record_success(provider_name)

    # ── Final adjustment: the TPM window reflects actual consumption ────────
    if reserved_tokens is not None:
        actual_total = (tokens_in or 0) + (tokens_out or 0)
        if actual_total > 0:
            rt.limiter.adjust_tokens(provider_name, actual_total - reserved_tokens)
            if actual_total > reserved_tokens * 1.25:
                logger.warning(
                    f"TPM estimate too low by {actual_total - reserved_tokens} tokens "
                    f"(réservé={reserved_tokens}, réel={actual_total}) — check "
                    f"token_chars_ratio/assumed_output_tokens | provider={provider_name} "
                    f"category={base_req.category} batch={len(merged_agents)} agents"
                )

    # ── Realignment of the agent_ids returned by the LLM ────────────────────
    # The model sometimes returns a malformed agent_id ("PERSONA 446264", or even the
    # persona's name) instead of the expected identifier. Without correction,
    # demultiplexing (results matched on exact agent_id) silently drops
    # these agents: they receive no recommendation. We realign on the real id
    # via its numeric part.
    _expected_ids = [str(a.agent_id) for a in merged_agents]
    _expected_set = set(_expected_ids)
    _by_digits: dict[str, str] = {}
    for _eid in _expected_ids:
        _digits = re.sub(r"\D", "", _eid)
        if _digits:
            _by_digits.setdefault(_digits, _eid)
    for agent_resp in llm_output.agents:
        aid = str(agent_resp.agent_id)
        if aid in _expected_set:
            continue
        resolved = _by_digits.get(re.sub(r"\D", "", aid))
        if resolved:
            logger.warning(
                f"agent_id malformed by the LLM, realigned | got={aid!r} → {resolved!r} | "
                f"provider={provider_name} category={base_req.category}"
            )
            agent_resp.agent_id = resolved
        else:
            logger.error(
                f"agent_id returned by the LLM matches no agent (result lost) | got={aid!r} | "
                f"expected={_expected_ids} provider={provider_name} category={base_req.category}"
            )

    # ── Domain metrics: delegated to the category ───────────────────────────
    # The gateway does not know what a transport mode is; the bundle that declared the
    # category observes the validated response and feeds the counters it wants.
    try:
        handle.observe(provider_name, items, llm_output, rt.metrics)
    except Exception as exc:  # a domain metric must never make a batch fail
        logger.warning(
            f"Category observe hook failed (ignored) | category={base_req.category} "
            f"provider={provider_name} error={exc!r}"
        )

    latency_ms = (time.monotonic() - start_ts) * 1000
    p5_llm_ms = latency_ms

    # Metrics: successful call + batching (agents received → 1 prompt sent)
    rt.metrics.incr(f"llm_calls_ok_total:{provider_name}")
    rt.metrics.incr(f"prompts_sent_total:{base_req.category}")
    rt.metrics.incr(f"agents_batched_total:{base_req.category}", amount=len(merged_agents))

    # Token metrics
    rt.metrics.incr(f"tokens_in_total:{provider_name}", amount=tokens_in)
    rt.metrics.incr(f"tokens_out_total:{provider_name}", amount=tokens_out)
    rt.metrics.incr("tokens_in_total:__all__", amount=tokens_in)
    rt.metrics.incr("tokens_out_total:__all__", amount=tokens_out)

    # Daily token quota (tpd_limit): after-the-fact count of the actual tokens.
    rt.limiter.record_tokens(provider_name, tokens_in + tokens_out)

    tokens_in_per_agent = tokens_in / len(merged_agents) if merged_agents else tokens_in
    if tokens_in_per_agent > settings.batching.assumed_prompt_tokens:
        logger.warning(
            f"[worker] tokens_in exceeds assumed_prompt_tokens | "
            f"tokens_in_per_agent={tokens_in_per_agent:.0f} "
            f"assumed={settings.batching.assumed_prompt_tokens} "
            f"provider={provider_name} batch_id={batch_id}"
        )

    # 6. Log of the LLM exchange (prompt + response + tokens) in workdir/llm_exchanges.jsonl
    # sim_ts = simulated timestamp of the batch, to break down consumption per simulation day.
    # It is the priority score provided by the category (for mobility: the smallest
    # departure_timestamp of the batch); None if the category defines none.
    _scores = [t.priority_score for t in tasks if t.priority_score != _FALLBACK_PRIORITY_SCORE]
    sim_ts = min(_scores) if _scores else None
    log_llm_exchange(
        task_id=batch_id,
        provider=provider_name,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        response=[a.model_dump() if hasattr(a, "model_dump") else a for a in llm_output.agents],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        category=base_req.category,
        sim_ts=sim_ts,
        telemetry=settings.telemetry,
        origine=base_req.origine,
    )

    # 7. Metrics telemetry
    log_llm_call(
        task_id=batch_id,
        provider=provider_name,
        status="success",
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        http_status=200,
    )

    # 7. Demultiplexing and persistence of the results
    # P5_5: demux computation time (dict build + result extraction, before Redis saves)
    _t_demux = time.monotonic()
    results_by_agent = {}
    for a in llm_output.agents:
        aid = a.get("agent_id") if isinstance(a, dict) else getattr(a, "agent_id", None)
        if aid:
            results_by_agent[aid] = a

    n = len(tasks)
    base_in,  rem_in  = divmod(tokens_in,  n)
    base_out, rem_out = divmod(tokens_out, n)

    # Build per-task results before saves so P5_5 is measured once
    task_bundles = []
    for i, t in enumerate(tasks):
        t_agent_ids = [a.agent_id for a in t.request.agents]
        t_results   = [results_by_agent[aid] for aid in t_agent_ids if aid in results_by_agent]
        t_tokens_in  = base_in  + (rem_in  if i == n - 1 else 0)
        t_tokens_out = base_out + (rem_out if i == n - 1 else 0)
        task_bundles.append((t, t_results, t_tokens_in, t_tokens_out))

    p5_5_ms = (time.monotonic() - _t_demux) * 1000

    for t, t_results, t_tokens_in, t_tokens_out in task_bundles:
        # P4_4: micro-batch wait = elapsed since task creation minus all worker phases
        # time.time() (epoch), not time.monotonic(): created_at is an epoch timestamp.
        # created_at may be naive (tasks created before migration) → forced to UTC.
        created = t.created_at if t.created_at.tzinfo else t.created_at.replace(tzinfo=UTC)
        elapsed_since_create = (time.time() - created.timestamp()) * 1000
        p4_4_ms = max(0.0, elapsed_since_create - p5_provider_wait_ms - p5_prompt_ms - p5_llm_ms - p5_5_ms)

        if len(t_results) < len(t.request.agents):
            manquants = set(t_agent_ids) - set(results_by_agent.keys())
            t.status = TaskStatus.FAILED
            t.error  = f"Réponse LLM incomplète : agent(s) manquant(s) dans le lot : {sorted(manquants)}"
            t.result = t_results
        else:
            t.status = TaskStatus.SUCCESS
            t.result = t_results
        t.provider_used = provider_name
        t.latency_ms    = latency_ms
        t.tokens_in     = t_tokens_in
        t.tokens_out    = t_tokens_out
        t.updated_at    = datetime.now(UTC)
        t.timing_p5     = {
            "P4_4_ms":    round(p4_4_ms, 2),
            "P5_1_ms":    round(p5_provider_wait_ms, 2),
            "P5_3_ms":    round(p5_prompt_ms, 2),
            "P5_4_ms":    round(p5_llm_ms, 2),
            "P5_5_ms":    round(p5_5_ms, 2),
            "provider":   provider_name,
            "retries":    retries,
            "tokens_in":  t_tokens_in,
            "tokens_out": t_tokens_out,
        }
        rt.store.save_sync(t)
        rt.store.publish_done_sync(t)

    _consigner_rejeu(rt, handle, task_bundles, provider_name)

    logger.info(
        f"Batch completed successfully | task_id={batch_id} tasks_merged={len(tasks)} "
        f"provider={provider_name} latency_ms={latency_ms:.1f} agents_count={len(llm_output.agents)}"
    )


def _consigner_rejeu(rt: WorkerRuntime, handle, task_bundles, provider_name: str) -> None:
    """Records each served task under the key of ITS prompt, in the space it names.

    The key's prompt is that of the task alone, rendered again: the batch's depends on the
    tasks merged with it, which are not the same from one arm to another. A task in which
    an agent got no response is not recorded — serving an incomplete response again would
    make it final. Never blocking: a missed replay costs a call, not a batch.
    """
    magasin = magasin_rejeu(rt.settings)
    if magasin is None:
        return
    for t, t_results, t_tokens_in, t_tokens_out in task_bundles:
        espace = espace_valide(t.request.espace_rejeu)
        if espace is None or len(t_results) != len(t.request.agents):
            continue
        try:
            messages = handle.render(handle.validate_items(t.request.agents), t.request.parameters)
            cle = cle_rejeu(t.request, messages)
            neuve = magasin.ecrire(espace, cle, enregistrement(
                cle=cle, espace=espace, request=t.request, provider=provider_name,
                agents=t_results, tokens_in=t_tokens_in, tokens_out=t_tokens_out,
                task_id=t.task_id,
            ))
            if neuve:
                rt.metrics.incr(f"rejeu_ab_consigne_total:{t.request.category}")
        except Exception as exc:
            logger.error(
                f"[ALARME] Replay: recording impossible, the next arm will pay for this call | "
                f"task_id={t.task_id} espace={espace} category={t.request.category} erreur={exc!r}"
            )


def _vider_file(rt: WorkerRuntime, batch_key: str, plafond: int = 10_000) -> list[Task]:
    """Pops EVERYTHING the batch queue holds — ticket 085, lot A.

    `BatchQueue.pop` is bounded by a number of AGENTS, not tasks: a single call leaves some
    behind as soon as the batch exceeds the limit. What remains would be picked up at the next
    dispatch and fail identically — precisely the re-arming this lot removes. So we loop
    until the queue is empty, under a cap that bounds the loop without ever being hit in practice.
    """
    tasks: list[Task] = []
    while len(tasks) < plafond:
        lot = rt.queue.pop(batch_key, 1000)
        if not lot:
            break
        tasks.extend(lot)
    return tasks


def _eligibles(
    statuses: dict[str, dict], force_provider: str | None, admises: list[str] | None = None
) -> list[str]:
    """Instances that can serve THIS batch: the pinned one, else the admitted ones, else all."""
    if force_provider is not None:
        return [n for n in statuses if n == force_provider]
    if admises:
        permises = set(admises)
        return [n for n in statuses if n in permises]
    return list(statuses)


def _providers_merely_busy(
    statuses: dict[str, dict], force_provider: str | None, admises: list[str] | None = None
) -> bool:
    """True if at least one eligible provider is merely BUSY, not down.

    Eligible: the forced provider, otherwise the admitted instances, otherwise all. Busy =
    neither disabled, nor in cooldown, nor at the daily quota — a full RPM/TPM window, smoothing
    or concurrency free up by themselves. An instance excluded by the restriction does not count:
    free or not, it will never serve this batch.
    """
    for name in _eligibles(statuses, force_provider, admises):
        st = statuses[name]
        if st.get("disabled") or st.get("cooldown") or st.get("quota_exhausted"):
            continue
        return True
    return False


# Cooldown of an instance after an HTTP 5xx: it is also the resume time announced
# to the client when the batch is returned to it.
_COOLDOWN_5XX_S = 60
# RPM window: resume announced when the instances are merely busy.
_FENETRE_RPM_S = 60


def _attente_client_restante(settings, force_provider: str | None, debut_attente: float) -> float:
    """Seconds before the client gives up on this batch, safety margin deducted.

    Client wait: the pinned instance's `wait_timeout` if declared (the SDK receives it
    per call), otherwise `resilience.client_wait_seconds`. The margin gives the result
    time to reach it before it gives up.
    """
    res = settings.resilience
    attente = res.client_wait_seconds
    if force_provider:
        cfg = settings.providers.get(force_provider)
        if cfg is not None and getattr(cfg, "wait_timeout", None):
            attente = float(cfg.wait_timeout)
    return attente - res.client_wait_margin_seconds - (time.time() - debut_attente)


@dataclass(frozen=True)
class _Verdict:
    """What a batch does when no eligible instance is selectable."""

    attendre: bool
    motif: str = ""                   # "occupe" | "refroidissement" if attendre
    genre: str | None = None          # error_kind returned to the client on give-up
    reprise_dans_s: float | None = None


def _verdict_saturation(
    statuses: dict[str, dict],
    eligibles: list[str],
    ttl_cooldown: dict[str, int],
    restant_s: float,
    *,
    attendre_si_occupe: bool,
    peut_rejouer: bool,
) -> _Verdict:
    """Wait, or return the batch to the client — and under which kind.

    We wait as long as the client is still waiting (`restant_s > 0`) AND an eligible instance
    can free up by itself in time: busy (full window), or cooling down and
    reopening before the client gives up. Otherwise the batch is returned:

    - all eligible ones at the daily quota → `quota_journalier`;
    - at least one overloaded (cooldown, disabled, window that does not free up in
      time) → `surcharge_fournisseur`, with the nearest known reopening;
    - no known eligible one → no kind (the historical case, which nothing qualifies).
    """
    if not eligibles:
        return _Verdict(attendre=False)
    en_panne = [n for n in eligibles if statuses[n].get("disabled") or statuses[n].get("cooldown")
                or statuses[n].get("quota_exhausted")]
    occupes = [n for n in eligibles if n not in en_panne]
    rouvrables = [n for n in eligibles if statuses[n].get("cooldown")
                  and not statuses[n].get("disabled") and not statuses[n].get("quota_exhausted")]
    delais = [float(ttl_cooldown.get(n, 0)) for n in rouvrables]
    if peut_rejouer and restant_s > 0:
        if occupes and attendre_si_occupe:
            return _Verdict(attendre=True, motif="occupe", reprise_dans_s=float(_FENETRE_RPM_S))
        if delais and min(delais) <= restant_s:
            return _Verdict(attendre=True, motif="refroidissement", reprise_dans_s=min(delais))
    if all(statuses[n].get("quota_exhausted") for n in eligibles):
        return _Verdict(attendre=False, genre="quota_journalier")
    candidats = delais + ([float(_FENETRE_RPM_S)] if occupes else [])
    return _Verdict(
        attendre=False,
        genre="surcharge_fournisseur",
        reprise_dans_s=min(candidats) if candidats else float(_COOLDOWN_5XX_S),
    )


def _reprise(settings, verdict: _Verdict, eligibles: list[str]) -> datetime | None:
    """Resume time announced to the client along with the verdict's kind."""
    if verdict.genre == "quota_journalier":
        fuseaux = {
            getattr(settings.providers.get(n), "quota_reset_tz", None) or DEFAUT_FUSEAU_QUOTA
            for n in eligibles
        }
        return min(next_quota_reset(tz) for tz in fuseaux)
    if verdict.reprise_dans_s is not None and verdict.genre:
        return datetime.now(UTC) + timedelta(seconds=verdict.reprise_dans_s)
    return None


def _fit_request_budget(
    provider_cfg,
    prompt_tokens_est: int,
    max_tokens: int,
    min_output_tokens: int,
    provider_name: str,
) -> int:
    """Checks the rendered prompt against the provider's per-request capacity.

    Returns max_tokens, possibly trimmed to fit under max_tokens_per_request
    (groq free tier providers count prompt + max_tokens in the 413 limit).
    Raises ProviderCapacityError if even min_output_tokens no longer fits in the budget.
    """
    if not (provider_cfg and provider_cfg.max_tokens_per_request):
        return max_tokens
    output_budget = provider_cfg.max_tokens_per_request - prompt_tokens_est
    if output_budget < min_output_tokens:
        raise ProviderCapacityError(provider_name, prompt_tokens_est, provider_cfg.max_tokens_per_request)
    return min(max_tokens, output_budget)


def _credits_epuises(
    celery_task,
    rt: WorkerRuntime,
    tasks: list[Task],
    batch_key: str,
    exc: ProviderClientError,
    force_provider: str | None,
    min_tpm_required: int | None,
    min_output_required: int | None,
    instances_admises: list[str] | None = None,
) -> None:
    """HTTP 402: the provider account has no credit left.

    Two differences with an ordinary 4xx:

    - **Duration.** A 30 s cooldown makes no sense: credits come back after a
      human action (billing), not after a minute. The instance is therefore DISABLED
      for its `disable_timeout`, which `/health` publishes as `available: false`.
    - **Visibility.** The event costs a whole campaign and was drowned as a WARNING in
      the worker log (outage 2026-09-07 on `cerebras_gpt-oss-120b`). It comes out as
      ERROR `[ALARME]`, on the **rising edge**: a single line at the first 402, not one
      per batch while the instance stays disabled.

    A caller that pinned its instance (experiment) gets a plain error: no switch
    to another model. A caller without pinning (GAMA) keeps the switch.
    """
    provider = exc.provider
    deja_hors_service = rt.limiter.is_disabled(provider)
    cfg = rt.settings.providers.get(provider)
    timeout = cfg.disable_timeout if cfg else 180
    rt.limiter.disable(provider, seconds=timeout)
    if not deja_hors_service:
        rt.metrics.incr("alarme:credits_epuises")
        logger.error(
            f"[ALARME] Credits exhausted (HTTP 402) on {provider} — instance disabled "
            f"{timeout}s, to be topped up on the provider side | batch_key={batch_key} "
            f"taches={len(tasks)} force_provider={force_provider or 'aucun'} "
            f"message={str(exc)[:200]}"
        )
    if force_provider:
        motif = (
            f"Crédits épuisés (HTTP 402) sur l'instance épinglée {force_provider!r} — "
            f"aucune bascule vers un autre modèle : {exc}"
        )
        for t in tasks:
            _fail_task(rt, t, motif)
        return
    _switch_provider_or_fail(
        celery_task, rt, tasks, batch_key, exc, provider,
        reason="Crédits épuisés (HTTP 402)",
        min_tpm_required=min_tpm_required,
        min_output_required=min_output_required,
            instances_admises=instances_admises,
    )


def _switch_provider_or_fail(
    celery_task,
    rt: WorkerRuntime,
    tasks: list[Task],
    batch_key: str,
    exc: Exception,
    provider: str,
    reason: str,
    min_tpm_required: int | None,
    min_output_required: int | None,
    final_error_msg: str | None = None,
    force_provider: str | None = None,
    instances_admises: list[str] | None = None,
) -> None:
    """Switches the batch to ANOTHER model rather than failing outright.

    Used for errors that are often provider-specific (parse error, unrecoverable
    4xx): the faulty provider is put in a short cooldown, then the batch is replayed
    WITHOUT force_provider, so that the SWRR rotation selects another model. Bounded to
    ~len(providers) attempts so as not to loop on a deterministic error (a truly
    invalid request ends up failing on every provider).

    EXCEPTION — non-null `force_provider`: the switch is REFUSED. A caller that pins
    an instance measures that very model (ticket 035 experiments, `decideurs.py`);
    answering it with another model invalidates the measurement. The client already refused
    the substituted response, but after the fact: the batch was lost and the run spun idle
    (run 2026-09-07_19_45_31, 8 requests, 8 refusals, 0 decision archived). So we fail
    here, plainly, with a reason that names the pinned instance.
    """
    settings = rt.settings
    if force_provider:
        rt.limiter.cooldown(provider, seconds=settings.resilience.provider_switch_cooldown_seconds)
        msg = (
            f"{reason} sur l'instance épinglée {force_provider!r} — aucune bascule "
            f"(décideur épinglé, substitution interdite) : {exc}"
        )
        logger.error(
            f"[ALARME] {reason} on {provider} — switch REFUSED, instance pinned by "
            f"the caller | batch_key={batch_key} force_provider={force_provider} "
            f"taches={len(tasks)} error={str(exc)[:200]}"
        )
        rt.metrics.incr("alarme:bascule_refusee")
        for t in tasks:
            _fail_task(rt, t, final_error_msg or msg)
        return
    rt.limiter.cooldown(provider, seconds=settings.resilience.provider_switch_cooldown_seconds)
    max_switches = min(celery_task.max_retries, max(1, len(settings.providers) - 1))
    if celery_task.request.retries < max_switches:
        logger.warning(
            f"{reason} on {provider} — switching to another model | "
            f"attempt={celery_task.request.retries + 1}/{max_switches} "
            f"error={str(exc)[:200]}"
        )
        rt.queue.requeue(batch_key, tasks)
        raise celery_task.retry(
            exc=exc,
            countdown=1,
            args=[batch_key],
            kwargs={
                "force_provider": None,
                "min_tpm_required": min_tpm_required,
                "min_output_required": min_output_required,
                # Ticket 084 — the restriction SURVIVES the switch. Without this line, it
                # vanished at the first incident: the replay went back to free rotation and
                # the batch could end up served by a model the caller had excluded, without
                # any trace saying so. This is the defect the lot removes.
                "instances_admises": instances_admises,
            },
        )
    terminal_msg = final_error_msg or f"{reason} — échec après {max_switches} bascule(s) de provider : {exc}"
    for t in tasks:
        _fail_task(rt, t, terminal_msg)


def _fail_task(
    rt: WorkerRuntime,
    task: Task,
    error_msg: str,
    *,
    error_kind: str | None = None,
    resume_at: datetime | None = None,
) -> None:
    """Marks the task as failed. `error_kind`/`resume_at` tell the caller WHAT it must
    do: a "quota_journalier" is not a busy gateway, it is waited out until
    `resume_at` instead of being retried every 30 s (incident of 2026-09-08)."""
    task.status     = TaskStatus.FAILED
    task.error      = error_msg
    task.error_kind = error_kind
    task.resume_at  = resume_at
    task.updated_at = datetime.now(UTC)
    rt.store.save_sync(task)
    rt.store.publish_done_sync(task)
    logger.error(
        f"Task failed | task_id={task.task_id} error={error_msg}"
        + (f" kind={error_kind}" if error_kind else "")
        + (f" resume_at={resume_at.isoformat(timespec='seconds')}" if resume_at else "")
    )


# 400 messages returned when max_tokens exceeds the model's completion cap.
# The limit N is captured to be learned (cf. learn_provider_max_output_tokens).
_MAX_TOKENS_LIMIT_PATTERNS = (
    # Groq : "`max_tokens` must be less than or equal to `8192`, the maximum..."
    re.compile(r"max_tokens`? must be less than or equal to `?(\d+)`?"),
    # OpenAI : "max_tokens is too large: 20000. This model supports at most 16384 completion tokens"
    re.compile(r"supports at most (\d+) (?:completion|output) tokens"),
    # Google : "max_output_tokens must be ... limited to 8192" / variants "limited to N"
    re.compile(r"max(?:_output)?_?tokens[^.]{0,80}?limited to (\d+)", re.IGNORECASE),
)


def _parse_max_tokens_limit(error_message: str) -> int | None:
    """Extracts the completion limit N from a 400 max_tokens error message.

    Returns None if the message matches no known format.
    """
    for pattern in _MAX_TOKENS_LIMIT_PATTERNS:
        m = pattern.search(error_message)
        if m:
            return int(m.group(1))
    return None


def _parse_ratelimit_reset_seconds(value: str | None, default: int = 60) -> int:
    """Parses the delay before reset (retry-after / x-ratelimit-reset-* header or 429 body).

    Supported formats:
      - Groq : "1h13m4s", "59m17.087999999s", "6m0s", "45s", "140ms"
        (TPD daily quotas return delays in hours)
      - standard retry-after: raw number of seconds ("13")
      - OpenAI : ISO 8601 "2026-05-28T17:01:00Z"

    Returns `default` if the value is missing or unparseable.
    Returned value clamped to [10, 3600].
    """
    if not value:
        return default

    value = value.strip()

    # Compound duration format: "XhYmZ.Ws", "Xs", "Xms" (Groq)
    m = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?', value)
    if m and any(m.groups()):
        hours   = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = float(m.group(3) or 0)
        millis  = float(m.group(4) or 0)
        total = int(hours * 3600 + minutes * 60 + seconds + millis / 1000) + 2  # +2s of margin
        return max(10, min(total, 3600))

    # Raw number of seconds (standard retry-after header)
    try:
        total = int(float(value)) + 2
        return max(10, min(total, 3600))
    except ValueError:
        pass

    # ISO 8601 timestamp format (OpenAI)
    try:
        reset_dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        delta = (reset_dt - datetime.now(UTC)).total_seconds()
        total = int(delta) + 2
        return max(10, min(total, 3600))
    except (ValueError, TypeError):
        pass

    return default
