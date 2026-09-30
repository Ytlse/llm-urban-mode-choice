"""
api/routes.py — Gateway endpoints (transport layer only).

HTTP contract unchanged (consumed by GAMA / the controller):
  POST /tasks                  → creates a task, returns the task_id immediately
  GET  /tasks/{task_id}        → polling: status + result if available
  GET  /tasks/{task_id}/wait   → long-poll (Redis Pub/Sub)
  GET  /health                 → healthcheck (providers' RPM status)
  GET  /metrics                → Prometheus export

Dependencies (store, queue, balancer…) are read from request.app.state.deps —
composed by create_app(), never imported as singletons.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import ValidationError

from llm_gateway.api.deps import GatewayDeps
from llm_gateway.api.metrics import AGENTS_RECEIVED
from llm_gateway.config import redacted_dump
from llm_gateway.core.batching import compute_batch_key
from llm_gateway.core.models import (
    _FALLBACK_PRIORITY_SCORE,
    AgentResponse,
    LLMRequest,
    Task,
    TaskStatus,
    TaskStatusResponse,
)
from llm_gateway.core.rejeu_ab import cle_rejeu, espace_valide, magasin_rejeu
from llm_gateway.prompts.registry import UnknownCategoryError
from llm_gateway.telemetry.logger import get_logger, log_llm_exchange

logger = get_logger(__name__)

router = APIRouter()


def _deps(request: Request) -> GatewayDeps:
    return request.app.state.deps


_ESPACES_REFUSES: set[str] = set()


async def _servir_par_rejeu(deps: GatewayDeps, handle, items, payload: LLMRequest, task: Task) -> bool:
    """Exact-prompt replay (core/rejeu_ab.py): serves the task if its prompt is already recorded.

    Once served, the task is done before entering the queue: no batch, no RPM slot, no call. The
    exchange log still records it, with provider prefixed `rejeu_ab:` and zero tokens,
    so that the comparison of the two arms reads the same log as before.
    """
    if not payload.espace_rejeu:
        if payload.rejeu_obligatoire:
            raise HTTPException(status_code=409, detail="Mandatory replay without a cache space")
        return False
    magasin = magasin_rejeu(deps.settings)
    espace = espace_valide(payload.espace_rejeu)
    if espace is None:
        if payload.rejeu_obligatoire:
            raise HTTPException(status_code=409, detail="Invalid replay space")
        if payload.espace_rejeu not in _ESPACES_REFUSES:
            _ESPACES_REFUSES.add(payload.espace_rejeu)
            logger.error(
                f"[ALARME] Replay space rejected (characters outside [A-Za-z0-9._-]): tasks "
                f"go to the provider | espace={payload.espace_rejeu!r}"
            )
        return False
    if magasin is None:
        if payload.rejeu_obligatoire:
            raise HTTPException(status_code=503, detail="Magasin de rejeu indisponible")
        return False
    messages = handle.render(items, payload.parameters)
    cle = cle_rejeu(payload, messages)
    rec = await asyncio.to_thread(magasin.lire, espace, cle)
    if rec is None:
        await asyncio.to_thread(deps.metrics.incr, f"rejeu_ab_absent_total:{payload.category}")
        if payload.rejeu_obligatoire:
            logger.error(
                "[ALARME] Préfixe commun interrompu : cache miss | espace={} "
                "categorie={} cle={} origine={}",
                espace, payload.category, cle, payload.origine,
            )
            raise HTTPException(
                status_code=409,
                detail={"erreur": "rejeu_obligatoire_absent", "categorie": payload.category,
                        "cle": cle, "espace": espace},
            )
        return False
    task.status = TaskStatus.SUCCESS
    task.result = [AgentResponse(**a) for a in rec.get("agents") or []]
    task.provider_used = rec.get("provider")
    task.latency_ms = 0.0
    task.tokens_in = 0
    task.tokens_out = 0
    task.rejeu = espace
    task.updated_at = datetime.now(UTC)
    await deps.store.save(task)
    await asyncio.to_thread(deps.metrics.incr, f"rejeu_ab_servi_total:{payload.category}")
    await asyncio.to_thread(
        log_llm_exchange,
        task_id=task.task_id,
        provider=f"rejeu_ab:{rec.get('provider')}",
        messages=[{"role": m.role, "content": m.content} for m in messages],
        response=rec.get("agents"),
        tokens_in=0,
        tokens_out=0,
        category=payload.category,
        sim_ts=task.priority_score if task.priority_score != _FALLBACK_PRIORITY_SCORE else None,
        telemetry=deps.settings.telemetry,
        origine=payload.origine,
    )
    return True


def _to_response(task: Task) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        result=task.result,
        error=task.error,
        error_kind=task.error_kind,
        resume_at=task.resume_at,
        provider_used=task.provider_used,
        latency_ms=task.latency_ms,
        timing_p5=task.timing_p5,
        rejeu=task.rejeu,
    )


@router.post(
    "/tasks",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a batched LLM task",
    description=(
        "Queues an LLM request for asynchronous processing by Celery. "
        "Similar requests (same category, same parameters) are automatically grouped into batches "
        "to optimise LLM calls.\n\n"
        "Immediately returns a unique `task_id` to query the status via `GET /tasks/{task_id}`."
    ),
    response_description="The generated task_id and the initial status of the task.",
)
async def create_task(payload: LLMRequest, request: Request) -> dict:
    """
    1. Validates the request (Pydantic)
    2. Creates the task with PENDING status and persists it
    3. Adds it to its batch queue and schedules the Celery dispatch
    4. Returns immediately
    """
    from llm_gateway.worker.task_worker import process_batch_task

    deps = _deps(request)
    # The category decides what a valid item is and the batch priority: the gateway
    # only knows the agent_id. Unknown category or invalid item → 422, not a task
    # that will fail in the worker.
    try:
        handle = deps.registry.get(payload.category)
        items = handle.validate_items(payload.agents)
    except UnknownCategoryError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from None
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid items for category {payload.category!r}: {exc.errors()}",
        ) from None
    priority_score = handle.priority_score(items)
    task = Task(request=payload, priority_score=priority_score)

    AGENTS_RECEIVED.labels(category=payload.category).inc(len(payload.agents))
    if await _servir_par_rejeu(deps, handle, items, payload, task):
        return {
            "task_id": task.task_id,
            "status": task.status,
            "provider_used": task.provider_used,
            "message": f"Tâche servie par rejeu (espace {task.rejeu}), sans appel au fournisseur.",
        }
    await deps.store.save(task)

    # Batch key based on the context and the global parameters: only perfectly
    # compatible tasks are merged.
    batch_key = compute_batch_key(payload)
    queue_size = await deps.queue.add(batch_key, task.task_id, priority_score)

    # Queue at threshold → immediate dispatch. Otherwise, a short delay is granted
    # (batch_delay_seconds) to accumulate other tasks; the SETNX flag guarantees
    # that exactly one delayed dispatch is scheduled per batch cycle, whatever
    # the arrival order of concurrent requests (queue_size alone is not enough:
    # two simultaneous requests can both observe queue_size == 2).
    # The threshold comes from get_dispatch_threshold (batch target, NOT the providers'
    # min, which is 1 and would make dispatch always immediate); the worker
    # will re-cap at pop according to the selected provider's batch_max_agents.
    settings = deps.settings
    batch_limit = settings.get_dispatch_threshold(
        payload.force_provider, instances_admises=payload.instances_admises
    )
    # Output budget of a task: the load balancer discards the providers whose
    # completion cap (max_output_tokens) cannot serve a single task.
    min_output_required = payload.parameters.get("max_tokens")
    loop = asyncio.get_event_loop()
    if queue_size >= batch_limit:
        await loop.run_in_executor(
            None,
            lambda: process_batch_task.delay(
                batch_key,
                payload.force_provider,
                payload.min_tpm_required,
                min_output_required,
                payload.instances_admises,
            ),
        )
    elif await deps.queue.try_mark_scheduled(batch_key, ttl=int(settings.batching.delay_seconds) + 30):
        await loop.run_in_executor(
            None,
            lambda: process_batch_task.apply_async(
                args=[
                    batch_key,
                    payload.force_provider,
                    payload.min_tpm_required,
                    min_output_required,
                    payload.instances_admises,
                ],
                countdown=settings.batching.delay_seconds,
            ),
        )

    logger.info(f"Task created and enqueued | task_id={task.task_id} category={payload.category}")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "provider_used": task.provider_used,
        "message": f"Tâche acceptée. Pollez GET /tasks/{task.task_id} pour le résultat.",
    }


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get the status and result of a task",
    description=(
        "Polling endpoint to check the state of a previously submitted task. "
        "If the task is finished (`status == 'success'`), the `result` field holds the LLM response."
    ),
    response_description="The current state of the task with its results, if any.",
)
async def get_task_status(task_id: str, request: Request) -> TaskStatusResponse:
    task = await _deps(request).store.get(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found or expired.",
        )

    return _to_response(task)


@router.get(
    "/tasks/{task_id}/wait",
    response_model=TaskStatusResponse,
    summary="Wait for a task to finish (Redis Pub/Sub long-poll)",
    description=(
        "Blocks until the task reaches a terminal state (success/failed) "
        "or the timeout expires. Notification latency ~10ms via Redis Pub/Sub. "
        "Reconnects automatically if the pubsub socket is interrupted before the timeout ends."
    ),
)
async def wait_for_task(task_id: str, request: Request, timeout: float = 120.0) -> TaskStatusResponse:
    timeout = min(max(timeout, 1.0), 300.0)
    task = await _deps(request).store.wait_done(task_id, timeout)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found or expired.",
        )

    return _to_response(task)


@router.get(
    "/health",
    summary="Check service health and RPM quotas",
    description="Returns the operating state of the API Gateway and the current state of the requests-per-minute (RPM) counters for each LLM provider.",
    response_description="Dictionary holding the global status and the provider metrics.",
)
async def health(request: Request) -> dict:
    return {
        "status": "ok",
        "providers": _deps(request).balancer.get_status(),
    }


@router.post(
    "/providers/{provider}/reset-quota",
    summary="Reset the quota-exhaustion lock of a provider",
    description="Removes the quota_exhausted flag set after a 429 so the instance can be tested again.",
)
async def reset_provider_quota(provider: str, request: Request) -> dict:
    deps = _deps(request)
    deps.limiter.clear_quota_exhausted(provider)
    return {"status": "ok", "provider": provider}


@router.post(
    "/providers/reset-quota",
    summary="Reset the quota-exhaustion locks",
    description="Removes the quota_exhausted flags for the specified providers or for all providers.",
)
async def reset_providers_quota(request: Request, payload: dict | None = None) -> dict:
    deps = _deps(request)
    providers = (payload or {}).get("providers")
    if providers is None:
        providers = list(deps.settings.providers.keys())
    for p in providers:
        deps.limiter.clear_quota_exhausted(p)
    return {"status": "ok", "reset": providers}


@router.get(
    "/errors/recent",
    summary="Latest LLM errors reported by the providers",
    description=(
        "Returns the latest LLM errors (raw message, provider, type, HTTP code) "
        "from the Redis ring buffer. Consumed by the 'Dernières erreurs' panel of the Grafana "
        "cockpit (Infinity datasource) — Prometheus stores only numeric data."
    ),
    response_description="JSON list of the errors, most recent first.",
)
async def recent_errors(request: Request, limit: int = 50) -> list[dict]:
    limit = min(max(limit, 1), 50)
    loop = asyncio.get_event_loop()
    metrics_sink = _deps(request).metrics
    return await loop.run_in_executor(None, metrics_sink.recent_errors, limit)


@router.get(
    "/config",
    summary="Effective gateway configuration, secrets masked",
    description=(
        "The settings as the process sees them (constructor, environment, file, "
        "profile, defaults) and the declared providers. API keys and tokens are masked."
    ),
)
async def effective_config(request: Request) -> dict:
    return redacted_dump(_deps(request).settings)


@router.get(
    "/config/providers",
    summary="Declared providers and their effective configuration, without secrets",
    description=(
        "For tools (dashboard, experiments) that used to read providers.yaml: the same "
        "information, as computed by the gateway (capacities, learned limits), without keys."
    ),
)
async def providers_config(request: Request) -> dict:
    settings = _deps(request).settings
    active = {
        name: {**cfg.model_dump(mode="json", exclude={"api_key"}), "has_api_key": bool(cfg.api_key.get_secret_value())}
        for name, cfg in settings.providers.items()
    }
    return {"declared": settings.declared_providers, "active": active}


@router.get(
    "/metrics",
    summary="Export Prometheus metrics",
    description="Exposes the internal metrics of the application in a format a Prometheus server can read.",
    response_description="Texte brut au format Prometheus.",
)
async def metrics():
    loop = asyncio.get_event_loop()
    # generate_latest reads Redis synchronously, so it is isolated in a thread
    content = await loop.run_in_executor(None, generate_latest, REGISTRY)
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)
