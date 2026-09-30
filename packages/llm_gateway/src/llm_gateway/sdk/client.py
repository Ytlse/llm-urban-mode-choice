"""
sdk.py — typed client SDK of the LLM gateway (phase 4 of the restructuring report).

Replaces the raw dicts of LLMClient.execute_async() with a pydantic TaskResult:
no more magic conventions ("EXPECTED_ERROR", "_post_ms" keys injected into the
response), no more dead _heartbeat task, and a single httpx.AsyncClient reused
across calls (keep-alive) instead of two created per task.

The gateway HTTP contract is unchanged: POST /tasks then GET /tasks/{id}/wait.

Usage:
    client = LLMGatewayClient(base_url="http://api:8000", wait_timeout=90.0)
    result = await client.execute(payload)          # dict or LLMRequest
    if result.ok:
        # itinary_multi_agent returns a distribution; the draw is done via
        # mobility_llm.mode_choice (normalize_option_probabilities + draw_index).
        probabilities = result.agents[0].probabilities
    await client.aclose()
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from prometheus_client import Gauge, Histogram
from pydantic import BaseModel

from llm_gateway.core.models import AgentResponse, LLMRequest, TaskStatus
from llm_gateway.telemetry.alarms import fire_alarme

# ---------------------------------------------------------------------------
# Consumer-side metrics (GAMA controller).
# The chosen modes/indices and task counters are already exposed on the worker side
# (llm_transport_mode_chosen_total, llm_chosen_index_total, llm_provider_calls_*);
# only the E2E, measurable only on the client side, lives here.
# ---------------------------------------------------------------------------

LLM_TASK_E2E_DURATION = Histogram(
    'llm_task_e2e_duration_seconds',
    'Total duration POST /tasks → final response (controller side), per category',
    ['category'],
    buckets=[1, 2, 5, 10, 30, 60, 120],
)

LLM_GATEWAY_CIRCUIT_OPEN = Gauge(
    'llm_gateway_circuit_open',
    'LLM gateway client circuit breaker (1=open: submissions are suspended '
    'until recovery, a probe re-tests periodically)',
)

LLM_GATEWAY_CIRCUIT_WAITERS = Gauge(
    'llm_gateway_circuit_waiters',
    'LLM submissions suspended behind the open circuit breaker (awaiting recovery)',
)

_TRANSIENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadTimeout)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class TaskTiming(BaseModel):
    """Timings measured on the client side + worker segments (P4/P5)."""
    post_ms: float = 0.0                          # duration of POST /tasks
    wait_ms: float = 0.0                          # duration of the /wait long-poll
    timing_p5: dict[str, Any] | None = None    # worker segments (P4_4, P5_1…)


class TaskResult(BaseModel):
    """Typed result of a gateway task — no more string comparisons."""
    status: TaskStatus
    agents: list[AgentResponse] = []
    error: str | None = None
    # Kind of failure when the gateway knows it: "quota_journalier" + the time the
    # window reopens. Lets the caller WAIT for the window instead of retrying every
    # 30 s a key closed for the day (incident of 2026-09-08).
    error_kind: str | None = None
    resume_at: datetime | None = None
    provider_used: str | None = None
    # Replay space that served the response without a provider call (None: served by it).
    rejeu: str | None = None
    timing: TaskTiming | None = None
    task_id: str | None = None

    @property
    def quota_journalier_epuise(self) -> bool:
        """Is the failure a daily quota confirmed by the provider?"""
        return self.error_kind == "quota_journalier"

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCESS and bool(self.agents)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMGatewayClient:
    """
    Asynchronous SDK of the gateway. A single httpx.AsyncClient reused across
    calls; aclose() when the consumer stops.
    """

    # Number of consecutive failures before raising the "gateway continuously failing" alarm
    _FAILURE_ALARM_THRESHOLD = 10

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        wait_timeout: float = 120.0,
        dialogue_log_file: str | None = None,   # text log of the payloads: potential personal data, opt-in
        transport: httpx.AsyncBaseTransport | None = None,
        backpressure_max_inflight: int = 0,
        backpressure_release_ratio: float = 0.2,
        circuit_failure_threshold: int = 10,
        circuit_probe_interval: float = 60.0,
        instances_admises: list[str] | None = None,
        origine: str | None = None,
        espace_rejeu: str | None = None,
        rejeu_strict_avant_ts: int | None = None,
        horloge_simulee: Callable[[], int] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._wait_timeout = wait_timeout
        self._dialogue_log_file = dialogue_log_file
        self._transport = transport  # injectable for tests (MockTransport)
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._consecutive_failures = 0
        # Backpressure: when the alarm fires (N consecutive failures), new
        # submissions are suspended until the in-flight stack falls back
        # below backpressure_release_ratio × backpressure_max_inflight.
        # backpressure_max_inflight=0 → backpressure disabled.
        self._backpressure_max_inflight = backpressure_max_inflight
        self._backpressure_release_ratio = backpressure_release_ratio
        self._backpressure_active = False
        self._inflight = 0
        # Circuit breaker: after N consecutive failures (lasting outage — quotas exhausted,
        # gateway/network down), submissions are SUSPENDED: each caller quietly
        # waits for recovery (quota renewal, service return) instead of burning
        # attempts doomed to fail and degrading decisions into default indices.
        # A half-open probe re-tests the gateway every circuit_probe_interval
        # seconds; the first success closes the circuit breaker again and all the
        # suspended calls resume on the nominal path.
        # circuit_failure_threshold=0 → circuit breaker disabled.
        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_probe_interval = circuit_probe_interval
        self._circuit_open_since: float | None = None
        self._circuit_next_probe_at = 0.0
        self._circuit_probe_inflight = False
        self._circuit_waiters = 0
        # Ticket 092 — the instance restriction belongs to the CLIENT, not to a call.
        # Ticket 084 set it on the decision payload only; the STM and LTM reflections
        # went out without it and were served in cascade. Measured on 2026-09-16: decisions
        # on gemini 3.1, memory consolidation on mistral, in an experiment whose subject
        # IS memory. Set here, it covers the three calls and any future call.
        self._instances_admises = list(instances_admises or [])
        # Set on the client for the same reason: every call, present or future, is signed.
        self._origine = origine
        # Exact-prompt replay space, set on all calls like the origin.
        self._espace_rejeu = espace_rejeu
        self._rejeu_strict_avant_ts = rejeu_strict_avant_ts
        self._horloge_simulee = horloge_simulee
        self._borne_franchie_annoncee = False
        # Read by the controller barrier even if a category catches its exception
        # locally: a cache miss can never turn into silence.
        self.strict_replay_failure: str | None = None
        if self._instances_admises:
            logger.info(
                f"[gateway] instance restriction active on ALL calls of this client: "
                f"{self._instances_admises}"
            )

    @property
    def circuit_open(self) -> bool:
        """True if the circuit breaker is open (LLM gateway considered in lasting outage)."""
        return self._circuit_open_since is not None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    # Read timeout aligned on the /wait long-poll (+ margin)
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self._wait_timeout + 30),
                        transport=self._transport,
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------

    async def execute(
        self, request: LLMRequest | dict[str, Any], *, wait_timeout: float | None = None
    ) -> TaskResult:
        """
        Submits a task and waits for its terminal state (server-side Pub/Sub long-poll).

        Does not raise on a task failure: the TaskResult carries status/error.
        Raises httpx.HTTPStatusError if the gateway refuses the submission (4xx/5xx).

        `wait_timeout` overrides the wait for THIS call. The client default applies to a
        remote provider; a local model serves one call at a time and makes the next ones
        wait, so that the wait of a task is not its generation time but the time spent in
        the queue. On 2026-09-08, a run on Muse Glimmer (28B, a generation of 40 to
        120 s, a single concurrent call) saw every task after the first one expire at
        120 s, then the circuit breaker open. The caller that pins an instance therefore
        passes it its `wait_timeout` (field of the providers file).
        """
        payload = request.model_dump(exclude_none=True) if isinstance(request, LLMRequest) else request
        category = payload.get("category", "unknown")

        # The client restriction only FILLS IN: a caller that carries its own stays
        # in control of its routing (case A4 of the contract).
        if self._instances_admises and not payload.get("instances_admises"):
            payload = {**payload, "instances_admises": list(self._instances_admises)}
        if self._origine and not payload.get("origine"):
            payload = {**payload, "origine": self._origine}
        if self._espace_rejeu and not payload.get("espace_rejeu"):
            payload = {**payload, "espace_rejeu": self._espace_rejeu}
        if self._rejeu_strict_avant_ts is not None:
            if self._horloge_simulee is None:
                raise RuntimeError("Common prefix: simulated clock not wired to the LLM client")
            if self._horloge_simulee() < self._rejeu_strict_avant_ts:
                payload = {**payload, "rejeu_obligatoire": True}
            else:
                # After the boundary (2026-09-28), the control no longer reads the treated store.
                # The store files each task's response under ITS prompt, but the treated arm got it
                # in a merged batch: the response of an agent who did not read the article may have
                # been written next to a reader's record. Serving it to the control would leak some
                # of the article. Without a space, nothing is served or recorded.
                payload = {k: v for k, v in payload.items() if k != "espace_rejeu"}
                if not self._borne_franchie_annoncee:
                    self._borne_franchie_annoncee = True
                    logger.info(
                        "[rejeu_ab] common prefix boundary crossed ({}): the control now pays "
                        "for all its calls, the treated store is no longer read.",
                        self._rejeu_strict_avant_ts,
                    )

        # Circuit breaker open: the submission is SUSPENDED until the gateway recovers
        # (no degraded decision — we wait for the quota renewal).
        # One of the suspended callers periodically becomes the probe (half-open)
        # that goes through to test whether the gateway has recovered.
        is_probe = await self._circuit_gate()

        # Backpressure: if the gateway alarm is active, wait for the in-flight stack
        # to drain before handing back control (submitting this task).
        # The probe is exempt: it must go out without delay.
        if not is_probe and self._backpressure_active and self._backpressure_max_inflight > 0:
            await self._await_backpressure_drain()

        self._inflight += 1
        _e2e_start = time.monotonic()
        try:
            try:
                task_id = await self._submit(payload)
            except httpx.HTTPStatusError as e:
                if payload.get("rejeu_obligatoire"):
                    self.strict_replay_failure = (
                        f"rejeu strict {category} refusé (HTTP {e.response.status_code}): "
                        f"{e.response.text[:500]}"
                    )
                # 4xx = invalid payload (programming error): propagated without
                # counting a failure. 5xx = sick gateway: failure counted (alarm,
                # backpressure, circuit breaker) then returned to the caller as TaskResult.
                if e.response.status_code < 500:
                    raise
                result = TaskResult(
                    status=TaskStatus.FAILED,
                    error=f"Gateway error {e.response.status_code} à la soumission",
                    timing=TaskTiming(),
                )
                self._observe(result, payload, category, _e2e_start, 0.0)
                return result
            except _TRANSIENT as e:
                # Gateway unreachable (network cut, service down): without this
                # count, a clean outage triggered NEITHER the alarm NOR the
                # circuit breaker (the exception short-circuited _observe).
                result = TaskResult(
                    status=TaskStatus.FAILED,
                    error=f"Gateway LLM injoignable ({type(e).__name__})",
                    timing=TaskTiming(),
                )
                self._observe(result, payload, category, _e2e_start, 0.0)
                return result
            post_ms = (time.monotonic() - _e2e_start) * 1000

            wait_start = time.monotonic()
            result = await self._wait(task_id, wait_timeout)
            wait_ms = (time.monotonic() - wait_start) * 1000

            result.task_id = task_id
            if result.timing is None:
                result.timing = TaskTiming()
            result.timing.post_ms = round(post_ms, 2)
            result.timing.wait_ms = round(wait_ms, 2)

            self._observe(result, payload, category, _e2e_start, wait_ms)
            return result
        finally:
            self._inflight -= 1
            if is_probe:
                self._circuit_probe_inflight = False

    async def _circuit_gate(self) -> bool:
        """Holds the caller as long as the circuit breaker is open.

        Returns True if this call becomes the probe (it goes through to test the
        gateway); False otherwise — either the circuit breaker was closed, or it has just
        closed again (another caller's probe succeeded) and the call resumes on the
        nominal path. No task fails or is degraded during the wait:
        the simulation waits for the quota renewal / the service return.
        """
        if not self.circuit_open:
            return False
        waiting = False
        try:
            while self.circuit_open:
                now = time.monotonic()
                if now >= self._circuit_next_probe_at and not self._circuit_probe_inflight:
                    self._circuit_probe_inflight = True
                    logger.info(
                        f"[circuit] Probe to the LLM gateway (circuit breaker half-open, "
                        f"{self._circuit_waiters} submission(s) waiting)"
                    )
                    return True
                if not waiting:
                    waiting = True
                    self._circuit_waiters += 1
                    LLM_GATEWAY_CIRCUIT_WAITERS.set(self._circuit_waiters)
                    logger.debug("[circuit] Submission suspended until the gateway recovers")
                await asyncio.sleep(min(1.0, max(0.1, self._circuit_next_probe_at - now)))
        finally:
            if waiting:
                self._circuit_waiters -= 1
                LLM_GATEWAY_CIRCUIT_WAITERS.set(self._circuit_waiters)
        return False

    async def _await_backpressure_drain(self) -> None:
        """Suspends the calling coroutine as long as the in-flight stack exceeds
        backpressure_release_ratio × backpressure_max_inflight. Once drained,
        disarms the backpressure: the pending backlog resumes all at once."""
        threshold = max(1, int(self._backpressure_max_inflight * self._backpressure_release_ratio))
        if self._inflight <= threshold:
            self._backpressure_active = False
            return
        logger.warning(
            f"[BACKPRESSURE] Alarme gateway active — soumissions LLM suspendues "
            f"jusqu'au drainage de la pile ({self._inflight} en vol → ≤ {threshold})"
        )
        while self._inflight > threshold:
            await asyncio.sleep(0.5)
        self._backpressure_active = False
        logger.info(
            f"[BACKPRESSURE] Pile drainée ({self._inflight} en vol ≤ {threshold}) — "
            f"reprise des soumissions LLM"
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    async def _submit(self, payload: dict[str, Any]) -> str:
        """POST /tasks — retry up to 3 times on transient network errors."""
        client = await self._http()
        for attempt in range(3):
            try:
                resp = await client.post(f"{self._base_url}/tasks", json=payload, timeout=30.0)
                resp.raise_for_status()
                return str(resp.json()["task_id"])
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"LLM gateway returned HTTP error | status={e.response.status_code} "
                    f"url={self._base_url}/tasks body={e.response.text[:300]}"
                )
                raise
            except _TRANSIENT as e:
                if attempt == 2:
                    logger.error(
                        f"Cannot connect to LLM gateway after 3 attempts | url={self._base_url} error={e}"
                    )
                    raise
                wait = 2 ** attempt
                logger.warning(
                    f"LLM gateway POST failed (attempt {attempt + 1}/3), retry in {wait}s "
                    f"| error={type(e).__name__}"
                )
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable: the last attempt re-raises or returns")  # pragma: no cover

    async def _wait(self, task_id: str, wait_timeout: float | None = None) -> TaskResult:
        """GET /tasks/{id}/wait — waits for the terminal state or the timeout.

        The httpx read timeout is set PER REQUEST: the shared client is built on the
        default wait, and a longer wait would otherwise be cut by it before the
        long-poll hands back control."""
        attente = self._wait_timeout if wait_timeout is None else float(wait_timeout)
        client = await self._http()
        try:
            resp = await client.get(
                f"{self._base_url}/tasks/{task_id}/wait",
                params={"timeout": attente},
                timeout=httpx.Timeout(attente + 30),
            )
        except _TRANSIENT as e:
            logger.error(f"LLM gateway wait request failed | task_id={task_id} error={e}")
            return TaskResult(status=TaskStatus.FAILED, error="Timeout expiré", timing=TaskTiming())

        if resp.status_code >= 500:
            logger.error(
                f"LLM gateway wait returned HTTP {resp.status_code} | task_id={task_id} body={resp.text[:300]}"
            )
            return TaskResult(status=TaskStatus.FAILED, error=f"Gateway error {resp.status_code}", timing=TaskTiming())
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error(
                f"LLM gateway returned non-JSON response | task_id={task_id} status={resp.status_code} body={resp.text[:300]}"
            )
            return TaskResult(status=TaskStatus.FAILED, error=f"Réponse gateway non-JSON (HTTP {resp.status_code})", timing=TaskTiming())

        status = TaskStatus(data["status"]) if data.get("status") in TaskStatus._value2member_map_ else TaskStatus.FAILED
        if status not in (TaskStatus.SUCCESS, TaskStatus.FAILED):
            # Long-poll budget exhausted without a terminal state
            return TaskResult(status=TaskStatus.FAILED, error="Timeout expiré", timing=TaskTiming())

        return TaskResult(
            status=status,
            agents=[AgentResponse(**a) for a in (data.get("result") or [])],
            error=data.get("error"),
            error_kind=data.get("error_kind"),
            resume_at=data.get("resume_at"),
            provider_used=data.get("provider_used"),
            rejeu=data.get("rejeu"),
            timing=TaskTiming(timing_p5=data.get("timing_p5")),
        )

    def _observe(self, result: TaskResult, payload: dict[str, Any], category: str, e2e_start: float, wait_ms: float) -> None:
        """Metrics + dialogue log — same behaviour as the old client."""
        if payload.get("rejeu_obligatoire") and not self.strict_replay_failure and (
            not result.ok or result.rejeu != payload.get("espace_rejeu")
        ):
            self.strict_replay_failure = (
                f"rejeu strict {category} non servi par cache : "
                f"état={result.status}, espace={result.rejeu}, erreur={result.error}"
            )
        LLM_TASK_E2E_DURATION.labels(category=category).observe(time.monotonic() - e2e_start)
        self._log_dialogue(payload, result)

        if result.ok:
            self._consecutive_failures = 0
            # The gateway answers again → any ongoing backpressure is disarmed.
            self._backpressure_active = False
            if self.circuit_open:
                downtime = time.monotonic() - (self._circuit_open_since or time.monotonic())
                self._circuit_open_since = None
                LLM_GATEWAY_CIRCUIT_OPEN.set(0)
                logger.info(
                    f"[circuit] LLM gateway recovered after {downtime:.0f}s of open circuit "
                    f"breaker — resuming the {self._circuit_waiters} suspended submission(s)"
                )
        else:
            error_msg = result.error or "No error detail"
            _log = logger.warning if "saturés" in error_msg or "indisponibles" in error_msg else logger.error
            _log(
                f"Task failed | task_id={result.task_id} category={category} "
                f"wait={wait_ms / 1000:.1f}s error={error_msg}"
            )
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._FAILURE_ALARM_THRESHOLD:
                # Arms the backpressure: the next submissions will wait for the
                # in-flight stack to drain before going out.
                self._backpressure_active = True
                if self._consecutive_failures == self._FAILURE_ALARM_THRESHOLD:
                    fire_alarme("gateway_llm")
                    logger.error(
                        f"[ALARME] Gateway LLM : {self._consecutive_failures} tâches échouées d'affilée "
                        f"(dernière : {error_msg}) — plus aucune décision LLM ne revient. "
                        f"Providers en rate-limit ou gateway indisponible : vérifier /health "
                        f"et docker compose logs api worker."
                    )
            now = time.monotonic()
            if self.circuit_open:
                # Probe (or request in flight at opening) failed: the next
                # probe will wait a full interval.
                self._circuit_next_probe_at = now + self._circuit_probe_interval
            elif (
                self._circuit_failure_threshold > 0
                and self._consecutive_failures >= self._circuit_failure_threshold
            ):
                self._circuit_open_since = now
                self._circuit_next_probe_at = now + self._circuit_probe_interval
                LLM_GATEWAY_CIRCUIT_OPEN.set(1)
                fire_alarme("gateway_llm_circuit")
                logger.error(
                    f"[ALARME] LLM gateway circuit breaker OPEN after {self._consecutive_failures} "
                    f"consecutive failures (last: {error_msg}). LLM submissions are SUSPENDED — "
                    f"the simulation waits for recovery (quota renewal / service return), "
                    f"no decision is degraded. A probe will re-test the gateway every "
                    f"{self._circuit_probe_interval:.0f}s; recovery is automatic."
                )

    def _log_dialogue(self, payload: dict[str, Any], result: TaskResult) -> None:
        """Records the request and the response as a detailed text dialogue."""
        if not self._dialogue_log_file:
            return
        try:
            with open(self._dialogue_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"⏱ TIMESTAMP : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"📌 CATEGORY  : {payload.get('category', 'unknown')}\n")
                f.write(f"{'-' * 60}\n")
                f.write("👤 >>> REQUEST (Payload) >>>\n")
                f.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
                f.write(f"{'-' * 60}\n")
                f.write("🤖 <<< RESPONSE (Result) <<<\n")
                content = [a.model_dump() for a in result.agents] if result.agents else result.model_dump(exclude={"agents"})
                f.write(json.dumps(content, indent=2, ensure_ascii=False, default=str) + "\n")
                f.write(f"{'=' * 60}\n")
        except Exception as e:
            logger.warning(f"Error while writing the dialogue log: {e}")
