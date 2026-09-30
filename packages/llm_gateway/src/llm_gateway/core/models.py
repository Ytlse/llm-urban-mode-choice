"""
core/models.py — Pydantic data models shared across all modules.

Pure domain: no I/O, no redis/celery/httpx/fastapi import.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# Incoming request (client → API Gateway)
# ---------------------------------------------------------------------------

class AgentItem(BaseModel):
    """An item of a batch: an identifier, and whatever the category will want to read in it.

    The gateway only knows ``agent_id`` (demultiplexing of responses). The other fields
    are kept as is (``extra="allow"``) and validated by the category's item model
    (cf. ``ports.category.CategorySpec.item_model``) at render time.
    """
    model_config = ConfigDict(extra="allow")

    agent_id: str

    @field_validator("agent_id", mode="before")
    @classmethod
    def _agent_id_as_str(cls, v: Any) -> Any:
        return str(v) if isinstance(v, int) else v


class LLMRequest(BaseModel):
    """Body of the POST /tasks request."""
    category: str = Field(..., description="Catégorie de la requête (selection itinéraire, ...)")
    agents: list[AgentItem] = Field(..., min_length=1, description="Items du lot (un persona, un document…), chacun avec son agent_id")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Paramètres additionnels pour le prompt")
    # Optional: force a specific provider (bypasses the load balancer)
    force_provider: str | None = None
    # Ticket 084 — LIST of instances allowed to serve this request, honoured AT SELECTION.
    #
    # ⚠ This field MUST be declared here. `LLMRequest` does not forbid extra fields: a
    # restriction set by the client without appearing in this model would be silently ignored
    # by FastAPI validation — no exception, no log, and a measurement taken under a
    # restriction that never existed. This is the reason for case B1 of the test contract.
    #
    # Not to be confused with the client filter `allowed_providers` (`llm_agent.py`), which
    # rejects an ALREADY billed response: this one prevents the call, the other observes it.
    instances_admises: list[str] | None = None
    # Optional: minimum TPM required — the load balancer excludes providers below this threshold
    min_tpm_required: int | None = None
    # Who issued the request (for the simulation: the run name). The exchange log is
    # written by the WORKER, shared by all clients: without this field, a run read in its
    # `llm_exchanges.jsonl` the calls of another client served while it was running (97
    # exchanges from a population of 1,000 agents in an arm of 20, on 2026-09-24).
    origine: str | None = None
    # Optional: exact-prompt replay space (core/rejeu_ab.py). A task whose prompt has
    # already been served in this space receives the recorded response, without a provider
    # call. The two arms of an A/B share it: they stay the same world as long as nothing
    # separates them. Outside the batch key: recording is per task, merging changes nothing.
    espace_rejeu: str | None = None
    # The common prefix of an A/B can never pay for a cache miss: a miss reveals a
    # prompt or schedule divergence and must interrupt the control arm.
    rejeu_obligatoire: bool = False
    context: str | None = Field(default=None, description="Contexte global de la ville (ex: trafic, météo)")


# ---------------------------------------------------------------------------
# Internal task (API → Broker → Worker)
# ---------------------------------------------------------------------------

_FALLBACK_PRIORITY_SCORE: float = 9_999_999_999.0


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    request: LLMRequest
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: list[AgentResponse] | None = None
    error: str | None = None
    # Nature of the failure, when it is known and usable by the caller. The only case
    # served today: "quota_journalier" — the provider refused for its daily quota,
    # and `resume_at` says when the window reopens. Without this field, the worker's error
    # message ("Providers saturés ou indisponibles") was classified "busy, it will come back"
    # on the experiments side, which waited indefinitely for a key closed for 7 h
    # (incident 2026-09-08).
    error_kind: str | None = None
    resume_at: datetime | None = None
    # Unix timestamp of earliest agent departure — lower score = higher priority
    priority_score: float = _FALLBACK_PRIORITY_SCORE

    # Telemetry metrics (filled in by the Worker)
    provider_used: str | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    # Pipeline timing segments measured inside the worker (P4_4, P5_1, P5_3, P5_4, P5_5)
    timing_p5: dict[str, Any] | None = None
    # Exact-prompt replay: the space that served the response, None if a provider served it.
    rejeu: str | None = None


# ---------------------------------------------------------------------------
# Structured output (LLM → Worker)
# ---------------------------------------------------------------------------

class OptionProbability(BaseModel):
    """Probability, for a trip option, that the persona picks it.

    The LLM scores **all** the proposed options (sum = 100) instead of choosing
    one: a random draw from this distribution produces the decision
    (cf. mobility_llm.mode_choice). `probability` is accepted as a % or as a
    fraction — downstream normalisation makes the scale irrelevant.
    """
    model_config = ConfigDict(extra="allow")

    index: int | None = None
    mode: str | None = None
    probability: float | None = None
    # Justification PER OPTION (2026-08-26). Previously carried by `AgentResponse.reason`,
    # a single sentence for the whole persona: it did not say why one option lost
    # against another. `extra="allow"` already tolerated it — it is declared so that
    # the contract lives in the model and not only in the prompt.
    reason: str | None = None

    @field_validator("probability", mode="before")
    @classmethod
    def coerce_probability(cls, v: Any) -> Any:
        """Tolerates "40 %" or "0,4": some models dress up their numbers."""
        if isinstance(v, str):
            cleaned = v.strip().replace("%", "").replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return v


class AgentResponse(BaseModel):
    """An element of the JSON array returned by the LLM."""
    model_config = ConfigDict(extra="allow")

    agent_id: str | int
    # Distribution over the proposed options — current format for the
    # itinary_multi_agent category.
    probabilities: list[OptionProbability] | None = None
    # `chosen_index`/`mode`: old format (the LLM chose one option). Kept so that a
    # response or cache produced before the switch can still be read.
    chosen_index: int | None = None
    mode: str | None = None
    reason: str | None = None
    summary: str | None = None

    @field_validator("agent_id")
    @classmethod
    def cast_agent_id_to_str(cls, v: Any) -> str:
        return str(v)


class LLMOutput(BaseModel):
    """Envelope validated on receipt of the LLM response."""
    agents: list[AgentResponse]


# ---------------------------------------------------------------------------
# API response (Worker → client via polling)
# ---------------------------------------------------------------------------

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    result: list[AgentResponse] | None = None
    error: str | None = None
    # Nature of the failure + window reopening (cf. Task.error_kind)
    error_kind: str | None = None
    resume_at: datetime | None = None
    # Metrics exposed to the client (useful for debug / monitoring)
    provider_used: str | None = None
    latency_ms: float | None = None
    timing_p5: dict[str, Any] | None = None
    rejeu: str | None = None


# ---------------------------------------------------------------------------
# Common interface between modules (adapters ↔ load_balancer ↔ worker)
# ---------------------------------------------------------------------------

class InternalMessage(BaseModel):
    """Normalised format passed to the adapters."""
    role: str   # "system" | "user" | "assistant"
    content: str | None = None
    trajectories: list[dict[str, Any]] = []
    history: list[str] = []


class InternalRequest(BaseModel):
    """What the Worker passes to the selected Adapter."""
    provider: str
    model: str | None = None          # If None, uses the provider's default
    messages: list[InternalMessage]
    response_schema: dict[str, Any]      # JSON Schema injected for Structured Output
    temperature: float = 0.7
    top_p: float | None = None           # not sent when None (cf. core.inference)
    max_tokens: int = 8192
    # Thinking depth, in thought tokens. None: nothing is requested, the provider applies
    # its default (which was the case for ALL calls until 2026-09-10 — `thoughtsTokenCount`
    # was counted without being controlled). 0 disables thinking, -1 lets the model
    # decide. Not every adapter can apply it: cf. `applique_reflexion`.
    thinking_budget: int | None = None
    # Thinking level — the CURRENT setting of the Gemini 3 API: `minimal`, `low`, `medium`,
    # `high` ("high" = maximum thinking). It replaces `thinking_budget`, which is still
    # accepted for backward compatibility but which the docs recommend dropping. The two
    # CANNOT coexist in a request: the provider returns 400. `core.inference` therefore
    # refuses it upstream, rather than discovering it in flight.
    thinking_level: str | None = None
