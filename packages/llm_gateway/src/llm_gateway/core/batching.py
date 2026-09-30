"""
core/batching.py — Pure logic for grouping requests into batches.

The batch key guarantees that only perfectly compatible tasks are
merged: same category, same parameters, same forced provider,
same TPM constraint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from llm_gateway.core.models import _FALLBACK_PRIORITY_SCORE, LLMRequest

#: Batch identifier forged by the worker: `batch_<8 hex>_<number of agents>`.
_MOTIF_LOT = re.compile(r"^batch_[0-9a-f]+_(\d+)$")


def compute_batch_key(request: LLMRequest) -> str:
    """Batch key: MD5(category + parameters + forced provider + min_tpm + allowed instances
    + origin).

    ⚠ **Every routing constraint must go in here** (ticket 084). A batch is served by ONE
    instance: two requests with different restrictions merged into the same batch would have
    one served by a provider it excluded, with no trace saying so. This is the only guard
    against that mix, and it is silent when forgotten.

    Allowed instances go in SORTED and deduplicated: it is a set, not a sequence, and two
    declarations of the same set in a different order must share their batch.

    The origin goes in too: a batch carries only ONE origin in the exchange log, and merging
    two clients would attribute it entirely to the first. `None` (a client that does not set
    it) keeps the historical key of batches without origin.
    """
    params_str = json.dumps(request.parameters, sort_keys=True)
    admises = (
        ",".join(sorted(set(request.instances_admises)))
        if request.instances_admises
        else None
    )
    hash_str = hashlib.md5(
        f"{request.category}:{params_str}:{request.force_provider}:"
        f"{request.min_tpm_required}:{admises}:{request.origine}".encode()
    ).hexdigest()
    return f"{request.category}:{hash_str}"


def compute_priority_score(scores: Iterable[float | None]) -> float:
    """Priority score of a batch = smallest score given — lower = more urgent.

    Scores come from the category (``CategorySpec.priority``); the gateway does not know
    what they measure (a departure timestamp for mobility). No score → fallback.
    """
    known = [s for s in scores if s is not None]
    return float(min(known)) if known else _FALLBACK_PRIORITY_SCORE


def compute_batch_max_agents(
    *,
    tpm_limit: int | None,
    rpm_limit: int,
    max_tokens_per_request: int | None,
    tokens_per_agent: int,
    plafond: int,
) -> int:
    """How many agents an instance can carry in ONE provider request.

    Extracted from `GatewaySettings._resolve`, which now calls it: this is the only
    definition of the batch ceiling. An experiment's cost estimate
    (`experiences/lots.py`) reuses it to convert trips into requests, and a formula
    copied over there would have drifted silently the day this one changes.

    `tokens_per_agent` = `assumed_prompt_tokens + assumed_output_tokens`; `plafond` =
    `max_batch_agents`. Without `tpm_limit`, the RPM serves as the bound (an instance with no
    declared token limit must not end up capped by a division by zero); without
    `max_tokens_per_request`, only the absolute ceiling applies.
    """
    tpm_bound = int(tpm_limit / tokens_per_agent) if tpm_limit else rpm_limit
    req_bound = (
        int(max_tokens_per_request / tokens_per_agent)
        if max_tokens_per_request
        else plafond
    )
    return max(1, min(tpm_bound, req_bound, rpm_limit, plafond))


def taille_de_lot(task_id: str | None) -> int | None:
    """Number of agents merged into a request, read from its batch identifier.

    The worker names each request `batch_<8 hex>_<n>` (cf. `worker/task_worker.py`) and
    `llm_exchanges.jsonl` keeps a trace of it: this is the only place where the exchange log
    says how many agents travelled together. Without this reading, a line's
    `tokens_in`/`tokens_out` — which are the BATCH's — pass for an agent's, and any estimate
    built on them is multiplied by the grouping factor.

    Returns `None` if the identifier does not follow the pattern: better to ignore the line
    than to count it as one agent, which would be exactly the error being fixed.
    """
    if not task_id:
        return None
    m = _MOTIF_LOT.match(str(task_id))
    if not m:
        return None
    n = int(m.group(1))
    return n if n > 0 else None
