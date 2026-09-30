"""Structured trace of concept operations — ticket 093, lot 3.

WHY ONE MORE SOURCE
-------------------
The four operations — created, confirmed, refined, contradicted — are already written, but only
in the readable journal `memoires/<agent>.md`, whose header itself says it is not a source of
measurement: the figures of a paper come from the memory metadata and from moves.csv, not from
a text formatted for reading. Honouring that requires a structured source,
and this file is it.

What this curve carries is the very mechanism of the study. Measured on the 2026-09-16 run, five
agents, ten days: **38 created, 61 confirmed, only 1 contradicted** — the single contradiction being
Corinne's, on the day of the shock. Without an injected event, the memory never revised anything. A
measurement that does not tell "no contradiction" from "contradictions are not counted"
would leave this result undecidable.

WHAT THIS MODULE GUARANTEES
---------------------------
1. **Nothing when switched off.** `agent.trace_concepts_enabled` is false by default.
2. **Nothing during replay.** A replayed operation is not an operation: the memory is
   frozen, it decides nothing. Counting it would inflate the curve accordingly — a defect already
   seen in `trace_rappel`, which lacks this safeguard and where 81 lines out of 196 are replay.
3. **Never an exception towards the caller.** A trace that brings down a sixty-day run
   would be worse than no trace at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger
from settings import _run_artifacts_disabled, settings
from urban_mobility_agents.utils.reprise import gel_actif

# The vocabulary of the four operations, as the journal and the measurements know it. An
# operation outside this list is traced anyway: it is counted separately downstream,
# so that a vocabulary changing silently is not read as a change in behaviour.
OPERATIONS = ("créé", "confirmé", "précisé", "contredit")


def tracer_operation(
    person_id: str,
    operation: str,
    quand: datetime | int | None,
    *,
    doc_id: str = "",
) -> None:
    """Writes one JSONL line per concept operation. Never raises.

    `quand` is the SIMULATED instant of the operation. `sim_day` is derived from it in UTC, as
    `trace_rappel` does: the two traces must match day by day, and two date conventions
    would make their overlay off by one day without anything saying so.
    """
    try:
        if not getattr(settings.agent, "trace_concepts_enabled", False):
            return
        if _run_artifacts_disabled():
            # A test that imports the memory chain does not open a run and does not drop
            # files into the directory of a running simulation (lesson of ticket 075).
            return
        if gel_actif():
            return
        instant = _epoch(quand)
        entree = {
            "sim_ts": instant,
            "sim_day": datetime.fromtimestamp(instant, tz=timezone.utc).strftime("%Y-%m-%d")
            if instant is not None
            else None,
            "person_id": str(person_id),
            "operation": str(operation),
            "doc_id": str(doc_id or ""),
        }
        with open(settings.app.trace_concepts_file, "a", encoding="utf-8") as flux:
            flux.write(json.dumps(entree, ensure_ascii=False, default=str) + "\n")
    except Exception as err:  # noqa: BLE001 — a trace never brings down a consolidation
        logger.warning(f"[trace_concepts] trace not written for {person_id} ({err})")


def _epoch(quand: datetime | int | None) -> int | None:
    if quand is None:
        return None
    if isinstance(quand, datetime):
        return int(quand.timestamp())
    try:
        return int(quand)
    except (TypeError, ValueError):
        return None
