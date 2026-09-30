"""Trace of the memories SERVED to a decision — ticket 077, lots E1 and D2.

The thirty-day run of ticket 075 left a question unanswered: **does memory
change decisions?** The journal said "10 souvenir(s) servi(s)" and their opening words,
without the score that selected them or the pool they came from. Reconstructing that afterwards
is impossible: the scores are written nowhere, and the buffer has changed.

Two things live here, because they are read at the same place and at the same instant.

**E1 — the trace.** One JSONL line per decision served by memory: the agent, the instant,
and the top-K with, for each memory, its identifier, its type, its pool of origin, its
composite score and its rank.

**D2 — concentration.** Recall reinforces itself: a served memory sees its strength
increase, hence its probability of being served again. Measured on the 075 run — 128 recalls
out of 133 serve ten memories, and always the same ones; the most served one is served fifty-five
times. These are the reflections of the first days, hence the most contaminated by the observations
that lot B corrects. Lot B removes the main cause; this measurement says whether that was enough.

⚠ **This module corrects nothing.** It measures, and raises an alarm. Changing the
reinforcement rule would mean changing a memory rule, which ticket 077 forbids itself.

Off by default (`agent.trace_rappel_enabled`): with a thousand agents, one line per decision and
per memory would make a file nobody would open.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from datetime import datetime, timezone

from loguru import logger
from settings import settings

# Observation window of the concentration, in number of recalls. Same order of magnitude as
# the pool window of lot 2: a strong concentration over one recall means nothing,
# the agent has only a handful of memories on the first day.
FENETRE_CONCENTRATION = 200

# Alarm thresholds. The high one fires, the low one re-arms: without hysteresis, a value
# oscillating around a single threshold would flood the log. Both are SHARES of the total served.
SEUIL_CONCENTRATION_HAUT = 0.80
SEUIL_CONCENTRATION_BAS = 0.65
# Number of memories whose cumulative share is examined. Ten, like the top-K served to the
# model: the question is "do the same ten always come back".
TETE_CONCENTRATION = 10


class _EtatConcentration:
    """Per-agent counter, bounded. Nothing grows without end here: this is a long process."""

    def __init__(self) -> None:
        self.servis: Counter[str] = Counter()
        self.fenetre: deque[str] = deque()
        self.en_alarme: bool = False

    def noter(self, doc_ids: list[str], candidats: int = 0) -> float | None:
        """Adds a recall and returns the concentration, or `None` if it is meaningless.

        ⚠ **A recall without choice does not count.** Defect found while running this module
        on 2026-09-15: the alarm fired at 100 % for four agents on the fifth simulated
        day. It told the truth and measured nothing — an agent that owns eleven memories and
        serves ten of them necessarily serves all of them, and the "concentration" only measured
        the SIZE of its memory. This is the recurring pattern of the repository, inverted: the
        absence of choice produced here the most alarming score instead of the most perfect one.

        A recall therefore enters the window only if the pool offered **strictly more**
        candidates than were served. Only then is always serving the same ones
        a selection, and not a fatality.
        """
        if candidats <= len(doc_ids):
            return None
        for doc_id in doc_ids:
            self.servis[doc_id] += 1
            self.fenetre.append(doc_id)
        while len(self.fenetre) > FENETRE_CONCENTRATION * TETE_CONCENTRATION:
            vieux = self.fenetre.popleft()
            self.servis[vieux] -= 1
            if self.servis[vieux] <= 0:
                del self.servis[vieux]
        total = sum(self.servis.values())
        if total < TETE_CONCENTRATION * 2:
            # Too little material: a concentration of 1.0 over twelve recalls is the
            # NORMAL behaviour of an agent just starting, not an anomaly.
            return None
        # Second safeguard, on the POOL and not on the number of recalls: as long as the agent
        # does not have clearly more memories than the top-K, "the same ten"
        # is not a choice.
        if len(self.servis) < TETE_CONCENTRATION * 2:
            return None
        tete = sum(n for _, n in self.servis.most_common(TETE_CONCENTRATION))
        return tete / total


_ETATS: dict[str, _EtatConcentration] = {}


def reinitialiser() -> None:
    """Resets the counters. For tests, and for tests only."""
    _ETATS.clear()


def _vivier(meta: dict) -> str:
    return str((meta or {}).get("vivier") or "?")


def tracer_rappel(
    person_id: str,
    query_at: int | None,
    servis: list,
    scores_par_doc: dict[str, float],
    candidats: int,
) -> None:
    """Writes the top-K trace and updates the concentration. Never raises.

    FAIL-OPEN by construction: a failing trace must not cost a decision.
    It is an observation output; it is on the critical path only by accident.
    """
    try:
        doc_ids = [
            str((r.metadata or {}).get("doc_id") or "") for r in servis
        ]
        etat = _ETATS.setdefault(person_id, _EtatConcentration())
        concentration = etat.noter([d for d in doc_ids if d], candidats)

        if concentration is not None:
            if concentration >= SEUIL_CONCENTRATION_HAUT and not etat.en_alarme:
                etat.en_alarme = True
                logger.error(
                    f"[ALARME] concentration des rappels à {concentration:.0%} pour "
                    f"{person_id} — les {TETE_CONCENTRATION} souvenirs les plus servis "
                    f"occupent la quasi-totalité du top-K sur les {FENETRE_CONCENTRATION} "
                    f"derniers rappels. Le rappel s'auto-renforce : ce que l'agent a appris "
                    f"tôt évince ce qu'il apprend ensuite."
                )
            elif concentration <= SEUIL_CONCENTRATION_BAS and etat.en_alarme:
                etat.en_alarme = False
                logger.info(
                    f"[viviers] recall concentration back down to "
                    f"{concentration:.0%} for {person_id} — alarm re-armed"
                )

        if not settings.agent.trace_rappel_enabled:
            return

        entree = {
            "sim_ts": query_at,
            "sim_day": datetime.fromtimestamp(query_at, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            if query_at
            else None,
            "person_id": str(person_id),
            "candidats": int(candidats),
            "concentration": round(concentration, 4)
            if concentration is not None
            else None,
            "servis": [
                {
                    "rang": rang,
                    "doc_id": (r.metadata or {}).get("doc_id"),
                    "type": (r.metadata or {}).get("memory_type"),
                    "vivier": _vivier(r.metadata),
                    "score": round(
                        float(
                            scores_par_doc.get(
                                str((r.metadata or {}).get("doc_id") or ""), 0.0
                            )
                        ),
                        4,
                    ),
                }
                for rang, r in enumerate(servis)
            ],
        }
        with open(settings.app.trace_rappel_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False, default=str) + "\n")
    except Exception as err:  # noqa: BLE001 — a trace never brings down a decision
        logger.warning(f"[trace_rappel] trace not written for {person_id} ({err})")
