"""Writing the measurements at the daily point — ticket 093, lot 3.

This module is the SEAM between the controller and the computation module
`scripts.analysis.mesures`, which is pure and knows neither Prometheus nor the simulation. The
same computation feeds the CSVs and the series: they cannot diverge.

WHEN
----
At the daily resume point, at 3 a.m. simulated, right after it is written — so on a closed
day, with empty short-term memory buffers, and with the state snapshot the point has just
produced.

FAIL-OPEN, NO DISCUSSION
------------------------
A failing measurement raises an `[ALARME]` and lets the run continue. Losing one day's
measurements costs one day of curve; bringing down a sixty-day run costs the run.

OFF BY DEFAULT
--------------
`agent.mesures_jour_enabled`. When off, this module reads nothing, writes nothing, and declares
no metric family — a family declared without data would read as a measurement of zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from settings import _run_artifacts_disabled, settings

_gauges: dict[str, Any] | None = None


def _gauges_prometheus() -> dict[str, Any]:
    """Declares the families on the FIRST write, never at import (cf. H1/H2)."""
    global _gauges
    if _gauges is None:
        from prometheus_client import Gauge

        from scripts.analysis.mesures.prometheus import _familles

        _gauges = _familles(lambda nom, aide, labels: Gauge(nom, aide, labels))
    return _gauges


def reinitialiser() -> None:
    """Forgets the declared gauges, and REMOVES them from the registry. Reserved for tests.

    Forgetting the reference without unregistering raised "Duplicated timeseries" on the
    next declaration — and since everything here is fail-open, the error was swallowed as an
    alarm: the run continued without any series, with nothing else saying so.
    """
    global _gauges
    if _gauges:
        from prometheus_client import REGISTRY

        for gauge in _gauges.values():
            try:
                REGISTRY.unregister(gauge)
            except KeyError:
                pass
    _gauges = None


def actif() -> bool:
    return bool(getattr(settings.agent, "mesures_jour_enabled", False)) and not (
        _run_artifacts_disabled()
    )


def ecrire_mesures_du_jour(workdir: Path | str | None = None) -> int:
    """Recomputes, rewrites the CSVs and sets the series. Returns the last closed day published (else 0).

    Never raises: the return value is 0 when nothing could be measured, and the alarm says why.
    """
    if not actif():
        return 0
    racine = Path(workdir or settings.workdir)
    try:
        from scripts.analysis.mesures.calcul import calculer
        from scripts.analysis.mesures.ecriture import ecrire
        from scripts.analysis.mesures.prometheus import publier

        mesures = calculer(racine)
        ecrire(mesures, racine / settings.app.mesures_dir)
        jour = publier(mesures, _gauges_prometheus())
        logger.info(
            f"[mesures] day {jour} closed — {len(mesures.choix_modal)} mode choice row(s), "
            f"{len(mesures.habitudes)} habit, {len(mesures.memoire)} memory; "
            f"{mesures.trajets_rejoues} replayed trip(s) and {mesures.rappels_rejoues} "
            f"replayed recall(s) discarded"
        )
        if not mesures.operations_tracees:
            logger.warning(
                "[mesures] operations_concept.jsonl absent : les colonnes d'opérations de "
                "concept restent VIDES. « Aucune contradiction » et « on ne mesure pas les "
                "contradictions » ne doivent pas se lire pareil — activez "
                "agent.trace_concepts_enabled."
            )
        return jour
    except Exception as exc:  # noqa: BLE001 — observation does not bring down the simulation
        logger.error(
            f"[ALARME] [mesures] cannot write for {racine} ({exc!r}) — the day "
            f"will have no row in the CSVs, and the series keep their previous value. "
            f"The run continues."
        )
        return 0
