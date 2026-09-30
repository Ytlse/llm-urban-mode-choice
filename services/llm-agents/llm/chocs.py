"""Declared shocks — ADAPTER to `llm/evenements/` (ticket 100, lot 1).

THIS MODULE NO LONGER HOLDS ANY LOGIC. Everything it carried — dataclasses, content guards,
exposure rules, cadence, counters, run day, trace, alarms — now lives in
`llm/evenements/`, where a press article takes the same path as an experienced shock.

WHY AN ADAPTER, AND WHY IT MUST GO
----------------------------------
Ticket 079 is delivered, measured, and its figures are in § 7.2 of the manuscript. A migration
that broke its 36 tests, or changed by a single character what `chocs.jsonl` contains,
would make those figures unverifiable. This adapter exists so the migration can be proven: the
079 tests run **without their expected values moving**, against the new code.

⚠ It goes away in **lot 6 of ticket 100**, after a short non-regression run on
`c6_voiture_suspecte` — the golden test replays a trace, not a run, and says nothing of the
GAMA path. The compatibility aliases go with it: `choc_id`, `vecu`, `RegistreChocs.choc`,
the `choc_id`/`vecu` keys of the trace, and the `choc.yaml`/`chocs.jsonl` links of the run
directory. Write no new code against this module.
"""

from __future__ import annotations

from llm.evenements import (  # noqa: F401 — façade de compatibilité
    CADENCES,
    CADENCE_PAR_DEFAUT,
    REGLES_EXPOSITION,
    CompteursJournee,
    incident_reseau_a_une_source,
    initialiser,
    registre,
    reinitialiser,
)
from llm.evenements.declaration import (  # noqa: F401
    EffetPhysique,
    Evenement,
    EvenementApplique,
    Exposition,
    JourDEvenement,
    RefusDEvenement,
    charger,
)
from llm.evenements.gardes import (  # noqa: F401
    MARQUEURS_CONSIGNE,
    MARQUEURS_INTENTION,
    MARQUEURS_VERDICT,
)
from llm.evenements.registre import RegistreEvenements

# ── Ticket 079 names ────────────────────────────────────────────────────────────────────────
# ALIASES, not subclasses: the 079 tests replace `RegistreChocs.jour_du_run` on the CLASS
# to play a given day without a simulator. A subclass would receive the fix without the
# real class seeing it, and vice versa — the test would pass while no longer checking
# anything, which is worse than seeing it fail.
RefusDeChoc = RefusDEvenement
RegistreChocs = RegistreEvenements
Choc = Evenement
JourDeChoc = JourDEvenement
ChocApplique = EvenementApplique

__all__ = [
    "CADENCES",
    "CADENCE_PAR_DEFAUT",
    "REGLES_EXPOSITION",
    "Choc",
    "ChocApplique",
    "CompteursJournee",
    "Exposition",
    "JourDeChoc",
    "RefusDeChoc",
    "RegistreChocs",
    "charger",
    "incident_reseau_a_une_source",
    "initialiser",
    "registre",
    "reinitialiser",
]
