"""Summary of the exact-prompt replay of a memory A/B: did the control really replay the treated arm?

The replay (`llm_gateway/core/rejeu_ab.py`) serves the control the answer the treated arm got,
as long as the prompt is the same word for word. Before the event, nothing separates the two arms:
every call of the control should therefore be served by replay. A call paid before the event says
that the arms diverged for a reason other than the event — and that is what this summary shows.

Read-only: the control's exchange log, and the date of the first injection read
from the treated arm's events.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PREFIXE_REJEU = "rejeu_ab:"


def lire_echanges(chemin: Path, origine: str | None = None) -> list[dict]:
    """Reads the exchanges and, if asked, isolates one run from the worker's shared log."""
    texte = chemin.read_text(encoding="utf-8")
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        while i < len(texte) and texte[i] in " \n\r\t":
            i += 1
        if i >= len(texte):
            return [e for e in out if origine is None or e.get("origine") == origine]
        obj, i = dec.raw_decode(texte, i)
        out.append(obj)


def date_premiere_injection(evenements: Path) -> str | None:
    """Simulated day (YYYY-MM-DD) of the treated arm's first injection, None if there is none."""
    instant = instant_premiere_injection(evenements)
    return instant[:10] if instant else None


def instant_premiere_injection(evenements: Path) -> str | None:
    """First simulated timestamp of the injection, kept to the second."""
    if not evenements.is_file():
        return None
    dates = []
    for ligne in evenements.read_text(encoding="utf-8").splitlines():
        try:
            h = json.loads(ligne).get("horodatage_simule")
        except ValueError:
            continue
        if h:
            dates.append(str(h))
    return min(dates) if dates else None


# Written by `llm/evenements/registre.py` next to `evenements.jsonl` (2026-09-28).
FICHIER_PREMIERS_SERVICES = "premiers_services.jsonl"


def instant_premier_service(evenements: Path) -> str | None:
    """First simulated instant at which the event's line entered a prompt of the treated arm.

    None if the file is missing (run from before 2026-09-28, or lived event, which serves no
    line) or has no readable line.
    """
    chemin = evenements.parent / FICHIER_PREMIERS_SERVICES
    if not chemin.is_file():
        return None
    instants = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        try:
            h = json.loads(ligne).get("instant_simule")
        except ValueError:
            continue
        if h:
            instants.append(str(h))
    return min(instants) if instants else None


def instant_debut_traitement(evenements: Path) -> tuple[str | None, str | None]:
    """Where the common prefix ends, and where that instant comes from: (instant, source).

    It is the first prompt of the treated arm that carries the event, not the injection timestamp.
    A next-day decision is computed the day before (a13 v5: the article of 27/03 entered a
    decision on 26/03 at 07:30, 16.5 h before the 00:00 injection). The earlier of the two
    is authoritative. `source` is `service`, `injection`, or None if there is neither.
    """
    candidats = [
        (instant, source)
        for instant, source in (
            (instant_premier_service(evenements), "service"),
            (instant_premiere_injection(evenements), "injection"),
        )
        if instant
    ]
    return min(candidats) if candidats else (None, None)


def bilan(echanges: list[dict], date_evenement: str | None) -> dict[str, Any]:
    """Control calls served by replay or paid, per category, and those paid BEFORE the event.

    An exchange without a simulated day (category without a priority timestamp) counts in the
    totals, not in "before": it cannot be dated, so it is not blamed.
    """
    par_categorie: dict[str, dict[str, int]] = defaultdict(
        lambda: {"servis": 0, "payes": 0}
    )
    payes_avant: list[dict[str, Any]] = []
    for e in echanges:
        cat = str(e.get("category") or "?")
        rejoue = str(e.get("provider") or "").startswith(PREFIXE_REJEU)
        par_categorie[cat]["servis" if rejoue else "payes"] += 1
        jour = e.get("sim_day")
        if not rejoue and date_evenement and jour and jour < date_evenement:
            payes_avant.append(
                {
                    "categorie": cat,
                    "jour": jour,
                    "agents": sorted(
                        str(r.get("agent_id", ""))
                        for r in e.get("response") or []
                        if isinstance(r, dict)
                    ),
                }
            )
    servis = sum(c["servis"] for c in par_categorie.values())
    payes = sum(c["payes"] for c in par_categorie.values())
    return {
        "date_evenement": date_evenement,
        "servis": servis,
        "payes": payes,
        "part_servie": round(servis / (servis + payes), 4) if servis + payes else None,
        "par_categorie": dict(par_categorie),
        "payes_avant_evenement": len(payes_avant),
        "premiers_payes_avant": sorted(payes_avant, key=lambda p: p["jour"])[:10],
    }


__all__ = [
    "FICHIER_PREMIERS_SERVICES", "PREFIXE_REJEU", "bilan", "date_premiere_injection",
    "instant_debut_traitement", "instant_premier_service", "instant_premiere_injection",
    "lire_echanges",
]
