"""Persistent FIFO queue of experiments (ticket 035, spec parallelisation_experiences).

Pure storage (no lock here): coordination — reading, promotion, removal — happens in
`reservations.py`, under the global lock shared with the reservation registry, so that
"test the free keys → reserve/dequeue" stays atomic (R6). An entry keeps the name of the
experiment, its **key set** computed at submission (disjointness pre-filter, R2b) and the
relaunch options; the actual promotion re-runs `lancer`, which recomputes and reserves.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiences.experience import dossier_experiences


def _chemin() -> Path:
    return dossier_experiences() / ".file.json"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def charger() -> list[dict]:
    """Queue entries, in FIFO submission order."""
    p = _chemin()
    if not p.is_file():
        return []
    try:
        contenu = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return contenu if isinstance(contenu, list) else []


def sauver(entrees: list[dict]) -> None:
    p = _chemin()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entrees, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def entree(exp: str, cles: set[str], args: dict) -> dict:
    """Builds a normalised queue entry."""
    return {
        "exp": exp,
        "cles": sorted(cles),
        "args": dict(args or {}),
        "soumis": _iso(),
    }


__all__ = ["charger", "entree", "sauver"]
