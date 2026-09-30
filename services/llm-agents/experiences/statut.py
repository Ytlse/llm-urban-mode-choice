"""Status of an experiment — `actif`, `archivee`, `invalide`.

Spec: `specs/hygiene-prompts-et-plateforme-experiences.md` §3.2 and §5.

This module NEVER deletes anything and moves no file. A status is a marker
placed next to the data (`data/experiences/<exp>/statut.json`): the runs, the traces,
the `scores.json` and the fingerprints stay where they are, readable and verifiable. What
changes is the default visibility in the registry and on the dashboard — and, for
`invalide`, the fact that the measurement must no longer be cited without its reason.

Three statuses, and nothing else:

- ``actif``    — default when no file exists. No marker to write for this case.
- ``archivee`` — of no current interest (definition never run, aborted run,
  naming duplicate). The data stays, the row leaves the default listing.
- ``invalide`` — the measurement is faulty for a named reason (typically an invalidated
  template, cf. `PromptManager` / `_invalidation`). The figure stays readable; its
  status is what says not to trust it.

Each transition is stacked in ``historique``: reactivating an experiment does not
erase the trace of its archiving, and one can always answer "why was this
experiment set aside, when, and by whom".
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIF = "actif"
ARCHIVEE = "archivee"
INVALIDE = "invalide"
STATUTS = (ACTIF, ARCHIVEE, INVALIDE)

F_STATUT = "statut.json"

# Statuses hidden by default in the registry and on the dashboard.
MASQUES_PAR_DEFAUT = (ARCHIVEE, INVALIDE)


class StatutInvalide(ValueError):
    """Unknown status, or experiment not found."""


def _maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def defaut() -> dict[str, Any]:
    """Status of an experiment without a marker: active, visible, no reason."""
    return {
        "statut": ACTIF,
        "motif": None,
        "le": None,
        "reference": None,
        "donnees": "conservees",
        "visible_par_defaut": True,
        "historique": [],
    }


def lire(dossier_experience: str | Path) -> dict[str, Any]:
    """Status of an experiment. Never raises: an unreadable marker counts as `actif`.

    A registry must stay listable even if a `statut.json` was hand-edited
    badly — the worst acceptable outcome is "show everything", not "show nothing".
    """
    p = Path(dossier_experience) / F_STATUT
    base = defaut()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return base
    if not isinstance(data, dict) or data.get("statut") not in STATUTS:
        return base
    base.update({k: v for k, v in data.items() if k in base})
    base["historique"] = list(data.get("historique") or [])
    base["visible_par_defaut"] = bool(
        data.get("visible_par_defaut", base["statut"] == ACTIF)
    )
    return base


def est_masquee(st: dict[str, Any]) -> bool:
    """True if the row must leave the default listing."""
    return st.get("statut") in MASQUES_PAR_DEFAUT and not st.get("visible_par_defaut")


def ecrire(
    dossier_experience: str | Path,
    statut: str,
    *,
    motif: str | None = None,
    reference: str | None = None,
    visible_par_defaut: bool | None = None,
) -> dict[str, Any]:
    """Sets a status. Atomic write, history stacked, no data touched.

    `motif` is MANDATORY for `archivee` and `invalide`: a marker without a reason is
    a marker nobody will be able to read back in six months.
    """
    dossier = Path(dossier_experience)
    if not (dossier / "experience.yaml").is_file():
        raise StatutInvalide(f"not an experiment: {dossier}")
    if statut not in STATUTS:
        raise StatutInvalide(
            f"statut {statut!r} inconnu (attendus : {', '.join(STATUTS)})"
        )
    if statut != ACTIF and not (motif or "").strip():
        raise StatutInvalide(f"un statut {statut!r} exige un motif")

    avant = lire(dossier)
    corps = {
        "statut": statut,
        "motif": (motif or None) if statut != ACTIF else None,
        "le": _maintenant(),
        "reference": reference,
        # States in black and white what the marker does not do: nothing is deleted.
        "donnees": "conservees",
        "visible_par_defaut": (
            (statut == ACTIF) if visible_par_defaut is None else bool(visible_par_defaut)
        ),
        "historique": [
            *avant["historique"],
            {
                "de": avant["statut"],
                "vers": statut,
                "le": _maintenant(),
                "motif": motif,
            },
        ],
    }
    cible = dossier / F_STATUT
    tmp = dossier / f"{F_STATUT}.tmp"
    tmp.write_text(
        json.dumps(corps, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    os.replace(tmp, cible)
    return corps


def statuts_par_experience(racine: str | Path) -> dict[str, dict[str, Any]]:
    """Status of each experiment under a root, keyed by directory name."""
    r = Path(racine)
    if not r.is_dir():
        return {}
    return {
        p.name: lire(p) for p in sorted(r.iterdir()) if (p / "experience.yaml").is_file()
    }


__all__ = [
    "ACTIF",
    "ARCHIVEE",
    "INVALIDE",
    "STATUTS",
    "F_STATUT",
    "StatutInvalide",
    "defaut",
    "ecrire",
    "est_masquee",
    "lire",
    "statuts_par_experience",
]
