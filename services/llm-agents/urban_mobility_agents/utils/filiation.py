"""Child runs — ticket 095, lot D.

THE PROBLEM, IN ONE NUMBER
--------------------------
The first fourteen days of the paired campaign of ticket 077 are identical in all three
arms: **44 decisions, 0 difference**. They are paid for three times. With eleven arms planned,
that is about a third of the cost of each additional arm thrown away.

THE PRINCIPLE
-------------
A PARENT run plays the common base up to the eve of the shock and freezes. Each arm starts
from that resume point, in ITS directory, with ITS settings. The mechanisms already exist:
resume points (ticket 075), replay with frozen memory, run identity (ticket 091). This
module only adds the parentage and its safeguards.

WHAT A CHILD IS ALLOWED TO CHANGE
---------------------------------
Nothing, except what it DECLARES. `CHAMPS_LIBRES` names the identity fields this arm varies
— typically `choc` and `mode_fenetre_changements`. Any other difference makes startup refuse,
naming the field, exactly like a misnamed resume.

The meaning of the rule: a child inherits its parent's MEMORY. If the population, the model
or the seeds differ, that memory describes another experiment than the one the child is about
to play — and nothing in the outputs would say so.

⚠ **No escape hatch.** As for the run identity, debugging cases are handled by hand, outside
the code: a workaround shipped in the product would end up being used for measurement.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from loguru import logger
from urban_mobility_agents.utils import identite_run, reprise

# Environment variables of the parentage.
ENV_PARENT = "RUN_PARENT"
ENV_CHAMPS_LIBRES = "CHAMPS_LIBRES"


class FiliationRefusee(RuntimeError):
    """The child cannot inherit from this parent."""


def parent_declare() -> str:
    """The named parent run, or an empty string."""
    return (os.environ.get(ENV_PARENT) or "").strip()


def champs_libres() -> tuple[str, ...]:
    """The identity fields this child declares it varies.

    An unknown field is REFUSED, not ignored: `fenetre_changement` instead of
    `fenetre_changements_jours` would let through a difference one believed declared.
    """
    brut = (os.environ.get(ENV_CHAMPS_LIBRES) or "").strip()
    if not brut:
        return ()
    demandes = tuple(c.strip() for c in brut.split(",") if c.strip())
    inconnus = [c for c in demandes if c not in identite_run.LIBELLES]
    if inconnus:
        raise FiliationRefusee(
            f"{ENV_CHAMPS_LIBRES} names identity fields that do not exist: "
            f"{inconnus}. Known fields: {sorted(identite_run.LIBELLES)}."
        )
    return demandes


def repertoire_parent(nom: str, racine: Path) -> Path:
    """The parent's directory: a path as is, otherwise a name under the runs root."""
    chemin = Path(nom)
    if chemin.is_dir():
        return chemin
    for candidat in (racine / nom, racine.parent / nom):
        if candidat.is_dir():
            return candidat
    raise FiliationRefusee(
        f"parent run not found: {nom!r} (looked up as is, then under {racine} and "
        f"{racine.parent})"
    )


def ecarts_interdits(
    parent: dict, courante: dict, libres: tuple[str, ...]
) -> list[str]:
    """The identity differences this child has NOT declared. Empty list = parentage allowed."""
    return [
        ecart
        for champ, ecart in _ecarts_par_champ(parent, courante).items()
        if champ not in libres
    ]


def _ecarts_par_champ(parent: dict, courante: dict) -> dict[str, str]:
    ecarts: dict[str, str] = {}
    for champ, libelle in identite_run.LIBELLES.items():
        a = parent.get(champ, identite_run.ABSENT)
        b = courante.get(champ, identite_run.ABSENT)
        if a is identite_run.ABSENT:
            ecarts[champ] = f"{libelle} : absent du run parent, {b!r} chez l'enfant"
        elif a != b:
            ecarts[champ] = f"{libelle} : {a!r} chez le parent, {b!r} chez l'enfant"
    return ecarts


def amorcer(workdir: Path, parent_dir: Path, identite_courante: dict) -> dict:
    """Checks the parentage, then copies the parent's resume point into the child.

    Returns the description of the copied point. Raises `FiliationRefusee` rather than start an
    arm that would inherit a memory produced under other settings.
    """
    libres = champs_libres()
    identite_parent = identite_run.lire(parent_dir)
    if identite_parent is None:
        raise FiliationRefusee(
            f"the parent run {parent_dir.name} carries no identity "
            f"({identite_run.FICHIER} missing or unreadable): impossible to tell which "
            f"experiment its memory comes from."
        )

    interdits = ecarts_interdits(identite_parent, identite_courante, libres)
    if interdits:
        raise FiliationRefusee(
            f"the child differs from its parent {parent_dir.name} on fields NOT declared "
            f"in {ENV_CHAMPS_LIBRES}:\n  - " + "\n  - ".join(interdits)
            + f"\nFields declared free: {list(libres) or 'aucun'}."
        )

    trouve = reprise.dernier_point(parent_dir)
    if trouve is None:
        raise FiliationRefusee(
            f"the parent run {parent_dir.name} has NO valid resume point: there is "
            f"nothing to inherit. A parent plays up to the eve of the shock, then freezes."
        )
    source, meta = trouve

    destination = Path(workdir) / reprise.POINTS / source.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    logger.info(
        f"[filiation] {Path(workdir).name} inherits from {parent_dir.name} at simulated day "
        f"{meta.get('jour_simule')} ({meta.get('horodatage_simule')}) — free fields: "
        f"{list(libres) or 'aucun'}"
    )
    return meta


def decisions(workdir: Path, jusqu_a: float | None = None) -> dict[tuple[str, str, float], str]:
    """The traced decisions of a run, indexed by (person, activity, instant).

    This is the ticket 090 trace, already written during the normal life of any run — no
    new instrumentation is needed to compare a child with its parent.
    """
    fichier = Path(workdir) / "decisions_rejeu.jsonl"
    vues: dict[tuple[str, str, float], str] = {}
    if not fichier.is_file():
        return vues
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            trace = json.loads(ligne)
        except ValueError:
            continue
        instant = float(trace.get("instant", 0.0))
        if jusqu_a is not None and instant > jusqu_a:
            continue
        cle = (str(trace.get("personne")), str(trace.get("activite")), instant)
        vues[cle] = str(trace.get("code_plan"))
    return vues


def ecarts_de_reproduction(
    parent_dir: Path, workdir: Path, jusqu_a: float | None
) -> list[str]:
    """The decisions that differ between parent and child on the COMMON days.

    Empty list = the child did replay its parent's base. A single difference and the saving
    is a lie: the two arms no longer share the baseline one believes they were given, and
    the comparison is between two different histories.
    """
    du_parent = decisions(parent_dir, jusqu_a)
    de_l_enfant = decisions(workdir, jusqu_a)
    ecarts: list[str] = []
    for cle, code in sorted(du_parent.items(), key=lambda kv: kv[0][2]):
        if cle not in de_l_enfant:
            ecarts.append(f"{cle[0]} · {cle[1]} : décidée chez le parent, absente chez l'enfant")
        elif de_l_enfant[cle] != code:
            ecarts.append(
                f"{cle[0]} · {cle[1]} : {code!r} chez le parent, {de_l_enfant[cle]!r} chez l'enfant"
            )
    return ecarts


def verifier_reproduction(parent_dir: Path, workdir: Path, jusqu_a: float | None) -> None:
    """Logs the reproduction check. A divergence raises a quantified `[ALARME]`."""
    ecarts = ecarts_de_reproduction(parent_dir, workdir, jusqu_a)
    communes = len(decisions(parent_dir, jusqu_a))
    if not ecarts:
        logger.info(
            f"[filiation] replay conforms: {communes} decision(s) of the parent "
            f"{parent_dir.name} reproduced identically."
        )
        return
    logger.error(
        f"[ALARME] [filiation] {len(ecarts)}/{communes} decision(s) of the common base DIFFER "
        f"between {parent_dir.name} and {Path(workdir).name} — the two arms no longer share the "
        f"baseline one believes they were given:\n  - " + "\n  - ".join(ecarts[:10])
        + ("\n  - …" if len(ecarts) > 10 else "")
    )
