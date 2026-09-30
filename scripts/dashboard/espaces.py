"""Workspaces of the experiment registry — spec `espaces-de-travail-experiences.md`.

A workspace is a **view**: it restricts what the registry shows, it files nothing away (R7).
No experiment file is read, written or moved here — this module only knows
names, and a name only serves here to filter a table.

**It fails open.** File absent, unreadable, malformed, entry without a name: the module
logs and returns what it can, down to the empty list. The registry then stays on
« Toutes les expériences » and remains usable (R13). A convenience file must never
close the door of the dashboard.

An experiment's name is computed from its parameters (spec `nommage-canonique-experiences`)
and so cannot carry its phase. The `phase` label of each entry is what makes the
link, and it lives only in this view (R6a).
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FICHIER = Path(__file__).resolve().parent / "espaces_experiences.yaml"

#: The workspace that is not one: always at the top of the menu, always present (R2, R3).
TOUTES = "Toutes les expériences"


def _sain(nom: object) -> Optional[str]:
    """The experiment name if it is usable as a label, else None.

    The file is written by hand: it is fallible by accident more than by
    malice. We still refuse anything that could escape `data/experiences/`
    should a caller decide to turn it into a path — separators, parent references, control
    characters — because a guard here costs three lines and a forgotten guard is paid for
    elsewhere (spec, § Sécurité).
    """
    if not isinstance(nom, str):
        return None
    n = nom.strip()
    if not n or n in (".", ".."):
        return None
    if "/" in n or "\\" in n or "\x00" in n:
        return None
    if any(unicodedata.category(c) == "Cc" for c in n):
        return None
    return n


def _lire(chemin: Optional[Path] = None) -> dict:
    """The raw content of the file, or {} — never an exception towards the caller (R13)."""
    p = Path(chemin) if chemin is not None else FICHIER
    if not p.is_file():
        logger.info("[espaces] no workspaces file (%s): only « %s » will be offered", p, TOUTES)
        return {}
    try:
        contenu = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning("[espaces] unreadable file (%s): %s — falling back to « %s »", p, e, TOUTES)
        return {}
    if not isinstance(contenu, dict):
        logger.warning("[espaces] unexpected content in %s (%s, expected a dictionary) — falling back to « %s »",
                       p, type(contenu).__name__, TOUTES)
        return {}
    return contenu


def espaces(chemin: Optional[Path] = None) -> list[dict]:
    """The defined workspaces, in file order. Empty list if nothing is usable.

    An entry without a usable experiment name is ignored WITH a message: silencing it
    would make a registry row disappear without anything saying so.
    """
    brut = _lire(chemin)
    liste = brut.get("espaces")
    if not isinstance(liste, list):
        if liste is not None:
            logger.warning("[espaces] key `espaces` of type %s (expected a list) — ignored", type(liste).__name__)
        return []
    resultat: list[dict] = []
    vus: set[str] = set()
    for i, e in enumerate(liste):
        if not isinstance(e, dict):
            logger.warning("[espaces] entry %d ignored: %s instead of a workspace", i, type(e).__name__)
            continue
        nom = _sain(e.get("nom"))
        if not nom:
            logger.warning("[espaces] workspace %d ignored: name absent or invalid (%r)", i, e.get("nom"))
            continue
        if nom == TOUTES or nom in vus:
            logger.warning("[espaces] workspace %r ignored: name reserved or already defined", nom)
            continue
        entrees = []
        brutes = e.get("entrees")
        if not isinstance(brutes, list):
            if brutes is not None:
                logger.warning("[espaces] workspace %r: `entrees` of type %s (expected a list)",
                               nom, type(brutes).__name__)
            brutes = []
        for j, x in enumerate(brutes):
            if isinstance(x, str):           # short form: the name alone, without a phase
                x = {"experience": x}
            if not isinstance(x, dict):
                logger.warning("[espaces] workspace %r, entry %d ignored: %s", nom, j, type(x).__name__)
                continue
            exp = _sain(x.get("experience"))
            if not exp:
                logger.warning("[espaces] workspace %r, entry %d ignored: invalid experiment name (%r)",
                               nom, j, x.get("experience"))
                continue
            entrees.append({
                "experience": exp,
                "phase": (x.get("phase") or "").strip() or None,
                "optionnel": bool(x.get("optionnel")),
                "note": (x.get("note") or "").strip() or None,
            })
        resultat.append({"nom": nom, "note": (e.get("note") or "").strip() or None, "entrees": entrees})
        vus.add(nom)
    return resultat


def noms(chemin: Optional[Path] = None) -> list[str]:
    """The drop-down menu labels: « Toutes les expériences » first, always (R2)."""
    return [TOUTES, *(e["nom"] for e in espaces(chemin))]


def espace(nom: Optional[str], chemin: Optional[Path] = None) -> Optional[dict]:
    """The workspace of that name, or None — including for `TOUTES`, which has no definition."""
    if not nom or nom == TOUTES:
        return None
    for e in espaces(chemin):
        if e["nom"] == nom:
            return e
    return None


def entrees(nom: Optional[str], chemin: Optional[Path] = None) -> list[dict]:
    """The entries of this workspace, in file order. Empty if the workspace is unknown."""
    e = espace(nom, chemin)
    return list(e["entrees"]) if e else []


def index(nom: Optional[str], chemin: Optional[Path] = None) -> dict[str, dict]:
    """The entries indexed by experiment name — what the registry queries per row."""
    return {x["experience"]: x for x in entrees(nom, chemin)}


def actif_valide(nom: Optional[str], chemin: Optional[Path] = None) -> str:
    """The workspace name to actually use, `TOUTES` if this one no longer exists (R14)."""
    if not nom or nom == TOUTES:
        return TOUTES
    if espace(nom, chemin) is None:
        logger.warning("[espaces] the remembered workspace %r no longer exists — back to « %s »", nom, TOUTES)
        return TOUTES
    return nom


def filtrer(lignes: list[dict], nom: Optional[str], chemin: Optional[Path] = None,
            cle: str = "experience") -> list[dict]:
    """The registry rows that this workspace lets through (R4). `TOUTES` filters nothing (R3)."""
    if not nom or nom == TOUTES:
        return list(lignes)
    dedans = index(nom, chemin)
    return [l for l in lignes if l.get(cle) in dedans]


def manquantes(presentes: set[str] | list[str], nom: Optional[str],
               chemin: Optional[Path] = None) -> list[str]:
    """The experiments cited by the workspace and absent from the disk (R9).

    This is NOT an error: the workspace fills up before the directories, and the list serves to
    announce it. It also works as a discreet alarm the day a directory disappears.
    """
    vues = set(presentes)
    return [x["experience"] for x in entrees(nom, chemin) if x["experience"] not in vues]
