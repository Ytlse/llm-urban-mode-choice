"""Who was exposed to a run's event, and when — read from the run, never assumed.

Analysis of 2026-09-25. `modal_variation_rate.py` split every run into four phases fixed on
a shock at days 8-9 ("Choc d'avarie moteur J8-9"), and the treated arm of `2026-09-24_17_50`
— an article read on day 11, drawn in a [9, 13] window per household — was read under a title
that described another protocol. This module gives each agent ITS window, from what the
run actually wrote:

- `evenements.jsonl` (or `chocs.jsonl` for a 079 run): the real exposures, dated;
- the run population (`population_<N>.json`): the households, so that a reader's co-residents
  are filed with them and not with the agents outside any exposed household;
- the declaration (`evenement.yaml` or `choc.yaml`): the identifier and the label, for the
  titles.

No runtime dependency: this module is read without `services/llm-agents` on the path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass(frozen=True)
class Fenetre:
    """An agent's exposure window: that of its household, for the run's event."""

    evenement_id: str
    household_id: str | None
    role: str          # "expose" (read it, went through it) | "co_resident"
    premier_jour: str  # simulated day "YYYY-MM-DD"
    dernier_jour: str


def lire_evenements(chemin_run: Path) -> list[dict[str, Any]]:
    """The exposure lines, `evenements.jsonl` first. Never both files."""
    for nom in ("evenements.jsonl", "chocs.jsonl"):
        fichier = Path(chemin_run) / nom
        if not fichier.is_file():
            continue
        lignes = []
        for brute in fichier.read_text(encoding="utf-8").splitlines():
            brute = brute.strip()
            if not brute:
                continue
            try:
                objet = json.loads(brute)
            except json.JSONDecodeError:
                continue
            if isinstance(objet, dict):
                lignes.append(objet)
        if lignes:
            return lignes
    return []


def menages(chemin_run: Path) -> dict[str, str]:
    """person_id → household_id, from the run population. Empty if it is missing."""
    for fichier in sorted(Path(chemin_run).glob("population_*.json")):
        if "_checkpoint_" in fichier.name:
            continue
        try:
            personnes = json.loads(fichier.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(personnes, list):
            continue
        return {
            str(p.get("person_id")): str((p.get("household") or {}).get("id"))
            for p in personnes
            if isinstance(p, dict) and (p.get("household") or {}).get("id")
        }
    return {}


def declaration(chemin_run: Path) -> dict[str, Any]:
    """`evenement.yaml`, otherwise `choc.yaml`. Empty if the run declares none."""
    for nom in ("evenement.yaml", "choc.yaml"):
        fichier = Path(chemin_run) / nom
        if not fichier.is_file():
            continue
        try:
            contenu = yaml.safe_load(fichier.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(contenu, dict):
            return contenu
    return {}


def libelle(chemin_run: Path) -> str | None:
    """E.g. "a09_vent_autan — Autan gales, parks closed", or `None` if undeclared."""
    decl = declaration(chemin_run)
    identifiant = decl.get("evenement") or decl.get("choc")
    if not identifiant:
        return None
    texte = decl.get("libelle")
    return f"{identifiant} — {texte}" if texte else str(identifiant)


def fenetres(chemin_run: Path,
             journee_de: Callable[[dict], str | None]) -> dict[str, Fenetre]:
    """person_id → its window. An agent absent from the dictionary is in no exposed household.

    `journee_de(ligne)` dates an exposure (the one of `evenement_par_jour.csv`, see
    `calcul.journee_de_l_exposition`). The `origine: entendu` lines of ticket 111 are not
    exposures: the informed member is a co-resident, and is filed as such.
    """
    lignes = [e for e in lire_evenements(chemin_run) if e.get("origine") != "entendu"]
    if not lignes:
        return {}
    foyer_de = menages(chemin_run)
    # (event, household) → first and last day, readers.
    par_foyer: dict[tuple[str, str], dict[str, Any]] = {}
    for e in lignes:
        agent = str(e.get("person_id") or "")
        identifiant = str(e.get("evenement_id") or e.get("choc_id") or "")
        journee = journee_de(e)
        if not (agent and identifiant and journee):
            continue
        foyer = foyer_de.get(agent) or f"seul:{agent}"
        info = par_foyer.setdefault((identifiant, foyer),
                                    {"premier": journee, "dernier": journee, "lecteurs": set()})
        info["premier"] = min(info["premier"], journee)
        info["dernier"] = max(info["dernier"], journee)
        info["lecteurs"].add(agent)

    resultat: dict[str, Fenetre] = {}
    for (identifiant, foyer), info in sorted(par_foyer.items()):
        seul = foyer.startswith("seul:")
        membres = [foyer.split(":", 1)[1]] if seul else sorted(
            p for p, h in foyer_de.items() if h == foyer)
        for p in membres:
            # An agent in two exposed households does not exist; two events on one household
            # keep the FIRST window, which is the one before any change.
            resultat.setdefault(p, Fenetre(
                evenement_id=identifiant, household_id=None if seul else foyer,
                role="expose" if p in info["lecteurs"] else "co_resident",
                premier_jour=info["premier"], dernier_jour=info["dernier"],
            ))
    return resultat
