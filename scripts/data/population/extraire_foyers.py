#!/usr/bin/env python3
"""Extracts whole HOUSEHOLDS from a sealed population — ticket 059, lot 6.

WHY ONE MORE SCRIPT
-------------------
`extraire_sous_population.py` samples INDIVIDUALS by predicate, and that was what was needed
to read five memories (ticket 075). Here the object is the household: extracting one member
without the other removes the measured quantity. The five agents of 075 belong to five
distinct households anyway, and the diffusion mechanism would be strictly unobservable there.

WHAT THE POPULATION CARRIES
---------------------------
Households of SIZE 2 whose two members are both mobile, split into two groups:

- **exposed** — the household will receive the article; only one of its two members will read
  it, and drawing the reader belongs to the `information` channel, not to this script;
- **controls** — the household will receive nothing, in the SAME run. The noise floor measured
  in ticket 095 is not zero (1 mode gap out of 31 paired decisions): comparing with a control
  run launched separately would bring that floor into the measured effect.

Size 2 by default, and it is a debugging choice: one reader, one co-resident, nothing
else to untangle. Households of four or five tell whether a statement reaches everyone or
stops at the first — a question for tiers P2 and P3, not for the plumbing. `--taille` and
`--adultes` open these tiers (2026-09-24: family of four, two adults, two children —
the reader is drawn among the adults, cf. `llm/evenements/exposition.py`). `--menage` designates
a household by its identifier rather than by draw, when the choice was reasoned (mobility
profiles suited to the article played): it remains subject to the same eligibility criteria.
`--taille` repeats to mix households of different sizes (2026-09-25: a couple and a
family of four). `--lecteur` designates who reads in a designated household, instead of the
draw among adults: an article on the metro must be read by someone who takes it. It is written
to the manifest (`expose[].lecteurs`), from where the cohort carries it into the played event.

DETERMINISM
-----------
Sorting follows household identifiers, in numeric order. The exposed/control assignment
is a stable hash of `graine:household_id`: two extractions yield the same groups, and
changing the seed shows in the manifest.

USAGE
-----
    services/llm-agents/.venv/bin/python -m scripts.data.population.extraire_foyers \
        --source data/population/population_1000_PANEL_v6/population.json \
        --sortie data/population/population_20_foyers_059 \
        --exposes 6 --temoins 4 --abonnement-tc

    # One designated household, family of four with two adults, no control household:
    services/llm-agents/.venv/bin/python -m scripts.data.population.extraire_foyers \
        --source data/population/population_1000_PANEL_v6/population.json \
        --sortie data/population/population_4_foyer_133048 \
        --taille 4 --adultes 2 --menage 133048 --temoins 0

    # Two designated households of different sizes, one designated reader in each:
    services/llm-agents/.venv/bin/python -m scripts.data.population.extraire_foyers \
        --source data/population/population_1000_PANEL_v6/population.json \
        --sortie data/population/population_6_foyers_a13 \
        --taille 2 --taille 4 --adultes 2 --menage 643030 --menage 534995 \
        --lecteur 1320713 --lecteur 1127260 --temoins 0 --motif "…"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TAILLE_FOYER = 2
AGE_ADULTE = 18   # the same threshold as the reader draw (llm/evenements/exposition.py)


def _traits(personne: dict) -> dict:
    return (personne.get("identity") or {}).get("traits_json") or {}


def _mobile(personne: dict) -> bool:
    return not personne.get("immobile", False)


def _sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def _cle_de_tri(identifiant: str) -> tuple[int, int, str]:
    """Numeric sort when the identifier is numeric, alphabetical otherwise.

    The v6 cohort has only numeric identifiers, but `int()` on an alphanumeric
    identifier would make the whole extraction fail on a population of a different shape —
    an incomprehensible refusal where a stable order is enough.
    """
    return (0, int(identifiant), "") if identifiant.isdigit() else (1, 0, identifiant)


def _rang(graine: int, household_id: str) -> float:
    """Stable rank in [0, 1[ — the exposed/control assignment depends on no execution order."""
    brut = hashlib.sha256(f"{graine}:{household_id}".encode("utf-8")).hexdigest()[:8]
    return int(brut, 16) / 0xFFFFFFFF


def _adulte(personne: dict) -> bool:
    try:
        return int(_traits(personne).get("age")) >= AGE_ADULTE
    except (TypeError, ValueError):
        return False


def foyers_eligibles(
    population: list[dict],
    *,
    abonnement_tc: bool,
    taille: int | Iterable[int] = TAILLE_FOYER,
    adultes: int | None = None,
) -> dict[str, list[dict]]:
    """Households of `taille` members (one size or several), all mobile (and exactly `adultes`
    adults if requested), sorted by identifier."""
    tailles = {taille} if isinstance(taille, int) else set(taille)
    par_menage: dict[str, list[dict]] = defaultdict(list)
    for personne in population:
        menage = (personne.get("household") or {}).get("id")
        if menage:
            par_menage[str(menage)].append(personne)

    retenus: dict[str, list[dict]] = {}
    for menage, membres in par_menage.items():
        if len(membres) not in tailles:
            continue
        if not all(_mobile(m) for m in membres):
            continue
        if adultes is not None and sum(_adulte(m) for m in membres) != adultes:
            continue
        if abonnement_tc and not any(_traits(m).get("has_pt_subscription") for m in membres):
            # The article played first targets public transport: a household where nobody
            # uses it has nothing to pass on, and its presence would dilute the measurement.
            continue
        retenus[menage] = sorted(membres, key=lambda p: _cle_de_tri(str(p["person_id"])))
    return {m: retenus[m] for m in sorted(retenus, key=_cle_de_tri)}


def choisir(
    population: list[dict],
    *,
    exposes: int,
    temoins: int,
    graine: int,
    abonnement_tc: bool,
    taille: int | Iterable[int] = TAILLE_FOYER,
    adultes: int | None = None,
    menages: list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, list[dict]]]:
    eligibles = foyers_eligibles(
        population, abonnement_tc=abonnement_tc, taille=taille, adultes=adultes
    )
    if menages:
        # DESIGNATED households: they are the exposed; controls are drawn from the rest.
        refuses = [m for m in menages if m not in eligibles]
        if refuses:
            raise SystemExit(
                f"REFUSED: household(s) {refuses} not eligible (size {taille}, "
                f"{'adults ' + str(adultes) + ', ' if adultes is not None else ''}all mobile"
                f"{', PT pass holder' if abonnement_tc else ''}). Designating a household does not "
                f"exempt it from the criteria: the population would say something other than its manifest."
            )
        reste = sorted(
            (m for m in eligibles if m not in menages), key=lambda m: (_rang(graine, m), _cle_de_tri(m))
        )
        if len(reste) < temoins:
            raise SystemExit(f"REFUS : {len(reste)} possible control households for {temoins} requested.")
        return list(menages), reste[:temoins], eligibles
    besoin = exposes + temoins
    if len(eligibles) < besoin:
        raise SystemExit(
            f"REFUSED: {len(eligibles)} eligible households for {besoin} requested → relax the "
            f"criterion (--abonnement-tc), or reduce --exposes / --temoins. An incomplete group "
            f"would make one of the two arms smaller than the other without anything saying so."
        )
    ordonnes = sorted(eligibles, key=lambda m: (_rang(graine, m), _cle_de_tri(m)))
    return ordonnes[:exposes], ordonnes[exposes:besoin], eligibles


def lecteurs_par_menage(
    lecteurs: list[str], exposes: list[str], eligibles: dict[str, list[dict]]
) -> dict[str, list[str]]:
    """Sorts designated readers by exposed household; refuses a reader who cannot read.

    A reader must be an ADULT of an exposed household: that is the draw rule
    (`llm/evenements/exposition.py`), and the simulator would discard them otherwise — at the
    cost of a household without a reader, discovered in the middle of the run.
    """
    ranges: dict[str, list[str]] = {}
    refus: list[str] = []
    for pid in lecteurs:
        foyer = next(
            (m for m in exposes for p in eligibles[m] if str(p["person_id"]) == str(pid)), None
        )
        membre = next((p for p in eligibles.get(foyer, []) if str(p["person_id"]) == str(pid)), None)
        if membre is None:
            refus.append(f"{pid} (hors des foyers exposés {exposes})")
        elif not _adulte(membre):
            refus.append(f"{pid} (mineur : {_traits(membre).get('age')} ans)")
        else:
            ranges.setdefault(foyer, []).append(str(pid))
    if refus:
        raise SystemExit(f"REFUS : reader(s) {refus}. A reader is an adult of an exposed household.")
    return ranges


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="sealed population.json")
    p.add_argument("--sortie", required=True, type=Path, help="directory to create")
    p.add_argument("--exposes", type=int, default=6, help="households that will receive the article")
    p.add_argument("--temoins", type=int, default=4, help="households that will receive nothing")
    p.add_argument("--graine", type=int, default=59)
    p.add_argument(
        "--taille", type=int, action="append", default=None,
        help=f"members per household (repeatable to mix sizes; {TAILLE_FOYER} if absent)",
    )
    p.add_argument(
        "--adultes", type=int, default=None,
        help=f"EXACT number of adults (≥ {AGE_ADULTE} years) per household; free if absent",
    )
    p.add_argument(
        "--menage", action="append", default=None,
        help="designated household (repeatable): becomes exposed instead of the draw",
    )
    p.add_argument(
        "--lecteur", action="append", default=None,
        help="designated reader (repeatable): an adult of a --menage household, instead of the draw",
    )
    p.add_argument(
        "--motif", default=None,
        help="why these households were designated (written to the manifest); required with --menage",
    )
    p.add_argument(
        "--abonnement-tc",
        action="store_true",
        help="accept only households with at least one public transport subscriber",
    )
    args = p.parse_args(argv)

    if args.menage and not args.motif:
        p.error("--menage requires --motif: a reasoned choice that does not state its reason is a hidden draw")
    if args.lecteur and not args.menage:
        p.error("--lecteur requires --menage: one does not designate who reads in a drawn household")
    tailles = sorted(set(args.taille or [TAILLE_FOYER]))
    population = json.loads(args.source.read_text(encoding="utf-8"))
    exposes, temoins, eligibles = choisir(
        population,
        exposes=args.exposes,
        temoins=args.temoins,
        graine=args.graine,
        abonnement_tc=args.abonnement_tc,
        taille=tailles,
        adultes=args.adultes,
        menages=args.menage,
    )
    lecteurs = lecteurs_par_menage(args.lecteur or [], exposes, eligibles)

    agents: list[dict] = []
    for menage in exposes + temoins:
        # Agents are copied AS IS: no normalisation, no field added. A test
        # population that diverges from the seal it comes from no longer proves anything — and
        # the role (exposed, control) lives in the manifest, not in the agent's data.
        agents.extend(eligibles[menage])

    args.sortie.mkdir(parents=True, exist_ok=True)
    fichier = args.sortie / "population.json"
    fichier.write_text(json.dumps(agents, ensure_ascii=False, indent=2), encoding="utf-8")

    lignes: list[str] = [
        "# Population de TEST — ticket 059, lot 6. Ce n'est PAS un sceau de cohorte.",
        "#",
        f"# {len(agents)} agents ne représentent rien : ils servent à observer si une information",
        "# lue par un seul membre d'un foyer atteint l'autre, et à quelle date. Ils ne mesurent",
        "# aucune part modale. La référence de l'article reste",
        "# data/population/population_1000_PANEL_v6.",
        "#",
        "# Les foyers témoins sont dans le MÊME fichier, donc dans le même run : le plancher de",
        "# bruit mesuré au ticket 095 n'est pas nul, et un témoin lancé séparément le ferait",
        "# entrer dans l'effet mesuré.",
        f"nom: {args.sortie.name}",
        f"extrait_le: '{datetime.now(timezone.utc).isoformat()}'",
        "source:",
        f"  fichier: {args.source.as_posix()}",
        f"  sha256: {_sha256(args.source)}",
        f"  n: {len(population)}",
        "population:",
        "  fichier: population.json",
        f"  sha256: {_sha256(fichier)}",
        f"  n: {len(agents)}",
        "selection:",
        "  methode: >-",
        f"    Foyers de taille {' ou '.join(map(str, tailles))} dont tous les membres sont mobiles"
        + (f", dont {args.adultes} adultes (≥ {AGE_ADULTE} ans)" if args.adultes is not None else "")
        + ("; exposés DÉSIGNÉS par --menage, témoins" if args.menage else ";")
        + " triés par rang stable",
        "    sha256(graine:household_id). Aucun tirage dépendant de l'ordre d'exécution.",
        "    Reproductible par scripts/data/population/extraire_foyers.py.",
        f"  graine: {args.graine}",
        f"  taille: {tailles[0] if len(tailles) == 1 else json.dumps(tailles)}",
        f"  adultes_exiges: {args.adultes if args.adultes is not None else 'null'}",
        f"  menages_designes: {json.dumps(args.menage or [])}",
        f"  lecteurs_designes: {json.dumps(args.lecteur or [])}",
        f"  motif: {json.dumps(args.motif or '', ensure_ascii=False)}",
        f"  abonnement_tc_exige: {bool(args.abonnement_tc)}",
        f"  foyers_eligibles: {len(eligibles)}",
        "groupes:",
    ]
    for role, menages in (("expose", exposes), ("temoin", temoins)):
        lignes.append(f"  {role}:")
        for menage in menages:
            membres = eligibles[menage]
            lignes.append(f"    - household_id: '{menage}'")
            if lecteurs.get(menage):
                lignes.append(f"      lecteurs: {json.dumps(lecteurs[menage])}")
            lignes.append("      membres:")
            for m in membres:
                t = _traits(m)
                lignes += [
                    f"        - person_id: '{m['person_id']}'",
                    f"          nom: {json.dumps(t.get('name', ''), ensure_ascii=False)}",
                    f"          age: {t.get('age')}",
                    f"          occupation: {json.dumps(t.get('main_occupation', ''), ensure_ascii=False)}",
                    f"          zone: {json.dumps(t.get('residence_zone', ''), ensure_ascii=False)}",
                    f"          abonnement_tc: {bool(t.get('has_pt_subscription'))}",
                    f"          voiture: {json.dumps(t.get('car_availability', ''), ensure_ascii=False)}",
                    f"          velo: {json.dumps(t.get('personal_bike', ''), ensure_ascii=False)}",
                ]
    (args.sortie / "MANIFEST.yaml").write_text("\n".join(lignes) + "\n", encoding="utf-8")

    print(
        f"{len(agents)} agents, {len(exposes)} exposed households and {len(temoins)} controls, "
        f"out of {len(eligibles)} eligible → {args.sortie}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
