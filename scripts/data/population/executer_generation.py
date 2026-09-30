#!/usr/bin/env python3
"""Ticket 074, C-1 — runs `generate_population.ipynb` STEP BY STEP, outside the notebook.

Why not `papermill`: it plays the notebook from end to end. But the chain lasts hours,
its steps have very uneven costs (eqasim, routing, OSMnx warm-up), and half of
them overwrite checkpoints. We want to be able to stop after each one, read its summary, and
decide on the next — that is also what the human review asks for.

Code cells are executed **in a single namespace**, in order, exactly
like the notebook: the variables of one step serve the next. The parameters of cell 2
are replaced by those passed on the command line, and the replacement is LOGGED — a
parameter changed silently is the simplest way to seal a cohort under a frame that
nobody wanted (it happened on 2026-09-03, a v2 exported under a v3 name).

    cd scripts/data/population
    ../../../services/llm-agents/.venv/bin/python executer_generation.py --jusqu-a 1
    ../../../services/llm-agents/.venv/bin/python executer_generation.py --de 2 --jusqu-a 3ter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ICI = Path(__file__).resolve().parent
CARNET = ICI / "generate_population.ipynb"

# What step 3ter SETS in the namespace and on which everything downstream depends:
# `POPULATION_SIZES = [SELECT_N]` and `POPULATION_TAG = SELECT_TAG`. In the notebook,
# these values survive because the kernel does not die. Here, a resume at step 4
# starts from a fresh namespace — and `pop_filename()` then returns the name of the POOL.
#
# It is not a loud error: step 4 starts routing the 33 420 pairs of the
# whole pool instead of the ~4 000 of the selected ones, for hours, and step 7
# would export the pool under the cohort's name. It happened on 2026-09-14.
# The state is therefore WRITTEN after 3ter and REREAD on resume, saying so.
ETAT = ICI / "Temp" / ".etat_generation.json"
ETAT_CHAMPS = ("POPULATION_SIZES", "POPULATION_TAG", "SELECT_N", "SELECT_TAG")

# Step name → index of the CODE cell carrying it, in notebook order.
# Read from the notebook of 2026-09-14; `--lister` redisplays the table as it is read,
# so that a shift shows before launching three hours of computation.
ETAPES: dict[str, int] = {
    "prelude": 1,        # certifi
    "parametres": 2,
    "chemins": 4,
    "imports": 6,
    "sante": 8,
    "1": 10,             # eqasim → Temp/1_raw
    "2": 12,             # activity validation → Temp/2_fixed
    "3": 14,             # public transport → Temp/3_pt_enriched
    "3bis": 16,          # AAV2020 zone + density → Temp/4_zone_enriched
    "3ter": 18,          # stratified selection
    "4": 20,             # travel time + schedules → Temp/5_scheduled
    "5": 22,             # merged with 4
    "6": 24,             # OSMnx warm-up
    "export": 26,
    "7": 28,             # final export → data/population/
    "8": 30,             # EMC² imputed traits
    "9": 32,             # completeness audit
}
ORDRE = list(ETAPES)


def _cellules_code(carnet: dict) -> list[str]:
    """Cell sources, indexed as in the notebook (markdown → empty string)."""
    return ["".join(c["source"]) if c["cell_type"] == "code" else ""
            for c in carnet["cells"]]


def _remplacer_parametres(source: str, valeurs: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrites the top-level assignments named in `valeurs`. Returns the changed lines."""
    lignes = source.splitlines()
    changees: list[str] = []
    restants = dict(valeurs)
    for i, ligne in enumerate(lignes):
        nu = ligne.strip()
        if not nu or nu.startswith("#"):
            continue
        nom = nu.split("=", 1)[0].strip() if "=" in nu else ""
        if nom in restants:
            commentaire = ""
            if "#" in ligne:
                commentaire = "  " + ligne[ligne.index("#"):]
            lignes[i] = f"{nom} = {restants[nom]}{commentaire}"
            changees.append(f"{nom} : {nu.split('=', 1)[1].split('#')[0].strip()} → "
                            f"{restants[nom]}")
            del restants[nom]
    if restants:
        raise SystemExit(
            f"REFUSED — parameter(s) not found in the parameters cell: "
            f"{sorted(restants)}. The notebook has changed; check before launching."
        )
    return "\n".join(lignes), changees


def _ecrire_etat(espace: dict) -> None:
    """Records what 3ter has just set, so that a resume finds it again."""
    etat = {champ: espace.get(champ) for champ in ETAT_CHAMPS}
    ETAT.parent.mkdir(parents=True, exist_ok=True)
    ETAT.write_text(json.dumps(etat, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"── selection state recorded in {ETAT.name}: "
          f"POPULATION_SIZES={etat['POPULATION_SIZES']} POPULATION_TAG="
          f"{etat['POPULATION_TAG']!r}")


def _etat_a_injecter(de: str, deja_passes: dict[str, str]) -> dict[str, str]:
    """Values set by 3ter to reinject when resuming AFTER it.

    Refuses rather than guesses: without a recorded state, a resume at step 4 works
    on the pool believing it works on the cohort, and nothing says so.
    """
    if ORDRE.index(de) <= ORDRE.index("3ter"):
        return {}
    # Having passed them by hand counts as state: it is the emergency exit that the refusal
    # below offers, and it must work.
    if {"POPULATION_SIZES", "POPULATION_TAG"} <= set(deja_passes):
        return {}
    if not ETAT.exists():
        raise SystemExit(
            f"REFUSÉ — resume at step {de} without selection state ({ETAT}).\n"
            "Step 3ter sets POPULATION_SIZES and POPULATION_TAG; without them, downstream "
            "would route and export the POOL under the cohort's name.\n"
            "Replay 3ter, or pass the values by hand:\n"
            "    --param \"POPULATION_SIZES=[1000]\" --param \"POPULATION_TAG='PANEL_v6'\"")
    etat = json.loads(ETAT.read_text(encoding="utf-8"))
    injecte = {c: repr(etat[c]) for c in ETAT_CHAMPS
               if etat.get(c) is not None and c not in deja_passes}
    if injecte:
        print(f"── resume after 3ter: state reread from {ETAT.name}")
    return injecte


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--de", default="prelude", help=f"first step ({'|'.join(ORDRE)})")
    p.add_argument("--jusqu-a", dest="jusqu_a", default="1", help="last step included")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="replaces a parameter of cell 2 (repeatable)")
    p.add_argument("--lister", action="store_true", help="shows the step table and exits")
    a = p.parse_args(argv)

    carnet = json.loads(CARNET.read_text(encoding="utf-8"))
    sources = _cellules_code(carnet)

    if a.lister:
        print(f"{'étape':10s} {'cellule':>7s}  première ligne de code")
        for nom, idx in ETAPES.items():
            tete = next((l for l in sources[idx].splitlines()
                         if l.strip() and not l.strip().startswith("#")), "(vide)")
            print(f"{nom:10s} {idx:7d}  {tete[:80]}")
        return 0

    for nom in (a.de, a.jusqu_a):
        if nom not in ETAPES:
            raise SystemExit(f"unknown step: {nom!r} (known: {', '.join(ORDRE)})")
    debut, fin = ORDRE.index(a.de), ORDRE.index(a.jusqu_a)
    if debut > fin:
        raise SystemExit(f"--de {a.de} comes after --jusqu-a {a.jusqu_a}")

    valeurs = dict(v.split("=", 1) for v in a.param)

    # The bootstrap cells (parameters, paths, imports) are ALWAYS replayed: they
    # cost nothing and without them the namespace is empty. Not replaying them would amount to
    # resuming in the middle of a sentence.
    amorce = [ETAPES[n] for n in ("prelude", "parametres", "chemins", "imports")]
    a_jouer = amorce + [ETAPES[n] for n in ORDRE[debut:fin + 1] if ETAPES[n] not in amorce]

    print("=" * 88)
    print(f"Population generation — steps {a.de} → {a.jusqu_a}"
          f"   ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    print("=" * 88)

    espace: dict = {"__name__": "__main__"}
    reprise = _etat_a_injecter(a.de, valeurs)
    depart_total = time.time()
    for idx in a_jouer:
        source = sources[idx]
        if not source.strip():
            continue
        etiquette = next((n for n, i in ETAPES.items() if i == idx), f"cellule {idx}")
        if idx == ETAPES["parametres"] and (valeurs or reprise):
            source, changees = _remplacer_parametres(source, {**reprise, **valeurs})
            print(f"\n── parameters replaced ({len(changees)}) ──")
            for c in changees:
                print(f"   {c}")
        print(f"\n{'─' * 88}\n── step {etiquette} (cell {idx}) ──")
        depart = time.time()
        try:
            exec(compile(source, f"<carnet:cellule {idx}>", "exec"), espace)
        except Exception:
            print(f"\n[ALARME] step {etiquette} INTERRUPTED after "
                  f"{time.time() - depart:.0f} s :", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"── étape {etiquette} terminée en {time.time() - depart:.0f} s")
        if etiquette == "3ter":
            _ecrire_etat(espace)

    print(f"\n{'=' * 88}")
    print(f"Steps {a.de} → {a.jusqu_a}: SUCCESS, in {time.time() - depart_total:.0f} s "
          f"({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
