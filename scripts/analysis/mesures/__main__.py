"""`python -m scripts.analysis.mesures <run>` — recomputes and rewrites the CSVs of a run."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.analysis.mesures.calcul import calculer
from scripts.analysis.mesures.ecriture import ecrire


def main(argv=None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("run", type=Path, help="run directory")
    parseur.add_argument("-o", "--sortie", type=Path, default=None,
                         help="CSV directory (default: <run>/mesures)")
    args = parseur.parse_args(argv)

    mesures = calculer(args.run)
    chemins = ecrire(mesures, args.sortie)

    jours = mesures.journees
    print(f"{len(jours)} lived day(s)"
          + (f", from day {jours[0].index} to day {jours[-1].index}" if jours else "")
          + f" · {mesures.trajets_rejoues} replayed trip(s) and {mesures.rappels_rejoues}"
          " replayed recall(s) set aside")
    if not mesures.operations_tracees:
        print("  ⚠ operations_concept.jsonl absent : les colonnes d'opérations restent VIDES "
              "(et non à zéro — « aucune contradiction » n'est pas « on ne mesure pas »).")
    for nom, chemin in chemins.items():
        lignes = max(0, len(chemin.read_text(encoding="utf-8").splitlines()) - 1)
        print(f"  {nom:16s} {lignes:>5} line(s) → {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
