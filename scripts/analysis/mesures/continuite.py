"""Checks that a cut did not betray the curves — ticket 093, § 4 (acceptance).

WHAT THIS CHECK CATCHES
-----------------------
The ticket's acceptance test takes three steps: two simulated days, a stop, a named resume
(ticket 091). The real test is what one reads afterwards: **a metric that jumps or doubles at
resume is a wrong metric**, and on a twenty-day run nobody would see it any more.

Three defects, then, and they can be spotted by eye on two days when one knows where to look:

1. **Doubling.** A key written twice, because the replayed day was appended instead
   of being recomputed.
2. **The gap.** A missing day between the first and the last — the resume picked up somewhere
   other than where it had left off.
3. **The jump.** A value of a day PRIOR to the cut that changed, although nothing that
   happened afterwards can concern it.

⚠ A missing day is not always a gap: the simulation skips weekends, whose
departures are postponed to Monday. The checker knows it, and only complains on working days.

USAGE
-----
    python -m scripts.analysis.mesures.continuite <avant> <apres>

where `<avant>` is a copy of the CSVs taken before the cut, and `<apres>` the CSV directory
after the resume. Non-zero exit if continuity is broken: the acceptance test fails loudly.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from scripts.analysis.mesures.ecriture import TABLES, Table


@dataclass
class Rupture:
    table: str
    genre: str  # "dédoublement", "trou", "saut"
    detail: str


@dataclass
class Verdict:
    ruptures: list[Rupture] = field(default_factory=list)
    tables_verifiees: list[str] = field(default_factory=list)
    lignes_comparees: int = 0

    @property
    def continu(self) -> bool:
        return not self.ruptures


def _lire(chemin: Path) -> list[dict[str, str]]:
    if not chemin.is_file():
        return []
    with chemin.open(newline="", encoding="utf-8") as flux:
        return list(csv.DictReader(flux))


def _cle(ligne: dict[str, str], table: Table) -> tuple[str, ...]:
    return tuple(ligne.get(colonne, "") for colonne in table.cle)


def _jours_ouvres_manquants(dates: Sequence[str]) -> list[str]:
    """WORKING days missing between the first and the last. Weekends do not count."""
    if len(dates) < 2:
        return []
    connues = {date.fromisoformat(d) for d in dates}
    debut, fin = min(connues), max(connues)
    manquants = []
    courant = debut
    while courant <= fin:
        if courant.isoweekday() <= 5 and courant not in connues:
            manquants.append(courant.isoformat())
        courant += timedelta(days=1)
    return manquants


def verifier(avant: Path | str, apres: Path | str) -> Verdict:
    """Compares two snapshots of the CSVs and returns the verdict."""
    avant, apres = Path(avant), Path(apres)
    verdict = Verdict()
    for nom, table in TABLES.items():
        lignes_apres = _lire(apres / table.fichier)
        if not lignes_apres:
            continue
        verdict.tables_verifiees.append(nom)

        vues: dict[tuple[str, ...], int] = {}
        for ligne in lignes_apres:
            clef = _cle(ligne, table)
            vues[clef] = vues.get(clef, 0) + 1
        for clef, compte in sorted(vues.items()):
            if compte > 1:
                verdict.ruptures.append(Rupture(
                    nom, "dédoublement",
                    f"la clé {clef} apparaît {compte} fois — un jour rejoué a été AJOUTÉ au "
                    f"lieu d'être recalculé"))

        dates = sorted({l["date_simulee"] for l in lignes_apres if l.get("date_simulee")})
        for manquant in _jours_ouvres_manquants(dates):
            verdict.ruptures.append(Rupture(
                nom, "trou", f"aucune ligne pour le jour ouvré {manquant}"))

        anciennes = {_cle(l, table): l for l in _lire(avant / table.fichier)}
        if not anciennes:
            continue
        nouvelles = {_cle(l, table): l for l in lignes_apres}
        for clef, ancienne in sorted(anciennes.items()):
            nouvelle = nouvelles.get(clef)
            verdict.lignes_comparees += 1
            if nouvelle is None:
                verdict.ruptures.append(Rupture(
                    nom, "trou", f"la ligne {clef}, écrite avant la coupure, a DISPARU"))
                continue
            for colonne in table.colonnes:
                if ancienne.get(colonne, "") != nouvelle.get(colonne, ""):
                    verdict.ruptures.append(Rupture(
                        nom, "saut",
                        f"{clef} · colonne « {colonne} » : "
                        f"{ancienne.get(colonne, '')!r} → {nouvelle.get(colonne, '')!r} — "
                        f"une valeur d'un jour antérieur à la coupure a changé"))
    return verdict


def main(argv=None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("avant", type=Path, help="CSVs copied BEFORE the cut")
    parseur.add_argument("apres", type=Path, help="CSVs AFTER the resume")
    args = parseur.parse_args(argv)

    verdict = verifier(args.avant, args.apres)
    print(f"{len(verdict.tables_verifiees)} table(s) checked, "
          f"{verdict.lignes_comparees} line(s) compared on either side of the cut.")
    if verdict.continu:
        print("✔ The curves are CONTINUOUS: no doubled key, no missing working day, "
              "no value of an earlier day modified.")
        return 0
    print(f"✘ Continuity BROKEN — {len(verdict.ruptures)} anomaly(ies):")
    for rupture in verdict.ruptures:
        print(f"  [{rupture.genre}] {rupture.table} — {rupture.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
