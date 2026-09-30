"""export_car_availability.py — Household car availability, measured on EMC².

`car_availability` (`all` / `some` / `none`) is derived, not observed: it compares the
household's number of cars to its number of adult licence holders
(`services/eqasim-toulouse/synthesis/population/enriched.py`). The synthetic population therefore sees
**too much sharing** as soon as the number of licences is overestimated — which is what
[ticket 017](../../docs/tickets/ticket_017_permis_progedo.md) measures.

This script publishes the reference distribution, recomputed from the microdata, with the
checks that make it enforceable. It fixes nothing: it provides the target of
[ticket 018](../../docs/tickets/ticket_018_partage_voiture_foyer.md).

## The derivation rule is copied from eqasim, not reinvented

    all   if cars >= household licences (adults only)
    some  if cars <  licences
    none  if cars == 0                    (takes precedence over the two above)

The restriction to adults comes from action A1.a of ticket 008: a licence inherited from an
adult donor by a child switched households from `all` to `some`, the car there
becoming « to be shared » while the extra driver is nine years old. Measuring the target
with a rule other than the simulator's would produce a gap that would not be a bias
but a difference of definition — the exact pattern that tickets 015 to 019 fix.

## The two validity checks

**Positive** — the same reading of the households file must reproduce the car ownership published by
the survey: 1.25 cars per household, and 19 / 45 / 35 % of households with zero / one / two
cars or more. If this check fails, the weighting or the filtering is wrong and the
derived distribution is worthless. The script **fails** rather than publish.

**Negative** — `P7` must be filled in for the persons who are asked the question. An
empty variable would produce zero holders per household, hence `all` everywhere: a
perfect and false result, exactly the vacuity the project tracks down. The script publishes the
non-response rate and the breakdown of modalities.

⚠ `P7 == 3` (« conduite accompagnée et leçons de conduite ») is **not** a holder.
Mistaking it for a `oui` would inflate the number of licences and hence sharing — it is
precisely the bias being measured.

## Two weightings, because they do not say the same thing

- **households** (`COE0`): the structure of the stock. It is the reading that compares with
  the published car ownership.
- **persons** (`COE1`): what an individual drawn at random experiences, hence the reading enforceable
  against a synthetic population of agents. It is the one ticket 018 cites.

Usage:
    python -m scripts.progedo_logit.export_car_availability [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV" / "fichiers_standards"
OUT = ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "car_availability_emc2.json"

LEVELS = ("all", "some", "none")

# Car ownership published by the survey (cf. scripts/data/population/population_emc2_2023.yaml).
# Serves as POSITIVE check: the reading must reproduce it, otherwise it is wrong.
PUBLISHED = {"cars_per_household": 1.25, "zero": 19.0, "one": 45.0, "two_plus": 35.0}
TOLERANCE_PT = 1.0
TOLERANCE_CARS = 0.03


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Households and persons of the standard file. Actual household key: `(ZFM, ECH)`.

    `ECH` alone is not unique from one zone to another — same remark as
    `export_bike_ownership`, and departing from it would mix households of different zones.
    """
    men = pd.read_csv(root / "Toulouse_2023_std_men.csv", dtype=str, sep=None,
                      engine="python")
    pers = pd.read_csv(root / "Toulouse_2023_std_pers.csv", dtype=str, sep=None,
                       engine="python")
    men["weight"] = pd.to_numeric(men["COE0"], errors="coerce")
    men["cars"] = pd.to_numeric(men["M6"], errors="coerce")
    pers["weight"] = pd.to_numeric(pers["COE1"], errors="coerce")
    pers["age"] = pd.to_numeric(pers["P4"], errors="coerce")
    return men, pers


def negative_control(pers: pd.DataFrame) -> dict:
    """Is `P7` alive? A dead variable would give `all` everywhere."""
    codes = pers["P7"].fillna("").str.strip()
    total = len(pers)
    adults = pers["age"] >= 18
    blank_adults = int((codes.eq("") & adults).sum())
    return {
        "modalities": {("(vide)" if k == "" else k): int(v)
                       for k, v in codes.value_counts().items()},
        "labels": {"1": "Oui", "2": "Non",
                   "3": "Conduite accompagnée et leçons (PAS un titulaire)"},
        "blank_share_pct": round(100 * float(codes.eq("").sum()) / total, 2),
        "blank_adults": blank_adults,
        "blank_adults_share_pct": round(100 * blank_adults / int(adults.sum()), 2),
        "verdict": ("variable vivante" if codes.eq("1").sum() > 0.3 * total
                    else "SUSPECTE — trop peu de titulaires, vérifier le codage"),
    }


def positive_control(men: pd.DataFrame) -> dict:
    """Does the reading reproduce the published car ownership?"""
    frame = men.dropna(subset=["weight", "cars"])
    total = float(frame["weight"].sum())
    measured = {
        "cars_per_household": float((frame["cars"] * frame["weight"]).sum() / total),
        "zero": 100 * float(frame.loc[frame["cars"] == 0, "weight"].sum()) / total,
        "one": 100 * float(frame.loc[frame["cars"] == 1, "weight"].sum()) / total,
        "two_plus": 100 * float(frame.loc[frame["cars"] >= 2, "weight"].sum()) / total,
    }
    deltas = {k: measured[k] - PUBLISHED[k] for k in PUBLISHED}
    ok = (abs(deltas["cars_per_household"]) <= TOLERANCE_CARS
          and all(abs(deltas[k]) <= TOLERANCE_PT for k in ("zero", "one", "two_plus")))
    return {"measured": {k: round(v, 3) for k, v in measured.items()},
            "published": PUBLISHED,
            "delta": {k: round(v, 3) for k, v in deltas.items()},
            "tolerance_pt": TOLERANCE_PT, "tolerance_cars": TOLERANCE_CARS,
            "passed": bool(ok)}


def derive(men: pd.DataFrame, pers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds `car_availability` to the households, and carries it over to the persons."""
    pers = pers.copy()
    # P7 == "1" only: supervised driving (3) is not a holder.
    pers["licensed_adult"] = (pers["P7"].fillna("").str.strip() == "1") & (pers["age"] >= 18)
    licenses = (pers.groupby(["ZFP", "ECH"])["licensed_adult"].sum()
                .rename("licenses").reset_index().rename(columns={"ZFP": "ZFM"}))
    hh = men.merge(licenses, on=["ZFM", "ECH"], how="left", validate="one_to_one")
    hh["licenses"] = hh["licenses"].fillna(0)
    hh = hh.dropna(subset=["weight", "cars"])

    level = np.where(hh["cars"] >= hh["licenses"], "all", "some")
    hh["car_availability"] = np.where(hh["cars"] == 0, "none", level)

    people = pers.merge(
        hh[["ZFM", "ECH", "car_availability"]].rename(columns={"ZFM": "ZFP"}),
        on=["ZFP", "ECH"], how="inner", validate="many_to_one").dropna(subset=["weight"])
    return hh, people


def distribution(frame: pd.DataFrame) -> dict:
    total = float(frame["weight"].sum())
    return {lvl: round(100 * float(frame.loc[frame["car_availability"] == lvl,
                                             "weight"].sum()) / total, 2)
            for lvl in LEVELS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--root", type=Path, default=DATA)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[ERREUR] microdata missing: {args.root}")
        return 1

    men, pers = load(args.root)
    negative = negative_control(pers)
    positive = positive_control(men)
    hh, people = derive(men, pers)

    by_household, by_person = distribution(hh), distribution(people)

    print("=== NEGATIVE CHECK — is the variable P7 alive? ===")
    for code, count in negative["modalities"].items():
        print(f"  {code:8} {count:6}  {negative['labels'].get(code, '')}")
    print(f"  non-response: {negative['blank_share_pct']} % of the whole, "
          f"{negative['blank_adults_share_pct']} % of adults")
    print(f"  → {negative['verdict']}")

    print("\n=== POSITIVE CHECK — published car ownership reproduced? ===")
    for key in ("cars_per_household", "zero", "one", "two_plus"):
        print(f"  {key:20} measured {positive['measured'][key]:7.2f}  "
              f"published {positive['published'][key]:6.2f}  "
              f"gap {positive['delta'][key]:+6.2f}")
    print(f"  → {'PASSÉ' if positive['passed'] else 'ÉCHEC'}")

    print("\n=== car_availability ===")
    print(f"  {'':22}{'all':>8}{'some':>8}{'none':>8}")
    print(f"  {'ménages (COE0)':22}"
          + "".join(f"{by_household[l]:8.1f}" for l in LEVELS))
    print(f"  {'personnes (COE1)':22}"
          + "".join(f"{by_person[l]:8.1f}" for l in LEVELS))
    print(f"\n  {len(hh)} households, {len(people)} matched persons")

    if not positive["passed"]:
        print("\n[ÉCHEC] the positive check does not pass: the reading does not reproduce the "
              "published car ownership, so the derived distribution is not enforceable. "
              "Nothing is written — fix the weighting or the filtering before publishing.")
        return 1

    payload = {
        "source": "EMC² Toulouse 2023, ProGEDO/ADISP lil-1750 — M6 (voitures), P7 (permis)",
        "derivation": "règle eqasim : all si voitures>=permis (majeurs), some si <, "
                      "none si voitures==0",
        "reference_by_household_coe0": by_household,
        "reference_by_person_coe1": by_person,
        "n_households": int(len(hh)),
        "n_persons": int(len(people)),
        "negative_control": negative,
        "positive_control": positive,
        "caveat": "Cible de NIVEAU seulement. Elle ne dit rien de la RIVALITÉ dans le "
                  "temps (deux membres d'un ménage à une voiture conduisant au même "
                  "instant) : c'est l'objet de l'option B du ticket 018, et aucune "
                  "distribution ne la mesure.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n  written → {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
