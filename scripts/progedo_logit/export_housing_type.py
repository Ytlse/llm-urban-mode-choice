"""export_housing_type.py — Law of the housing type by fine zone and household size.

The EMC² reference breaks down modal shares by **housing type**, but the population
generation chain does not produce this trait: neither eqasim nor the INSEE tables
used by the notebook (AAV zoning, density grid) carry the information.
The only source that carries it for the Toulouse scope is the survey itself,
variable `M1` of the households file (« Type d'habitat »), whose modalities are
exactly those of the published breakdown.

This script extracts from it the law needed to impute the trait at population
generation (action A2, revised by **ticket 019**):

- **law conditional on the fine zone**, weighted by the adjustment coefficients
  of **households** (`COE0`): a household occupies one dwelling and draws once. Before
  ticket 019 the weighting was that of persons (`COEP`), which *by coincidence*
  compensated for the absence of household size in the conditioning — so the
  weighting must never be changed without the size lever, see `_internal_check`
  which publishes both measurements side by side;
- **household size lever** at scope level, `P(M1 | size) / P(M1)` for
  classes 1, 2, 3, 4 and more. The module applies it to the zone law then
  renormalises (odds-ratio transfer). Serving the raw law
  `P(M1 | zone, size)` was ruled out: the 2 145 (zone, size) cells count
  **3 households at the median** and only 18 reach 30 observations;
- **hierarchical smoothing** zone → sampling sector → whole scope. A fine
  zone counts 12 surveyed households at the median, a sector counts 122: serving the
  raw law of a zone with 2 respondents would pass sampling noise off as
  geography. The fallback weight, `PRIOR_WEIGHT`, is set to the median count of a
  zone — at the median, the zone and its sector therefore weigh the same;
- **no threshold hides anything**: the surveyed count of each zone (in households and
  in persons) is written in the resource next to its smoothed law, and any cell
  (size × modality) under `THIN_CELL` observations is flagged, not silently smoothed.

The script finally publishes the **EMC² internal test** required by ticket 019 as an
executable criterion: each surveyed household gets its zone's law corrected by the
lever of its size, and it is compared with its real `M1`. This measurement — mean
absolute error over the 20 cells (5 modalities × 4 sizes) — tells whether the mechanism
beats the previous one; it is written in the resource to be rereadable without the data.

What the script writes in `mobility_core/data/zf_housing_type.json`: the modalities, the
overall law, the law by sector, the law by zone, the size levers and their
counts, the validation block, and a `meta` provenance block. **No
microdata** — only aggregated laws, like the fine-zone layer exported
by `export_zone_layer.py`, and with the same status: outside the repository,
regenerable, never committed.

Usage:
    python -m scripts.progedo_logit.export_housing_type [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mobility_core.housing_type import (
    DEFAULT_RESOURCE,
    MODALITY_KEYS,
    SECTOR_PREFIX_LEN,
    SIZE_MAX,
    rake,
)
from scripts.progedo_logit.build_mode_choice_dataset import find_project_root, load_raw

# `M1` recoding (households file) → keys of `cerema_values.yaml`. The survey labels
# are: 1 Individuel isolé, 2 Individuel accolé, 3 Petit collectif
# (R+1 à R+3), 4 Grand collectif (R+4 et plus), 5 Autres.
HOUSING = {
    "1": "individuel_isole",
    "2": "individuel_accole",
    "3": "petit_habitat_collectif",
    "4": "grand_habitat_collectif",
    "5": "autres",
}

# Fallback weight, in surveyed **households**. Set to the median count of a fine zone:
# a median zone then weighs as much as its sector, a well-surveyed zone dominates its
# sector, a zone with 2 respondents fades into it. The unit followed the weighting
# (ticket 019): 18 persons at the median, but 12 households.
PRIOR_WEIGHT = 12.0

# Fallback weight before ticket 019, in surveyed persons (median 18). Kept
# to replay the previous mechanism in the internal test, and for nothing else.
PREVIOUS_PRIOR = 18.0

# Flagging threshold for a (size × modality) cell of the lever block. Ticket
# 019 requires it: "any cell under 30 weighted observations is flagged, not
# silently smoothed". The count compared with the threshold is that of **surveyed
# households**: the `COE0` weights are extrapolation coefficients to the population (a
# household weighs ~60), a threshold of 30 set on their sum would never trigger.
THIN_CELL = 30

# Maximum mean absolute error of the EMC² internal test, in points, over the 20 cells
# (5 modalities × 4 sizes). Acceptance criterion of ticket 019; the earlier
# mechanism scored 3.00 pt, the delivered mechanism 0.75.
MAX_MEAN_ABS_ERROR_PT = 1.0


def load_survey(progedo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two tables of the test: the surveyed **households**, and the **persons**.

    Households carry the housing type, the `COE0` weight, the household size and its
    lever class: they are what builds the served law. Household key = (fine
    zone, sample) — `ECH` alone is not unique from one zone to another, same rule
    as `build_household`. The size is **rebuilt from the persons file**
    (`M4`/`M5` are entirely empty in the standard file).

    Persons carry the housing type of their household and the `COEP` weight: it is the
    exact table that the mechanism before ticket 019 used, and it now only serves
    to replay it in the internal test — no longer something compared from memory.
    """
    pers, men, _ = load_raw(progedo_dir)

    sizes = pers.groupby(["ZFP", "ECH"]).size().rename("size")
    key = pd.MultiIndex.from_arrays([men["ZFM"], men["ECH"]])

    households = pd.DataFrame({
        "ZF": men["ZFM"],
        "housing": men["M1"].map(HOUSING),
        "weight": pd.to_numeric(men["COE0"], errors="coerce"),
        "size": key.map(sizes),
    })
    n0 = len(households)
    households = households.dropna(subset=["ZF", "housing", "weight", "size"])
    households = households[households["weight"] > 0]
    households["bucket"] = np.minimum(households["size"].astype(int), SIZE_MAX)
    print(f"Households: {n0} at the start → {len(households)} with housing type, size "
          f"and weighting")
    print("  by size: " + " / ".join(
        f"{size}{'+' if size == SIZE_MAX else ''} : "
        f"{int((households['bucket'] == size).sum())}"
        for size in range(1, SIZE_MAX + 1)))

    housing_of = (men.assign(housing=men["M1"].map(HOUSING))
                  .drop_duplicates(["ZFM", "ECH"])
                  .set_index(["ZFM", "ECH"])["housing"])
    persons = pd.DataFrame({
        "ZF": pers["ZFP"],
        "housing": pd.MultiIndex.from_arrays([pers["ZFP"], pers["ECH"]]).map(housing_of),
        "weight": pd.to_numeric(pers["COEP"], errors="coerce"),
    })
    persons = persons.dropna(subset=["ZF", "housing", "weight"])
    persons = persons[persons["weight"] > 0]
    print(f"Persons: {len(persons)} with housing type and weighting (they only "
          f"serve to replay the mechanism before ticket 019)")
    return households.reset_index(drop=True), persons.reset_index(drop=True)


def _shares(frame: pd.DataFrame) -> np.ndarray:
    """Weighted shares in the order of `MODALITY_KEYS`, summing to 1."""
    mass = frame.groupby("housing")["weight"].sum()
    vector = np.array([float(mass.get(key, 0.0)) for key in MODALITY_KEYS])
    total = vector.sum()
    return vector / total if total > 0 else vector


def _smooth(observed: np.ndarray, n: float, prior: np.ndarray,
            prior_weight: float | None = None) -> np.ndarray:
    """Observed law pulled towards its fallback, in proportion to the surveyed count.

    `prior_weight` is only made explicit by the person-weighted comparison variant
    (`_person_weighted`), whose count is in persons and not in households.
    """
    weight = PRIOR_WEIGHT if prior_weight is None else prior_weight
    return (n * observed + weight * prior) / (n + weight)


def build_geography(households: pd.DataFrame,
                    persons_per_zone: pd.Series) -> tuple[dict, dict, np.ndarray]:
    """Smoothed laws by zone and by sector, plus the overall law of the scope."""
    households = households.assign(sector=households["ZF"].str[:SECTOR_PREFIX_LEN])
    overall = _shares(households)

    sectors: dict[str, dict] = {}
    for sector, frame in households.groupby("sector"):
        sectors[str(sector)] = {
            "n": int(len(frame)),
            "shares": [round(float(v), 6)
                       for v in _smooth(_shares(frame), len(frame), overall)],
        }

    zones: dict[str, dict] = {}
    for zf, frame in households.groupby("ZF"):
        sector = str(zf)[:SECTOR_PREFIX_LEN]
        prior = np.array(sectors[sector]["shares"]) if sector in sectors else overall
        zones[str(zf)] = {
            "n": int(len(frame)),
            "n_persons": int(persons_per_zone.get(str(zf), 0)),
            "shares": [round(float(v), 6)
                       for v in _smooth(_shares(frame), len(frame), prior)],
        }
    return zones, sectors, overall


def build_size_leverage(households: pd.DataFrame, overall: np.ndarray) -> dict:
    """Levers `P(M1 | size) / P(M1)`, with the counts of each cell.

    The lever is estimated **at scope level**, not by zone: this is the transfer
    assumption of ticket 019, and it is what makes the law servable — the
    (zone, size) cell counts 3 households at the median.

    A modality with zero mass at scope level gets a lever of 1 (neutral) rather
    than a division by zero: it is absent from all laws anyway.
    """
    out: dict[str, dict] = {}
    for bucket in range(1, SIZE_MAX + 1):
        frame = households[households["bucket"] == bucket]
        shares = _shares(frame)
        leverage = np.divide(shares, overall, out=np.ones_like(shares),
                            where=overall > 0)
        cells = frame.groupby("housing").agg(n=("weight", "size"),
                                            weighted_n=("weight", "sum"))
        out[str(bucket)] = {
            "n": int(len(frame)),
            "weighted_n": round(float(frame["weight"].sum()), 1),
            "shares": [round(float(v), 6) for v in shares],
            "leverage": [round(float(v), 6) for v in leverage],
            "cells": [
                {
                    "modality": key,
                    "n": int(cells["n"].get(key, 0)),
                    "weighted_n": round(float(cells["weighted_n"].get(key, 0.0)), 1),
                    "thin": bool(int(cells["n"].get(key, 0)) < THIN_CELL),
                }
                for key in MODALITY_KEYS
            ],
        }
    return out


def _law_of(zf: str, zones: dict, sectors: dict, overall: np.ndarray) -> np.ndarray:
    """Geographic law served for a zone — same fallback as `HousingTypeTable`."""
    node = zones.get(str(zf))
    if node is not None:
        return np.array(node["shares"], dtype=float)
    sector = sectors.get(str(zf)[:SECTOR_PREFIX_LEN])
    if sector is not None:
        return np.array(sector["shares"], dtype=float)
    return overall


def _imputed_by_size(households: pd.DataFrame, zones: dict, sectors: dict,
                     overall: np.ndarray,
                     leverage: dict[int, np.ndarray] | None) -> dict[int, np.ndarray]:
    """Mean imputed law by size class, `COE0`-weighted.

    We compare **laws**, not draws: the hash draw reproduces the law
    up to sampling noise (checked in `test_housing_type.py`), and measuring
    on the law avoids making an acceptance criterion depend on a set of seeds.
    """
    out: dict[int, np.ndarray] = {}
    for bucket, frame in households.groupby("bucket"):
        tilt = None if leverage is None else leverage[int(bucket)]
        weights = frame["weight"].to_numpy(dtype=float)
        laws = np.array([rake(_law_of(zf, zones, sectors, overall), tilt)
                         for zf in frame["ZF"]], dtype=float)
        out[int(bucket)] = (laws * weights[:, None]).sum(axis=0) / weights.sum()
    return out


def _variant(households: pd.DataFrame, zones: dict, sectors: dict,
             overall: np.ndarray, leverage: dict[int, np.ndarray] | None,
             label: str) -> dict:
    """A variant of the mechanism, measured inside EMC²."""
    imputed = _imputed_by_size(households, zones, sectors, overall, leverage)
    weights_by_size = households.groupby("bucket")["weight"].sum()
    total = float(weights_by_size.sum())

    cells, by_size = [], []
    for bucket in sorted(imputed):
        frame = households[households["bucket"] == bucket]
        observed = _shares(frame)
        for index, key in enumerate(MODALITY_KEYS):
            cells.append({
                "size": bucket,
                "modality": key,
                "observed_pct": round(float(100 * observed[index]), 2),
                "imputed_pct": round(float(100 * imputed[bucket][index]), 2),
                "abs_error_pt": round(
                    float(100 * abs(observed[index] - imputed[bucket][index])), 2),
            })
        by_size.append({
            "size": bucket,
            "n": int(len(frame)),
            "individuel_isole_observed_pct": round(float(100 * observed[0]), 2),
            "individuel_isole_imputed_pct": round(float(100 * imputed[bucket][0]), 2),
        })

    marginal = sum(float(weights_by_size[b]) * imputed[b] for b in imputed) / total
    return {
        "label": label,
        "mean_abs_error_pt": round(
            float(np.mean([cell["abs_error_pt"] for cell in cells])), 3),
        "by_size": by_size,
        "cells": cells,
        "overall_marginal_observed_pct": [
            round(float(100 * v), 2) for v in _shares(households)],
        "overall_marginal_imputed_pct": [round(float(100 * v), 2) for v in marginal],
    }


def internal_check(households: pd.DataFrame, persons: Optional[pd.DataFrame],
                   zones: dict, sectors: dict, overall: np.ndarray,
                   size_leverage: dict) -> dict:
    """The EMC² internal test of ticket 019, and the two mechanisms it replaces.

    Three variants are measured on the same households:

    1. **zone only, person weighting** — the mechanism before ticket 019;
    2. **zone only, household weighting** — to show that the weighting is NOT
       the issue: on its own, it degrades. Whoever measured only that one would
       conclude that one must go back to `COEP`;
    3. **zone with household weighting + size lever** — the delivered mechanism.

    The test is *in-sample*: the laws are estimated on the households that are also
    used to evaluate them. It therefore measures not a generalisation ability but the
    **fidelity of the mechanism** — a mechanism that cannot reproduce the gradient of
    the population on which it is estimated will reproduce it nowhere, and this is
    exactly what the previous one did (3.00 pt in-sample).
    """
    leverage = {int(size): np.array(node["leverage"], dtype=float)
                for size, node in size_leverage.items()}

    # Variant 1: the zone law as it was weighted before ticket 019.
    # Rebuilt here rather than read from the old resource, so that the
    # comparison bears on the single measured change and not on a dated file.
    baselines = []
    if persons is not None and len(persons):
        before = _person_weighted(persons)
        baselines.append(_variant(households, *before, None,
                                  "zone seule, pondération personnes "
                                  "(avant ticket 019)"))
    households_only = _variant(households, zones, sectors, overall, None,
                               "zone seule, pondération ménages")
    raked = _variant(households, zones, sectors, overall, leverage,
                     "zone en pondération ménages + levier de taille (livré)")

    return {
        "note": "Chaque ménage enquêté reçoit la loi de sa zone (corrigée du levier de "
                "sa taille pour la variante livrée) ; on la compare à son M1 réel. "
                "Mesure EN PLACE, sans biais de périmètre : c'est la fidélité du "
                "mécanisme, pas sa généralisation.",
        "max_mean_abs_error_pt": MAX_MEAN_ABS_ERROR_PT,
        "passes": raked["mean_abs_error_pt"] <= MAX_MEAN_ABS_ERROR_PT,
        "delivered": raked,
        "baselines": baselines + [households_only],
    }


def _person_weighted(persons: pd.DataFrame) -> tuple[dict, dict, np.ndarray]:
    """The zone laws as the mechanism before ticket 019 built them.

    **Exact** replay: one row per surveyed person, `COEP` weighting, smoothing
    count in persons, fallback weight at 18. Rebuilt here rather than reread from
    the old resource, so that the comparison bears on the mechanism and not on a
    dated file — but it is indeed the same table and the same smoothing code.
    """
    overall = _shares(persons)
    sectors: dict[str, dict] = {}
    for sector, frame in persons.groupby(persons["ZF"].str[:SECTOR_PREFIX_LEN]):
        sectors[str(sector)] = {
            "n": int(len(frame)),
            "shares": list(_smooth(_shares(frame), len(frame), overall, PREVIOUS_PRIOR)),
        }
    zones: dict[str, dict] = {}
    for zf, frame in persons.groupby("ZF"):
        prior = np.array(sectors[str(zf)[:SECTOR_PREFIX_LEN]]["shares"])
        zones[str(zf)] = {
            "n": int(len(frame)),
            "shares": list(_smooth(_shares(frame), len(frame), prior, PREVIOUS_PRIOR)),
        }
    return zones, sectors, overall


def build_table(households: pd.DataFrame,
                persons: Optional[pd.DataFrame] = None) -> dict:
    """The complete resource: geography, size levers, validation, provenance.

    `persons` only serves the replay of the mechanism before ticket 019 in the
    validation block, and to publish the person count of each zone. Without it, the
    resource is complete but the comparison point is missing.
    """
    per_zone = (persons.groupby("ZF").size() if persons is not None and len(persons)
                else pd.Series(dtype=int))
    zones, sectors, overall = build_geography(households, per_zone)
    size_leverage = build_size_leverage(households, overall)
    validation = internal_check(households, persons, zones, sectors,
                                overall, size_leverage)

    counts = households.groupby("ZF").size()
    cells = households.groupby(["ZF", "bucket"]).size()
    return {
        "version": 2,
        "trait": "housing_type",
        "modalities": list(MODALITY_KEYS),
        "sizes": list(range(1, SIZE_MAX + 1)),
        "global": [round(float(v), 6) for v in overall],
        "size_leverage": size_leverage,
        "sectors": sectors,
        "zones": zones,
        "validation": validation,
        "meta": {
            "source": "EMC² Toulouse 2023 (ProGEDO lil-1750), fichiers standards "
                      "ménages (M1 « Type d'habitat », COE0) et personnes (taille du "
                      "ménage reconstituée)",
            "weighting": "COE0 — coefficient de redressement du ménage enquêté. Un "
                         "ménage occupe un logement et tire une fois ; la marginale "
                         "personnes se reconstitue par le conditionnement sur la taille",
            "conditioning": "zone fine × taille du ménage (1, 2, 3, 4+), la taille "
                            f"entrant par un levier P(M1|taille)/P(M1) estimé au "
                            f"périmètre puis renormalisé (ticket 019)",
            "smoothing": f"zone → secteur de tirage ({SECTOR_PREFIX_LEN} premiers "
                         f"caractères du code ZF) → périmètre, poids du repli "
                         f"{PRIOR_WEIGHT} ménages",
            "thin_cell_threshold": THIN_CELL,
            "n_households": int(len(households)),
            "n_zones": len(zones),
            "n_sectors": len(sectors),
            "median_households_per_zone": float(np.median(counts)) if len(counts) else 0.0,
            "zones_under_5_households": int((counts < 5).sum()),
            "n_zone_size_cells": int(len(cells)),
            "median_households_per_zone_size_cell": (
                float(np.median(cells)) if len(cells) else 0.0),
            "zone_size_cells_over_30": int((cells >= 30).sum()),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def report(table: dict) -> None:
    """What the reader must see without opening the JSON."""
    print("\nOverall law (households, COE0-weighted):")
    for key, share in zip(MODALITY_KEYS, table["global"]):
        print(f"  {key:26s} {100 * share:5.2f} %")

    print("\nSize levers P(M1|size)/P(M1) — 1 = neutral:")
    header = "  " + " ".join(f"{key[:12]:>13s}" for key in MODALITY_KEYS)
    print(f"  {'taille':>6s} {'n':>6s}" + header)
    for size in table["sizes"]:
        node = table["size_leverage"][str(size)]
        line = " ".join(f"{value:13.3f}" for value in node["leverage"])
        print(f"  {size:>6d} {node['n']:>6d}   {line}")
    thin = [(size, cell["modality"], cell["n"])
            for size in table["sizes"]
            for cell in table["size_leverage"][str(size)]["cells"] if cell["thin"]]
    if thin:
        print(f"  cells under {THIN_CELL} surveyed households (flagged, not smoothed):")
        for size, modality, n in thin:
            print(f"    size {size} × {modality:26s} n = {n}")
    else:
        print(f"  no cell under {THIN_CELL} surveyed households")

    print("\nEMC² internal test — share of detached houses by household size:")
    delivered = table["validation"]["delivered"]
    print(f"  {'taille':>6s} {'observé':>9s} {'imputé':>9s} {'écart':>7s}")
    for row in delivered["by_size"]:
        observed = row["individuel_isole_observed_pct"]
        imputed = row["individuel_isole_imputed_pct"]
        print(f"  {row['size']:>6d} {observed:8.1f}% {imputed:8.1f}% "
              f"{imputed - observed:+7.1f}")
    print("\n  mean absolute error over the 20 cells (5 modalities × 4 sizes):")
    for variant in table["validation"]["baselines"]:
        print(f"    {variant['mean_abs_error_pt']:5.2f} pt   {variant['label']}")
    print(f"    {delivered['mean_abs_error_pt']:5.2f} pt   {delivered['label']}")
    verdict = "TENU" if table["validation"]["passes"] else "NON TENU"
    print(f"  ticket 019 criterion (≤ {MAX_MEAN_ABS_ERROR_PT} pt): {verdict}")

    print("\n  overall marginal — the geography must not move:")
    print("    observed: " + " / ".join(
        f"{v:.1f}" for v in delivered["overall_marginal_observed_pct"]))
    print("    imputed : " + " / ".join(
        f"{v:.1f}" for v in delivered["overall_marginal_imputed_pct"]))

    meta = table["meta"]
    print(f"\n{meta['n_zones']} zones, {meta['n_sectors']} sectors, median "
          f"{meta['median_households_per_zone']:.0f} households/zone, "
          f"{meta['zones_under_5_households']} zones under 5 households (they fade "
          f"behind their sector)")
    print(f"{meta['n_zone_size_cells']} (zone, size) cells, median "
          f"{meta['median_households_per_zone_size_cell']:.0f} households, "
          f"{meta['zone_size_cells_over_30']} with 30 households or more — this is why "
          f"size enters through a scope lever and not through a raw crossing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output file (default: {DEFAULT_RESOURCE})")
    args = parser.parse_args()

    root = find_project_root()
    progedo_dir = (root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
                   / "fichiers_standards")

    households, persons = load_survey(progedo_dir)
    table = build_table(households, persons)

    out = args.out or DEFAULT_RESOURCE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")

    report(table)
    print(f"→ {out}")
    if not table["validation"]["passes"]:
        print("\n[ALARME] EMC² internal test above the ticket 019 threshold: the law "
              "is written, but it does not meet the acceptance criterion.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
