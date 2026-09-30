"""export_equipment_propensity.py — Lot 1 of tickets 016 and 017, in a single pass.

Two persona traits are currently **copied** from the ENTD 2008 donor matched to the
person (`services/eqasim-toulouse/synthesis/population/enriched.py`), and both are wrong in
the same way: the total roughly holds, the breakdown is reversed, and the cause is
the same — `matching_attributes` uses an age class `[14, 29, 44, 59, 74]` that
covers ages 15 to 29 as a single block, whereas the survey sees the propensity drop by a
factor of 2 inside that class.

| Trait | Measured gap (ticket) |
|---|---|
| `has_pt_subscription` | students **−38.9 pt**, retirees **+12.0 pt**, overall −3.9 |
| `has_driving_license` | aged 18-24 **+27.3 pt**, overall +5.6 |

Both tickets state that "lots 1 and 2 are shared: same source file, same
restriction `PENQ = 1`, same weighting `COEP`, same cause, same correction pattern.
Handling them together halves the cost; handling them separately means writing the same
loader twice." This script is that pooling: one loader, two targets,
two resources.

## What it writes

`mobility_core/data/pt_subscription.json` and `mobility_core/data/driving_license.json`:
logit coefficients, occupation vocabulary, median density of the scope, out-of-sample
acceptance tables, provenance block. **No microdata** — this is what
allows committing the resources although their source is restricted-access
(ProGEDO/ADISP `lil-1750`).

## Two method decisions, written down rather than suffered

**1. Cross-validation is grouped by household, never by person.** Two members
of the same household share car ownership, zone and context: a split by
person would put the same household on both sides and overestimate generalisation.
67 % of households have only one surveyed person, which bounds from the start what can be
learnt about the intra-household correlation — ticket 016 asks to publish it, not to
keep it quiet.

**2. The fare thresholds are fitted then arbitrated, not decreed.** The Tisséo rule
"under 26" and the senior eligibility (65, or 62 for retirees) say
*where* to place a break in the age curve; they do not say it is worth it.
The script fits each trait **with and without** these thresholds and publishes both
out-of-sample AUCs. The decision rule is written here, before seeing the result:

    the thresholds are kept if the out-of-sample AUC grouped by household gains
    at least KNOT_MIN_GAIN; otherwise they are removed and the removal is printed.

This is the only way to prevent a "domain-flavoured" covariate from staying because
it looks relevant.

Usage:
    services/llm-agents/.venv/bin/python -m scripts.progedo_logit.export_equipment_propensity
    services/llm-agents/.venv/bin/python -m scripts.progedo_logit.export_equipment_propensity --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from mobility_core.equipment_propensity import (
    DRIVING_LICENSE,
    FEATURE_BASE,
    FEATURE_KNOTS,
    PT_SUBSCRIPTION,
    RESOURCE_DIR,
    TraitSpec,
    design_vector,
    write_resource,
)
from scripts.progedo_logit.build_mode_choice_dataset import (
    GENDER,
    MAIN_OCCUPATION,
    build_geo,
    find_project_root,
    load_raw,
)

# Regularisation. Weak penalisation: we want the survey's law, not a parsimonious
# model — the goal is to reproduce published strata, and a tight `C` would
# crush precisely the sparsely populated modalities the ticket asks to hold.
PROPENSITY_C = 1e3
CV_SPLITS = 5

# Minimum out-of-sample AUC gain to keep the fare thresholds. Set before
# seeing the result (see the header). 0.002 is the order of the CV noise on ~15 000
# persons: below it, a gain cannot be told apart from a splitting artefact.
KNOT_MIN_GAIN = 0.002

# A cell under this number of weighted observations settles nothing: it is
# flagged in the tables, not silently smoothed.
THIN_CELL = 30.0

# Binding targets, recomputed on `lil-1750` by tickets 016 and 017. They are
# NOT recomputed here: the script measures what its law reproduces, and the
# comparison with these values is what tells whether lot 1 holds.
AGE_BANDS_PT = ((5, 18, "5-17"), (18, 25, "18-24"), (25, 35, "25-34"),
                (35, 50, "35-49"), (50, 65, "50-64"), (65, 200, "65+"))
AGE_BANDS_LICENSE = ((18, 25, "18-24"), (25, 35, "25-34"), (35, 50, "35-49"),
                     (50, 65, "50-64"), (65, 75, "65-74"), (75, 200, "75+"))

TARGETS_PT = {
    "overall": 25.8,
    "occupation": {"Étudiant": 74.3, "Scolaire (jusqu'au Bac)": 33.3,
                   "Chômeur/recherche d'emploi": 28.8, "Personne au foyer": 24.0,
                   "Travail à temps partiel": 21.5, "Retraité": 17.7,
                   "Travail à plein temps": 14.8},
    "age": {"5-17": 32.9, "18-24": 63.3, "25-34": 22.7, "35-49": 15.2,
            "50-64": 14.8, "65+": 18.9},
    "cars": {"0": 61.8, "1": 25.5, "2+": 16.1},
    "gender": {"Female": 28.0, "Male": 23.6},
}

TARGETS_LICENSE = {
    "overall": 85.9,
    "occupation": {"Travail à plein temps": 94.8, "Retraité": 92.6,
                   "Travail à temps partiel": 86.5,
                   "Chômeur/recherche d'emploi": 69.4, "Personne au foyer": 63.9,
                   "Étudiant": 59.2},
    "age": {"18-24": 58.1, "25-34": 84.3, "35-49": 91.8, "50-64": 94.3,
            "65-74": 93.8, "75+": 88.9},
    "gender": {"Male": 88.6, "Female": 83.4},
}


# ── Loading — a single one, for both traits ───────────────────────────────────

def load_people(root: Path) -> pd.DataFrame:
    """Surveyed persons, with both targets and the shared covariates.

    Restriction `PENQ = 1` and weighting `COEP`: `COEP` is zero for the
    non-surveyed, so any weighted statistic is already correctly restricted —
    but we filter explicitly rather than rely on that coincidence.

    The real household key is `(ZF, ECH)`: `ECH` alone is not unique from one zone to
    another. Same rule as `build_household` and `export_bike_ownership`.
    """
    progedo = root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
    pers, men, _ = load_raw(progedo / "fichiers_standards")

    men = men.copy()
    men["cars"] = pd.to_numeric(men["M6"], errors="coerce")
    geo, _, _ = build_geo(root / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg", men)
    zone = geo.reindex(men["ZFM"]).reset_index(drop=True)
    men["density"] = zone["density_hh_km2"].values
    men["dist_center"] = zone["dist_center_km"].values
    indexed = men.set_index(["ZFM", "ECH"])

    pers = pers.copy()
    key = pd.MultiIndex.from_arrays([pers["ZFP"], pers["ECH"]])
    people = pd.DataFrame({
        "hh_id": [f"{z}|{e}" for z, e in zip(pers["ZFP"], pers["ECH"])],
        "PENQ": pers["PENQ"].values,
        "age": pd.to_numeric(pers["P4"], errors="coerce").values,
        "gender": pers["P2"].map(GENDER).values,
        "main_occupation": pers["P9"].map(MAIN_OCCUPATION).values,
        "weight": pd.to_numeric(pers["COEP"], errors="coerce").values,
        # Both targets, recoded here and nowhere else.
        #
        # `P7 == 3` = "supervised driving and driving lessons": 266 persons,
        # median age 18, 155 of them adults. They are NOT licence holders — the
        # `.eq("1")` says so, but it is written down so that nobody rereads it as an
        # oversight.
        "has_pt_subscription": pers["P12"].eq("6").values,
        "has_driving_license": pers["P7"].eq("1").values,
        "cars": key.map(indexed["cars"]).values,
        "density": key.map(indexed["density"]).values,
        "dist_center": key.map(indexed["dist_center"]).values,
    })

    before = len(people)
    people = people[(people["PENQ"] == "1") & (people["weight"] > 0)]
    people = people.dropna(
        subset=["age", "gender", "main_occupation", "cars", "dist_center", "weight"]
    ).reset_index(drop=True)
    print(f"Persons: {before} → {len(people)} surveyed and usable")
    solo = int((people.groupby("hh_id").size() == 1).sum())
    n_hh = people.hh_id.nunique()
    print(f"Households with a single surveyed person: {solo}/{n_hh} "
          f"({100 * solo / n_hh:.0f} %) — identification bound: the residual "
          f"intra-household correlation will not be reproduced, only measured")
    return people


# ── Fitting ──────────────────────────────────────────────────────────────────

def matrix(people: pd.DataFrame, occupations: tuple[str, ...],
           features: tuple[str, ...], median_density: float) -> pd.DataFrame:
    """Design matrix, built by the **module's** function.

    Never a copy of the formula: this is what guarantees that training and
    application see exactly the same vector. A copy would drift at the first
    recoding change.
    """
    rows = [
        design_vector(age, gender, occupation, cars, density, dist,
                      occupations, features, median_density)
        for age, gender, occupation, cars, density, dist in zip(
            people["age"], people["gender"], people["main_occupation"],
            people["cars"], people["density"], people["dist_center"])
    ]
    return pd.DataFrame(rows, columns=list(features))


def fit(people: pd.DataFrame, target: str, occupations: tuple[str, ...],
        features: tuple[str, ...], median_density: float) -> tuple[LogisticRegression,
                                                                   np.ndarray, float]:
    """Weighted logit + out-of-sample prediction grouped by household."""
    X = matrix(people, occupations, features, median_density)
    y = people[target].astype(int)
    weight = people["weight"]
    model = LogisticRegression(max_iter=5000, C=PROPENSITY_C).fit(
        X, y, sample_weight=weight)

    oof = np.zeros(len(people))
    for train, test in GroupKFold(n_splits=CV_SPLITS).split(
            X, y, groups=people["hh_id"]):
        fold = LogisticRegression(max_iter=5000, C=PROPENSITY_C).fit(
            X.iloc[train], y.iloc[train], sample_weight=weight.iloc[train])
        oof[test] = fold.predict_proba(X.iloc[test])[:, 1]
    auc_out = float(roc_auc_score(y, oof, sample_weight=weight))
    return model, oof, auc_out


def fit_with_knot_arbitration(people: pd.DataFrame, spec: TraitSpec,
                              occupations: tuple[str, ...],
                              median_density: float) -> dict:
    """Fits without then with the fare thresholds, and decides by the written rule."""
    base = tuple(FEATURE_BASE) + tuple(f"occ_{o}" for o in occupations)
    with_knots = tuple(FEATURE_BASE) + tuple(FEATURE_KNOTS) + tuple(
        f"occ_{o}" for o in occupations)

    print(f"\n── {spec.key} " + "─" * max(0, 56 - len(spec.key)))
    _, oof_base, auc_base = fit(people, spec.key, occupations, base, median_density)
    model_k, oof_k, auc_k = fit(people, spec.key, occupations, with_knots,
                                median_density)
    gain = auc_k - auc_base
    keep = gain >= KNOT_MIN_GAIN
    print(f"Out-of-sample AUC (household-grouped CV): no thresholds {auc_base:.4f} | "
          f"with thresholds {auc_k:.4f} | gain {gain:+.4f}")
    print(f"Rule (gain ≥ {KNOT_MIN_GAIN}) → fare thresholds "
          f"{'RETENUS' if keep else 'RETIRÉS'}"
          + ("" if keep else " — the age break is already carried by the "
                             "continuous terms and the occupations"))

    features = with_knots if keep else base
    if keep:
        model, oof, auc = model_k, oof_k, auc_k
    else:
        model, oof, auc = fit(people, spec.key, occupations, features, median_density)
    auc_in = float(roc_auc_score(
        people[spec.key].astype(int),
        model.predict_proba(matrix(people, occupations, features,
                                   median_density))[:, 1],
        sample_weight=people["weight"]))
    print(f"In-sample AUC {auc_in:.4f} | out-of-sample {auc:.4f}")
    return {
        "law": {
            "features": list(features),
            "occupations": list(occupations),
            "intercept": round(float(model.intercept_[0]), 8),
            "coefficients": [round(float(v), 8) for v in model.coef_[0]],
            "median_density": round(median_density, 6),
        },
        "fit": {
            "n_persons": int(len(people)),
            "restriction": "PENQ = 1 (personnes enquêtées)",
            "weighting": "COEP — coefficient de redressement de la personne enquêtée",
            "regularisation_C": PROPENSITY_C,
            "cv": f"GroupKFold({CV_SPLITS}) groupée par ménage — un découpage par "
                  f"personne mettrait le même foyer des deux côtés",
            "auc_in_sample": round(auc_in, 4),
            "auc_out_of_sample_grouped_by_household": round(auc, 4),
            "knots_offered": list(FEATURE_KNOTS),
            "knots_retained": keep,
            "knot_auc_gain": round(float(gain), 5),
            "knot_min_gain_rule": KNOT_MIN_GAIN,
        },
        "oof": oof,
    }


# ── Acceptance ───────────────────────────────────────────────────────────────

def _band(age: float, bands) -> str:
    for low, high, label in bands:
        if low <= age < high:
            return label
    return "hors bandes"


def _cars_class(cars: float) -> str:
    return "0" if cars <= 0 else ("1" if cars < 2 else "2+")


def stratum_table(people: pd.DataFrame, target: str, oof: np.ndarray,
                  keys: pd.Series, order, targets: dict) -> list[dict]:
    """Observed / out-of-sample predicted / target, by stratum, with counts.

    The displayed gap is **predicted − observed**: it is what the law misses on the
    survey itself, out of sample. The target column also tells whether the script's
    observed value matches the value published by the ticket — if it does not, it is
    not the law that is at fault but the definition of the stratum, and it must be seen.
    """
    weight = people["weight"].values
    y = people[target].astype(float).values
    rows = []
    for label in order:
        mask = (keys == label).values
        w = weight[mask]
        if w.sum() <= 0:
            continue
        observed = 100.0 * float((w * y[mask]).sum() / w.sum())
        predicted = 100.0 * float((w * oof[mask]).sum() / w.sum())
        rows.append({
            "stratum": label,
            "n_persons": int(mask.sum()),
            "n_weighted": round(float(w.sum()), 1),
            "observed_pct": round(observed, 2),
            "predicted_oof_pct": round(predicted, 2),
            "gap_pt": round(predicted - observed, 2),
            "ticket_target_pct": targets.get(label),
            "thin_cell": bool(w.sum() < THIN_CELL),
        })
    return rows


def validation(people: pd.DataFrame, spec: TraitSpec, oof: np.ndarray,
               bands, targets: dict) -> dict:
    """The acceptance tables of the tickets, measured out of sample."""
    scope = people if spec.key == "has_pt_subscription" else people[people.age >= 18]
    idx = scope.index
    sub_oof = oof[idx]
    weight = scope["weight"].values
    y = scope[spec.key].astype(float).values
    overall_obs = 100.0 * float((weight * y).sum() / weight.sum())
    overall_pred = 100.0 * float((weight * sub_oof).sum() / weight.sum())

    out = {
        "scope": ("5 ans et plus" if spec.key == "has_pt_subscription"
                  else "18 ans et plus"),
        "overall": {
            "observed_pct": round(overall_obs, 2),
            "predicted_oof_pct": round(overall_pred, 2),
            "gap_pt": round(overall_pred - overall_obs, 2),
            "ticket_target_pct": targets["overall"],
        },
        "by_occupation": stratum_table(
            scope, spec.key, sub_oof, scope["main_occupation"],
            list(targets["occupation"]), targets["occupation"]),
        "by_age": stratum_table(
            scope, spec.key, sub_oof,
            scope["age"].map(lambda a: _band(a, bands)),
            list(targets["age"]), targets["age"]),
        "by_gender": stratum_table(
            scope, spec.key, sub_oof, scope["gender"],
            list(targets["gender"]), targets["gender"]),
    }
    if "cars" in targets:
        out["by_household_cars"] = stratum_table(
            scope, spec.key, sub_oof, scope["cars"].map(_cars_class),
            list(targets["cars"]), targets["cars"])
    # Intra-household correlation: a VALIDATION criterion, not an identity (ticket 016).
    # If the law stays at its independence level, the car-ownership covariate is not
    # doing its job, and that must be said rather than believing it useful.
    multi = scope.groupby("hh_id").filter(lambda g: len(g) >= 2)
    if len(multi):
        obs_all = multi.groupby("hh_id")[spec.key].all()
        out["intra_household"] = {
            "households_with_2plus_respondents": int(obs_all.size),
            "all_equipped_pct_observed": round(100.0 * float(obs_all.mean()), 2),
            "note": "à comparer au niveau d'indépendance ; un modèle qui y reste "
                    "signale que la motorisation ne porte pas la corrélation",
        }
    return out


def report(spec: TraitSpec, block: dict) -> None:
    v = block["validation"]
    print(f"\nAcceptance — {spec.key} ({v['scope']})")
    o = v["overall"]
    print(f"  overall: observed {o['observed_pct']:.1f} % | predicted oos "
          f"{o['predicted_oof_pct']:.1f} % | gap {o['gap_pt']:+.1f} | "
          f"ticket target {o['ticket_target_pct']}")
    for name, key in (("occupation", "by_occupation"), ("âge", "by_age"),
                      ("genre", "by_gender"), ("motorisation", "by_household_cars")):
        if key not in v:
            continue
        print(f"  by {name}:")
        for row in v[key]:
            flag = "  ⚠ cellule mince" if row["thin_cell"] else ""
            target = row["ticket_target_pct"]
            target_txt = f"cible {target:5.1f}" if target is not None else "cible   —  "
            print(f"    {row['stratum']:28s} n={row['n_persons']:5d} "
                  f"obs {row['observed_pct']:5.1f} % | pred {row['predicted_oof_pct']:5.1f} % "
                  f"| gap {row['gap_pt']:+5.1f} | {target_txt}{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="fits and prints the acceptance tables, writes no resources")
    args = parser.parse_args(argv)

    root = find_project_root()
    progedo = root / "data" / "PROGEDO 2023"
    if not progedo.is_dir():
        print(f"[erreur] PROGEDO data missing: {progedo} "
              f"(restricted access lil-1750)", file=sys.stderr)
        return 1

    people = load_people(root)
    occupations = tuple(sorted(set(MAIN_OCCUPATION.values())))
    median_density = float(people["density"].median())
    print(f"Vocabulary of occupations ({len(occupations)}): {', '.join(occupations)}")
    print(f"Median density of the scope: {median_density:.1f} households/km²")

    written = []
    for spec, bands, targets in ((PT_SUBSCRIPTION, AGE_BANDS_PT, TARGETS_PT),
                                 (DRIVING_LICENSE, AGE_BANDS_LICENSE,
                                  TARGETS_LICENSE)):
        block = fit_with_knot_arbitration(people, spec, occupations, median_density)
        block["validation"] = validation(people, spec, block.pop("oof"), bands,
                                         targets)
        report(spec, block)
        if not args.dry_run:
            path = RESOURCE_DIR / spec.resource
            write_resource(path, spec, block["law"],
                           {**block["validation"], "fit": block["fit"]},
                           {"source": "EMC² Toulouse 2023 — ProGEDO/ADISP lil-1750, "
                                      "fichier standard `pers`",
                            "tickets": ["016", "017"]})
            written.append(path)

    if args.dry_run:
        print("\n--dry-run: no resource written.")
    else:
        for path in written:
            print(f"\nWritten: {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
