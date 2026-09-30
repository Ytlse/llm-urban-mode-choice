"""export_bike_ownership.py — The three stages of bike equipment, learned on EMC².

Produces `mobility_core/data/bike_ownership.json`: the coefficients of the two logits of
ticket 015, plus the validation tables used to judge the result.

**Stage 1 — how many bikes in the household.** Multinomial logit on `k = M21` clipped at
`4+`, 10 783 households, weighting `COE0`. Covariates: household size, number of cars
(`M6`), and the residence zone through its household density and distance to the centre.
Neither `M1` (housing type) nor `M2` (dwelling occupancy) — the reasons are written in
`packages/mobility_core/src/mobility_core/bike_ownership.py`, and they differ: `M1` is less
informative than the zone it is imputed from, `M2` does not exist on the persona side.

**Stage 2 — who, within the household, holds the bikes.** Binary logit on `P20 ∈ {plusieurs
jours/semaine, plusieurs jours/mois, occasionnellement}`, the practice declared as a
**driver** — the best available indicator of "whose bike is this", and there is no
other: the survey never asks who owns what. Restricted to `PENQ = 1`
(15 775 persons out of 20 890), weighting `COEP` — which is exactly 0 for the
non-surveyed, so any `COEP`-weighted statistic is already correctly restricted.

**Identification limit, to be accepted and not worked around**: 67 % of households have
only one surveyed person (7 238 out of 10 783). We can estimate `P(practice |
covariates)`; we can **not** observe who, among three siblings, rides.
Attribution is therefore independent conditionally on `k`, with no intra-household
correlation modelled — and the script computes and writes it, rather than leave it unsaid.

**Cross-validation is grouped by household, never by person**: two members of the
same household share `k`, the leak would be mechanical.

What the script writes: **coefficients** and **aggregated tables**, no
microdata — same status as `export_housing_type.py` and the fine-zone layer: outside
the repository, regenerable, never committed.

Usage:
    python -m scripts.progedo_logit.export_bike_ownership [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from mobility_core.bike_ownership import (
    DEFAULT_RESOURCE,
    K_CLASSES,
    K_MAX,
    MIN_AGE_ELECTRIC,
    MIN_AGE_ELIGIBLE,
    PROPENSITY_BASE_FEATURES,
    SIZE_MAX,
    STOCK_FEATURES,
    VAE_SHARE,
    Member,
    assign,
    propensity_design,
    stock_design,
)
from mobility_core.housing_type import MODALITY_KEYS, HousingTypeTable, draw
from scripts.progedo_logit.build_mode_choice_dataset import (
    GENDER,
    MAIN_OCCUPATION,
    build_geo,
    find_project_root,
    load_raw,
)

# `P20` modalities that count as "rides a bike as a driver".
# `4` = « Jamais » (never), and blanks are the non-surveyed (already dropped by `PENQ`).
PRACTICE_CODES = ("1", "2", "3")

# `M1` recoding → `core.housing_type` keys, only for the validation tables
# (stage 1 does NOT use housing type — see the module).
HOUSING = {
    "1": "individuel_isole", "2": "individuel_accole",
    "3": "petit_habitat_collectif", "4": "grand_habitat_collectif", "5": "autres",
}

# Regularisation. Deliberately weak on stage 1 (counts are large and we want the
# observed law, not a law shrunk towards uniform) and standard on stage 2,
# where the occupation dummies have thin cells.
STOCK_C = 1e3
PROPENSITY_C = 1.0
CV_SPLITS = 5

# Clipping of the household-size BUCKETS in the validation tables. Distinct from
# `SIZE_MAX` (the clipping of the model dummies): the law of `k` caps at `4+`
# because the survey shows it really caps there (2.62 / 2.64 / 2.42 bikes
# for sizes 4 / 5 / 6), but the HOLDER rate keeps falling beyond that,
# the denominator growing without the numerator (63 % / 52 % / 40 %). A "5+" bucket
# would therefore mix two very different regimes, and its value would depend entirely
# on the relative weight of sizes 5 and 6 — which is not the same in the survey and in a
# synthetic population. The buckets therefore go up to `6+` on both sides.
SIZE_BUCKET_MAX = 6

# Cell flagging threshold. The ticket requires it: "any cell under 30
# weighted observations is flagged, not silently smoothed".
THIN_CELL = 30.0


# ── Loading ──────────────────────────────────────────────────────────────────

def load_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Survey households and persons, with `k`, e-bikes, household size and zone.

    The real household key is `(ZFM, ECH)` — `ECH` alone is not unique from one zone to
    another, same rule as `build_household`.

    `ML21` (number of electrically assisted bikes) does **not** exist in the
    standard file, where `M22` is entirely empty: the original file is needed, whose key
    is `(MP2, ECH)` with a 4-character `ECH` where the standard one has 5 (the
    first being the sample type). The match is checked below on
    `M20 == M21`, rather than assumed from the row order.
    """
    progedo = root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
    pers, men, _ = load_raw(progedo / "fichiers_standards")

    original = pd.read_csv(
        progedo / "fichiers_originaux"
        / "04a_EMC2_Toulouse_2023_Men_coef_ML21_21062023.csv",
        dtype=str, sep=None, engine="python")
    for column in original.columns:
        original[column] = original[column].str.strip().replace({"": np.nan})

    men = men.copy()
    men["k_raw"] = pd.to_numeric(men["M21"], errors="coerce")
    men["weight"] = pd.to_numeric(men["COE0"], errors="coerce")
    men["cars"] = pd.to_numeric(men["M6"], errors="coerce")
    men["housing"] = men["M1"].map(HOUSING)
    men["_ech4"] = men["ECH"].str[1:]

    vae = original.assign(
        n_vae=pd.to_numeric(original["ML21"], errors="coerce"),
        m20=pd.to_numeric(original["M20"], errors="coerce"),
    ).rename(columns={"MP2": "ZFM", "ECH": "_ech4"})
    men = men.merge(vae[["ZFM", "_ech4", "n_vae", "m20"]],
                    on=["ZFM", "_ech4"], how="left", validate="one_to_one")
    matched = men["m20"].notna()
    if not matched.all():
        raise SystemExit(
            f"Incomplete match with the original file: {(~matched).sum()} households "
            "without ML21. The key (MP2, ECH) has changed — check the lil-1750 delivery.")
    disagree = int((men["m20"] != men["k_raw"]).sum())
    if disagree:
        raise SystemExit(
            f"Suspicious match with the original file: {disagree} households where M20 "
            "(original) differs from M21 (standard). They are supposedly the same variable; "
            "do not export an e-bike fleet on a wrong join.")
    print(f"ML21 matched on {len(men)} households (M20 == M21 everywhere)")

    # Household size and eligible members aged 5 and over, counted on the persons file.
    pers = pers.copy()
    pers["age"] = pd.to_numeric(pers["P4"], errors="coerce")
    sizes = pers.groupby(["ZFP", "ECH"]).size().rename("size")
    eligibles = (pers[pers["age"] >= MIN_AGE_ELIGIBLE]
                 .groupby(["ZFP", "ECH"]).size().rename("n_eligible"))
    key = pd.MultiIndex.from_arrays([men["ZFM"], men["ECH"]])
    men["size"] = key.map(sizes)
    men["n_eligible"] = key.map(eligibles).fillna(0)

    geo, _, _ = build_geo(root / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg", men)
    zone = geo.reindex(men["ZFM"]).reset_index(drop=True)
    men["density"] = zone["density_hh_km2"].values
    men["dist_center"] = zone["dist_center_km"].values

    # Persons: demographics, declared practice, and the equipment of their household.
    key_p = pd.MultiIndex.from_arrays([pers["ZFP"], pers["ECH"]])
    indexed = men.set_index(["ZFM", "ECH"])
    people = pd.DataFrame({
        "hh_id": [f"{z}|{e}" for z, e in zip(pers["ZFP"], pers["ECH"])],
        "ZF": pers["ZFP"].values,
        "PENQ": pers["PENQ"].values,
        "age": pers["age"].values,
        "gender": pers["P2"].map(GENDER).values,
        "main_occupation": pers["P9"].map(MAIN_OCCUPATION).values,
        "weight": pd.to_numeric(pers["COEP"], errors="coerce").values,
        "practises": pers["P20"].isin(PRACTICE_CODES).values,
        "k_raw": key_p.map(indexed["k_raw"]).values,
        "size": key_p.map(indexed["size"]).values,
        "density": key_p.map(indexed["density"]).values,
        "dist_center": key_p.map(indexed["dist_center"]).values,
    })

    n_men, n_pers = len(men), len(people)
    men = men.dropna(subset=["k_raw", "weight", "size", "dist_center"]).reset_index(drop=True)
    people = people[(people["PENQ"] == "1") & (people["weight"] > 0)]
    people = people.dropna(
        subset=["k_raw", "size", "age", "dist_center", "weight"]).reset_index(drop=True)
    print(f"Households: {n_men} → {len(men)} usable")
    print(f"Persons   : {n_pers} → {len(people)} surveyed and usable")
    solo = int((people.groupby("hh_id").size() == 1).sum())
    print(f"Households with a single surveyed person: {solo}/{people.hh_id.nunique()} "
          f"({100 * solo / people.hh_id.nunique():.0f} %) — attribution therefore cannot "
          f"model any intra-household correlation")
    return men, people


# ── Design matrices, through the module functions ────────────────────────────
# We go through `stock_design` / `propensity_design` rather than copying the
# formulas: this is what guarantees that training and application see
# exactly the same vector. A copy would drift at the first change.

def stock_matrix(men: pd.DataFrame, median_density: float) -> pd.DataFrame:
    rows = [
        stock_design(int(size), cars,
                     median_density if pd.isna(density) else float(density),
                     float(dist))
        for size, cars, density, dist in zip(
            men["size"], men["cars"], men["density"], men["dist_center"])
    ]
    return pd.DataFrame(rows, columns=list(STOCK_FEATURES))


def propensity_matrix(people: pd.DataFrame, occupations: tuple[str, ...],
                      median_density: float) -> pd.DataFrame:
    rows = [
        propensity_design(int(k), int(size), age, gender, occupation,
                          median_density if pd.isna(density) else float(density),
                          float(dist), occupations)
        for k, size, age, gender, occupation, density, dist in zip(
            people["k_raw"], people["size"], people["age"], people["gender"],
            people["main_occupation"], people["density"], people["dist_center"])
    ]
    columns = list(PROPENSITY_BASE_FEATURES) + [f"occ_{o}" for o in occupations]
    return pd.DataFrame(rows, columns=columns)


# ── Stage 1 ──────────────────────────────────────────────────────────────────

def fit_stock(men: pd.DataFrame, median_density: float) -> tuple[dict, np.ndarray]:
    """Multinomial logit on clipped `k`. Returns the resource block and the predicted laws."""
    X = stock_matrix(men, median_density)
    y = men["k_raw"].clip(upper=K_MAX).astype(int)
    model = LogisticRegression(max_iter=5000, C=STOCK_C).fit(
        X, y, sample_weight=men["weight"])
    if tuple(int(c) for c in model.classes_) != K_CLASSES:
        raise SystemExit(
            f"Unexpected stage 1 classes: {model.classes_} instead of {K_CLASSES}. "
            "A clipped k must cover 0..K_MAX without a gap.")

    # Cross-validation grouped by household. Here the household IS the observation, so the
    # grouping is trivial; the fold remains an honest out-of-sample fold for
    # calibration, and we write it so that the resource says what it was judged on.
    folds = GroupKFold(n_splits=CV_SPLITS)
    oof = np.zeros((len(men), len(K_CLASSES)))
    groups = men["ZFM"].astype(str) + "|" + men["ECH"].astype(str)
    for train, test in folds.split(X, y, groups=groups):
        fold = LogisticRegression(max_iter=5000, C=STOCK_C).fit(
            X.iloc[train], y.iloc[train], sample_weight=men["weight"].iloc[train])
        oof[test] = fold.predict_proba(X.iloc[test])

    doc = {
        "target": "k = M21 écrêté à 4+, nombre de vélos du ménage",
        "features": list(STOCK_FEATURES),
        "classes": list(K_CLASSES),
        "intercepts": [round(float(v), 8) for v in model.intercept_],
        "coefficients": [[round(float(v), 8) for v in row] for row in model.coef_],
        "n_households": int(len(men)),
        "weighting": "COE0 — coefficient de redressement du ménage",
        "regularisation_C": STOCK_C,
    }
    return doc, oof


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def fit_propensity(people: pd.DataFrame, occupations: tuple[str, ...],
                   median_density: float) -> tuple[dict, np.ndarray]:
    """Binary logit on the declared practice. Returns the resource block and the OOF."""
    X = propensity_matrix(people, occupations, median_density)
    y = people["practises"].astype(int)
    weight = people["weight"]
    model = LogisticRegression(max_iter=5000, C=PROPENSITY_C).fit(
        X, y, sample_weight=weight)

    folds = GroupKFold(n_splits=CV_SPLITS)
    oof = np.zeros(len(people))
    for train, test in folds.split(X, y, groups=people["hh_id"]):
        fold = LogisticRegression(max_iter=5000, C=PROPENSITY_C).fit(
            X.iloc[train], y.iloc[train], sample_weight=weight.iloc[train])
        oof[test] = fold.predict_proba(X.iloc[test])[:, 1]

    auc_in = float(roc_auc_score(y, model.predict_proba(X)[:, 1], sample_weight=weight))
    auc_out = float(roc_auc_score(y, oof, sample_weight=weight))
    print(f"Stage 2 — in-sample AUC {auc_in:.4f} | out-of-sample (grouped by household) "
          f"{auc_out:.4f}")
    doc = {
        "target": "P20 ∈ {plusieurs jours/semaine, plusieurs jours/mois, "
                  "occasionnellement} — vélo, conducteur",
        "features": list(X.columns),
        "classes": [1],
        "intercepts": [round(float(model.intercept_[0]), 8)],
        "coefficients": [[round(float(v), 8) for v in model.coef_[0]]],
        "n_persons": int(len(people)),
        "restriction": "PENQ = 1 (personnes enquêtées)",
        "weighting": "COEP — coefficient de redressement de la personne enquêtée",
        "regularisation_C": PROPENSITY_C,
        "auc_in_sample": round(auc_in, 4),
        "auc_out_of_sample_grouped_by_household": round(auc_out, 4),
        "cv": f"GroupKFold({CV_SPLITS}) groupée par ménage — un split par personne "
              f"fuirait, deux membres d'un foyer partageant k",
    }
    return doc, oof


# ── Validation tables ────────────────────────────────────────────────────────

def _share(frame: pd.DataFrame, mask: pd.Series, weight: str = "weight") -> float:
    total = frame[weight].sum()
    return float(100.0 * frame.loc[mask, weight].sum() / total) if total > 0 else float("nan")


def stock_validation(men: pd.DataFrame, oof: np.ndarray) -> dict:
    """What stage 1 must reproduce, and what it reproduces out of sample."""
    weight = men["weight"].values
    equipped_pred = 1.0 - oof[:, 0]
    expected_k = oof @ np.array(K_CLASSES, dtype=float)
    total = weight.sum()
    eligible = men["n_eligible"].values
    raw_k = men["k_raw"].values
    clipped_k = men["k_raw"].clip(upper=K_MAX).values
    out: dict = {
        "overall": {
            "equipped_pct_observed": round(_share(men, men["k_raw"] > 0), 2),
            "equipped_pct_predicted": round(float(100 * (weight * equipped_pred).sum() / total), 2),
            "bikes_per_household_observed": round(
                float((weight * clipped_k).sum() / total), 4),
            "bikes_per_household_predicted": round(
                float((weight * expected_k).sum() / total), 4),
        },
        # What clipping at `4+` costs, measured rather than assumed.
        #
        # The ticket's acceptance criterion asks for **1.22 bikes/household (± 0.05)**.
        # That is the published figure, computed on unclipped `M21`. A model clipped at
        # `4+` — the clipping the ticket itself specifies, and on which all its
        # reference tables are built — structurally cannot reach it:
        # it caps at 1.151, i.e. 0.065 below, which eats up the whole tolerance.
        # The criterion is therefore **restated**: the binding target is the clipped mean.
        #
        # This is not a concession for comfort, and here is the measurement that shows it:
        # on the quantity the trait actually carries — the *attributable* bikes,
        # `min(k, eligible aged 5 and over)`, since a bike without a holder does not
        # appear in the JSON — clipping costs only 0.011 bike per household and affects
        # only ~1 % of households. The 4.1 % of households with 5 bikes or more have on
        # average fewer than 5 eligible members: their surplus bikes would in any
        # case have had no one to carry them.
        "clipping_cost": {
            "k_max": K_MAX,
            "bikes_per_household_unclipped": round(float((weight * raw_k).sum() / total), 4),
            "bikes_per_household_clipped": round(float((weight * clipped_k).sum() / total), 4),
            "attributable_per_household_unclipped": round(
                float((weight * np.minimum(raw_k, eligible)).sum() / total), 4),
            "attributable_per_household_clipped": round(
                float((weight * np.minimum(clipped_k, eligible)).sum() / total), 4),
            "households_losing_an_attributable_bike_pct": round(float(
                100 * weight[(raw_k > K_MAX) & (eligible > K_MAX)].sum() / total), 2),
            # Weighted standard deviation of attributable bikes per household. Served so
            # that the application can bound the sampling noise of its mean:
            # on 7 measurable households (population of 10 agents), σ/√n is 0.45 bike and
            # requiring ± 0.05 would measure nothing but chance.
            "attributable_sd": round(float(np.sqrt(
                np.average((np.minimum(clipped_k, eligible)
                            - np.average(np.minimum(clipped_k, eligible),
                                         weights=weight)) ** 2,
                           weights=weight))), 4),
            "note": "Le critère publié « 1,22 vélo/ménage » porte sur M21 non écrêté ; "
                    "un modèle écrêté à 4+ plafonne à la valeur `..._clipped`, qui est "
                    "la cible opposable. L'écrêtage est sans effet sur le trait produit "
                    "(cf. `attributable_*`, 0,011 d'écart) parce que l'attribution est "
                    "de toute façon bornée par le nombre de membres éligibles.",
        },
        "by_household_size": [],
        "by_housing_observed": [],
    }
    for size, frame in men.groupby(men["size"].clip(upper=SIZE_BUCKET_MAX)):
        index = frame.index
        out["by_household_size"].append({
            "size": int(size),
            "n": int(len(frame)),
            "weighted_n": round(float(frame["weight"].sum()), 1),
            "equipped_pct_observed": round(_share(frame, frame["k_raw"] > 0), 2),
            "equipped_pct_predicted": round(float(
                100 * (frame["weight"].values * equipped_pred[index]).sum()
                / frame["weight"].sum()), 2),
            # Served for the direct standardisation on the application side: the
            # measurable households of a synthetic population are not a neutral sample
            # of household sizes (a one-person household is always complete), so
            # the target must be recomposed on the breakdown actually measured.
            "bikes_per_household_observed": round(float(
                (frame["weight"] * frame["k_raw"].clip(upper=K_MAX)).sum()
                / frame["weight"].sum()), 4),
            # Bikes actually ATTRIBUTABLE, `min(k, eligible aged 5 and over)`.
            # It is this quantity — and not the stock — that the individual trait carries:
            # a bike without a holder does not appear in the JSON. Comparing the bikes
            # attributed in a population with the survey stock confuses two quantities
            # (0.33 versus 0.44 among people living alone, the gap being the bikes that
            # no one can carry).
            "attributable_per_household_observed": round(float(
                (frame["weight"] * np.minimum(frame["k_raw"].clip(upper=K_MAX),
                                              frame["n_eligible"])).sum()
                / frame["weight"].sum()), 4),
            "thin": bool(frame["weight"].sum() < THIN_CELL),
        })
    for housing, frame in men.groupby("housing"):
        index = frame.index
        out["by_housing_observed"].append({
            "housing": str(housing),
            "n": int(len(frame)),
            "weighted_n": round(float(frame["weight"].sum()), 1),
            "equipped_pct_observed": round(_share(frame, frame["k_raw"] > 0), 2),
            "equipped_pct_predicted": round(float(
                100 * (frame["weight"].values * equipped_pred[index]).sum()
                / frame["weight"].sum()), 2),
            "thin": bool(frame["weight"].sum() < THIN_CELL),
        })
    return out


def diluted_housing_reference(men: pd.DataFrame, root: Path, draws: int = 8) -> dict:
    """The **binding** target of equipment by housing type, and why it differs.

    The ticket asks to reproduce 71 % (detached house) → 38 % (large apartment block),
    the published curve. This criterion is **out of reach by construction** on a
    synthetic population: the persona's housing type is itself imputed from the law of
    its fine zone, and it matches the real housing type only one time in two. Crossing
    the **true survey** number of bikes with the **imputed** housing type is enough to
    crush the amplitude from 33 to ~19 points: this is regression dilution, it caps
    what the measurement can see, and no model of `k` can undo it.

    This function computes that ceiling, by replaying the housing imputation of
    `enrich_housing_type` on the survey households (whose true `k` is known), and
    averaging over several draws so that the target does not depend on a seed.
    It is this curve that the population validation must aim for; the published
    curve stays written alongside, as the source figure it is.

    Since **ticket 019**, the replayed imputation is conditioned on **household
    size** in addition to the zone: it is the same law as the one served in production,
    and the agreement rate with the observed housing type is measured here rather than
    recited. It sets the extent of the dilution, hence the height of the ceiling.
    """
    # `FileNotFoundError`: resource never exported. `ValueError`: resource exported
    # for a module version other than the installed one — this is the case during the
    # rollout of ticket 019, whose module requires v2 while the exporter still produces
    # v1. Both are the same functional case: "no usable housing
    # law". We degrade and say so; we do not bring down the whole export of the
    # bike model for a side target.
    try:
        table = HousingTypeTable.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[avertissement] Loi du type de logement inutilisable — la cible diluée "
              f"par habitat n'est PAS calculée, et le contrôle de cet axe sera donc "
              f"absent du rapport d'enrichissement (pas « réussi » : absent).\n"
              f"  cause : {exc}\n"
              f"  suite : `make housing-type`, puis relancer `make bike-ownership`.")
        return {}

    weight = men["weight"].values
    # Two quantities per modality, and **both are needed** — this is the trap of this axis.
    #
    # `households`  : share of equipped HOUSEHOLDS (`k > 0`). This is the definition of the
    #                 published curve ("Equipped households, detached house: 70.9 %").
    # `holders`     : share of PERSONS given a bike, under the rules of the mechanism
    #                 (`min(min(k, K_MAX), eligible) / size`).
    #
    # The two differ by 2 to 10 points **and the gap follows household size**: the
    # families live in houses, and a household of four with one bike is "equipped"
    # while only one of its members is given one. Serving the household share as the
    # target of a per-person measurement therefore produces a negative bias on ALL
    # modalities, the stronger the more family-oriented the housing (−10.0 pts in detached
    # house versus −3.2 in large apartment block). It is a unit mix-up, not a model defect.
    accumulated: dict[str, list[float]] = {key: [] for key in MODALITY_KEYS}
    accumulated_holders: dict[str, list[float]] = {key: [] for key in MODALITY_KEYS}
    attributable = np.minimum(men["k_raw"].clip(upper=K_MAX), men["n_eligible"]).values
    sizes_arr = men["size"].values
    # Agreement rate between imputed and observed housing type. It is THIS that governs
    # the dilution, and it changes with each improvement of the imputation (47.6 % with
    # the zone-only law, 50.2 % after the size raking of ticket 019). We therefore
    # measure it instead of freezing it in a comment that would silently go stale.
    agreement: list[float] = []
    # The housing law is served by (zone, **household size**) since ticket 019:
    # we therefore impute here with the real size of each surveyed household, exactly as
    # `enrich_housing_type` will do on the population. Measuring the dilution with a law
    # other than the one actually applied would give a wrong target.
    sizes = men["size"].tolist()
    for seed in range(draws):
        # Draw independent of the production one (distinct salt through the offset):
        # we want the expectation of the crossing, not the realisation of a given file.
        rng = np.random.default_rng(seed)
        uniforms = rng.random(len(men))
        imputed = [draw(table.shares_for(zf, size), u)
                   for zf, size, u in zip(men["ZFM"], sizes, uniforms)]
        series = pd.Series(imputed, index=men.index)
        agreement.append(float((series.values == men["housing"].values).mean()))
        for key in MODALITY_KEYS:
            mask = (series == key).values
            if not mask.any():
                continue
            accumulated[key].append(
                float(100 * weight[mask & (men["k_raw"].values > 0)].sum()
                      / weight[mask].sum()))
            denominator = float((weight[mask] * sizes_arr[mask]).sum())
            if denominator > 0:
                accumulated_holders[key].append(
                    float(100 * (weight[mask] * attributable[mask]).sum() / denominator))

    observed = {str(h): round(_share(f, f["k_raw"] > 0), 2)
                for h, f in men.groupby("housing")}
    return {
        "note": "Cible opposable à une population synthétique. L'habitat du persona "
                "étant lui-même IMPUTÉ, le croisement est dilué : même avec le k VRAI "
                "de l'enquête, l'amplitude tombe sous celle de la courbe publiée. "
                "Comparer la population à la courbe PUBLIÉE reviendrait à exiger du "
                "modèle qu'il sur-corrige pour compenser le bruit de l'axe de mesure. "
                "La dilution est RECALCULÉE à chaque export avec la loi d'habitat du "
                "moment : elle se resserre à mesure que cette imputation s'améliore "
                "(ticket 019).",
        "imputed_vs_observed_agreement_pct": (
            round(float(np.mean(agreement)) * 100, 1) if agreement else None),
        "draws": draws,
        "published_on_observed_housing": observed,
        # Share of equipped HOUSEHOLDS, to compare with the published curve (same unit).
        "attainable_households_equipped_pct": {
            key: round(float(np.mean(values)), 2)
            for key, values in accumulated.items() if values
        },
        # Share of PERSONS given a bike: it is THIS ONE that the enrichment report
        # holds against the population, because the `personal_bike` trait is individual.
        "attainable_on_imputed_housing": {
            key: round(float(np.mean(values)), 2)
            for key, values in accumulated_holders.items() if values
        },
        "unit": "attainable_on_imputed_housing = part de PERSONNES dotées d'un vélo "
                "(min(min(k, K_MAX), éligibles) / taille). La courbe publiée et "
                "`attainable_households_equipped_pct` sont, elles, des parts de "
                "MÉNAGES équipés : ne pas comparer les deux unités.",
        # Amplitude as a share of HOUSEHOLDS, hence directly comparable to the 33.4 pts published.
        "attainable_spread_pts": round(float(
            np.mean(accumulated["individuel_isole"])
            - np.mean(accumulated["grand_habitat_collectif"])), 2)
        if accumulated["individuel_isole"] and accumulated["grand_habitat_collectif"]
        else None,
    }


def practice_validation(people: pd.DataFrame, oof: np.ndarray) -> dict:
    """The table `P(practice | k, size)`, observed and predicted out of sample."""
    cells = []
    k = people["k_raw"].clip(upper=K_MAX)
    size = people["size"].clip(upper=SIZE_MAX)
    for stock in range(1, K_MAX + 1):
        for members in range(1, SIZE_MAX + 1):
            mask = (k == stock) & (size == members)
            frame = people[mask]
            if frame.empty:
                continue
            weight = frame["weight"]
            cells.append({
                "k": stock,
                "size": members,
                "n": int(len(frame)),
                "weighted_n": round(float(weight.sum()), 1),
                "practice_pct_observed": round(
                    float(100 * (weight * frame["practises"]).sum() / weight.sum()), 2),
                "practice_pct_predicted": round(
                    float(100 * (weight * oof[mask.values]).sum() / weight.sum()), 2),
                "thin": bool(weight.sum() < THIN_CELL),
            })
    overall = people["weight"]
    return {
        "overall_practice_pct_observed": round(
            float(100 * (overall * people["practises"]).sum() / overall.sum()), 2),
        "overall_practice_pct_predicted": round(
            float(100 * (overall * oof).sum() / overall.sum()), 2),
        "by_k_and_size": cells,
    }


def holder_targets(men: pd.DataFrame, people: pd.DataFrame) -> dict:
    """The **person**-level targets: holders, size gradient, riders per bike.

    "Persons given a bike" is computed identically to what the
    mechanism will produce — `min(k, size)` bikes attributed per household, weighting
    `COE0` — so that the comparison bears on the same quantity.

    The share of persons **living in** an equipped household (63.2 % in `COEP`) is written
    alongside: it is the definition on which the mode choice policy was
    trained, and the gap with ~51 % is precisely the consumer constraint that
    lot 3 must address.
    """
    weight, k, size = men["weight"], men["k_raw"], men["size"]
    eligible = men["n_eligible"]
    # Two definitions, and both are needed.
    #
    # `holders_pct` is the ticket's PUBLISHED figure (50.9 %): `min(k, size)` on the
    # raw `k`, without clipping or age condition.
    #
    # `holders_pct_mechanism` is the same quantity computed **under the rules the
    # mechanism actually obeys**: `k` clipped at `K_MAX`, and attribution bounded by the
    # members aged 5 and over (`min(k, eligible)`), since a bike without a holder does
    # not appear in the JSON. It is this one that is binding for a population: the
    # published figure would ask the mechanism to produce holders it deliberately
    # refuses to produce — three-year-old children on bikes.
    #
    # The gap is negligible in aggregate (~0.7 point) and considerable for large
    # households, where young children are many and the stock exceeds the clipping: at
    # size 5, 59.7 % published versus ~52 % under the rules of the mechanism.
    attributable = np.minimum(k.clip(upper=K_MAX), eligible)
    holders = (weight * np.minimum(k, size)).sum() / (weight * size).sum()
    holders_mech = (weight * attributable).sum() / (weight * size).sum()
    by_size = []
    for value, frame in men.groupby(size.clip(upper=SIZE_BUCKET_MAX)):
        w, kk, ss = frame["weight"], frame["k_raw"], frame["size"]
        att = np.minimum(kk.clip(upper=K_MAX), frame["n_eligible"])
        by_size.append({
            "size": int(value),
            "n": int(len(frame)),
            "weighted_n": round(float(w.sum()), 1),
            "holders_pct": round(float(100 * (w * np.minimum(kk, ss)).sum()
                                       / (w * ss).sum()), 2),
            "holders_pct_mechanism": round(float(100 * (w * att).sum()
                                                 / (w * ss).sum()), 2),
            "thin": bool(w.sum() < THIN_CELL),
        })

    # Riders per household and per bike, COEP numerator on the surveyed, COE0
    # denominator on households — two weightings, as the ticket's check requires.
    practising = (people.assign(_p=people["weight"] * people["practises"])
                  .groupby("hh_id")["_p"].sum())
    eligible_5p = men["n_eligible"]
    hh_id = men["ZFM"].astype(str) + "|" + men["ECH"].astype(str)
    men_practising = hh_id.map(practising).fillna(0.0)
    per_bike = []
    for stock in range(1, K_MAX + 1):
        mask = k.clip(upper=K_MAX) == stock
        frame_weight = weight[mask]
        if frame_weight.sum() <= 0:
            continue
        per_bike.append({
            "k": stock,
            "n": int(mask.sum()),
            "persons_5plus_per_household": round(
                float((frame_weight * eligible_5p[mask]).sum() / frame_weight.sum()), 2),
            "practising_per_household": round(
                float(men_practising[mask].sum() / frame_weight.sum()), 2),
            "practising_per_bike": round(
                float(men_practising[mask].sum() / (frame_weight * stock).sum()), 2),
        })

    person_weight = people["weight"]
    return {
        "holders_pct": round(float(100 * holders), 2),
        "holders_definition": "Σ w·min(k, taille) / Σ w·taille, pondération COE0 — le "
                              "chiffre publié du ticket (50,9 %)",
        "holders_pct_mechanism": round(float(100 * holders_mech), 2),
        "holders_mechanism_definition": "Σ w·min(min(k, K_MAX), éligibles 5+) / Σ w·"
                                        "taille — la même grandeur sous les règles que "
                                        "le mécanisme obéit (écrêtage de k, pas de vélo "
                                        "sous 5 ans). C'est la cible OPPOSABLE.",
        "living_in_equipped_household_pct": round(float(
            100 * person_weight[people["k_raw"] > 0].sum() / person_weight.sum()), 2),
        "living_in_equipped_definition": "part de personnes dans un ménage à k > 0, "
                                         "pondération COEP — la définition sur laquelle "
                                         "la politique de choix modal a été entraînée "
                                         "(has_bike = M21 > 0)",
        "holders_by_household_size": by_size,
        "practising_per_bike": per_bike,
        "vae_share_of_fleet_pct": round(float(
            100 * (weight * men["n_vae"].fillna(0)).sum()
            / (weight * k).sum()), 2),
        "households_with_at_least_one_vae_pct": round(
            _share(men, men["n_vae"].fillna(0) > 0), 2),
    }


def mechanism_check(men: pd.DataFrame, people: pd.DataFrame,
                    occupations: tuple[str, ...], median_density: float,
                    propensity_doc: dict) -> dict:
    """The attribution mechanism replayed **on the survey households**.

    This is the check that counts: `k`, size and `P20` are all known there, so we
    can verify that the weighted draw without replacement reproduces the table
    `P(practice | k, size)` — which the module points out is not an identity
    of the Efraimidis–Spirakis scheme but a criterion to be checked afterwards.

    This same replay produces the **constructed** `has_bike` indicator that lot 3 must
    substitute for `M21 > 0` to retrain the mode choice policy: without it, the
    policy consumes "there is a bike in the household" (63.2 % of persons) where the
    persona carries a nominative attribution (~51 %), and the learnt coefficient
    applies to something other than what it measures.
    """
    from mobility_core.bike_ownership import LogitModel

    model = LogitModel.from_doc(propensity_doc)
    columns = list(propensity_doc["features"])
    rows = propensity_matrix(people, occupations, median_density)
    people = people.assign(_p=[model.probability(dict(zip(columns, row)))
                               for row in rows.to_numpy()])

    held: dict[int, bool] = {}
    for hh_id, frame in people.groupby("hh_id"):
        stock = int(frame["k_raw"].iloc[0])
        nominal = int(frame["size"].iloc[0])
        present = [Member(index=int(i), propensity=float(p),
                          eligible=bool(a >= MIN_AGE_ELIGIBLE))
                   for i, p, a in zip(frame.index, frame["_p"], frame["age"])]
        # Absent slots: the nominal household counts `size` persons, the survey only
        # describes part of them (67 % of households have a single respondent). Without
        # them, the k bikes would concentrate on the respondents alone and over-equip them —
        # exactly the bias of partially present households on the population side.
        mean_p = float(np.mean([m.propensity for m in present])) if present else 0.0
        absent = [Member(index=-1 - j, propensity=mean_p, eligible=True)
                  for j in range(max(0, nominal - len(frame)))]
        chosen = assign(present + absent, stock, hh_id)
        for member in present:
            held[member.index] = member.index in chosen

    people = people.assign(_held=[held.get(i, False) for i in people.index])
    weight = people["weight"]
    cells = []
    k = people["k_raw"].clip(upper=K_MAX)
    size = people["size"].clip(upper=SIZE_MAX)
    for stock in range(1, K_MAX + 1):
        for members in range(1, SIZE_MAX + 1):
            mask = (k == stock) & (size == members)
            frame = people[mask]
            if frame.empty:
                continue
            w = frame["weight"]
            cells.append({
                "k": stock, "size": members, "n": int(len(frame)),
                "practice_pct_observed": round(
                    float(100 * (w * frame["practises"]).sum() / w.sum()), 2),
                "held_pct_mechanism": round(
                    float(100 * (w * frame["_held"]).sum() / w.sum()), 2),
                "thin": bool(w.sum() < THIN_CELL),
            })
    held_pct = float(100 * (weight * people["_held"]).sum() / weight.sum())
    practice_pct = float(100 * (weight * people["practises"]).sum() / weight.sum())
    # Share of holders too young for an e-bike. It is what renormalises stage 3:
    # without it, applying 7.67 % to those aged 14 and over alone puts the fleet below the
    # target, in proportion to the bikes held by children (see `electric_probability`).
    held_weight = (weight * people["_held"]).sum()
    under_age = float(
        (weight * (people["_held"] & (people["age"] < MIN_AGE_ELECTRIC))).sum()
        / held_weight) if held_weight > 0 else 0.0
    return {
        "note": "Attribution rejouée sur les ménages de l'enquête (k, taille et P20 "
                "connus). `held_pct_mechanism` est l'indicateur has_bike CONSTRUIT sur "
                "lequel la politique de choix modal doit être ré-entraînée, à la place "
                "de M21 > 0.",
        "constructed_has_bike_pct": round(held_pct, 2),
        "practice_pct": round(practice_pct, 2),
        # Two distinct quantities, not to be confused — the ticket cites the second one.
        #
        # `dormant_gross`: the share of the population that holds a bike WITHOUT
        # riding it. This is the dormant mass in the proper sense, the one represented
        # on purpose because a bike in the garage is a bike.
        #
        # `dormant_net`: the holders − riders gap. It is smaller, because
        # there are ALSO riders without an attributed bike: bike-share users
        # (outside the ticket's scope) and the 7.9 % of riders living in a
        # household with zero bikes. The flow thus runs both ways, and only the net
        # can be read on the two totals.
        "dormant_gross_pts": round(float(
            100 * (weight * (people["_held"] & ~people["practises"])).sum()
            / weight.sum()), 2),
        "dormant_net_pts": round(held_pct - practice_pct, 2),
        "practising_without_bike_pts": round(float(
            100 * (weight * (~people["_held"] & people["practises"])).sum()
            / weight.sum()), 2),
        "under_age_holder_share": round(under_age, 6),
        "by_k_and_size": cells,
    }


# ── Output ───────────────────────────────────────────────────────────────────

def report(doc: dict) -> None:
    stock = doc["validation"]["stock"]
    print("\n── Stage 1: how many bikes in the household "
          "(out-of-sample) ────────────────")
    over = stock["overall"]
    print(f"  equipped households observed {over['equipped_pct_observed']:5.1f} %  "
          f"predicted {over['equipped_pct_predicted']:5.1f} %")
    print(f"  bikes per household observed {over['bikes_per_household_observed']:5.2f}    "
          f"predicted {over['bikes_per_household_predicted']:5.2f}")
    clip = stock["clipping_cost"]
    print(f"  [clipping {clip['k_max']}+] published stock "
          f"{clip['bikes_per_household_unclipped']:.3f} → binding target "
          f"{clip['bikes_per_household_clipped']:.3f} ; but ATTRIBUTABLE bikes "
          f"{clip['attributable_per_household_unclipped']:.3f} → "
          f"{clip['attributable_per_household_clipped']:.3f} "
          f"({clip['households_losing_an_attributable_bike_pct']:.2f} % of households "
          f"lose a carryable bike)")
    print(f"  {'taille':>8} {'observé':>9} {'prédit':>9} {'écart':>7}   n")
    for row in stock["by_household_size"]:
        print(f"  {row['size']:>8} {row['equipped_pct_observed']:8.1f}% "
              f"{row['equipped_pct_predicted']:8.1f}% "
              f"{row['equipped_pct_predicted'] - row['equipped_pct_observed']:+7.1f}"
              f"   {row['n']}{'  [cellule mince]' if row['thin'] else ''}")
    print(f"  {'habitat (observé, hors modèle)':>30} {'observé':>9} {'prédit':>9}")
    for row in stock["by_housing_observed"]:
        print(f"  {row['housing']:>30} {row['equipped_pct_observed']:8.1f}% "
              f"{row['equipped_pct_predicted']:8.1f}%"
              f"{'  [cellule mince]' if row['thin'] else ''}")

    diluted = doc["validation"].get("housing_reference") or {}
    if diluted.get("attainable_on_imputed_housing"):
        print("\n── Housing: the binding target, and why it is not the published one ─────")
        print(f"  {'modalité':>30} {'publiée':>9} {'ménages':>10} {'personnes':>11}")
        print(f"  {'(unité)':>30} {'ménages':>9} {'ménages':>10} {'personnes':>11}")
        households = diluted.get("attainable_households_equipped_pct") or {}
        for key, value in diluted["attainable_on_imputed_housing"].items():
            published = diluted["published_on_observed_housing"].get(key)
            shown = f"{published:8.1f}%" if published is not None else f"{'—':>9}"
            hh = households.get(key)
            hh_shown = f"{hh:9.1f}%" if hh is not None else f"{'—':>10}"
            print(f"  {key:>30} {shown} {hh_shown} {value:10.1f}%")
        print("  The « personnes » column is the target held against the population: the\n"
              "  `personal_bike` trait is individual. The gap between the last two columns\n"
              "  follows household size — a household of four with one bike is « equipped »,\n"
              "  but only one of its members is given one.")
        published = diluted["published_on_observed_housing"]
        published_spread = (published["individuel_isole"]
                            - published["grand_habitat_collectif"])
        accord = diluted.get("imputed_vs_observed_agreement_pct")
        print(f"  reachable amplitude detached − large apartment block: "
              f"{diluted['attainable_spread_pts']:.1f} pts, versus "
              f"{published_spread:.1f} published pts")
        print(f"  The gap IS the dilution of the imputed housing type"
              + (f" (imputed/observed agreement: {accord:.1f} %)" if accord else "")
              + ": it shrinks\n  as that imputation gains in precision "
                "(ticket 019). Nothing to fix on the bike side.")

    practice = doc["validation"]["practice"]
    print("\n── Stage 2: P(practice | k, size) — observed / predicted "
          "out-of-sample ────")
    print(f"  riders  observed {practice['overall_practice_pct_observed']:5.2f} %  "
          f"predicted {practice['overall_practice_pct_predicted']:5.2f} %")
    for stock_value in range(1, K_MAX + 1):
        cells = [c for c in practice["by_k_and_size"] if c["k"] == stock_value]
        line = "  ".join(
            f"{c['practice_pct_observed']:5.1f}/{c['practice_pct_predicted']:5.1f}"
            f"{'!' if c['thin'] else ' '}(n={c['n']:4d})" for c in cells)
        print(f"  k={stock_value}  {line}")

    mech = doc["validation"]["mechanism"]
    print("\n── Attribution replayed on the survey ──────────────────────────────────")
    print(f"  constructed has_bike: {mech['constructed_has_bike_pct']:5.2f} %   "
          f"riders: {mech['practice_pct']:5.2f} %")
    print(f"  dormant bikes: {mech['dormant_gross_pts']:5.2f} pts gross "
          f"(hold a bike without riding it), {mech['dormant_net_pts']:5.2f} pts "
          f"net — the gap\n    is filled by the "
          f"{mech['practising_without_bike_pts']:.2f} pts who ride WITHOUT an "
          f"attributed bike (bike-share,\n    households with zero bikes): the flow runs "
          f"both ways. The ticket cites the net.")
    print("  (it is `constructed_has_bike_pct` that the mode choice policy must "
          "learn,\n   and not the "
          f"{doc['validation']['targets']['living_in_equipped_household_pct']:.1f} % of "
          "persons living in an equipped household)")

    targets = doc["validation"]["targets"]
    print("\n── Person-level targets (acceptance criteria) ──────────────────────────")
    print(f"  persons given a bike: {targets['holders_pct']:.2f} %")
    print("  size gradient: " + " / ".join(
        f"{row['holders_pct']:.1f}" for row in targets["holders_by_household_size"]))
    print("  riders per bike: " + " / ".join(
        f"{row['practising_per_bike']:.2f}" for row in targets["practising_per_bike"]))
    print(f"  e-bike share of the fleet: {targets['vae_share_of_fleet_pct']:.2f} % "
          f"(module constant: {100 * VAE_SHARE:.2f} %)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output file (default: {DEFAULT_RESOURCE})")
    args = parser.parse_args()

    root = find_project_root()
    men, people = load_frames(root)

    # Median density, served in the resource: it is what the module will
    # substitute for the 81 fine zones without a surveyed household. Not zero, which would
    # describe a desert where the information is simply missing.
    median_density = float(men["density"].median())
    occupations = tuple(sorted(set(MAIN_OCCUPATION.values())))

    stock_doc, stock_oof = fit_stock(men, median_density)
    propensity_doc, propensity_oof = fit_propensity(people, occupations, median_density)
    # The attribution replay serves twice: it validates the mechanism AND it measures the
    # share of holders too young for an e-bike, which renormalises stage 3.
    mechanism = mechanism_check(men, people, occupations, median_density, propensity_doc)

    doc = {
        "version": 1,
        "trait": "personal_bike",
        "stock": stock_doc,
        "propensity": propensity_doc,
        "occupations": list(occupations),
        "median_density": round(median_density, 6),
        "vae_share_of_fleet": VAE_SHARE,
        "under_age_holder_share": mechanism["under_age_holder_share"],
        "validation": {
            "stock": stock_validation(men, stock_oof),
            "housing_reference": diluted_housing_reference(men, root),
            "practice": practice_validation(people, propensity_oof),
            "mechanism": mechanism,
            "targets": holder_targets(men, people),
            "thin_cell_weighted_n": THIN_CELL,
        },
        "meta": {
            "source": "EMC² Toulouse 2023 (ProGEDO lil-1750) — fichiers standards "
                      "ménages (M21, M1, M6, COE0) et personnes (P20, P4, P2, P9, "
                      "COEP), fichier original (ML21, vélos à assistance électrique, "
                      "absent du standard où M22 est vide)",
            "stage1_covariates": "taille du ménage, nombre de VP (M6), densité de "
                                 "ménages et distance à l'hypercentre de la zone fine. "
                                 "NI l'habitat (M1) — moins informatif que la zone dont "
                                 "il est imputé côté persona — NI l'occupation du "
                                 "logement (M2), que le persona ne porte pas.",
            "stage2_covariates": "k, taille du ménage, âge, genre, occupation, densité "
                                 "et distance au centre. AUCUNE distance de "
                                 "déplacement : un stock doit être invariant au trajet.",
            "eligibility": f"membres de {MIN_AGE_ELIGIBLE} ans et plus (champ de la "
                           f"question P20) ; VAE à partir de {14} ans",
            "out_of_scope": "vélo en libre-service (MODP ∈ {10, 18}, 7 % des trajets "
                            "vélo) ; stationnement (M23, P18A manquant à 65 %) ; "
                            "week-end — P20 ne porte que du lundi au vendredi, un "
                            "cycliste de loisir dominical est vu « Jamais » et "
                            "l'attribution le sous-estime sans qu'on puisse mesurer de "
                            "combien",
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    report(doc)
    out = args.out or DEFAULT_RESOURCE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
