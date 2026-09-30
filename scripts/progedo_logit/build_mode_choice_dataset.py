"""build_mode_choice_dataset.py — Training set of the mode-choice policy.

Builds, from the EMC² Toulouse 2023 survey (ProGEDO / lil-1750), the dataset that
will serve to train the statistical policy served in simulation (ticket 005, phase 1).

This script **replaces** `prepare_progedo_logit.ipynb` for any training use.
Three substantive differences with the notebook:

1. **Distance.** The notebook exports `distance_km = D12/1000`. This variable is
   contaminated: for walking, D11 is exactly `declared duration × 58 m/min` and D12
   is the network distance *of the mode used* — knowing the distance already means
   knowing the mode (PR-AUC 0.985 versus 0.804 with a mode-neutral distance, cf.
   `explore_progedo_walk_shapley.ipynb` §7). And at decision time in simulation,
   there is no « trip distance »: there are k OTP options each having its
   own. We therefore use `od_km`, distance between fine-zone centroids, mode-neutral
   and computable on both sides.

2. **Weighting.** The goal is to reproduce *modal shares*: an unweighted training
   biases them. `COEP` (adjustment coefficient of the surveyed person)
   is exported as `sample_weight`.

3. **The feature contract.** A variable enters the dataset only if it is
   computable at decision time in simulation, from the persona
   (`traits_json`), the activity context, or the geometry. `feature_spec.json`
   freezes the list, the types and the modalities: it is reread at training time and at
   runtime, and any divergence raises an error instead of silently producing
   wrong predictions.

**Validity domain**: the survey covers only working days (`JOUR ∈ 1..5`).
The policy trained here is a *weekday* model; applying it on Saturday or
Sunday is an out-of-domain extrapolation. This is declared in `feature_spec.json`
(`domain.weekday_only`) so that the runtime can guard against it.

Usage:
    python scripts/progedo_logit/build_mode_choice_dataset.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- Version of the feature contract ----------------------------------------
# To be incremented at each change of the list, the order or the typing of the
# features. The runtime refuses to load a model whose version differs.
SPEC_VERSION = 2

TEST_SIZE = 0.25
SPLIT_SEED = 0


# ---------------------------------------------------------------------------
# Recoding dictionaries (ProGEDO → project value space)
# ---------------------------------------------------------------------------
# The labels come from `lil-1750-Documentation/LABELS/`. The target values
# are those observed in `traits_json` of `data/population/*.json`: this is what
# makes the survey and the synthetic population comparable.

GENDER = {"1": "Male", "2": "Female"}

MAIN_OCCUPATION = {
    "1": "Travail à plein temps",
    "2": "Travail à temps partiel",
    "3": "Étudiant",  # work-study / internship
    "4": "Étudiant",
    "5": "Scolaire (jusqu'au Bac)",
    "6": "Chômeur/recherche d'emploi",
    "7": "Retraité",
    "8": "Personne au foyer",
    "9": "Autre",
}
EMPLOYED_CODES = {"1", "2"}
STUDIES_CODES = {"3", "4", "5"}

SOCIOPRO = {
    "01": "Farmer",
    "02": "Craftsperson or Shop Owner",
    "03": "Executive or Higher Intellectual Professional",
    "04": "Intermediate Professional",
    "05": "Employee",
    "06": "Manual Worker",
    "07": "Student",
    "08": "Other Inactive",
    "09": "Other Inactive",
}

# MODP → target {car, bike, walk, transit}.
# Active micro-mobility (roller/scooter/wheelchair) attached to walk; motorised
# two-wheelers, taxi/ride-hailing and van to car; plane/river/farm vehicles out of scope.
#
# Accepted structural limit: `train` and `motorbike`, which exist on the simulation side
# in CANONICAL_MODES, are merged here into transit and car. The policy will
# never be able to tell them apart (cf. ticket 005 §4).
MODE_GROUP = {
    "01": "walk", "93": "walk", "94": "walk", "96": "walk", "97": "walk",
    "10": "bike", "11": "bike", "12": "bike", "17": "bike", "18": "bike",
    "21": "car", "22": "car", "61": "car", "62": "car", "81": "car", "82": "car",
    "13": "car", "14": "car", "15": "car", "16": "car", "19": "car", "20": "car",
    "31": "transit", "32": "transit", "33": "transit", "34": "transit",
    "37": "transit", "38": "transit", "39": "transit",
    "41": "transit", "42": "transit", "43": "transit",
    "51": "transit", "52": "transit", "53": "transit", "54": "transit",
    "71": "transit",
}

TARGET_CLASSES = ["bike", "car", "transit", "walk"]


def purpose_from_code(code: str) -> str:
    """ProGEDO purpose (D5A destination / D2A origin) → project purpose."""
    if code in ("01", "02"):
        return "home"
    if code in ("11", "12", "13", "14", "81"):
        return "work"
    if code in ("21", "22", "23", "24", "25", "26", "27", "28", "29", "96", "97"):
        return "education"
    if code in ("30", "31", "32", "33", "34", "35", "82", "98"):
        return "shop"
    if code in ("51", "52", "53", "54"):
        return "leisure"
    return "other"


# ---------------------------------------------------------------------------
# Feature definition — the contract
# ---------------------------------------------------------------------------
# `source` documents where the value will come from in simulation. It is the
# admission criterion: without a runtime source, the variable is excluded whatever its
# predictive power.

FEATURE_SPEC: list[dict] = [
    # --- persona (traits_json) ---
    {"name": "age", "kind": "numeric", "source": "persona"},
    {"name": "gender", "kind": "categorical", "source": "persona"},
    {"name": "household_size", "kind": "numeric", "source": "persona"},
    {"name": "has_driving_license", "kind": "bool", "source": "persona"},
    {"name": "has_pt_subscription", "kind": "bool", "source": "persona"},
    {"name": "number_of_cars", "kind": "numeric", "source": "persona"},
    {"name": "car_availability", "kind": "categorical", "source": "persona"},
    {"name": "has_bike", "kind": "bool", "source": "persona"},
    {"name": "socioprofessional_class", "kind": "categorical", "source": "persona"},
    {"name": "main_occupation", "kind": "categorical", "source": "persona"},
    {"name": "employed", "kind": "bool", "source": "persona"},
    {"name": "studies", "kind": "bool", "source": "persona"},
    # --- activity context ---
    {"name": "purpose", "kind": "categorical", "source": "context"},
    {"name": "purpose_origin", "kind": "categorical", "source": "context"},
    {"name": "departure_hour", "kind": "numeric", "source": "context"},
    # --- geometry ---
    {"name": "od_km", "kind": "numeric", "source": "geo"},
    {"name": "same_zone", "kind": "bool", "source": "geo"},
    {"name": "dist_center_orig_km", "kind": "numeric", "source": "geo"},
    {"name": "dist_center_dest_km", "kind": "numeric", "source": "geo"},
    {"name": "density_orig", "kind": "numeric", "source": "geo"},
    {"name": "density_dest", "kind": "numeric", "source": "geo"},
]

FEATURES = [f["name"] for f in FEATURE_SPEC]

# Variables kept in the parquet for diagnostics but **forbidden to the model**.
DIAGNOSTIC_ONLY = ["distance_km", "crow_km", "duration_min"]

# Traceability (outside the model).
KEYS = ["ZF", "ECH", "PER", "NDEP", "hh_id"]

# Features without which a row is not usable.
CRITICAL = [
    "age", "gender", "has_pt_subscription", "socioprofessional_class",
    "main_occupation", "car_availability", "number_of_cars", "od_km",
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    root = Path(__file__).resolve()
    while not (root / "data" / "PROGEDO 2023").exists() and root != root.parent:
        root = root.parent
    if not (root / "data" / "PROGEDO 2023").exists():
        raise SystemExit("Racine du projet introuvable (dossier 'data/PROGEDO 2023').")
    return root


def load_raw(progedo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the three standard files, all as str.

    ProGEDO codes have significant leading zeros ('01' ≠ '1') and empty cells
    that carry meaning (not surveyed): automatic numeric parsing would
    destroy them. The recoding is explicit below.
    """
    frames = []
    for name in ("pers", "men", "depl"):
        df = pd.read_csv(progedo_dir / f"Toulouse_2023_std_{name}.csv", dtype=str)
        for c in df.columns:
            df[c] = df[c].str.strip().replace({"": np.nan})
        frames.append(df)
    pers, men, depl = frames
    print(f"pers: {pers.shape} | men: {men.shape} | depl: {depl.shape}")
    return pers, men, depl


def build_geo(sig_zf: Path, men: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Geographic layers per fine zone: density, distance to the city core, centroid.

    Returns (GEO, XYS, REF) — GEO carries the derived features, XYS the Lambert 93
    coordinates and the area needed to compute `od_km`, REF the geographic
    reference to publish in the spec so that the runtime computes identically.
    """
    # The CRS is captured before the column subset: without the geometry,
    # the GeoDataFrame falls back to a DataFrame and loses `.crs`.
    layer = gpd.read_file(sig_zf)
    crs = layer.crs
    zf = layer[["ZF", "XL93", "YL93", "SURF_M2"]].copy()
    zf["ZF"] = zf["ZF"].astype(str).str.strip()

    # Household density estimated from the household adjustment coefficients.
    weights = men.assign(COE0=pd.to_numeric(men["COE0"], errors="coerce")).groupby("ZFM")["COE0"].sum()
    zf["area_km2"] = zf["SURF_M2"] / 1e6
    zf["density_hh_km2"] = zf["ZF"].map(weights) / zf["area_km2"]

    # City core = centroid of the fine zones of sector 01 (Capitole).
    core = zf[zf["ZF"].str.startswith("1011")]
    cx, cy = core["XL93"].mean(), core["YL93"].mean()
    zf["dist_center_km"] = np.hypot(zf["XL93"] - cx, zf["YL93"] - cy) / 1000

    # Published in the spec: the runtime must use THIS centre, and not the constant
    # hard-coded in `move_logger.py` (43.6047 / 1.4442), from which it is
    # ~820 m away. Two competing definitions of the centre would shift dist_center_*.
    center_wgs84 = (
        gpd.GeoSeries(gpd.points_from_xy([cx], [cy]), crs=crs)
        .to_crs("EPSG:4326")
    )
    ref = {
        "crs": crs.to_string(),
        "zf_layer": sig_zf.name,
        "n_zones": int(len(zf)),
        "hypercenter": {
            "definition": "centroïde des zones fines du secteur 01 (Capitole)",
            "x_l93": round(float(cx), 1),
            "y_l93": round(float(cy), 1),
            "lat": round(float(center_wgs84.y.iloc[0]), 6),
            "lon": round(float(center_wgs84.x.iloc[0]), 6),
        },
        # od_km formula to replicate exactly at runtime: a distance computed
        # on the exact coordinates rather than on the centroids gives a factor of 2
        # on intra-zone trips (cf. ticket 005 §2.1).
        "od_km": {
            "inter_zone": "distance entre centroïdes de zones fines (L93), en km",
            "intra_zone": "0.5 * sqrt(SURF_M2) / 1000",
        },
    }
    print(f"City core: L93 X={cx:.0f} Y={cy:.0f} "
          f"| WGS84 lat={ref['hypercenter']['lat']:.4f} lon={ref['hypercenter']['lon']:.4f}")

    geo = zf.set_index("ZF")[["density_hh_km2", "dist_center_km"]]
    xys = zf.set_index("ZF")[["XL93", "YL93", "SURF_M2"]]
    return geo, xys, ref


def build_household(men: pd.DataFrame) -> pd.DataFrame:
    """Household equipment. Actual household key = (ZFM, ECH) — ECH alone is not unique.

    `n_bikes` (= `M21`) is kept: since ticket 015 it no longer serves to set
    `has_bike` directly, but it is the **stock** from which the nominal assignment starts
    (cf. `build_has_bike`).
    """
    out = pd.DataFrame({
        "ZF": men["ZFM"],
        "ECH": men["ECH"],
        "number_of_cars": pd.to_numeric(men["M6"], errors="coerce"),
        "n_bikes": pd.to_numeric(men["M21"], errors="coerce"),
    })
    return out.drop_duplicates(["ZF", "ECH"])


def build_has_bike(person: pd.DataFrame, geo: pd.DataFrame,
                   model) -> pd.Series:
    """`has_bike` **built identically to inference** (ticket 015, lot 3).

    It is the consumer's constraint, and it is structural. The mode-choice
    policy consumes `has_bike`; it is its 2nd most influential variable for the bike
    decision (mean |SHAP| 0.74, behind distance at 1.63). Until now it learned it on
    `M21 > 0`, that is « **there is a bike in the household** » — true for 63.2 % of
    persons. Yet the persona now carries a **nominal assignment**, true for
    ~50 %. The two sides were not talking about the same thing, and the learned coefficient
    applied to something other than what it measures.

    The way out is to rebuild the same variable on both sides: we apply here to the
    survey households — where `k`, the size and `P20` are known — exactly the
    assignment rule of stage 2, the one `enrich_personal_bike` applies to the
    population. Same definition at training and at inference. It is the price of a
    single individual field, and it is affordable.

    The draw stays deterministic (hashing on the survey household key), so the
    training set is reproducible without a seed.
    """
    from mobility_core.bike_ownership import (
        K_MAX, MIN_AGE_ELIGIBLE, Member, assign)

    zone = geo.reindex(person["ZF"]).reset_index(drop=True)
    propensity = [
        model.propensity_of(
            k=int(min(k, K_MAX)) if pd.notna(k) else 0,
            household_size=int(size) if pd.notna(size) else 1,
            age=age if pd.notna(age) else None,
            gender=gender if isinstance(gender, str) else None,
            main_occupation=occupation if isinstance(occupation, str) else None,
            density_hh_km2=None if pd.isna(density) else float(density),
            dist_center_km=0.0 if pd.isna(dist) else float(dist),
        )
        for k, size, age, gender, occupation, density, dist in zip(
            person["n_bikes"], person["household_size"], person["age"],
            person["gender"], person["main_occupation"],
            zone["density_hh_km2"], zone["dist_center_km"])
    ]

    held = pd.Series(False, index=person.index)
    frame = person.assign(_p=propensity,
                          _hh=person["ZF"].astype(str) + "|" + person["ECH"].astype(str))
    for hh_id, group in frame.groupby("_hh", sort=False):
        stock = group["n_bikes"].iloc[0]
        if pd.isna(stock) or stock <= 0:
            continue
        # All household members are in the survey file: unlike
        # the synthetic population, there is no missing slot to fill. The
        # nominal size IS the number of rows.
        members = [
            Member(index=int(i), propensity=float(p),
                   eligible=bool(pd.notna(a) and a >= MIN_AGE_ELIGIBLE))
            for i, p, a in zip(group.index, group["_p"], group["age"])
        ]
        for index in assign(members, int(min(stock, K_MAX)), hh_id):
            held.loc[index] = True
    return held


def build_person(pers: pd.DataFrame, household: pd.DataFrame) -> pd.DataFrame:
    """Demographics and individual equipment. Person key = (ZFP, ECH, PER)."""
    hh_size = pers.groupby(["ZFP", "ECH"]).size().rename("household_size").reset_index()
    licensed = (
        pers.assign(_lic=(pers["P7"] == "1"))
        .groupby(["ZFP", "ECH"])["_lic"].sum()
        .rename("n_licensed").reset_index()
    )

    out = pd.DataFrame({
        "ZF": pers["ZFP"],
        "ECH": pers["ECH"],
        "PER": pers["PER"],
        "PENQ": pers["PENQ"],
        "age": pd.to_numeric(pers["P4"], errors="coerce"),
        "gender": pers["P2"].map(GENDER),
        "has_driving_license": pers["P7"] == "1",
        "has_pt_subscription": pers["P12"].map({"4": False, "6": True}),
        "socioprofessional_class": pers["PCSC"].map(SOCIOPRO),
        "main_occupation": pers["P9"].map(MAIN_OCCUPATION),
        "employed": pers["P9"].isin(EMPLOYED_CODES),
        "studies": pers["P9"].isin(STUDIES_CODES),
        # Adjustment coefficient of the surveyed person: it is what carries
        # the representativeness of trips (cf. ticket 005, E5).
        "sample_weight": pd.to_numeric(pers["COEP"], errors="coerce"),
    })

    for agg in (hh_size, licensed):
        out = out.merge(
            agg, left_on=["ZF", "ECH"], right_on=["ZFP", "ECH"], how="left"
        ).drop(columns="ZFP")
    out = out.merge(household, on=["ZF", "ECH"], how="left")

    def car_availability(row):
        """Car supply relative to the household's drivers (project semantics)."""
        cars = row["number_of_cars"]
        if pd.isna(cars):
            return np.nan
        if cars == 0:
            return "none"
        lic = row["n_licensed"]
        if pd.isna(lic) or lic == 0:
            return "all"  # car available, no driver constraint
        return "all" if cars >= lic else "some"

    out["car_availability"] = out.apply(car_availability, axis=1)
    # `n_bikes` stays until `build_has_bike`, which needs it as the household stock.
    return out.drop(columns="n_licensed")


# Granularity of zone codes: the `zf_zones.gpkg` layer is indexed on 9-digit codes
# whose last 3 are ALWAYS `000` (785 zones, checked), while the
# trips file codes the origin (`D3`) and the destination (`D7`) at the
# sub-zone level — `102103503`, `127105205`. Comparing the two as is resolved only
# **51.1 %** of ODs, and the attrition was not uniform: 75.5 % of the trips of
# 10-14 year-olds went through versus 48.4 % of those of 30-49 year-olds, which divided by nearly
# three the public transport share learned for this cohort. The model was therefore
# badly trained exactly on the ages where the summary page shows its largest gap.
#
# Bringing the code down to the layer's granularity raises resolution to **95.8 %**, and this
# is not an abusive matching: on the 24,365 trips thus recovered, the
# obtained distance correlates at **0.984** with the declared crow-fly distance (`D11`),
# versus 0.992 on the trips already resolved, with the same median bias (+0.18 km
# versus +0.14) and the same tail (0.3 % of gaps beyond 5 km versus 0.2 %). In other
# words the truncated codes locate trips as well as the others.
#
# The remaining 4.2 % are truly outside the survey scope (prefixes 98x, 93x, 909 —
# other départements, distant municipalities) and must stay unresolved: inventing
# a zone for them would be an out-of-domain extrapolation, exactly what part 3 refuses.
ZONE_CODE_WIDTH = 6


def zone_key(codes: pd.Series) -> pd.Series:
    """Zone code brought down to the layer's granularity (6 digits + `000`)."""
    return codes.astype(str).str[:ZONE_CODE_WIDTH] + "0" * (9 - ZONE_CODE_WIDTH)


def build_trips(depl: pd.DataFrame, geo: pd.DataFrame, xys: pd.DataFrame) -> pd.DataFrame:
    """Trips enriched with the origin-destination geometry."""
    out = pd.DataFrame({
        "ZF": depl["ZFD"],
        "ECH": depl["ECH"],
        "PER": depl["PER"],
        "NDEP": depl["NDEP"],
        "mode": depl["MODP"].map(MODE_GROUP),
        "purpose": depl["D5A"].map(purpose_from_code),
        "purpose_origin": depl["D2A"].map(purpose_from_code),
        "departure_hour": (pd.to_numeric(depl["D4"], errors="coerce") // 100) % 24,
        "ZF_orig": zone_key(depl["D3"]),
        "ZF_dest": zone_key(depl["D7"]),
        # Diagnostic only — contaminated, never in the model (ticket 005 §1).
        "distance_km": pd.to_numeric(depl["D12"], errors="coerce") / 1000,
        "crow_km": pd.to_numeric(depl["D11"], errors="coerce") / 1000,
        "duration_min": pd.to_numeric(depl["D9"], errors="coerce"),
    })

    for side in ("orig", "dest"):
        joined = geo.reindex(out[f"ZF_{side}"]).reset_index(drop=True)
        out[f"density_{side}"] = joined["density_hh_km2"].values
        out[f"dist_center_{side}_km"] = joined["dist_center_km"].values

    # Mode-neutral OD distance. For an intra-zone trip the distance between
    # centroids is 0: it is replaced by a characteristic length of the zone
    # (0.5 × √area). `same_zone` stays alongside so that the model knows the
    # value is imputed rather than measured.
    o = xys.reindex(out["ZF_orig"]).reset_index(drop=True)
    d = xys.reindex(out["ZF_dest"]).reset_index(drop=True)
    inter_km = np.hypot(o["XL93"].values - d["XL93"].values,
                        o["YL93"].values - d["YL93"].values) / 1000
    intra = (out["ZF_orig"].values == out["ZF_dest"].values)
    out["od_km"] = np.where(intra, 0.5 * np.sqrt(o["SURF_M2"].values) / 1000, inter_km)
    out["same_zone"] = intra
    return out


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Train/test split **by household**.

    A split by trip would leak: the trips of the same individual share
    their characteristics, and those of the same household its car equipment.
    """
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        .split(df, groups=df["hh_id"])
    )
    split = pd.Series("train", index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    return split


def build_feature_spec(clean: pd.DataFrame, geo_ref: dict) -> dict:
    """Serialises the feature contract, categorical modalities included.

    The runtime rereads this file: freezing the modalities here guarantees that
    the encoding of a category will be identical at training and in simulation.
    """
    features = []
    for spec in FEATURE_SPEC:
        entry = dict(spec)
        if spec["kind"] == "categorical":
            entry["categories"] = sorted(clean[spec["name"]].dropna().unique().tolist())
        features.append(entry)

    return {
        "spec_version": SPEC_VERSION,
        "source": "EMC² Toulouse 2023 (ProGEDO / lil-1750)",
        "target": {"name": "mode", "classes": TARGET_CLASSES},
        "sample_weight": "sample_weight",
        "features": features,
        # Geographic reference: the runtime must reproduce these definitions
        # identically, otherwise the geo features are shifted (ticket 005 §2.1).
        "geo_reference": geo_ref,
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "split": {"by": "hh_id", "test_size": TEST_SIZE, "seed": SPLIT_SEED},
        "domain": {
            # The survey covers only working days (JOUR ∈ 1..5): applying the
            # policy on a Saturday or a Sunday is an out-of-domain extrapolation.
            "weekday_only": True,
            "city": "Toulouse",
            # od_km is computable only if both fine zones are within the
            # survey scope.
            "requires_survey_perimeter": True,
        },
        "notes": [
            "distance_km/crow_km/duration_min sont contaminées : diagnostic uniquement.",
            "train et motorbike sont fusionnés dans transit et car : la politique ne "
            "peut pas les distinguer.",
            "has_bike (spec v2, ticket 015) est l'attribution NOMINATIVE d'un vélo, "
            "reconstruite par la règle de l'étage 2 appliquée aux ménages de l'enquête "
            "— et non « le foyer déclare au moins un vélo » (M21 > 0) comme en v1. "
            "~50 % des personnes contre 63,2 %. C'est la définition que porte "
            "traits_json.personal_bike, donc la seule qui rende l'entraînement et "
            "l'inférence comparables.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: scripts/progedo_logit/)")
    args = parser.parse_args()

    root = find_project_root()
    progedo_dir = root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV" / "fichiers_standards"
    sig_zf = (root / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
              / "EMC2_Toulouse_2023_ZF_26052023.shp")
    out_dir = args.out_dir or (root / "scripts" / "progedo_logit")
    out_dir.mkdir(parents=True, exist_ok=True)

    pers, men, depl = load_raw(progedo_dir)
    geo, xys, geo_ref = build_geo(sig_zf, men)
    household = build_household(men)
    person = build_person(pers, household)

    # `has_bike` built identically to inference (ticket 015, lot 3). It requires the
    # bike equipment resource: without it the indicator cannot be rebuilt,
    # and falling back on `M21 > 0` would silently produce a training set that
    # measures something other than what the persona carries. We therefore refuse, rather than
    # fall back.
    from mobility_core.bike_ownership import BikeOwnershipModel
    bike_model = BikeOwnershipModel.load()
    person["has_bike"] = build_has_bike(person, geo, bike_model)
    print(f"has_bike built (nominal assignment): "
          f"{100 * person['has_bike'].mean():.2f} % of persons; "
          f"for reference, « the household has a bike » (M21 > 0) amounts to "
          f"{100 * (person['n_bikes'].fillna(0) > 0).mean():.2f} %")
    person = person.drop(columns="n_bikes")

    trips = build_trips(depl, geo, xys)

    df = trips.merge(person, on=["ZF", "ECH", "PER"], how="left")
    n0 = len(df)

    # --- Filters, traced one by one: each one costs rows, we want to know how many.
    steps = []
    df = df[df["mode"].notna()]
    steps.append(("mode exploitable", len(df)))
    df = df[df["PENQ"] == "1"]
    steps.append(("personne enquêtée", len(df)))
    df = df.dropna(subset=CRITICAL)
    steps.append(("features critiques", len(df)))
    df = df[df["sample_weight"].notna()]
    steps.append(("pondération", len(df)))

    print(f"\nRows: {n0} at the start")
    for label, n in steps:
        print(f"  after {label:22s} : {n}")

    df = df.reset_index(drop=True)
    df["hh_id"] = df["ZF"].astype(str) + "_" + df["ECH"].astype(str)
    df["split"] = assign_split(df)

    clean = df[KEYS + FEATURES + DIAGNOSTIC_ONLY + ["mode", "sample_weight", "split"]].copy()

    # Explicit typing: the parquet must be reread without ambiguity.
    for spec in FEATURE_SPEC:
        col = spec["name"]
        if spec["kind"] == "bool":
            clean[col] = clean[col].astype(bool)
        elif spec["kind"] == "categorical":
            clean[col] = clean[col].astype("category")
    clean["age"] = clean["age"].astype(int)
    for col in ("household_size", "number_of_cars", "departure_hour"):
        clean[col] = clean[col].astype("Int64")

    # --- Checks ------------------------------------------------------------
    print("\nModal shares (raw):")
    print(clean["mode"].value_counts(normalize=True).round(4).to_string())
    weighted = (clean.groupby("mode", observed=True)["sample_weight"].sum()
                / clean["sample_weight"].sum())
    print("\nModal shares (COEP-weighted):")
    print(weighted.round(4).to_string())

    missing = clean[FEATURES].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nRemaining missing values (LightGBM handles them natively):")
        print(missing.to_string())

    print(f"\nSplit: train={(clean['split'] == 'train').sum()} "
          f"test={(clean['split'] == 'test').sum()} "
          f"| ménages={clean['hh_id'].nunique()}")
    overlap = (set(clean.loc[clean['split'] == 'train', 'hh_id'])
               & set(clean.loc[clean['split'] == 'test', 'hh_id']))
    assert not overlap, f"Leak: {len(overlap)} households in both splits"

    # --- Writing -----------------------------------------------------------
    parquet_path = out_dir / "progedo_mode_choice_v2.parquet"
    spec_path = out_dir / "feature_spec.json"
    clean.to_parquet(parquet_path, index=False)
    spec_path.write_text(
        json.dumps(build_feature_spec(clean, geo_ref), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nWritten:\n - {parquet_path} ({len(clean)} rows, {clean.shape[1]} columns)"
          f"\n - {spec_path} (spec v{SPEC_VERSION}, {len(FEATURES)} features)")


if __name__ == "__main__":
    main()
