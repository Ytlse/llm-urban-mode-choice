"""export_terminal_time.py — The terminal time of a car trip, measured on EMC².

`services/llm-agents/config/terminal_time.yaml` applies 2 to 10 minutes of access and
parking per car trip, sourced from the literature (COMPASS tables, Shoup,
Cerema). The survey the project takes as its target measures **11 to 14 times less**, and
it measures it directly: the trips file carries `T2` (walking at departure),
`T6` (walking at arrival) and `T11` (time spent looking for parking).

This script extracts from it the **empirical law**, by ring and by trip end.

## Why a law and not a mean

The survey mean is **below one minute** (0.36 min of access in Toulouse). Yet the
rendering of the options imposes multiples of 60 s — it is structural, the invariant
"displayed total = sum of the sub-steps" depends on it. Serving the mean would force
showing 0 minutes everywhere, which would erase a very real tail: 2 to 4 % of trips
really take 5 minutes or more. The law keeps both, the mean **and** the tail.

And it is **not a bell curve**. The distribution is massed at zero (87 to 96 % depending
on the ring) and stretched to the right. A Gaussian would produce negative values and
destroy the mass at zero; it is the observed histogram that is served, not a
parametric shape chosen for its convenience.

## What the validity check established

The legitimate doubt was that walking to the car might be coded as a **separate walking
leg**, in which case `T2`/`T6` would be 0 by construction and the comparison would be
empty. Checked: of the 24 481 trips that include a car leg, **none**
carries a walking leg. Terminal walking can therefore only be in `T2`/`T6`. And
the instrument works — on public transport legs, of identical
structure, `T2 + T6` gives 6 minutes at the median. The survey can record a terminal
time; it records ~0.6 min of it for the car.

## What this script is not

It is **not** the fitting that decision T2 of ticket 013 forbids. T2 forbids
tuning this parameter *on a calibration score*. Here it is **re-sourced on the survey
measurement**, which is precisely what its own `provenance: sourced` calls for. The
distinction can be checked: no value below was chosen by looking at a
modal share.

Usage:
    python -m scripts.progedo_logit.export_terminal_time [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.progedo_logit.build_mode_choice_dataset import find_project_root

DEFAULT_RESOURCE = (Path(__file__).resolve().parents[2] / "packages" / "mobility_core" / "src" / "mobility_core" / "data"
                    / "terminal_time_emc2.json")

# `T3` of the trips file: mode used. 21 = private car driver.
# Only the driver is kept: the passenger does not look for a space.
CAR_DRIVER = "21"

# Comparison modes, to publish the validity check in the resource.
TRANSIT = ("31", "32", "33")
BIKE = ("11", "17")

# Rings, in EMC² reading order. These are the modalities of
# `population_reference.COURONNES`, and since ticket 028 the strata are set by the
# survey TABLE (fine zone → sampling sector → ring), no longer by the distance to the
# centre: terminal time and residence finally speak of the same partition.
CROWNS = ("Toulouse", "1ere couronne", "2eme couronne", "3eme couronne")

# Tail clipping, in minutes. Beyond 20 min the survey only carries a
# few legs (and two values at 207 min, clearly data-entry outliers):
# keeping them would let the noise carry the mean.
MAX_MINUTES = 20

# Minimum count of a (ring × end) cell to publish its law. Below it, the
# resource serves the overall law rather than a histogram over a few dozen
# legs — and it says so, rather than smoothing it silently.
MIN_CELL = 200


def crown_of_zone():
    """Fine zone → ring, through the survey TABLE (ticket 028).

    Before: the centroid of each fine zone was classified by its distance to the centre
    (`geo_reference.residence_zone`, 8 / 20 / 40 km). That is not the definition of the
    survey, which partitions by list of communes, and ticket 020 measured the gap:
    24.4 % of homes change ring between the two. The terminal time laws
    were therefore stratified on a partition that neither the targets nor the log use.

    `CouronneTable.couronne_of_zf` attaches a fine zone through its sampling sector — the
    first three digits of the code — and it is the sector that carries the ring in the
    survey. Measured 100 % identical to the geometric classification (ticket 021, lot 0). A
    code unknown to the table returns `None`: the row leaves the strata and stays in the
    overall law, as before — we do not guess a ring.
    """
    from mobility_core.residence_zone import CouronneTable

    return CouronneTable.load().couronne_of_zf


def load_legs(root: Path) -> pd.DataFrame:
    """Survey legs, with their two terminal times and their rings."""
    path = (root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
            / "fichiers_standards" / "Toulouse_2023_std_traj.csv")
    legs = pd.read_csv(path, dtype=str)
    for column in legs.columns:
        legs[column] = legs[column].str.strip().replace({"": np.nan})
    for column in ("T2", "T6", "T11"):
        legs[column] = pd.to_numeric(legs[column], errors="coerce").fillna(0.0)

    crown_of = crown_of_zone()
    # `access` = walking at departure; `egress` = walking at arrival + parking search.
    # The search is counted at arrival because that is where one searches.
    legs["access"] = legs["T2"].clip(0, MAX_MINUTES).round().astype(int)
    legs["egress"] = (legs["T6"] + legs["T11"]).clip(0, MAX_MINUTES).round().astype(int)
    # `T4`/`T5`: fine zones of departure and arrival **of the mechanised mode**. Access
    # depends on the origin (where the vehicle is parked), egress on the destination
    # (where a space must be found) — same convention as the config file.
    legs["access_crown"] = legs["T4"].map(crown_of)
    legs["egress_crown"] = legs["T5"].map(crown_of)
    car = legs["T3"] == CAR_DRIVER
    # Counter of legs without a ring: they stay in the overall law, never
    # in a stratum. A high figure would flag a stale table, not a normal case.
    unmapped = int((legs.loc[car, "access_crown"].isna()
                    | legs.loc[car, "egress_crown"].isna()).sum())
    print(f"Legs: {len(legs)} in total, {int(car.sum())} as car driver, "
          f"of which {unmapped} without a ring at one end ({100.0 * unmapped / max(int(car.sum()), 1):.1f} %)")
    return legs


def histogram(values: pd.Series) -> dict:
    """Law in whole minutes: `{minutes: probability}`, plus its moments."""
    counts = values.value_counts().sort_index()
    total = int(counts.sum())
    return {
        "n": total,
        "mean_min": round(float(values.mean()), 4),
        "median_min": float(values.median()),
        "p90_min": float(values.quantile(0.90)),
        # The keys are strings: this is JSON, and an integer becomes a string there
        # anyway. The consumer converts back.
        "pmf": {str(int(k)): round(int(v) / total, 6) for k, v in counts.items()},
    }


def validity_check(legs: pd.DataFrame) -> dict:
    """The check that authorises the comparison, published with the law.

    If walking to the car were a separate walking leg, `T2`/`T6` would be 0
    by construction and the law would measure emptiness. We therefore check that trips
    with a car leg have **no** walking leg, and that a mode whose real terminal walking
    is known (public transport) does show it.
    """
    key = ["ZFT", "ECH", "PER", "NDEP"]
    grouped = legs.assign(
        _car=legs["T3"] == CAR_DRIVER,
        _foot=legs["T3"] == "01",
        _transit=legs["T3"].isin(TRANSIT),
    ).groupby(key)[["_car", "_foot", "_transit"]].sum()
    with_car = grouped[grouped["_car"] > 0]

    def terminal(codes) -> pd.Series:
        frame = legs[legs["T3"].isin(codes)]
        return frame["access"] + frame["egress"]

    return {
        "question": "La marche vers la voiture est-elle codée comme un trajet à pied "
                    "distinct ? Si oui, T2/T6 vaudraient 0 par construction.",
        "trips_with_car_leg": int(len(with_car)),
        "of_which_no_foot_leg_pct": round(float(100 * (with_car["_foot"] == 0).mean()), 1),
        "verdict": "La marche terminale ne peut être que dans T2/T6 : aucun déplacement "
                   "voiture ne porte de jambe à pied.",
        "instrument_works": {
            "note": "Contrôle positif — un mode dont la marche terminale est réelle doit "
                    "l'afficher. Les TC, de structure identique (aucune jambe à pied), "
                    "la portent bien.",
            "transit_terminal_median_min": float(terminal(TRANSIT).median()),
            "transit_terminal_mean_min": round(float(terminal(TRANSIT).mean()), 2),
            "car_terminal_median_min": float(terminal([CAR_DRIVER]).median()),
            "car_terminal_mean_min": round(float(terminal([CAR_DRIVER]).mean()), 2),
            "bike_terminal_mean_min": round(float(terminal(BIKE).mean()), 2),
        },
    }


# Modes for which a law is served, with the matching `T3` codes and the spatialisation.
# Bike is NOT spatialised: only 2 047 legs, hence cells per ring that are
# too thin. Inventing a ring variation would be disguised fitting.
LAW_MODES = {
    "car": {"codes": (CAR_DRIVER,), "spatialise": True},
    "bicycle": {"codes": BIKE, "spatialise": False},
}


def build_ends(frame: pd.DataFrame, spatialise: bool) -> dict:
    """`access`/`egress` laws of a mode, by ring if the count allows it."""
    ends: dict[str, dict] = {}
    for end, crown_column in (("access", "access_crown"), ("egress", "egress_crown")):
        overall = histogram(frame[end])
        per_crown: dict[str, dict] = {}
        if spatialise:
            for crown in CROWNS:
                cell = frame[frame[crown_column] == crown]
                if len(cell) < MIN_CELL:
                    # Flagged, not silently smoothed: the consumer will fall back on
                    # the overall law and will know why.
                    per_crown[crown] = {"n": int(len(cell)), "thin": True}
                    continue
                per_crown[crown] = {**histogram(cell[end]), "thin": False}
        ends[end] = {"overall": overall, "by_crown": per_crown}
    return ends


def build(legs: pd.DataFrame) -> dict:
    modes = {
        mode: {
            "spatialise": spec["spatialise"],
            "n_legs": int(legs["T3"].isin(spec["codes"]).sum()),
            "ends": build_ends(legs[legs["T3"].isin(spec["codes"])],
                               spec["spatialise"]),
        }
        for mode, spec in LAW_MODES.items()
    }
    return {
        "version": 2,
        "trait": "terminal_time",
        "unit": "minutes entières",
        "crowns": list(CROWNS),
        "modes": modes,
        # Kept for the readers that only need the car.
        "ends": modes["car"]["ends"],
        "validity": validity_check(legs),
        "meta": {
            "source": "EMC² Toulouse 2023 (ProGEDO lil-1750), fichier trajets — T2 "
                      "(marche au départ), T6 (marche à l'arrivée), T11 (durée de "
                      "recherche du stationnement)",
            "scope": "conducteur de véhicule particulier (T3 = 21) ; le passager ne "
                     "cherche pas de place",
            "crown_definition": "mobility_core.residence_zone.CouronneTable — zone "
                                "fine → secteur de tirage → couronne, la liste de "
                                "communes de l'enquête (ticket 028) ; même définition "
                                "que le trait `residence_zone` des personas",
            "clip_minutes": MAX_MINUTES,
            "min_cell": MIN_CELL,
            "not_a_calibration_fit": "Aucune valeur n'a été choisie en regardant une "
                                     "part modale. Le paramètre reste exogène (ticket "
                                     "013, décision T2) ; il est re-sourcé, pas ajusté.",
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def report(doc: dict) -> None:
    validity = doc["validity"]
    print(f"\n── Validity check ──────────────────────────────────────────────────")
    print(f"  trips with a car leg: {validity['trips_with_car_leg']}, "
          f"of which {validity['of_which_no_foot_leg_pct']} % without any walking leg")
    works = validity["instrument_works"]
    print(f"  median terminal — PT {works['transit_terminal_median_min']:.0f} min "
          f"(mean {works['transit_terminal_mean_min']:.2f}) versus car "
          f"{works['car_terminal_median_min']:.0f} min "
          f"(mean {works['car_terminal_mean_min']:.2f})")
    print(f"  → the instrument does record a terminal time; the car has little of it.")

    # Values currently applied, so that the gap can be read at a glance.
    applied = {"access": {"Toulouse": 3, "1ere couronne": 2, "2eme couronne": 2,
                          "3eme couronne": 1},
               "egress": {"Toulouse": 7, "1ere couronne": 4, "2eme couronne": 3,
                          "3eme couronne": 1}}
    print(f"\n── bike — overall law (not spatialised, 2 047 legs) ───────────────────")
    for end in ("access", "egress"):
        cell = doc["modes"]["bicycle"]["ends"][end]["overall"]
        print(f"  {end:8s} n={cell['n']:5d}  mean {cell['mean_min']:.2f} min "
              f"| tt2 applied 1 min")
    for end in ("access", "egress"):
        print(f"\n── {end} — survey law versus applied value ─────────────────────────")
        print(f"  {'couronne':16s} {'n':>6s} {'moyenne':>8s} {'appliqué':>9s} "
              f"{'facteur':>8s}   law (minutes: share)")
        for crown in doc["crowns"]:
            cell = doc["ends"][end]["by_crown"].get(crown) or {}
            if cell.get("thin"):
                print(f"  {crown:16s} {cell.get('n', 0):6d}   thin cell → overall law")
                continue
            pmf = cell["pmf"]
            top = " ".join(f"{k}:{100 * v:.0f}%" for k, v in list(pmf.items())[:5])
            factor = applied[end][crown] / max(cell["mean_min"], 1e-9)
            print(f"  {crown:16s} {cell['n']:6d} {cell['mean_min']:8.2f} "
                  f"{applied[end][crown]:9d} {factor:7.0f}×   {top}")


def emit_yaml(doc: dict) -> str:
    """`modes:` block of `terminal_time.yaml`, generated from the measured law.

    The block is **embedded in the YAML** rather than read from the JSON resource, and
    this is a deployment constraint, not a choice: the `osmnx` replicas only mount
    `config/` and do not have `mobility_core` on their path (see the lazy import of
    `osmnx_direct`). A config that depended on `mobility_core/data/` would kill them at
    the next `docker compose build`.

    It is therefore **generated**, not copied: `make terminal-time --emit-config` re-emits
    it from the survey, which keeps a hundred numbers from drifting by hand.
    """
    # Rendering labels: taken verbatim from tt2 — only the DURATIONS change, and
    # emitting them here keeps the block self-contained (the validator requires them).
    labels = {
        "car": {"access": "Rejoindre la voiture", "main": "Conduite",
                "egress": "Stationnement et marche jusqu'à '{destination}'",
                "egress_sans_destination": "Stationnement et marche jusqu'à la destination",
                "terminal": "d'accès et de stationnement"},
        "bicycle": {"access": "Déverrouiller le vélo", "main": "Trajet à vélo",
                    "egress": "Attacher le vélo à '{destination}'",
                    "egress_sans_destination": "Attacher le vélo à l'arrivée",
                    "terminal": "d'accès et d'attache"},
    }
    lines: list[str] = []
    for mode in ("car", "bicycle"):
        spec = doc["modes"][mode]
        lines.append(f"  {mode}:   # {spec['n_legs']} trajets enquêtés")
        for end in ("access", "egress"):
            node = spec["ends"][end]
            lines.append(f"    {end}_law:")
            for crown in list(doc["crowns"]) + ["default"]:
                cell = (node["by_crown"].get(crown) if crown != "default"
                        else node["overall"])
                thin = crown != "default" and (cell or {}).get("thin")
                if crown == "default" or thin or not cell:
                    cell = node["overall"]
                    note = "  # repli : cellule mince" if thin else ""
                else:
                    note = ""
                lines.append(f"      {crown}:{note}"
                             f"   # n={cell['n']}, moyenne {cell['mean_min']:.2f} min")
                for minutes, probability in sorted(cell["pmf"].items(),
                                                   key=lambda kv: int(kv[0])):
                    lines.append(f"        {minutes}: {probability}")
        # `spatialise` stays true for the car: the law IS by ring. The bike
        # keeps an overall law, for lack of a measurable ring effect on 2 047
        # legs — inventing one would be disguised fitting.
        lines.append(f"    provenance: sourced")
        lines.append(f"    spatialise: {'true' if spec['spatialise'] else 'false'}")
        lines.append(f"    labels:")
        for key, value in labels[mode].items():
            lines.append(f"      {key}: \"{value}\"")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output file (default: {DEFAULT_RESOURCE})")
    parser.add_argument("--emit-config", type=Path, default=None,
                        help="Also writes the `modes:` block of terminal_time.yaml")
    args = parser.parse_args()

    root = find_project_root()
    legs = load_legs(root)
    doc = build(legs)
    report(doc)

    out = args.out or DEFAULT_RESOURCE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    if args.emit_config:
        args.emit_config.write_text(emit_yaml(doc), encoding="utf-8")
        print(f"→ {args.emit_config} (`modes:` block to insert into terminal_time.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
