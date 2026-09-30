"""
core/geo_reference.py — The Toulouse hypercentre, read and not redeclared.

The city centre serves as reference in two places that must talk about the same point:
the `dist_center_orig_km` / `dist_center_dest_km` variables of the mode choice model
(baked into the fine-zone layer, cf. `zone_resolver`) and the
"Lieu de résidence" column of the move-log, which classes each agent as Toulouse / 1st /
2nd / 3rd ring according to its distance to the centre.

**The reference is `scripts/progedo_logit/feature_spec.json`**, block
`geo_reference.hypercenter`: the value there is computed from the EMC² survey
data (centroid of the fine zones of sector 01, Capitole) by
`scripts/progedo_logit/build_mode_choice_dataset.py`, and published with the feature
contract. No other module may redeclare one: `move_logger.py` carried
a second one (43.6047 / 1.4442), 820 m away, which shifted the residence
rings relative to the distances to the centre seen in training.

**The spec may be missing at run time.** It is produced from the restricted-access
PROGEDO microdata (`data/PROGEDO 2023/`), outside the repository: a machine or a
container without these data does not have the file. The fallback is then the spec value
copied as a constant below (`FALLBACK_GEO_REFERENCE`) — never the old
competing constant — and it is logged once.

This module is the only reader of the `geo_reference` block: `zone_resolver.load`
uses it for its safeguard (it refuses to serve a layer whose geographic
reference diverges from the spec's), and `move_logger` for its ring
classification.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from loguru import logger

from mobility_core.resources import find_repo_file

# Literal copy of `geo_reference.hypercenter` from spec v1 (EMC² Toulouse 2023).
# Used only as a fallback when the spec cannot be found; any update of the spec
# must be carried over here, and the `test_move_logger_hypercenter.py` test fails if the
# two diverge while the spec is present.
FALLBACK_GEO_REFERENCE = {
    "crs": "IGNF:LAMB93",
    "hypercenter": {
        "definition": "centroïde des zones fines du secteur 01 (Capitole)",
        "x_l93": 574406.1,
        "y_l93": 6278824.6,
        "lat": 43.597347,
        "lon": 1.444997,
    },
}

# The spec is not copied into the package: it lives in the repository
# (`scripts/progedo_logit/feature_spec.json`). Search order (cf. `resources`):
#   1. the environment variable, which always takes precedence;
#   2. the root designated by MOBILITY_CORE_REPO_ROOT, then a walk up from the
#      current directory;
#   3. the `controller` container, where `scripts/` is mounted under /app/scripts.
FEATURE_SPEC_ENV = "MODE_CHOICE_FEATURE_SPEC"
FEATURE_SPEC_RELATIVE = "scripts/progedo_logit/feature_spec.json"


def find_feature_spec() -> Path | None:
    """First `feature_spec.json` found at the standard locations, or `None`."""
    return find_repo_file(FEATURE_SPEC_RELATIVE, env_var=FEATURE_SPEC_ENV)


def read_geo_reference(feature_spec: Path) -> dict:
    """`geo_reference` block of a given spec. Raises if the file is unreadable.

    Strict version, for the caller that already knows which spec it wants to compare
    (`zone_resolver.load`). A spec without a `geo_reference` block returns `{}`: it is an
    older spec, not a read error.
    """
    spec = json.loads(Path(feature_spec).read_text(encoding="utf-8"))
    return spec.get("geo_reference", {}) or {}


@lru_cache(maxsize=1)
def geo_reference() -> dict:
    """Effective geographic reference: the spec's, or the documented fallback.

    Cached: a run calls `hypercenter()` for every logged trip,
    which is no reason to reread the spec every time. `geo_reference.
    cache_clear()` lets tests replay the resolution.
    """
    path = find_feature_spec()
    if path is None:
        logger.warning(
            "feature_spec.json not found: hypercentre falls back to the published "
            f"spec value ({FALLBACK_GEO_REFERENCE['hypercenter']['lat']} / "
            f"{FALLBACK_GEO_REFERENCE['hypercenter']['lon']}). The PROGEDO data "
            "are restricted-access, their absence is a normal case."
        )
        return FALLBACK_GEO_REFERENCE

    try:
        reference = read_geo_reference(path)
    except (OSError, ValueError) as exc:
        logger.warning(
            f"feature_spec.json unreadable ({path}): {exc}. Hypercentre falls back to "
            "the published spec value."
        )
        return FALLBACK_GEO_REFERENCE

    center = reference.get("hypercenter") or {}
    if center.get("lat") is None or center.get("lon") is None:
        logger.warning(
            f"feature_spec.json ({path}) does not publish a usable hypercentre: "
            "falling back to the published spec value."
        )
        return FALLBACK_GEO_REFERENCE

    logger.info(
        f"Hypercentre read from {path} | lat={center['lat']} lon={center['lon']} "
        f"({center.get('definition', 'sans définition')})"
    )
    return reference


def hypercenter() -> tuple[float, float]:
    """Hypercentre `(lat, lon)` in WGS84 — the only way to get it at runtime."""
    center = geo_reference()["hypercenter"]
    return float(center["lat"]), float(center["lon"])


# Ring bounds, in km from the hypercentre. They carry the `lieu_residence` LABELS of
# the EMC² reference, but they are NOT its definition:
# the survey splits by list of communes (ticket 020, axis A2 — 24.4% of personas
# reclassified). Since ticket 028, nothing in production reads them any more: the
# residence is read on the persona (ticket 021) and terminal time classifies by
# ring membership, its laws being stratified by the survey table
# (`terminal_time_emc2.json`, `meta.crown_definition`, tt4). They remain as a CONTROL
# for the measurement scripts — cf. `mobility_core.residence_zone` and the docstring of
# `residence_zone` below. Do not move them: an archived trace measured them.
COURONNE_BOUNDS_KM: tuple[tuple[float, str], ...] = (
    (8.0,  "Toulouse"),
    (20.0, "1st ring"),
    (40.0, "2nd ring"),
)
COURONNE_OUTER = "3rd ring"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    import math

    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def residence_zone(lat: float | None, lon: float | None) -> str:
    """METRIC classification of a point — **audit control, zero production callers**.

    Returns ``Toulouse`` / ``1ere`` / ``2eme`` / ``3eme couronne`` from the distance
    to the hypercentre alone (8 / 20 / 40 km). Empty string when the point is unknown — empty
    is not a category, exactly as an empty probability cell is not
    a 0.

    ⚠ **THIS IS NOT THE DEFINITION OF ANYTHING IN THE EMC² SENSE.** This function long
    claimed to serve "the `lieu_residence` categories of the EMC² reference"; that was
    false, and it cost two years of misread modal shares by zone. The survey splits its
    scope by **list of communes**, not by metric rings. Ticket 020
    quantified the gap: **24.4%** of personas change ring, **66** "Toulousains"
    actually live in Blagnac, Balma or Colomiers, and **45** homes filed in the 3rd
    ring are outside the survey scope — because "beyond 40 km" has no
    upper bound.

    **Who classifies what, from now on.** The RESIDENCE ring is read on the persona
    (`residence_zone` trait, `mobility_core.residence_zone`, ticket 021); TERMINAL
    TIME classifies its origin and destination points by ring membership
    (`CommunalZones`, ticket 028), and its laws are stratified by the survey table
    (`terminal_time_emc2.json`, `meta.crown_definition`, `tt4`). Neither `move_logger`, nor
    `osmnx_direct`, nor `export_terminal_time` imports this function, and a test
    enforces it for each: the distance fallback is impossible by construction, not
    merely discouraged.

    **Why it stays.** As a COMPARATOR: `audit_perimetre` (historical axis A2),
    `enrich_residence_zone --check` and `measure_couronne_v7` check it against the communal
    classification to quantify what the old definition made the modal shares say, and
    an archived trace must remain replayable identically. Deleting it would erase the
    control; calling it in production would restore the defect.

    **What the divergence cost, for the record.** Under `tt3`, log and terminal time
    could designate two rings for the same home: **34 s per trip end**
    on the worst observed pair (Toulouse vs 1st ring, 66 cases). Ticket 028
    closed it by re-exporting the resource — and by re-stratifying, it gave a ring back
    to ~5,300 survey trips that the centroid left out of the strata (25.6% → 3.9%).
    """
    if lat is None or lon is None:
        return ""
    center_lat, center_lon = hypercenter()
    d = haversine_km(center_lat, center_lon, lat, lon)
    for bound, name in COURONNE_BOUNDS_KM:
        if d < bound:
            return name
    return COURONNE_OUTER
