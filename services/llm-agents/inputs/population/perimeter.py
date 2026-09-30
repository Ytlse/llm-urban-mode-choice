"""Admission scope of the population at loading (ticket 031, part 2).

The study scope is that of the EMC² 2023 survey: the **453 communes** of six départements,
delimited by the commune polygon (`mobility_core/data/couronne_perimetre.geojson`, table
`mobility_core/data/commune_couronne.json`). Until 2026-09-03, `_prepare_population` filtered
on a RECTANGLE — `TOULOUSE_OSM_ROUTES_30K_BBOX`, the largest rectangle inscribed in the 30 km
OSMnx graph — and discarded any agent whose home or A SINGLE activity left the
rectangle: 79 agents of v3, the whole 3rd ring of v4, hence a rejected seal.

What this module decides, and in this order:

1. **The home defines the scope, and the commune decides.** `household.commune_id`
   (filled in for all personas since v4) ∈ 453 → admitted; filled in and outside the 453 →
   rejected. Without a commune, the `residence_zone` trait (ticket 021) decides: a ring → admitted,
   `hors périmètre` → rejected. Without either, GEOMETRY decides (is the home inside
   the polygon?) and an `[ALARME]` says that the scope was not checked by commune —
   this is not a degraded fallback (same scope, another measure), but a non-enriched
   population must be visible at every loading.
2. **An activity outside the polygon does not discard the agent** (school or workplace outside
   the scope: author's decision, question 3 of ticket 031). It is counted, logged, and
   an `[ALARME]` fires on the rising edge above `ACTIVITY_OUTSIDE_ALARM_SHARE` (1% of
   located activities: beyond that, it is the population or the scope that changed, not a few
   peripheral destinations). The polygon's OSMnx graph does not cover these points: their trip
   falls back on the node nearest to the edge, which the counter makes visible.
3. **A sealed file loads whole or is refused** — `sealed_population_complete`: the headcount
   after filtering must be exactly `population_size`, otherwise `[ALARME]` and nothing is loaded.

The second filter of the chain (`world/population.py` → `eqasim_loader.perimeter_verdict`, on the
`residence_zone` trait) is already by scope and stays as is: both say the same thing,
the home commune is in the 453 or it is not.
"""

from __future__ import annotations

import functools
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models import BBox

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from mobility_core.residence_zone import TRAIT_KEY as RESIDENCE_TRAIT_KEY

PERIMETER_LABEL = "453 communes, six départements, polygone communal"

# Share of located activities (excluding home) outside the polygon beyond which the alarm
# fires. 1%: recommendation of open question no. 3 of ticket 031 (beyond, extend the graph).
ACTIVITY_OUTSIDE_ALARM_SHARE = 0.01

# Values of `household.commune_id` that count as "not filled in" (eqasim v3 export:
# "undefined").
_UNDEFINED = frozenset({"", "undefined", "none", "null", "nan"})

# Number of rejection examples detailed in the log (the full count is always given).
_MAX_REJECT_EXAMPLES = 10


@dataclass
class PerimeterStats:
    """What the filter saw and decided — logged in one line, and returned to the caller."""

    total: int = 0
    kept: int = 0
    admitted_by_commune: int = 0
    admitted_by_trait: int = 0
    admitted_by_geometry: int = 0
    rejected_commune_outside: int = 0
    rejected_trait_outside: int = 0
    rejected_geometry_outside: int = 0
    rejected_no_home: int = 0
    activities_located: int = 0       # non-home activities with coordinates
    activities_outside: int = 0       # … of which outside the polygon of the 453 communes
    agents_with_activity_outside: int = 0
    duration_s: float = 0.0

    @property
    def rejected(self) -> int:
        return self.total - self.kept

    @property
    def unverified_by_commune(self) -> int:
        """Personas judged by geometry for lack of commune and trait."""
        return self.admitted_by_geometry + self.rejected_geometry_outside

    @property
    def activities_outside_share(self) -> float:
        return self.activities_outside / self.activities_located if self.activities_located else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rejected"] = self.rejected
        d["activities_outside_share"] = round(self.activities_outside_share, 5)
        return d


class PopulationPerimeter:
    """The 453 communes (INSEE table) and their dissolved polygon (WGS84), loaded once."""

    def __init__(self, communes, polygon, label: str = PERIMETER_LABEL) -> None:
        from shapely import prepare

        self.communes = communes          # mobility_core.residence_zone.CommuneTable
        self.polygon = polygon            # shapely (Multi)Polygon, EPSG:4326
        prepare(self.polygon)
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        self.bbox = BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)
        self.label = label

    @classmethod
    def load(cls, geojson: Optional[Path] = None, commune_table: Optional[Path] = None) -> "PopulationPerimeter":
        """Load the ring geometry and the commune table. Never a fallback on a rectangle."""
        from mobility_core.residence_zone import (DEFAULT_GEOJSON, CommuneTable,
                                                     ResidenceZoneError)

        path = Path(geojson) if geojson else DEFAULT_GEOJSON
        if not path.exists():
            raise ResidenceZoneError(
                f"ring geometry missing: {path} (`make communes-couronnes`). Without it, "
                "the admission scope cannot be computed — nothing is loaded in its place.")
        import geopandas as gpd

        t0 = time.monotonic()
        layer = gpd.read_file(path).to_crs(4326)
        inconnues = sorted(set(map(str, layer["couronne"])) - set(COURONNES))
        if inconnues:
            raise ResidenceZoneError(f"unexpected rings in {path.name} : {inconnues}.")
        polygon = layer.union_all() if hasattr(layer, "union_all") else layer.unary_union
        communes = CommuneTable.load(commune_table)
        perimeter = cls(communes, polygon)
        logger.info(
            f"[perimetre] {perimeter.label} : {len(communes)} communes, polygon "
            f"{polygon.geom_type} lon {perimeter.bbox.min_lon:.3f}→{perimeter.bbox.max_lon:.3f} "
            f"lat {perimeter.bbox.min_lat:.3f}→{perimeter.bbox.max_lat:.3f}, loaded in "
            f"{time.monotonic() - t0:.1f}s from {path.name}")
        return perimeter

    # ── Membership tests ─────────────────────────────────────────────────────

    def contains(self, lon: float, lat: float) -> bool:
        from shapely import contains_xy

        return bool(contains_xy(self.polygon, lon, lat))

    def home_verdict(self, entry: dict) -> tuple[bool, str]:
        """`(admitted, reason)` of an eqasim record (raw dict).

        Reason if admitted: `commune`, `trait` or `geometrie` — what decided. Reason if rejected:
        `sans domicile`, `commune hors périmètre`, `trait hors périmètre`, `zone inconnue (…)`,
        `géométrie hors polygone`.
        """
        identity = entry.get("identity") or {}
        home = identity.get("home") or {}
        lon, lat = home.get("lon"), home.get("lat")
        if lon is None or lat is None:
            return False, "sans domicile"

        insee = str((entry.get("household") or {}).get("commune_id") or "").strip()
        if insee.lower() not in _UNDEFINED:
            if self.communes.contains(insee):
                return True, "commune"
            return False, "commune hors périmètre"

        zone = (identity.get("traits_json") or {}).get(RESIDENCE_TRAIT_KEY)
        if zone:
            if zone in COURONNES:
                return True, "trait"
            if zone == OUT_OF_PERIMETER:
                return False, "trait hors périmètre"
            return False, f"zone inconnue ({zone})"

        if self.contains(float(lon), float(lat)):
            return True, "geometrie"
        return False, "géométrie hors polygone"

    def activities_outside(self, entry: dict) -> tuple[int, int]:
        """`(outside polygon, located)` among the non-home activities of the record."""
        outside = located = 0
        for act in (entry.get("identity") or {}).get("activities") or []:
            if act.get("purpose") == "home":
                continue
            loc = act.get("location") or {}
            lon, lat = loc.get("lon"), loc.get("lat")
            if lon is None or lat is None:
                continue
            located += 1
            if not self.contains(float(lon), float(lat)):
                outside += 1
        return outside, located


@functools.lru_cache(maxsize=1)
def load_population_perimeter() -> PopulationPerimeter:
    """The repository's scope, loaded once per process."""
    return PopulationPerimeter.load()


# ── Filter ───────────────────────────────────────────────────────────────────

_activity_alarm_on = False


def filter_population(raw: list, perimeter: PopulationPerimeter,
                      source: str = "population") -> tuple[list, PerimeterStats]:
    """Keep the records whose home is inside the scope; count the rest.

    Logs a summary line (explicit success included), up to ten detailed rejections,
    an `[ALARME]` if personas had to be judged by geometry, and an `[ALARME]` on the rising
    edge if the share of activities outside the polygon exceeds `ACTIVITY_OUTSIDE_ALARM_SHARE`.
    """
    global _activity_alarm_on

    t0 = time.monotonic()
    stats = PerimeterStats(total=len(raw))
    kept: list = []
    rejects: Counter = Counter()
    examples: list[str] = []

    for entry in raw:
        admis, motif = perimeter.home_verdict(entry)
        if not admis:
            rejects[motif] += 1
            if motif == "sans domicile":
                stats.rejected_no_home += 1
            elif motif == "commune hors périmètre":
                stats.rejected_commune_outside += 1
            elif motif == "géométrie hors polygone":
                stats.rejected_geometry_outside += 1
            else:
                stats.rejected_trait_outside += 1
            if len(examples) < _MAX_REJECT_EXAMPLES:
                home = (entry.get("identity") or {}).get("home") or {}
                insee = (entry.get("household") or {}).get("commune_id")
                examples.append(f"{entry.get('person_id', '?')} ({motif}, commune={insee}, "
                                f"lat={home.get('lat')}, lon={home.get('lon')})")
            continue

        kept.append(entry)
        stats.kept += 1
        if motif == "commune":
            stats.admitted_by_commune += 1
        elif motif == "trait":
            stats.admitted_by_trait += 1
        else:
            stats.admitted_by_geometry += 1

        outside, located = perimeter.activities_outside(entry)
        stats.activities_located += located
        stats.activities_outside += outside
        if outside:
            stats.agents_with_activity_outside += 1

    stats.duration_s = round(time.monotonic() - t0, 3)

    detail = ", ".join(f"{n} {m}" for m, n in rejects.most_common()) or "aucun rejet"
    logger.info(
        f"[{source}] scope filter ({perimeter.label}) : {stats.total} → {stats.kept} agents "
        f"in {stats.duration_s:.2f}s — admitted by commune {stats.admitted_by_commune}, by trait "
        f"{stats.admitted_by_trait}, by geometry {stats.admitted_by_geometry} ; discarded "
        f"{stats.rejected} ({detail}) ; activities outside polygon {stats.activities_outside} / "
        f"{stats.activities_located} located ({100 * stats.activities_outside_share:.2f} %, "
        f"{stats.agents_with_activity_outside} agent(s) affected)")
    for ex in examples:
        logger.warning(f"[{source}] agent discarded from the scope: {ex}")

    if stats.unverified_by_commune:
        # No rising edge: this is a state of the population, it must be visible at every loading.
        logger.error(
            f"[ALARME] [{source}] {stats.unverified_by_commune}/{stats.total} persona(s) without "
            f"`household.commune_id` or trait `{RESIDENCE_TRAIT_KEY}` : their scope was judged "
            f"by the polygon geometry ({stats.admitted_by_geometry} admitted, "
            f"{stats.rejected_geometry_outside} rejected), not by the home commune. "
            "Population older than v4: regenerate or enrich it (`make residence-zone`).")

    share = stats.activities_outside_share
    if share > ACTIVITY_OUTSIDE_ALARM_SHARE:
        if not _activity_alarm_on:
            _activity_alarm_on = True
            logger.error(
                f"[ALARME] [{source}] {stats.activities_outside} activité(s) sur "
                f"{stats.activities_located} ({100 * share:.2f} %) hors du polygone des 453 communes, "
                f"au-dessus du seuil de {100 * ACTIVITY_OUTSIDE_ALARM_SHARE:.0f} %. Les agents sont "
                "gardés (le domicile fait le périmètre) mais le graphe OSMnx du polygone ne couvre pas "
                "ces points : leurs trajets se rabattent sur le bord du graphe. Une population ou un "
                "périmètre a changé — étendre le polygone du graphe ou revoir la population.")
    elif _activity_alarm_on:
        _activity_alarm_on = False
        logger.info(f"[{source}] share of activities outside polygon back under the threshold "
                    f"({100 * share:.2f} %) — alarm cleared")
    return kept, stats


def sealed_population_complete(sealed_path: str, kept: int, population_size: int,
                               stats: Optional[PerimeterStats] = None) -> bool:
    """A seal is taken whole: `kept == population_size`, otherwise `[ALARME]` and refusal.

    Resampling 1,000 agents from a sealed file of 1,000 from which the filter discarded 12
    would amount to publishing a population that is no longer the MANIFEST's — without anything
    signalling it.
    """
    if kept == population_size:
        logger.info(f"[population] Sealed population loaded whole: {kept}/{population_size} "
                    f"agents, 0 discarded at loading — {sealed_path}")
        return True
    ecartes = f" ({stats.rejected} écartés par le filtre de périmètre)" if stats is not None else ""
    logger.error(
        f"[ALARME] [population] Population scellée {sealed_path} : {kept} agents après filtre de "
        f"périmètre pour population_size={population_size}{ecartes}. Un sceau ne se rogne pas : "
        "alignez population_size sur l'effectif scellé, ou corrigez la population (domiciles hors "
        "des 453 communes). Rien n'est chargé.")
    return False


# ── World footprint ──────────────────────────────────────────────────────────

def world_extent(stops_bbox: Optional[BBox], perimeter: Optional[PopulationPerimeter] = None) -> BBox:
    """World envelope: polygon of the 453 communes ∪ footprint of the GTFS stops (± buffer).

    `WorldGrid` asserts that every location is inside its rectangle: before 2026-09-03 this
    rectangle was that of the Tisséo stops ± 0.05°, which only contains 221 of the 453 communes —
    a 3rd-ring home made the assertion fail there. The world now covers the whole
    scope; the union with the stops keeps every GTFS stop inside the grid.
    """
    perimeter = perimeter or load_population_perimeter()
    p = perimeter.bbox
    if stops_bbox is None:
        extent = BBox(min_lon=p.min_lon, min_lat=p.min_lat, max_lon=p.max_lon, max_lat=p.max_lat)
    else:
        extent = BBox(
            min_lon=min(p.min_lon, stops_bbox.min_lon), min_lat=min(p.min_lat, stops_bbox.min_lat),
            max_lon=max(p.max_lon, stops_bbox.max_lon), max_lat=max(p.max_lat, stops_bbox.max_lat),
        )
    logger.info(
        f"[world] world footprint = polygon of the 453 communes ∪ GTFS stops: "
        f"lon {extent.min_lon:.4f}→{extent.max_lon:.4f}, lat {extent.min_lat:.4f}→{extent.max_lat:.4f} "
        f"(polygon only: {p.min_lon:.4f}→{p.max_lon:.4f}, {p.min_lat:.4f}→{p.max_lat:.4f})")
    return extent
