"""congestion_zones.py — The congestion zone of each node of the OSMnx graph (ticket 031, decision 4).

Three zones, set on the NODES of the graph (``zone`` attribute); an edge takes the zone of
its origin node:

- ``city`` — the commune of Toulouse (the geocoded ``boundary`` of the graph): TomTom "city"
  congestion profile (``city_raw`` of ``config/osmnx.yaml``);
- ``agglo`` — the urban area outside Toulouse: union of the ``Toulouse``, ``1ere couronne``
  and ``2eme couronne`` rings of the EMC² survey (``mobility_core/data/couronne_perimetre.geojson``), minus
  the city: "urban area" profile (``metro_raw``);
- ``outside`` — the rest of the polygon of the 453 communes (the 3rd ring) and beyond: factor 1.0.

WHY. Until 2026-09-03, a trip one end of which touched Toulouse received the "city"
factor over its whole length, and any other trip the "urban area" factor — including a
3rd-ring village → village trip, at 1.84 on a Monday at 8 a.m. With the graph of the polygon of the 453
communes, half of the routed kilometres are rural: the congested duration becomes the sum
of the free-flow times of the edges, each multiplied by the factor of ITS zone at the departure time.

The zones are computed once: when building the polygon graph
(`build_osmnx_perimeter_graph.py`) and, for the historical 30 km graph, lazily at
first loading (`_GraphStore`, `route_worker`), then cached in the pickle. A graph
without zones and without an available urban-area geometry is an explicit error, not a fallback.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ZONE_CITY, ZONE_AGGLO, ZONE_OUTSIDE = "city", "agglo", "outside"
ZONES = (ZONE_CITY, ZONE_AGGLO, ZONE_OUTSIDE)
NODE_ZONE_KEY = "zone"
AGGLO_COURONNES = ("Toulouse", "1st ring", "2nd ring")

# Where to find the ring geometry depending on where the code runs: the resource of the
# mobility_core package when it is importable (repository, controller), otherwise the file mounted
# in the osmnx replicas (`/app/mobility_core/data`) or under /opt.
def _packaged_geojson() -> str:
    try:
        from mobility_core.resources import data_path
        return str(data_path("couronne_perimetre.geojson"))
    except ImportError:  # pragma: no cover - image osmnx sans le paquet
        return ""


GEOJSON_CANDIDATES = (
    os.environ.get("AGGLO_GEOJSON", ""),
    _packaged_geojson(),
    "/opt/mobility_core/data/couronne_perimetre.geojson",
    "/app/mobility_core/data/couronne_perimetre.geojson",
)


class ZoneError(RuntimeError):
    """Missing geometry or node without a zone: we refuse to guess a congestion factor."""


def geojson_path(explicit: Optional[str] = None) -> Path:
    for cand in (explicit, *GEOJSON_CANDIDATES):
        if cand and Path(cand).exists():
            return Path(cand)
    raise ZoneError(
        "ring geometry not found (couronne_perimetre.geojson): searched "
        + ", ".join(c for c in (explicit, *GEOJSON_CANDIDATES) if c)
        + ". Without it, the congestion zones of the graph cannot be computed; mount "
          "mobility_core/data or set AGGLO_GEOJSON.")


def agglo_polygon(explicit: Optional[str] = None):
    """Union of the Toulouse + 1st + 2nd rings (WGS84) — the urban area in the survey's sense."""
    import geopandas as gpd

    path = geojson_path(explicit)
    g = gpd.read_file(path)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    sel = g[g["couronne"].isin(AGGLO_COURONNES)]
    if len(sel) != len(AGGLO_COURONNES):
        raise ZoneError(f"{path} : expected rings {AGGLO_COURONNES}, found {sorted(sel['couronne'])}")
    return sel.union_all() if hasattr(sel, "union_all") else sel.unary_union


def assign_node_zones(G, city_geom, agglo_geom) -> Counter:
    """Set ``zone`` on each node of ``G`` (coordinates ``x`` = lon, ``y`` = lat). Return the counts."""
    import numpy as np
    from shapely import contains_xy, prepare

    nodes = list(G.nodes)
    if not nodes:
        return Counter()
    xs = np.fromiter((G.nodes[n]["x"] for n in nodes), dtype=float, count=len(nodes))
    ys = np.fromiter((G.nodes[n]["y"] for n in nodes), dtype=float, count=len(nodes))
    prepare(city_geom)
    prepare(agglo_geom)
    in_city = contains_xy(city_geom, xs, ys)
    in_agglo = contains_xy(agglo_geom, xs, ys)
    counts: Counter = Counter()
    for n, c, a in zip(nodes, in_city, in_agglo):
        zone = ZONE_CITY if c else ZONE_AGGLO if a else ZONE_OUTSIDE
        G.nodes[n][NODE_ZONE_KEY] = zone
        counts[zone] += 1
    return counts


def has_zones(G) -> bool:
    return all(NODE_ZONE_KEY in data for _, data in G.nodes(data=True))


def ensure_zones(graphs: dict, boundary, geojson: Optional[str] = None,
                 log: Optional[logging.Logger] = None) -> Optional[dict]:
    """Compute the zones of the graphs that have none. Return ``{mode: counts}`` or ``None`` if nothing to do.

    ``boundary``: GeoDataFrame of the commune of Toulouse (``boundary_<clé>.pkl``)."""
    log = log or logger
    missing = [mode for mode, G in graphs.items() if not has_zones(G)]
    if not missing:
        return None
    city = boundary.geometry.iloc[0]
    agglo = agglo_polygon(geojson)
    counts = {}
    for mode in missing:
        counts[mode] = assign_node_zones(graphs[mode], city, agglo)
        log.info("congestion zones set on graph %s : %s", mode, dict(counts[mode]))
    return counts
