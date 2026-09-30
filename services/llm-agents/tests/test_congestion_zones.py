"""Congestion applies per zone of the edge (city / metropolitan area / outside).

    services/llm-agents/.venv/bin/python -m pytest services/llm-agents/tests/test_congestion_zones.py -q
"""

import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import box

_LLM_AGENTS = Path(__file__).resolve().parents[1]
if str(_LLM_AGENTS) not in sys.path:
    sys.path.insert(0, str(_LLM_AGENTS))

from trip_helper import osmnx_direct as od                                   # noqa: E402
from trip_helper.congestion_zones import (NODE_ZONE_KEY, ZONE_AGGLO,          # noqa: E402
                                          ZONE_CITY, ZONE_OUTSIDE, ZoneError,
                                          assign_node_zones, ensure_zones, has_zones)

LUNDI_8H = datetime(2024, 1, 8, 8, 0)     # Monday: "Mon" column of the TomTom tables


def _graph():
    """Four nodes aligned eastwards, three 100 s edges: city → metro area → outside."""
    G = nx.MultiDiGraph(crs="EPSG:4326")
    coords = {1: (1.440, 43.600), 2: (1.450, 43.600), 3: (1.460, 43.600), 4: (1.470, 43.600)}
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y)
    for u, v in ((1, 2), (2, 3), (3, 4)):
        G.add_edge(u, v, key=0, length=1000.0, travel_time=100.0, highway="primary")
    G.nodes[1][NODE_ZONE_KEY] = ZONE_CITY
    G.nodes[2][NODE_ZONE_KEY] = ZONE_AGGLO
    G.nodes[3][NODE_ZONE_KEY] = ZONE_OUTSIDE
    G.nodes[4][NODE_ZONE_KEY] = ZONE_OUTSIDE
    return G


def test_facteurs_par_zone_lundi_8h():
    ville = od._CITY_FF / float(od._CITY_DF.loc["08:00", "Mon"])
    agglo = od._METRO_FF / float(od._METRO_DF.loc["08:00", "Mon"])
    assert od._zone_factor(ZONE_CITY, LUNDI_8H) == pytest.approx(ville)
    assert od._zone_factor(ZONE_AGGLO, LUNDI_8H) == pytest.approx(agglo)
    assert od._zone_factor(ZONE_OUTSIDE, LUNDI_8H) == 1.0
    assert ville > 1.5 and agglo > 1.5        # peak hour: both profiles congest
    with pytest.raises(ZoneError):
        od._zone_factor(None, LUNDI_8H)        # a node without a zone has no guessable factor


def test_duree_congestionnee_est_la_somme_par_arete():
    """One edge per zone: the city and the metro area are congested, the outside is not."""
    G = _graph()
    gdf = ox.routing.route_to_gdf(G, [1, 2, 3, 4], weight="travel_time")
    # Third value: the number of edges with accidents, zero here.
    cong, free, n_acc = od._congested_travel_time(G, gdf, LUNDI_8H)
    assert n_acc == 0
    ville = od._zone_factor(ZONE_CITY, LUNDI_8H)
    agglo = od._zone_factor(ZONE_AGGLO, LUNDI_8H)
    assert free == pytest.approx(300.0)
    assert cong == pytest.approx(100.0 * ville + 100.0 * agglo + 100.0 * 1.0)
    # The overall factor is the free-flow-time weighted mean — strictly between 1 and city.
    assert 1.0 < cong / free < ville


def test_trajet_hors_agglomeration_a_un_facteur_de_un():
    G = _graph()
    gdf = ox.routing.route_to_gdf(G, [3, 4], weight="travel_time")
    # Third value: the number of edges with accidents, zero here.
    cong, free, n_acc = od._congested_travel_time(G, gdf, LUNDI_8H)
    assert n_acc == 0
    assert cong == pytest.approx(free) == pytest.approx(100.0)


def test_route_sync_congestionne_la_seule_part_agglomeree():
    """End to end with the production code: origin near node 1 (city), destination near
    node 4 (outside) → duration = Σ edges × zone factor, not a single factor."""
    from models import Location
    G = _graph()
    o = Location(lat=43.6, lon=1.4401)
    d = Location(lat=43.6, lon=1.4699)
    r = od._route_sync(G, None, o, d, "drive", LUNDI_8H)
    ville = od._zone_factor(ZONE_CITY, LUNDI_8H)
    agglo = od._zone_factor(ZONE_AGGLO, LUNDI_8H)
    attendu = int(100.0 * ville + 100.0 * agglo + 100.0)
    assert r["duration_s"] == attendu
    assert r["distance_m"] == pytest.approx(3000.0)
    # The same trip by bike is not congested.
    assert od._route_sync(G, None, o, d, "bike", LUNDI_8H)["duration_s"] == 300


def test_assign_node_zones_et_ensure_zones():
    import geopandas as gpd
    G = _graph()
    for n in G.nodes:
        del G.nodes[n][NODE_ZONE_KEY]
    assert not has_zones(G)
    city = box(1.435, 43.59, 1.445, 43.61)          # contains node 1
    agglo = box(1.435, 43.59, 1.455, 43.61)         # contains nodes 1 and 2
    counts = assign_node_zones(G, city, agglo)
    assert dict(counts) == {ZONE_CITY: 1, ZONE_AGGLO: 1, ZONE_OUTSIDE: 2}
    assert has_zones(G)
    # ensure_zones redoes nothing on an already zoned graph…
    boundary = gpd.GeoDataFrame(geometry=[city], crs="EPSG:4326")
    assert ensure_zones({"drive": G}, boundary) is None
    # … and sets the zones of a graph that has none, with the real metro-area geometry.
    H = _graph()
    for n in H.nodes:
        del H.nodes[n][NODE_ZONE_KEY]
    done = ensure_zones({"drive": H}, boundary)
    assert done is not None and sum(done["drive"].values()) == 4
    assert H.nodes[1][NODE_ZONE_KEY] == ZONE_CITY
    assert H.nodes[4][NODE_ZONE_KEY] in (ZONE_AGGLO, ZONE_OUTSIDE)   # at 1.47° E: real metro area
