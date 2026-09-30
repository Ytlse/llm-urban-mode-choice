"""Loading filters by the home COMMUNE, no longer by rectangle.

What is locked here:

- **the commune decides**: a 3rd-ring home 45 km from the Capitole, outside the 30 km
  rectangle, is admitted because its commune is one of the 453; a home in central Toulouse whose
  `household.commune_id` says Paris is rejected — the commune is authoritative, not the geometry;
- **cascading fallback, never silent**: without a commune, the `residence_zone` trait; without
  either, the polygon geometry AND an `[ALARME]`;
- **an activity outside the polygon does not drop the agent**: it is counted, and the alarm fires
  on the rising edge above the threshold;
- **a seal loads whole or is refused**;
- **the v4 seal loads whole**: 1,000 / 1,000 admitted by commune, 0 activity outside the polygon —
  the "0 v4 agent dropped at load" measurement;
- **the OSMnx graph key is configured** (polygon of the 453 communes by default) and a missing
  graph is an error, not a download.

    services/llm-agents/.venv/bin/python -m pytest services/llm-agents/tests/test_perimetre_chargement.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geography import (PERIMETER_CACHE_KEY, PRODUCTION_CACHE_KEY_30KM,
                       TOULOUSE_OSM_ROUTES_30K_BBOX)
from models import BBox

import inputs.population.perimeter as perimeter_mod
from inputs.population.perimeter import (ACTIVITY_OUTSIDE_ALARM_SHARE, PopulationPerimeter,
                                         filter_population, sealed_population_complete,
                                         world_extent)

_REPO = Path(__file__).resolve().parents[3]
SEALED_V4 = _REPO / "data" / "population" / "population_1000_PANEL_v4" / "population.json"

CENTRE = (43.6045, 1.4440)          # Capitole, Toulouse (31555)
LOIN_DEDANS = (43.2104, 1.4223)     # Villeneuve-du-Latou (09338), 3rd ring, ~45 km south
DEHORS = (43.6050, 2.2400)          # Castres (81065), outside the 453 communes
PARIS = (48.8566, 2.3522)


@pytest.fixture(scope="module")
def perimeter() -> PopulationPerimeter:
    try:
        return PopulationPerimeter.load()
    except Exception as exc:  # restricted-access resources missing
        pytest.skip(f"scope resources missing: {exc}")


@pytest.fixture
def alarmes():
    """Captures loguru ERRORs — `caplog` does not see them (no propagation)."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="ERROR")
    yield messages
    logger.remove(sink)


def _entry(pid: str, home: tuple[float, float], commune=None, zone=None, acts=()) -> dict:
    lat, lon = home
    traits = {"age": 30}
    if zone is not None:
        traits["residence_zone"] = zone
    activities = [{"purpose": "home", "location": {"lat": lat, "lon": lon}}]
    for purpose, (alat, alon) in acts:
        activities.append({"purpose": purpose, "location": {"lat": alat, "lon": alon}})
    entry = {"person_id": pid,
             "identity": {"traits_json": traits, "home": {"lat": lat, "lon": lon},
                          "activities": activities}}
    if commune is not None:
        entry["household"] = {"id": "h1", "commune_id": commune}
    return entry


def _in_rectangle(lat: float, lon: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = TOULOUSE_OSM_ROUTES_30K_BBOX
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


# ── The commune decides ────────────────────────────────────────────────────────

def test_la_commune_admet_un_domicile_hors_du_rectangle_de_30_km(perimeter):
    assert not _in_rectangle(*LOIN_DEDANS), "the test case must lie outside the old rectangle"
    admis, motif = perimeter.home_verdict(_entry("p", LOIN_DEDANS, commune="09338"))
    assert admis and motif == "commune"


def test_une_commune_hors_des_453_est_ecartee_meme_en_plein_toulouse(perimeter):
    admis, motif = perimeter.home_verdict(_entry("p", CENTRE, commune="75056"))
    assert not admis and motif == "commune hors périmètre"


def test_undefined_compte_comme_commune_manquante(perimeter):
    admis, motif = perimeter.home_verdict(_entry("p", CENTRE, commune="undefined", zone="Toulouse"))
    assert admis and motif == "trait"


# ── The fallback, cascading ─────────────────────────────────────────────────────

def test_sans_commune_le_trait_decide(perimeter):
    assert perimeter.home_verdict(_entry("p", LOIN_DEDANS, zone="3rd ring")) == (True, "trait")
    admis, motif = perimeter.home_verdict(_entry("p", CENTRE, zone="outside perimeter"))
    assert not admis and motif == "trait hors périmètre"
    admis, motif = perimeter.home_verdict(_entry("p", CENTRE, zone="4eme couronne"))
    assert not admis and "zone inconnue" in motif


def test_sans_commune_ni_trait_la_geometrie_decide_et_alarme(perimeter, alarmes):
    kept, stats = filter_population([_entry("a", CENTRE), _entry("b", PARIS)], perimeter, source="test")
    assert [e["person_id"] for e in kept] == ["a"]
    assert stats.admitted_by_geometry == 1 and stats.rejected_geometry_outside == 1
    assert stats.unverified_by_commune == 2
    assert any("[ALARME]" in m and "commune_id" in m for m in alarmes), \
        "a population without commune or trait must raise an alarm at every load"


def test_sans_domicile_pas_d_admission(perimeter):
    e = _entry("p", CENTRE, commune="31555")
    e["identity"]["home"] = None
    assert perimeter.home_verdict(e) == (False, "sans domicile")


# ── Activities outside the polygon ──────────────────────────────────────────────

def test_une_activite_hors_polygone_garde_l_agent_et_se_compte(perimeter):
    e = _entry("p", CENTRE, commune="31555", acts=[("work", PARIS), ("shop", CENTRE)])
    kept, stats = filter_population([e], perimeter, source="test")
    assert len(kept) == 1
    assert (stats.activities_outside, stats.activities_located) == (1, 2)
    assert stats.agents_with_activity_outside == 1


def test_alarme_activites_hors_polygone_sur_front_montant(perimeter, alarmes):
    perimeter_mod._activity_alarm_on = False
    dehors = [_entry("p", CENTRE, commune="31555", acts=[("work", PARIS)])]
    filter_population(dehors, perimeter, source="test")
    filter_population(dehors, perimeter, source="test")
    n_alarmes = sum(1 for m in alarmes if "[ALARME]" in m and "hors du polygone" in m)
    assert n_alarmes == 1, "a single alarm on crossing, not one per load"
    assert perimeter_mod._activity_alarm_on
    # Back below the threshold: the alarm re-arms.
    dedans = [_entry("p", CENTRE, commune="31555", acts=[("work", CENTRE)] * 200)]
    _, stats = filter_population(dedans, perimeter, source="test")
    assert stats.activities_outside_share <= ACTIVITY_OUTSIDE_ALARM_SHARE
    assert not perimeter_mod._activity_alarm_on


# ── The seal ─────────────────────────────────────────────────────────────────

def test_un_sceau_incomplet_est_refuse(alarmes):
    assert sealed_population_complete("sceau.json", 988, 1000) is False
    assert any("[ALARME]" in m and "sceau" in m.lower() for m in alarmes)


def test_un_sceau_entier_est_accepte_et_le_dit(alarmes):
    assert sealed_population_complete("sceau.json", 1000, 1000) is True
    assert not [m for m in alarmes if "[ALARME]" in m]


@pytest.mark.skipif(not SEALED_V4.exists(), reason="sealed v4 population missing")
def test_le_sceau_v4_se_charge_entier(perimeter, alarmes):
    """The measurement: 0 v4 agent dropped at load — and the old rectangle did drop some."""
    raw = json.loads(SEALED_V4.read_text(encoding="utf-8"))
    kept, stats = filter_population(raw, perimeter, source="test-v4")
    assert stats.total == 1000
    assert stats.kept == 1000 and stats.rejected == 0
    assert stats.admitted_by_commune == 1000, "all v4 personas carry household.commune_id"
    assert stats.activities_outside == 0
    assert not [m for m in alarmes if "[ALARME]" in m]
    assert sealed_population_complete(str(SEALED_V4), len(kept), 1000, stats) is True
    # What the 30 km rectangle dropped: at least the far 3rd ring.
    hors_rectangle = sum(1 for e in raw
                         if not _in_rectangle(e["identity"]["home"]["lat"], e["identity"]["home"]["lon"]))
    assert hors_rectangle > 0, "the test measures nothing if v4 fits in the old rectangle"


# ── The world extent ───────────────────────────────────────────────────────

def test_l_emprise_du_monde_couvre_le_polygone_et_les_arrets(perimeter):
    stops = BBox(min_lon=1.10, min_lat=43.34, max_lon=1.74, max_lat=43.80)
    extent = world_extent(stops, perimeter)
    p = perimeter.bbox
    assert extent.min_lon <= min(p.min_lon, stops.min_lon) and extent.max_lon >= max(p.max_lon, stops.max_lon)
    assert extent.min_lat <= min(p.min_lat, stops.min_lat) and extent.max_lat >= max(p.max_lat, stops.max_lat)
    # The test's 3rd-ring home is inside the world — the stops rectangle excluded it.
    assert extent.min_lat <= LOIN_DEDANS[0] <= extent.max_lat
    assert not (stops.min_lat <= LOIN_DEDANS[0] <= stops.max_lat)


# ── The OSMnx graph key ───────────────────────────────────────────────────

def test_la_cle_du_graphe_est_celle_du_polygone_par_defaut(monkeypatch):
    from settings import settings
    from trip_helper.osmnx_direct import graph_key, graph_label

    assert PERIMETER_CACHE_KEY == "444ca7e6a515"
    assert PRODUCTION_CACHE_KEY_30KM == "ecb40f20a303"
    monkeypatch.setattr(settings.gtfs, "osmnx_graph_key", None, raising=False)
    assert graph_key() == PERIMETER_CACHE_KEY
    assert "453" in graph_label(graph_key())
    monkeypatch.setattr(settings.gtfs, "osmnx_graph_key", PRODUCTION_CACHE_KEY_30KM, raising=False)
    assert graph_key() == PRODUCTION_CACHE_KEY_30KM
    assert "audit" in graph_label(graph_key())


def test_un_graphe_absent_est_une_erreur_pas_un_telechargement(tmp_path, monkeypatch, alarmes):
    from settings import settings
    from trip_helper.osmnx_direct import GraphMissingError, _GraphStore

    monkeypatch.setattr(settings.gtfs, "osmnx_cache_dir", str(tmp_path), raising=False)
    with pytest.raises(GraphMissingError):
        _GraphStore._build_sync("deadbeef0000")
    assert any("[ALARME]" in m and "deadbeef0000" in m for m in alarmes)
    assert not list(tmp_path.glob("graphs_*.pkl")), "nothing must have been downloaded or written"
