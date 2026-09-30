"""Activities outside the polygon of the 453 municipalities, and "same node" fallback per mode.

    services/llm-agents/.venv/bin/python -m pytest services/llm-agents/tests/test_perimetre_activites_et_repli.py -q
"""

import sys
from pathlib import Path

import pytest

_LLM_AGENTS = Path(__file__).resolve().parents[1]
if str(_LLM_AGENTS) not in sys.path:
    sys.path.insert(0, str(_LLM_AGENTS))

from models import Location                                              # noqa: E402
from population_utils import drop_out_of_perimeter_activities            # noqa: E402
from trip_helper.osmnx_direct import (_FALLBACKS, SAME_NODE_DETOUR,      # noqa: E402
                                      _crow_flies_m, _same_node_fallback)


def _classify(lat, lon):
    # Test convention: longitude > 2.0 = outside the 453 municipalities; 0 = unknown.
    if lon == 0:
        return ""
    return "hors périmètre" if lon > 2.0 else "3eme couronne"


def _person(purposes_lons):
    acts, t = [], 0.0
    for i, (purpose, lon) in enumerate(purposes_lons):
        acts.append({"purpose": purpose, "start_time": t, "end_time": t + 3600.0,
                     "scheduled_start_time": 100.0, "location": {"lat": 43.6, "lon": lon}})
        t += 3600.0
    return {"person_id": "p1", "identity": {"activities": acts}}


def test_une_activite_hors_polygone_est_retiree_et_comptee():
    p = _person([("home", 1.4), ("work", 2.5), ("shop", 1.5), ("home", 1.4)])
    assert drop_out_of_perimeter_activities(p, _classify) == 1
    acts = p["identity"]["activities"]
    assert [a["purpose"] for a in acts] == ["home", "shop", "home"]
    # The preceding home absorbs the time slot of the removed activity.
    assert acts[0]["end_time"] == 7200.0 and acts[0]["scheduled_start_time"] is None
    assert p["perimetre"]["activites_hors_perimetre_supprimees"] == 1


def test_le_domicile_n_est_jamais_retire_et_le_compte_vaut_zero_sinon():
    p = _person([("home", 2.5), ("work", 1.5), ("home", 2.5)])
    assert drop_out_of_perimeter_activities(p, _classify) == 0
    assert [a["purpose"] for a in p["identity"]["activities"]] == ["home", "work", "home"]
    assert p["perimetre"] == {"activites_hors_perimetre_supprimees": 0}


def test_un_point_inconnu_ne_decide_rien():
    p = _person([("home", 1.4), ("leisure", 0), ("home", 1.4)])
    assert drop_out_of_perimeter_activities(p, _classify) == 0
    assert len(p["identity"]["activities"]) == 3


def test_activite_retiree_en_tete_ouvre_la_journee_sur_la_suivante():
    p = _person([("work", 2.5), ("home", 1.4)])
    assert drop_out_of_perimeter_activities(p, _classify) == 1
    acts = p["identity"]["activities"]
    assert acts[0]["purpose"] == "home" and acts[0]["start_time"] == 0.0


@pytest.mark.parametrize("mode,speed_kph", [("walk", 5), ("bike", 14), ("drive", 30)])
def test_repli_meme_noeud_a_la_vitesse_du_mode(mode, speed_kph):
    """200 m as the crow flies: 10 s for all modes before (70 km/h); now 260 m of detour
    at the mode's fallback speed — 187 s on foot, 67 s by bike, 31 s by car."""
    assert _FALLBACKS[mode] == speed_kph
    o = Location(lat=43.6000, lon=1.4400)
    d = Location(lat=43.6018, lon=1.4400)           # ≈ 200 m due north
    crow = _crow_flies_m(o, d)
    assert 195 < crow < 205
    r = _same_node_fallback(o, d, mode)
    assert r["distance_m"] == pytest.approx(crow * SAME_NODE_DETOUR)
    assert r["duration_s"] == round(crow * SAME_NODE_DETOUR / (speed_kph * 1000 / 3600))
    assert r["duration_s"] >= 1


def test_repli_meme_noeud_dure_au_moins_une_seconde():
    o = Location(lat=43.6, lon=1.44)
    assert _same_node_fallback(o, o, "drive") == {"duration_s": 1, "distance_m": 0.0}
