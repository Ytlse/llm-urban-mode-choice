"""Tuesday plays Tuesday's supply, read from the GTFS without recomputation.

Rule: supply is identical ⇔ the timetable is identical; otherwise a trip is served
from the set only if no differing departure falls within its window (departure → departure + 4 h).
Mini-feed: `line:1` runs Monday AND Tuesday (S_both); `line:2` has a run at 10:15 on Monday
(S_lun) replaced by a run at 10:45 on Tuesday (S_mar). Set prepared for Monday 2026-03-16.
"""

import asyncio
import json
from collections import Counter

import pytest

from experiences import jeu as J
from experiences.offre_jour import difference_grille, services_actifs, verifier_offre_jour_gtfs
from experiences.population import charger_population, info_population
from models import Location, Transit, TransitLocation, TravelPlan
from tests.test_trip_set import GYM, HOME, WORK, _pas_de_locale, _population, _sceller
from urban_mobility_agents.simulation_controller import SimulationLoopV1

TOL = {"walk": "insensible", "bike": "insensible", "car": "heure", "transit": {"pas_min": 10}, "rail": {"pas_min": 10}}
BASE = J.jour_base_ts("2026-03-16")


def _leg(mode, o, d, t0, t1, route=None, stop_id=None, transfer=False):
    return Transit(start_time=t0 * 1000, end_time=t1 * 1000, duration=t1 - t0, distance=1200.0, mode=mode,
                   start_location=TransitLocation(stop="A" if stop_id else "", stop_id=stop_id, lat=o.lat, lon=o.lon),
                   end_location=TransitLocation(stop="B" if stop_id else "", stop_id=None, lat=d.lat, lon=d.lon),
                   transit_route=route or f"__DIRECT_{mode.upper()}__", is_transfer=transfer)


class MoteurGTFS:
    """Walk + a bus whose route depends on the destination: WORK → line:1 (08:31 at SP_1), GYM → line:2 (10:15 at SP_2)."""
    def __init__(self):
        self.appels = 0

    async def get_itineraries(self, origin, destination, departure_time, **_):
        self.appels += 1
        dep = departure_time
        plans = [TravelPlan(id=f"foot-{dep}", start_location=origin, end_location=destination, start_time=dep * 1000,
                            end_time=(dep + 1800) * 1000, duration=1800, legs=[_leg("foot", origin, destination, dep, dep + 1800)])]
        if abs(destination.lat - WORK["lat"]) < 1e-9:
            board = BASE + 8 * 3600 + 31 * 60
            legs = [_leg("foot", origin, destination, dep, board, transfer=True), _leg("bus", origin, destination, board, board + 900, "line:1", "stop_point:SP_1")]
        else:
            board = BASE + 10 * 3600 + 15 * 60
            legs = [_leg("foot", origin, destination, dep, board, transfer=True), _leg("bus", origin, destination, board, board + 600, "line:2", "stop_point:SP_2")]
        plans.append(TravelPlan(id=f"bus-{dep}", start_location=origin, end_location=destination, start_time=dep * 1000,
                                end_time=legs[-1].end_time, duration=(legs[-1].end_time // 1000) - dep, legs=legs))
        return plans


@pytest.fixture
def banc(tmp_path, monkeypatch):
    pop = tmp_path / "pop"; pop.mkdir()
    (pop / "population.json").write_text(json.dumps(_population()), encoding="utf-8")
    _sceller(pop)
    personnes, info = charger_population(pop)
    prep = J.JeuEnPreparation.ouvrir(tmp_path / "jeu", "jeu_g", info, "2026-03-16", dependances={"commit": "abc"})
    asyncio.run(J.preparer(prep, personnes, MoteurGTFS(), fabrique_locale=_pas_de_locale, progression_s=100))
    prep.clore(J.deplacements_attendus(personnes, "2026-03-16"), len(personnes))
    gtfs = tmp_path / "gtfs"; gtfs.mkdir()
    (gtfs / "calendar_dates.txt").write_text(
        "service_id,date,exception_type\nS_both,20260316,1\nS_both,20260317,1\nS_lun,20260316,1\nS_mar,20260317,1\n")
    (gtfs / "trips.txt").write_text("route_id,service_id,trip_id,direction_id\nline:1,S_both,T1,0\nline:2,S_lun,T2,0\nline:2,S_mar,T3,0\n")
    (gtfs / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,08:31:00,08:31:00,stop_point:SP_1,1\nT1,08:46:00,08:46:00,stop_point:SP_9,2\n"
        "T2,10:15:00,10:15:00,stop_point:SP_2,1\n"
        "T3,10:45:00,10:45:00,stop_point:SP_2,1\n")          # Tuesday: the 10:15 run no longer exists, the 10:45 one replaces it
    from settings import settings
    monkeypatch.setattr(settings.data, "jeu_tolerances_horaires", dict(TOL))
    monkeypatch.setattr(settings.app, "log_file", str(tmp_path / "run" / "app.log")); (tmp_path / "run").mkdir()
    return {"jeu": J.Jeu.charger(tmp_path / "jeu"), "gtfs": gtfs, "info": info, "personnes": personnes, "jeu_dir": tmp_path / "jeu"}


def test_services_actifs(banc):
    assert services_actifs(banc["gtfs"], "2026-03-16") == {"S_both", "S_lun"}
    assert services_actifs(banc["gtfs"], "2026-03-17") == {"S_both", "S_mar"}


def test_difference_grille(banc):
    d = difference_grille(banc["gtfs"], "2026-03-16", "2026-03-17")
    assert d["courses_seulement_a"] == 1 and d["courses_seulement_b"] == 1          # T2 (Monday) vs T3 (Tuesday)
    assert d["evenements"] == {("line:2", "stop_point:SP_2", 10 * 3600 + 15 * 60): 1, ("line:2", "stop_point:SP_2", 10 * 3600 + 45 * 60): -1}
    assert difference_grille(banc["gtfs"], "2026-03-16", "2026-03-16")["evenements"] == {}


def test_verification_gtfs_par_fenetre(banc):
    res = verifier_offre_jour_gtfs(banc["jeu"], "2026-03-17", banc["gtfs"])
    # Differing departures at 10:15 and 10:45. p3/a1 leaves 08:30 (window → 12:30): affected;
    # p2/b1 leaves 10:10: affected; p3/a2 leaves 17:15: outside window → served. The day is NOT equivalent.
    # The closed day adds two trips, both servable: the closing trip of
    # p2 (b1 → b0) leaves at 23:45, outside window, and that of p3 (a2 → a0) goes from HOME to HOME, so with
    # no transit proposal at all — nothing to recompute there. 5 trips = 2 invalid + 3 valid.
    assert res["equivalent"] is False and res["grille"]["evenements_differents"] == 2
    assert set(res["invalides_cles"]) == {"p3|a1", "p2|b1"} and res["valides"] == 3
    assert res["deplacements"] == 5
    assert res["grille"]["par_route"] == {"line:2": 2} and set(res["grille"]["par_heure"]) == {"10h"}
    # A shorter window (1 h) spares p3/a1 (08:30 → 09:30).
    court = verifier_offre_jour_gtfs(banc["jeu"], "2026-03-17", banc["gtfs"], fenetre_s=3600)
    assert set(court["invalides_cles"]) == {"p2|b1"}
    # Identical timetable (Monday itself): equivalent, everything served.
    lundi = verifier_offre_jour_gtfs(banc["jeu"], "2026-03-16", banc["gtfs"])
    assert lundi["equivalent"] is True and lundi["invalides"] == 0


def test_controleur_sert_ou_recalcule_selon_la_verification(banc):
    c = SimulationLoopV1.__new__(SimulationLoopV1)
    c.jeu = None; c.jeu_tolerances = {}; c._jeu_stats = Counter(); c._jeu_propositions_par_source = Counter()
    c._jeu_alarme_sterile_on = False; c._jeu_stats_depuis_ecriture = 0
    c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3"); a1 = next(a for a in p3.identity.activities if a.id == "a1")
    p2 = next(p for p in banc["personnes"] if p.person_id == "p2"); b1 = next(a for a in p2.identity.activities if a.id == "b1")
    mardi = 86400
    # Nothing verified for Tuesday: nothing is assumed, transit is recomputed.
    assert c._propositions_du_jeu(p3, a1, c.jeu.ligne("p3", "a1").depart_ts + mardi)["a_recalculer"] == {"transit"}
    assert c._jeu_stats["offre_jour_non_verifiee"] == 1
    # After a declared GTFS verification: p3/a2 (17:15, outside the changes window) served from the set;
    # p3/a1 and p2/b1 recomputed (a differing departure falls within their window).
    a2 = next(a for a in p3.identity.activities if a.id == "a2")
    empreinte = c.jeu.empreinte
    c.jeu.declarer_verification_jour("2026-03-17", verifier_offre_jour_gtfs(c.jeu, "2026-03-17", banc["gtfs"]))
    assert c._propositions_du_jeu(p3, a2, c.jeu.ligne("p3", "a2").depart_ts + mardi)["a_recalculer"] == set()
    assert c._propositions_du_jeu(p3, a1, c.jeu.ligne("p3", "a1").depart_ts + mardi)["a_recalculer"] == {"transit"}
    assert c._propositions_du_jeu(p2, b1, c.jeu.ligne("p2", "b1").depart_ts + mardi)["a_recalculer"] == {"transit"}
    assert c._jeu_stats["offre_jour_grille_differente"] == 2
    assert J.Jeu.charger(banc["jeu_dir"]).empreinte == empreinte, "the set stays intact: the verification lives alongside it"
    # Monday (the set's day): served without verifying anything.
    assert c._propositions_du_jeu(p2, b1, c.jeu.ligne("p2", "b1").depart_ts)["a_recalculer"] == set()
    # A Wednesday never verified: recompute.
    assert c._propositions_du_jeu(p3, a1, c.jeu.ligne("p3", "a1").depart_ts + 2 * mardi)["a_recalculer"] == {"transit"}


def test_ligne_valide_le(banc):
    jeu = banc["jeu"]
    assert jeu.ligne_valide_le("2026-03-16", ("p2", "b1")) is True
    assert jeu.ligne_valide_le("2026-03-17", ("p2", "b1")) is None
    jeu.declarer_verification_jour("2026-03-17", verifier_offre_jour_gtfs(jeu, "2026-03-17", banc["gtfs"]))
    assert jeu.ligne_valide_le("2026-03-17", ("p2", "b1")) is False and jeu.ligne_valide_le("2026-03-17", ("p3", "a2")) is True
    assert jeu.jours_equivalents() == []                     # not 100 %: the day is not equivalent, only valid ones are served
