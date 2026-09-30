"""A single way to decide.

One test per rule, named by its number. The business rules of the vehicle chain are
not re-tested here (that is `test_vehicle_chain.py`, D3): we check that the single decision
applies them, JUSTIFIES them, TRACES them and ORDERS them.
"""

import asyncio
import os
import re
import time
from pathlib import Path

import pytest

from models import Location, Person, PersonalIdentity, PersonState, Transit, TransitLocation, TravelPlan
from experiences import decision as d

HOME = Location(lat=43.6000, lon=1.4400)
WORK = Location(lat=43.6100, lon=1.4500)
GYM = Location(lat=43.6200, lon=1.4600)
FAR = Location(lat=43.6300, lon=1.4400)   # ≈ 3.3 km from home as the crow flies


def _plan(code: str, *modes: str, start=HOME, end=WORK, duration=600) -> TravelPlan:
    loc = TransitLocation(stop="", lat=start.lat, lon=start.lon)
    legs = [
        Transit(start_time=0, end_time=duration, duration=duration, distance=1000.0, mode=m,
                start_location=loc, end_location=loc, is_transfer=(m == "foot" and len(modes) > 1),
                transit_route=code if (m not in ("foot",) or len(modes) == 1) else None)
        for i, m in enumerate(modes)
    ]
    return TravelPlan(id=code, start_location=start, end_location=end, start_time=0, end_time=duration,
                      duration=duration, legs=legs)


def K(code: str) -> str:
    """`TravelPlan.get_code()` of a test plan: route^stop^stop with empty stops."""
    return f"{code}^^"


def _person(pid="p1", **traits) -> Person:
    base = {"personal_bike": "vélo normal", "number_of_cars": 1, "has_driving_license": True,
            "age": 35, "household_size": 1}
    base.update(traits)
    return Person(person_id=pid, identity=PersonalIdentity(name="t", traits_json=base, home=HOME),
                  state=PersonState())


def _props(*specs) -> list[d.Proposition]:
    """specs: (code, modes...)"""
    return [d.Proposition(_plan(spec[0], *spec[1:])) for spec in specs]


STANDARD = (("W", "foot"), ("B", "bicycle"), ("C", "car"), ("T", "foot", "bus", "foot"))


def _ctx(activity_id="a1", purpose="work", destination=WORK, **kw) -> d.ContexteDecision:
    return d.ContexteDecision(timestamp=1773648000, activity_id=activity_id, purpose=purpose,
                              departure_time=1773648000, from_location=HOME, destination=destination, **kw)


class DecideurFixe:
    """Test decision-maker: returns a fixed index (or by code), and counts its calls."""
    nom = "fixe"
    sans_quota = True

    def __init__(self, index=0, code=None, erreur=None):
        self.index, self.code, self.erreur, self.appels = index, code, erreur, 0

    async def choisir(self, person, ctx, presentees):
        self.appels += 1
        if self.erreur:
            return d.ReponseDecideur(index=None, erreur=self.erreur)
        idx = self.index
        if self.code is not None:
            idx = next(i for i, p in enumerate(presentees) if p.code == self.code)
        return d.ReponseDecideur(index=idx, fournisseur="test", distribution={"walking": 1.0},
                                 reponse_brute="[]", poids=[1.0 / len(presentees)] * len(presentees))


def _run(coro):
    return asyncio.run(coro)


# ── D1: no other implementation of the filter outside the single function ─────────────
def test_D1_aucune_autre_implementation_du_filtre():
    racine = Path(__file__).resolve().parents[1]
    autorises = {racine / "urban_mobility_agents" / "vehicle_chain.py", racine / "experiences" / "decision.py"}
    appels = re.compile(r"\b(_vehicle_available|_vehicle_unavailable_reason|_vehicles_parked_at)\(")
    fautifs = []
    for py in racine.rglob("*.py"):
        if ".venv" in py.parts or "tests" in py.parts or py in autorises:
            continue
        if appels.search(py.read_text(encoding="utf-8", errors="ignore")):
            fautifs.append(str(py.relative_to(racine)))
    assert fautifs == [], f"the exit lock is reimplemented/called outside the single decision: {fautifs}"


# ── D2: four reasons, walking and PT never dropped ───────────────────────────────────
def test_D2_non_possede():
    r = d.eligibilite(_person(personal_bike="Pas de vélo"), HOME, _props(*STANDARD), "work", WORK)
    assert {(e.code, e.motif) for e in r.ecartees} == {(K("B"), "non_possede")}
    assert {p.code for p in r.eligibles} == {K("W"), K("C"), K("T")}


def test_D2_pas_de_conducteur():
    mineur_seul = _person(age=12, has_driving_license=False, household_size=1)
    r = d.eligibilite(mineur_seul, HOME, _props(*STANDARD), "work", WORK)
    assert (K("C"), "pas_de_conducteur") in {(e.code, e.motif) for e in r.ecartees}
    assert ("car", "no_driver") in r.evenements
    # with an adult in the household: passenger, the car stays
    mineur_accompagne = _person(age=12, has_driving_license=False, household_size=3)
    r2 = d.eligibilite(mineur_accompagne, HOME, _props(*STANDARD), "work", WORK)
    assert K("C") in {p.code for p in r2.eligibles}


def test_D2_vehicule_ailleurs():
    p = _person()
    p.state.planning_vehicle_at["car"] = WORK          # the car sleeps at work
    r = d.eligibilite(p, HOME, _props(*STANDARD), "shop", GYM)
    assert (K("C"), "vehicule_ailleurs") in {(e.code, e.motif) for e in r.ecartees}
    assert ("car", "unavailable") in r.evenements and r.contrainte == "sortie_bloquee"
    assert {K("W"), K("T")} <= {q.code for q in r.eligibles}


def test_D2_retour_force():
    p = _person()
    p.state.planning_vehicle_at["bike"] = FAR          # bike parked at the start point (3 km from home)
    p.state.planning_vehicle_at["car"] = FAR
    props = [d.Proposition(_plan(c, *m, start=FAR, end=HOME)) for c, *m in STANDARD]
    r = d.eligibilite(p, FAR, props, "home", HOME)
    assert {q.code for q in r.eligibles} == {K("B"), K("C")}             # both vehicles are there: the decision-maker chooses
    assert {(e.code, e.motif) for e in r.ecartees} == {(K("W"), "retour_force"), (K("T"), "retour_force")}
    assert r.contrainte == "retour_force"


def test_D2_marche_et_tc_jamais_ecartes_par_le_filtre():
    p = _person(personal_bike="Pas de vélo", number_of_cars=0)
    r = d.eligibilite(p, HOME, _props(*STANDARD), "work", WORK)
    assert {q.code for q in r.eligibles} == {K("W"), K("T")}


# ── D4: the chain of the day ──────────────────────────────────────────────────────
def test_D4_chaine_de_la_journee():
    p = _person()
    d.avancer_chaine(p, _plan("C", "car", start=HOME, end=WORK), HOME, WORK, "work")   # home → work by car
    d.avancer_chaine(p, _plan("W", "foot", start=WORK, end=GYM), WORK, GYM, "sport")   # work → sport on foot
    r = d.eligibilite(p, GYM, _props(*STANDARD), "shop", FAR)                            # 3rd trip from the sport place
    assert (K("C"), "vehicule_ailleurs") in {(e.code, e.motif) for e in r.ecartees}
    assert p.state.planning_vehicle_at["car"] == WORK


# ── D5: no solution / single choice, decision-maker not called ─────────────────────────
def test_D5_sans_solution_et_choix_unique():
    dec = DecideurFixe()
    p = _person(personal_bike="Pas de vélo", number_of_cars=0)
    rien = _run(d.decider(p, _ctx(), _props(("B", "bicycle"), ("C", "car")), dec))
    assert rien.methode == "sans_solution" and rien.retenue is None and dec.appels == 0
    assert {e["motif"] for e in rien.trace["ecartees"]} == {"non_possede"}
    seule = _run(d.decider(p, _ctx(), _props(("W", "foot"), ("C", "car")), dec))
    assert seule.methode == "choix_unique" and seule.retenue.code == K("W") and dec.appels == 0


# ── D6: the trace carries the five elements ─────────────────────────────────────────────
def test_D6_trace_complete():
    dec = DecideurFixe(code=K("T"))
    p = _person(personal_bike="Pas de vélo")
    res = _run(d.decider(p, _ctx(), _props(*STANDARD), dec))
    t = res.trace
    assert d.valider_trace(t) == []
    assert [x["code"] for x in t["presentees"]] and t["retenue"]["code"] == K("T")
    assert t["ecartees"] == [{"code": K("B"), "mode": "bicycle", "motif": "non_possede"}]
    assert t["distribution"] == {"walking": 1.0} and t["reponse_brute"] == "[]"
    assert set(t["sources"]) == {x["code"] for x in t["presentees"]}
    del t["reponse_brute"]
    assert d.valider_trace(t) == ["reponse_brute"]


# ── D7: deterministic order, independent of the order received, seed → order ──────────────
def test_D7_ordre_de_presentation():
    props = _props(("A", "foot"), ("B", "bicycle"), ("C", "car"), ("D", "foot", "bus", "foot"),
                   ("E", "foot", "metro", "foot"), ("F", "foot", "tram", "foot"))
    o1 = [p.code for p in d.ordre_presentation(props, 42, "p1", "a1")]
    o2 = [p.code for p in d.ordre_presentation(list(reversed(props)), 42, "p1", "a1")]
    assert o1 == o2, "same seed, same trip ⇒ same order whatever the order received"
    o3 = [p.code for p in d.ordre_presentation(props, 43, "p1", "a1")]
    assert o3 != o1, "another seed changes the order"
    # The decision does not depend on the order: a decision-maker naming a code picks the same plan.
    dec = DecideurFixe(code=K("D"))
    r1 = _run(d.decider(_person(), _ctx(graine_ordre=42), props, dec))
    r2 = _run(d.decider(_person(), _ctx(graine_ordre=43), props, dec))
    assert r1.retenue.code == r2.retenue.code == K("D")
    assert [x["code"] for x in r1.trace["presentees"]] != [x["code"] for x in r2.trace["presentees"]]


# ── D10: unusable response → uniform fallback, never a decision-maker decision ─────
def test_D10_repli_uniforme():
    class Muet(DecideurFixe):
        async def choisir(self, person, ctx, presentees):
            self.appels += 1
            return d.ReponseDecideur(index=None)          # empty vector, no network error
    res = _run(d.decider(_person(), _ctx(), _props(*STANDARD), Muet()))
    assert res.methode == "repli_uniforme" and res.retenue is not None and res.est_decision
    assert res.trace["methode"] == "repli_uniforme"


def test_erreur_du_decideur_n_est_pas_une_decision():
    res = _run(d.decider(_person(), _ctx(), _props(*STANDARD), DecideurFixe(erreur="quota")))
    assert res.methode == "erreur" and not res.est_decision and res.retenue is None
    assert res.trace["erreur"] == "quota"


# ── cap: the options dropped by the cap are justified ───────────────────────────────
def test_plafond_motive():
    props = _props(("W", "foot"), *[(f"T{i}", "foot", "bus", "foot") for i in range(8)])
    retenues, ecartees = d.plafonner(props, 6)
    assert len(retenues) == 6 and len(ecartees) == 3 and {e.motif for e in ecartees} == {"plafond"}
    assert K("W") in {p.code for p in retenues}, "the fastest of each group is kept first"


# ── D11: pure with respect to the process time zone ───────────────────────────────────────
def test_D11_independante_du_fuseau():
    def _trace():
        return _run(d.decider(_person(), _ctx(), _props(*STANDARD), DecideurFixe(code=K("T")))).trace
    old = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "UTC"; time.tzset(); t_utc = _trace()
        os.environ["TZ"] = "Europe/Paris"; time.tzset(); t_paris = _trace()
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()
    assert t_utc == t_paris
