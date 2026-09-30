"""Tests of the chain consistency of personal vehicles (bike AND car).

A vehicle is a **place**: it stays parked where the agent left it. Three rules,
identical for the bike and the car:

1. **exit lock** — the mode is offered only if the vehicle is at the departure point;
2. **parking** — after the choice, the vehicle follows the agent if used, otherwise it stays;
3. **return lock** — a trip home departing from a place where a vehicle
   is parked is restricted to that mode (the agent brings its vehicle back).

Unlike `urban_mobility_agents/agents/tests/test_personal_bike.py` (which
copies the logic for lack of sys.path), these tests import the real functions of the
controller — the conftest of this folder puts the repo and `services/llm-agents/` on the path.

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_vehicle_chain.py
"""

import pytest

from models import (
    Activity,
    Location,
    Person,
    PersonalIdentity,
    PersonState,
    Transit,
    TransitLocation,
    TravelPlan,
)
from urban_mobility_agents.simulation_controller import (
    RETURN_LOCK_MIN_DISTANCE_KM,
    _can_drive,
    _is_car_passenger,
    _orphaned_vehicles,
    _owns_bike,
    _owns_car,
    _park_vehicles,
    _primary_mode,
    _road_distance_km,
    _same_place,
    _vehicle_available,
    _vehicle_position,
    _vehicles_parked_at,
)

HOME = Location(lat=43.6000, lon=1.4400)
WORK = Location(lat=43.6100, lon=1.4500)
GYM = Location(lat=43.6200, lon=1.4600)
# ~250 m from home as the crow flies: below the return-lock threshold (A3), even
# after the 1.3 detour factor.
SHOP = Location(lat=43.6020, lon=1.4415)


def _plan(*modes: str, start: Location = HOME, end: Location = WORK) -> TravelPlan:
    """Plan with one leg per mode; the `foot` legs of a PT plan are transfers."""
    loc = TransitLocation(stop="", lat=start.lat, lon=start.lon)
    legs = [
        Transit(
            start_time=0, end_time=600, duration=600, distance=1000.0, mode=mode,
            start_location=loc, end_location=loc,
            is_transfer=(mode == "foot" and len(modes) > 1),
        )
        for mode in modes
    ]
    return TravelPlan(
        id="p", start_location=start, end_location=end,
        start_time=0, end_time=600, legs=legs,
    )


def _person(**traits) -> Person:
    """Agent at home, vehicles parked at home (empty state dict).

    By default: **adult holding a licence, living alone**. Since the passenger
    mode, driving requires age and licence — a persona
    silent on these two traits would never again be offered the car, and the
    tests of the three rules would bear on an agent unable to drive. Living
    alone, it is not a passenger either: the passenger cases are explicit.
    """
    base = {"personal_bike": "vélo normal", "number_of_cars": 1,
            "has_driving_license": True, "age": 40, "household_size": 1}
    base.update(traits)
    return Person(
        person_id="p1",
        identity=PersonalIdentity(name="Test", traits_json=base, home=HOME),
        state=PersonState(),
    )


# ── Ownership ────────────────────────────────────────────────────────────────

class TestPossession:
    def test_pas_de_velo(self):
        assert _owns_bike({"personal_bike": "Pas de vélo"}) is False

    def test_casse_indifferente(self):
        assert _owns_bike({"personal_bike": "PAS DE VÉLO"}) is False

    def test_velo_normal(self):
        assert _owns_bike({"personal_bike": "vélo normal"}) is True

    def test_vae(self):
        assert _owns_bike({"personal_bike": "VAE"}) is True

    def test_velo_champ_absent_defaut_false_et_alarme(self, monkeypatch):
        """Missing field ⇒ NO bike, and the alarm fires.

        The default used to be the opposite, for backward compatibility: a population without
        `personal_bike` thus put 100 % of the agents on a bike **silently**, on the mode
        whose modal share is the most scrutinised in the project. The fallback now
        deprives the agent of a mode rather than offering one it does not have, and it is
        loud."""
        import urban_mobility_agents.vehicle_chain as controller  # the alarm flag lives in vehicle_chain

        fired = []
        monkeypatch.setattr(controller, "fire_alarme", fired.append)
        monkeypatch.setattr(controller, "_bike_trait_alarm_on", False)
        errors = []
        monkeypatch.setattr(controller.logger, "error",
                            lambda message, *a, **k: errors.append(message))

        assert _owns_bike({}) is False
        assert fired == ["personal_bike_absent"]
        assert any("[ALARME]" in message and "personal_bike" in message
                   for message in errors)

    def test_lalarme_champ_absent_ne_sonne_quune_fois(self, monkeypatch):
        """Otherwise it is emitted at every decision of every agent and drowns `make error`.
        The Prometheus counter, for its part, does count all the cases."""
        import urban_mobility_agents.vehicle_chain as controller  # the alarm flag lives in vehicle_chain

        fired = []
        monkeypatch.setattr(controller, "fire_alarme", fired.append)
        monkeypatch.setattr(controller, "_bike_trait_alarm_on", False)
        errors = []
        monkeypatch.setattr(controller.logger, "error",
                            lambda message, *a, **k: errors.append(message))

        for _ in range(5):
            assert _owns_bike({}) is False
        assert len(errors) == 1
        assert len(fired) == 5

    def test_un_velo_declare_ne_declenche_aucune_alarme(self, monkeypatch):
        import urban_mobility_agents.vehicle_chain as controller  # the alarm flag lives in vehicle_chain

        fired = []
        monkeypatch.setattr(controller, "fire_alarme", fired.append)
        monkeypatch.setattr(controller, "_bike_trait_alarm_on", False)
        assert _owns_bike({"personal_bike": "vélo normal"}) is True
        assert _owns_bike({"personal_bike": "Pas de vélo"}) is False
        assert fired == []

    def test_voiture_selon_number_of_cars(self):
        assert _owns_car({"number_of_cars": 1}) is True
        assert _owns_car({"number_of_cars": 2}) is True
        assert _owns_car({"number_of_cars": 0}) is False

    def test_voiture_champ_absent_defaut_false(self):
        """Opposite of the bike: without data, no car (it is the reference eqasim field)."""
        assert _owns_car({}) is False

    def test_voiture_champ_null(self):
        assert _owns_car({"number_of_cars": None}) is False


# ── Initial position: missing key ⇒ home ──────────────────────────────────────

class TestPositionInitiale:
    def test_etat_par_defaut_vide(self):
        assert PersonState().planning_vehicle_at == {}

    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_vehicule_au_domicile_par_defaut(self, mode):
        assert _vehicle_position(_person(), mode) == HOME

    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_position_memorisee_prime(self, mode):
        person = _person()
        person.state.planning_vehicle_at[mode] = WORK
        assert _vehicle_position(person, mode) == WORK


# ── Rule 1: exit lock ─────────────────────────────────────────────────────────

class TestVerrouDeSortie:
    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_disponible_au_point_de_depart(self, mode):
        assert _vehicle_available(_person(), mode, HOME) is True

    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_indisponible_gare_ailleurs(self, mode):
        """The core of the fix: ownership without presence ⇒ no option."""
        person = _person()
        person.state.planning_vehicle_at[mode] = WORK
        assert _vehicle_available(person, mode, HOME) is False
        assert _vehicle_available(person, mode, WORK) is True

    def test_velo_non_possede_meme_au_domicile(self):
        assert _vehicle_available(_person(personal_bike="Pas de vélo"), "bike", HOME) is False

    def test_voiture_non_possedee_meme_au_domicile(self):
        assert _vehicle_available(_person(number_of_cars=0), "car", HOME) is False

    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_domicile_inconnu_degrade_vers_possession(self, mode):
        """Without a home, we do not know where the vehicle is: former behaviour."""
        person = _person()
        person.identity.home = None
        assert _vehicle_available(person, mode, WORK) is True

    @pytest.mark.parametrize("mode", ["bike", "car"])
    def test_origine_inconnue_bloque(self, mode):
        assert _vehicle_available(_person(), mode, None) is False


# ── Rule 2: parking after the trip ────────────────────────────────────────────

class TestStationnement:
    def test_trajet_a_velo_le_velo_suit(self):
        person = _person()
        _park_vehicles(person, _plan("bicycle"), HOME, WORK)
        assert _vehicle_position(person, "bike") == WORK
        assert _vehicle_position(person, "car") == HOME  # the car has not moved

    def test_trajet_en_voiture_la_voiture_suit(self):
        person = _person()
        _park_vehicles(person, _plan("car"), HOME, WORK)
        assert _vehicle_position(person, "car") == WORK
        assert _vehicle_position(person, "bike") == HOME

    def test_trajet_en_bus_rien_ne_bouge(self):
        person = _person()
        _park_vehicles(person, _plan("foot", "bus", "foot"), HOME, WORK)
        assert person.state.planning_vehicle_at == {}

    def test_retour_au_domicile_libere_la_cle(self):
        """Invariant "missing key ⇒ home": the return purges the entry."""
        person = _person()
        _park_vehicles(person, _plan("car"), HOME, WORK)
        _park_vehicles(person, _plan("car", start=WORK, end=HOME), WORK, HOME)
        assert person.state.planning_vehicle_at == {}
        assert _vehicle_position(person, "car") == HOME

    def test_plus_de_retour_implicite_au_domicile(self):
        """Regression: going home by bus no longer makes the bike reappear at home."""
        person = _person()
        _park_vehicles(person, _plan("bicycle"), HOME, WORK)
        _park_vehicles(person, _plan("foot", "bus", "foot", start=WORK, end=HOME), WORK, HOME)
        assert _vehicle_position(person, "bike") == WORK
        assert _vehicle_available(person, "bike", HOME) is False

    def test_plan_absent_rien_ne_bouge(self):
        person = _person()
        _park_vehicles(person, None, HOME, WORK)
        assert person.state.planning_vehicle_at == {}

    def test_plan_vide_compte_comme_marche(self):
        person = _person()
        assert _primary_mode(_plan()) == "walk"
        _park_vehicles(person, _plan(), HOME, WORK)
        assert person.state.planning_vehicle_at == {}

    def test_vehicule_absent_du_depart_ne_se_teleporte_pas(self):
        """Defence in depth: a car plan departing from a place without a car."""
        person = _person()
        person.state.planning_vehicle_at["car"] = GYM
        _park_vehicles(person, _plan("car", start=HOME, end=WORK), HOME, WORK)
        assert _vehicle_position(person, "car") == GYM


# ── Rule 3: return lock ───────────────────────────────────────────────────────

class TestVerrouDeRetour:
    def test_vehicule_gare_ici_doit_etre_ramene(self):
        person = _person()
        person.state.planning_vehicle_at["car"] = WORK
        assert _vehicles_parked_at(person, WORK) == {"car"}

    def test_les_deux_vehicules_gares_ici(self):
        person = _person()
        person.state.planning_vehicle_at["car"] = WORK
        person.state.planning_vehicle_at["bike"] = WORK
        assert _vehicles_parked_at(person, WORK) == {"bike", "car"}

    def test_vehicule_gare_ailleurs_non_concerne(self):
        person = _person()
        person.state.planning_vehicle_at["car"] = GYM
        assert _vehicles_parked_at(person, WORK) == set()

    def test_depart_du_domicile_rien_a_ramener(self):
        """Vehicles at home have nothing to bring back: no lock on departure."""
        assert _vehicles_parked_at(_person(), HOME) == set()

    def test_non_possede_jamais_a_ramener(self):
        person = _person(personal_bike="Pas de vélo", number_of_cars=0)
        person.state.planning_vehicle_at["bike"] = WORK
        assert _vehicles_parked_at(person, WORK) == set()


# ── Orphans: vehicle left at an intermediate stop ─────────────────────────────

class TestOrphelins:
    def test_aucun_orphelin_au_depart(self):
        assert _orphaned_vehicles(_person()) == set()

    def test_voiture_restee_au_travail(self):
        person = _person()
        person.state.planning_vehicle_at["car"] = WORK
        assert _orphaned_vehicles(person) == {"car"}

    def test_non_possede_jamais_orphelin(self):
        person = _person(number_of_cars=0)
        person.state.planning_vehicle_at["car"] = WORK
        assert _orphaned_vehicles(person) == set()


# ── Position comparison ───────────────────────────────────────────────────────

class TestSamePlace:
    def test_identique(self):
        assert _same_place(HOME, Location(lat=43.6, lon=1.44)) is True

    def test_arrondi_de_serialisation_tolere(self):
        assert _same_place(HOME, Location(lat=43.6000001, lon=1.4400001)) is True

    def test_lieux_distincts(self):
        assert _same_place(HOME, WORK) is False

    def test_none(self):
        assert _same_place(None, HOME) is False
        assert _same_place(HOME, None) is False


# ── Complete day chains ───────────────────────────────────────────────────────

class TestChaineJournee:
    def test_bus_aller_puis_plus_de_vehicule_au_bureau(self):
        """home → work by bus: neither bike nor car available at the office."""
        person = _person()
        _park_vehicles(person, _plan("foot", "bus", "foot"), HOME, WORK)
        assert _vehicle_available(person, "bike", WORK) is False
        assert _vehicle_available(person, "car", WORK) is False

    def test_voiture_aller_voiture_disponible_toute_la_chaine(self):
        person = _person()
        _park_vehicles(person, _plan("car"), HOME, WORK)
        assert _vehicle_available(person, "car", WORK) is True
        _park_vehicles(person, _plan("car", start=WORK, end=GYM), WORK, GYM)
        assert _vehicle_available(person, "car", GYM) is True
        # …but the bike stayed at home all along.
        assert _vehicle_available(person, "bike", GYM) is False

    def test_voiture_au_travail_velo_au_domicile_sont_exclusifs(self):
        """The agent cannot drive from the office AND pedal from home."""
        person = _person()
        _park_vehicles(person, _plan("car"), HOME, WORK)
        assert _vehicle_available(person, "car", HOME) is False
        assert _vehicle_available(person, "bike", WORK) is False

    def test_boucle_complete_remet_tout_au_domicile(self):
        person = _person()
        _park_vehicles(person, _plan("bicycle"), HOME, WORK)
        assert _vehicles_parked_at(person, WORK) == {"bike"}  # return lock active
        _park_vehicles(person, _plan("bicycle", start=WORK, end=HOME), WORK, HOME)
        assert _orphaned_vehicles(person) == set()
        assert _vehicle_available(person, "bike", HOME) is True
        assert _vehicle_available(person, "car", HOME) is True

    def test_etape_intermediaire_orpheline_la_voiture(self):
        """home → work by car, work → gym on foot, gym → home by bus."""
        person = _person()
        _park_vehicles(person, _plan("car"), HOME, WORK)
        _park_vehicles(person, _plan("foot", start=WORK, end=GYM), WORK, GYM)
        assert _vehicles_parked_at(person, GYM) == set()  # no lock: nothing parked here
        _park_vehicles(person, _plan("foot", "bus", "foot", start=GYM, end=HOME), GYM, HOME)
        assert _orphaned_vehicles(person) == {"car"}  # residual case, caught up at home


# ── Passenger mode: the child goes to school by car ──────────

def _child(**traits) -> Person:
    """12-year-old child, household of 4 persons with 3 cars — typical passenger."""
    base = {"age": 12, "has_driving_license": False, "household_size": 4,
            "number_of_cars": 3}
    base.update(traits)
    return _person(**base)


class TestPeutConduire:
    def test_adulte_avec_permis(self):
        assert _can_drive({"age": 40, "has_driving_license": True}) is True

    def test_mineur_meme_avec_permis(self):
        """Hard lock: a badly generated population hands licences to
        nine-year-old children. Age decides."""
        assert _can_drive({"age": 12, "has_driving_license": True}) is False

    def test_adulte_sans_permis(self):
        assert _can_drive({"age": 40, "has_driving_license": False}) is False

    def test_traits_muets(self):
        assert _can_drive({}) is False


class TestPassager:
    def test_enfant_du_foyer_motorise(self):
        assert _is_car_passenger(_child()) is True

    def test_enfant_sans_voiture_au_foyer(self):
        assert _is_car_passenger(_child(number_of_cars=0)) is False

    def test_adulte_sans_permis_vivant_seul(self):
        """Nobody to drive them: this is not a passenger."""
        p = _person(age=40, has_driving_license=False, household_size=1)
        assert _is_car_passenger(p) is False

    def test_adulte_sans_permis_en_famille(self):
        p = _person(age=40, has_driving_license=False, household_size=3)
        assert _is_car_passenger(p) is True

    def test_conducteur_n_est_pas_passager(self):
        assert _is_car_passenger(_person(household_size=4)) is False

    def test_voiture_proposee_sans_test_de_position(self):
        """It is not their car: it does not matter where the household left it."""
        child = _child()
        child.state.planning_vehicle_at["car"] = GYM
        assert _vehicle_available(child, "car", HOME) is True
        assert _vehicle_available(child, "car", WORK) is True

    def test_non_conducteur_non_passager_jamais_de_voiture(self):
        """The hard lock: no minor or unlicensed person drives any more."""
        seul = _person(age=40, has_driving_license=False, household_size=1)
        assert _vehicle_available(seul, "car", HOME) is False
        enfant_sans_voiture = _child(number_of_cars=0)
        assert _vehicle_available(enfant_sans_voiture, "car", HOME) is False

    def test_le_velo_reste_ouvert_aux_mineurs(self):
        """The licence only conditions the car."""
        assert _vehicle_available(_child(), "bike", HOME) is True

    def test_la_voiture_ne_se_gare_pas_a_destination(self):
        """A third party drives and leaves: the car does not sleep at school."""
        child = _child()
        _park_vehicles(child, _plan("car", start=HOME, end=WORK), HOME, WORK)
        assert child.state.planning_vehicle_at == {}
        assert _vehicle_position(child, "car") == HOME

    def test_pas_de_retour_force_pour_le_passager(self):
        """The easiest point to break: the child dropped at school must
        not be told to bring the car back. Nothing being parked there, there is nothing
        to bring back — the direct consequence of the previous test."""
        child = _child()
        _park_vehicles(child, _plan("car", start=HOME, end=WORK), HOME, WORK)
        assert _vehicles_parked_at(child, WORK) == set()
        assert _orphaned_vehicles(child) == set()

    def test_le_conducteur_ne_change_pas_de_comportement(self):
        """Non-regression: for an adult with a licence, the three rules are
        strictly those from before the passenger mode."""
        adulte = _person(household_size=4, number_of_cars=3)
        assert _vehicle_available(adulte, "car", HOME) is True
        _park_vehicles(adulte, _plan("car"), HOME, WORK)
        assert _vehicle_position(adulte, "car") == WORK
        assert _vehicle_available(adulte, "car", HOME) is False
        assert _vehicles_parked_at(adulte, WORK) == {"car"}

    def test_velo_du_passager_suit_normalement(self):
        """Only the car is concerned: their bike stays where they leave it."""
        child = _child()
        _park_vehicles(child, _plan("bicycle"), HOME, WORK)
        assert _vehicle_position(child, "bike") == WORK
        assert _vehicles_parked_at(child, WORK) == {"bike"}


# ── Distance threshold of the return lock ────────────────────

class TestSeuilRetourCourt:
    def test_distance_routiere_facteur_de_detour(self):
        """Crow-flies distance × 1.3, the convention of `_estimate_fallback_duration`."""
        assert _road_distance_km(HOME, HOME) == 0.0
        assert _road_distance_km(HOME, None) is None
        assert 0.2 < _road_distance_km(HOME, SHOP) < RETURN_LOCK_MIN_DISTANCE_KM

    def test_trajet_long_au_dessus_du_seuil(self):
        assert _road_distance_km(HOME, WORK) > RETURN_LOCK_MIN_DISTANCE_KM

    def test_le_verrou_reste_pertinent_au_dela_du_seuil(self):
        """The threshold does not disarm rule 3: beyond it, the vehicle must be brought back."""
        person = _person()
        person.state.planning_vehicle_at["car"] = WORK
        assert _vehicles_parked_at(person, WORK) == {"car"}
        assert _road_distance_km(WORK, HOME) > RETURN_LOCK_MIN_DISTANCE_KM


# ── Catch-up at home (controller method) ──────────────────────────────────────

class TestSettleVehiclesAtHome:
    def _fresh(self):
        """Minimal instance: only _settle_vehicles_at_home is called, without I/O."""
        from urban_mobility_agents.simulation_controller import SimulationLoopV1
        ctrl = SimulationLoopV1.__new__(SimulationLoopV1)
        ctrl._vehicle_home_returns = 0
        ctrl._vehicle_orphan_returns = 0
        ctrl._vehicle_orphan_alarm_on = False
        return ctrl

    @staticmethod
    def _act(purpose: str) -> Activity:
        return Activity(id="a", start_time=0, end_time=3600, purpose=purpose, location=HOME)

    def test_activite_non_domicile_ignoree(self):
        ctrl, person = self._fresh(), _person()
        person.state.planning_vehicle_at["car"] = WORK
        ctrl._settle_vehicles_at_home(person, self._act("work"))
        assert ctrl._vehicle_home_returns == 0
        assert _vehicle_position(person, "car") == WORK

    def test_retour_sans_orphelin(self):
        ctrl, person = self._fresh(), _person()
        ctrl._settle_vehicles_at_home(person, self._act("home"))
        assert (ctrl._vehicle_home_returns, ctrl._vehicle_orphan_returns) == (1, 0)

    def test_orphelin_ramene_au_domicile(self):
        ctrl, person = self._fresh(), _person()
        person.state.planning_vehicle_at["car"] = WORK
        ctrl._settle_vehicles_at_home(person, self._act("home"))
        assert (ctrl._vehicle_home_returns, ctrl._vehicle_orphan_returns) == (1, 1)
        assert person.state.planning_vehicle_at == {}
        assert _vehicle_available(person, "car", HOME) is True

    def test_purpose_casse_indifferente(self):
        ctrl, person = self._fresh(), _person()
        ctrl._settle_vehicles_at_home(person, self._act("HOME"))
        assert ctrl._vehicle_home_returns == 1
