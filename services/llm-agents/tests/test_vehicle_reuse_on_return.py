"""A vehicle left somewhere can be taken again when the agent comes back to it.

Question asked on 2026-09-11: *"and if the agent goes back to the vehicle's position, can it
use it again?"* The answer is **yes**, and this file pins it down, because no test
held it.

The rule is **positional, not a single-use lock**: a vehicle mode can be offered
if and only if the vehicle is parked **where the trip starts**. Nothing is consumed, nothing
is marked "already used". An agent who goes to the office by car, leaves it another way, then
comes back to the office, finds its car again.

Without this behaviour, the chain constraint would become a permanent ban: the
first non-vehicle trip would deprive the agent of its vehicle for the whole day. That would be
as wrong as the defect it fixes — the "ghost bike" that followed the agent everywhere.

**The case really occurs.** In the v5 cohort, 177 of the 894 mobile persons come back at
least once to a place other than home during their day, for 190 returns in total.

**What the rule does NOT do**, and it is deliberate: it compares points, not
neighbourhoods. The tolerance is 10⁻⁶ degree, about 11 cm — enough to absorb serialisation
rounding, nothing more. Two activities 200 m apart are two places, and the car
of one is of no use to the other. In v5, 746 pairs of activities of the same person are
less than 500 m apart without being at the same point; only 3 are less than 50 m apart. Access
walking is not modelled, and this test declares it rather than leaving it to be discovered.
"""

from __future__ import annotations

import pytest

from models import Location
from urban_mobility_agents.vehicle_chain import (
    MOTIF_VEHICULE_AILLEURS,
    _park_vehicles,
    _vehicle_unavailable_reason,
)

HOME = {"lon": 1.44, "lat": 43.60, "public_transport": True, "zone": "centre"}
BUREAU = {"lon": 1.45, "lat": 43.61, "public_transport": True, "zone": "nord"}
SPORT = {"lon": 1.46, "lat": 43.62, "public_transport": False, "zone": "est"}


def _lieu(d: dict) -> Location:
    return Location(lon=d["lon"], lat=d["lat"], public_transport=d["public_transport"], zone=d["zone"])


def _acte(i: int, purpose: str, loc: dict) -> dict:
    return {
        "id": str(i),
        "scheduled_start_time": 3600.0 * (i + 7),
        "start_time": 3600.0 * (i + 7),
        "end_time": 3600.0 * (i + 8),
        "purpose": purpose,
        "location": loc,
    }


@pytest.fixture
def agent():
    """A motorised person, with a licence, whose day goes back through the office."""
    from inputs.population.eqasim_loader import EqasimJSONPopulationLoader

    traits = {
        "age": 35,
        "gender": "Female",
        "main_occupation": "actif",
        "household_size": 1,
        "number_of_cars": 1,
        "has_driving_license": True,
        "personal_bike": "vélo normal",
        "residence_zone": "Toulouse",
    }
    brut = [
        {
            "person_id": "p",
            "identity": {
                "traits_json": traits,
                "home": HOME,
                "activities": [
                    _acte(0, "home", HOME),
                    _acte(1, "work", BUREAU),
                    _acte(2, "leisure", SPORT),
                    _acte(3, "work", BUREAU),
                ],
            },
            "state": {"last_activity_index": 0},
            "is_llm_based": True,
        }
    ]
    return EqasimJSONPopulationLoader().load_population_from_data(brut, max_size=1, bbox=None)[0]


class _PlanVoiture:
    """The minimum `_park_vehicles` expects: a trip whose mode is the car."""

    legs = [type("Leg", (), {"mode": "car", "is_transfer": False})()]


def test_la_voiture_part_du_domicile(agent):
    assert _vehicle_unavailable_reason(agent, "car", _lieu(HOME)) is None


def test_la_voiture_reste_ou_on_la_gare(agent):
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    assert _vehicle_unavailable_reason(agent, "car", _lieu(BUREAU)) is None
    assert _vehicle_unavailable_reason(agent, "car", _lieu(HOME)) == MOTIF_VEHICULE_AILLEURS


def test_ailleurs_la_voiture_nest_pas_proposable(agent):
    """The defect the chain fixes: the ghost vehicle that followed the agent."""
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    assert _vehicle_unavailable_reason(agent, "car", _lieu(SPORT)) == MOTIF_VEHICULE_AILLEURS


def test_revenir_sur_le_lieu_de_stationnement_rend_la_voiture(agent):
    """THE point of the question: the constraint is not a single-use lock."""
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    assert _vehicle_unavailable_reason(agent, "car", _lieu(SPORT)) == MOTIF_VEHICULE_AILLEURS
    # The agent comes back to the office, by any mode: its car is still waiting there.
    assert _vehicle_unavailable_reason(agent, "car", _lieu(BUREAU)) is None


def test_la_reprise_ne_depend_pas_du_mode_par_lequel_on_revient(agent):
    """One can come back on foot, by bus or by bike: the POSITION decides."""
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    velo = type("Plan", (), {"legs": [type("Leg", (), {"mode": "bicycle", "is_transfer": False})()]})()
    _park_vehicles(agent, velo, _lieu(BUREAU), _lieu(SPORT))  # leave by bike
    _park_vehicles(agent, velo, _lieu(SPORT), _lieu(BUREAU))  # come back by bike
    assert _vehicle_unavailable_reason(agent, "car", _lieu(BUREAU)) is None


def test_deux_lieux_proches_restent_deux_lieux(agent):
    """The rule compares POINTS, not neighbourhoods: access walking does not exist.

    Declared rather than discovered. In the v5 cohort, 746 pairs of activities of the same
    person are less than 500 m apart without being at the same point.
    """
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    # ~200 m north of the office: another place, hence no car.
    voisin = Location(lon=BUREAU["lon"], lat=BUREAU["lat"] + 0.0018, public_transport=True, zone="nord")
    assert _vehicle_unavailable_reason(agent, "car", voisin) == MOTIF_VEHICULE_AILLEURS


def test_un_arrondi_de_serialisation_ne_perd_pas_le_vehicule(agent):
    """The 10⁻⁶ degree tolerance absorbs rounding, and nothing else (~11 cm)."""
    _park_vehicles(agent, _PlanVoiture(), _lieu(HOME), _lieu(BUREAU))
    arrondi = Location(
        lon=BUREAU["lon"] + 1e-9, lat=BUREAU["lat"] - 1e-9, public_transport=True, zone="nord"
    )
    assert _vehicle_unavailable_reason(agent, "car", arrondi) is None
