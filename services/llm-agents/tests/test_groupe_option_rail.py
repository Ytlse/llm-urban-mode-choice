"""Rail forms its own option group, without pushing out the car.

Measured on 2026-09-04 on the 2,580 points of the sealed v4 population: of the **440 points where
a direct rail itinerary exists**, 122 (27.7 %) lost it to a faster bus + train,
both sharing the "transit" group. The agent therefore never saw the train
as a choice. Decision of the repo author: rail becomes a separate group, and the
option cap stays at 6.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import Location, TransitLocation, Transit, TravelPlan  # noqa: E402
from urban_mobility_agents.simulation_controller import (  # noqa: E402
    GROUPE_RAIL, _select_candidates, _selection_group)

_L = Location(lat=43.60, lon=1.44)
_T = TransitLocation(lat=43.60, lon=1.44, stop="s")


def _plan(nom: str, duree: int, modes: list[str]) -> TravelPlan:
    return TravelPlan(
        id=nom, start_location=_L, end_location=_L, start_time=0, end_time=duree,
        duration=duree,
        legs=[Transit(start_time=0, end_time=duree, start_location=_T, end_location=_T,
                      mode=m) for m in modes],
    )


def test_le_train_direct_a_son_propre_groupe():
    assert _selection_group(_plan("t", 100, ["foot", "rail", "foot"])) == GROUPE_RAIL


def test_un_bus_plus_train_reste_dans_le_groupe_du_collectif():
    """The survey hierarchy puts the bus (rank 4) above the train (rank 8): a
    mixed trip is a bus trip, and it must not take the train's slot."""
    groupe = _selection_group(_plan("bt", 90, ["foot", "bus", "rail", "foot"]))
    assert groupe == "transit"
    assert groupe != GROUPE_RAIL


def test_les_autres_categories_gardent_leur_groupe():
    assert _selection_group(_plan("v", 100, ["car"])) == "car"
    assert _selection_group(_plan("b", 100, ["bicycle"])) == "bike"
    assert _selection_group(_plan("m", 100, ["foot"])) == "walk"
    assert _selection_group(_plan("bus", 100, ["foot", "bus", "foot"])) == "transit"


def test_le_train_direct_n_est_plus_ecarte_par_un_bus_plus_rapide():
    """The case of the 122 points: a bus + train faster than a direct train. Before, the two
    shared the group and only the faster one went through first."""
    bus_train = _plan("bus+train", 80, ["foot", "bus", "rail", "foot"])
    train_direct = _plan("train", 95, ["foot", "rail", "foot"])
    voiture = _plan("voiture", 60, ["car"])

    retenus = _select_candidates([bus_train, train_direct, voiture], 6)

    assert train_direct in retenus, "the direct train must be offered"
    assert {p.id for p in retenus} == {"bus+train", "train", "voiture"}


def test_les_cinq_groupes_tiennent_dans_le_plafond_de_six():
    """The cap stays at 6: the five groups must fit, otherwise splitting off rail would have
    pushed out the car or walking — this is why metro and tram are NOT split off too."""
    plans = [
        _plan("collectif", 10, ["foot", "bus", "foot"]),
        _plan("metro", 20, ["foot", "metro", "foot"]),      # same group as public transit
        _plan("train", 30, ["foot", "rail", "foot"]),
        _plan("velo", 40, ["bicycle"]),
        _plan("voiture", 50, ["car"]),
        _plan("marche", 60, ["foot"]),
    ]
    retenus = _select_candidates(plans, 6)
    groupes = {_selection_group(p) for p in retenus}

    assert groupes == {"transit", GROUPE_RAIL, "bike", "car", "walk"}
    for attendu in ("train", "velo", "voiture", "marche"):
        assert attendu in {p.id for p in retenus}, f"{attendu} pushed out by the cap"


def test_le_plafond_reste_opposable():
    """Splitting a group must not make the cap be exceeded."""
    plans = [_plan(f"t{i}", 10 + i, ["foot", "bus", "foot"]) for i in range(10)]
    plans.append(_plan("train", 5, ["foot", "rail", "foot"]))
    retenus = _select_candidates(plans, 6)
    assert len(retenus) == 6
    assert "train" in {p.id for p in retenus}
