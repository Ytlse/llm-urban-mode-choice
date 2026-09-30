"""The main mode follows the survey, and the vehicle mode stays separate.

Two invariants, whose violation raises no exception:

1. **The log follows the survey hierarchy.** `move_logger._plan_transport_mode`
   tested the car FIRST, whereas it is at rank 19 of the published annex
   (AUAT/CEREMA report p. 53), below ranks 1 to 13 of public transport. The survey codes 760
   of its 770 mixed car + transit trips as "transports collectifs".

2. **A main mode is not a vehicle mode.** The vehicle
   chain asks "where is the car", not "what is the main mode". Confusing
   the two makes the car of a park-and-ride trip vanish from the return lock.
   `_primary_mode` and `_vehicle_mode` must therefore DIVERGE on a mixed plan, and that is
   what this file pins down.

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_hierarchie_modes.py
"""

import pytest

from models import Location, TransitLocation, Transit, TravelPlan
from urban_mobility_agents.simulation_controller import _primary_mode, _vehicle_mode
from urban_mobility_agents.utils.move_logger import (_BUS_MODES, _CANONICAL_FR,
                                                     _CAR_MODES, _RAIL_MODES,
                                                     _plan_transport_mode)


def _jambe(mode: str, is_transfer: bool = False) -> Transit:
    depart = TransitLocation(stop=f"{mode}-A", lat=43.5, lon=1.4)
    arrivee = TransitLocation(stop=f"{mode}-B", lat=43.6, lon=1.45)
    return Transit(
        start_time=0, end_time=600_000, start_location=depart, end_location=arrivee,
        is_transfer=is_transfer, transit_route="X", shape_id=None, duration=600,
        distance=5_000.0, mode=mode)


def _plan(*modes: str) -> TravelPlan:
    jambes = [_jambe(m) for m in modes]
    return TravelPlan(id="-".join(modes) or "vide", start_location=Location(lat=43.5, lon=1.4),
                      end_location=Location(lat=43.6, lon=1.45), start_time=0,
                      end_time=600_000, legs=jambes)


# ── 1. The log follows the hierarchy ─────────────────────────────────────────

@pytest.mark.parametrize("modes,attendu", [
    (("bus",), "Transports_collectifs"),
    (("rail",), "Train"),
    (("car",), "Voiture Privée"),
    (("bicycle",), "Vélo"),
    (("foot",), "Marche"),
    (("school_bus",), "Transports_collectifs"),
    # The two notches that matter, and the direction in which they decide.
    (("bus", "rail"), "Transports_collectifs"),     # bus rank 4 < TER rank 8
    (("metro", "rail"), "Transports_collectifs"),   # metro rank 1
    (("car", "bus"), "Transports_collectifs"),      # car rank 19, below all public transport
    (("car", "rail"), "Train"),
    (("car", "bicycle"), "Voiture Privée"),         # car rank 19 < bike rank 23
    (("bicycle", "bus"), "Transports_collectifs"),
])
def test_le_libelle_du_journal_suit_l_ordre_de_l_enquete(modes, attendu):
    assert _plan_transport_mode(_plan(*modes)) == attendu


def test_les_jambes_de_transfert_ne_comptent_pas():
    """A walking transfer is not a walking trip (terminal legs, T13)."""
    plan = TravelPlan(id="transferts", start_location=Location(lat=43.5, lon=1.4),
                      end_location=Location(lat=43.6, lon=1.45), start_time=0,
                      end_time=600_000,
                      legs=[_jambe("foot", is_transfer=True), _jambe("bus"),
                            _jambe("foot", is_transfer=True)])
    assert _plan_transport_mode(plan) == "Transports_collectifs"


def test_un_plan_sans_jambe_est_de_la_marche():
    """Rank 36 of the annex: "Marche à pied UNIQUEMENT", and it is measured."""
    assert _plan_transport_mode(_plan()) == "Marche"
    assert _primary_mode(_plan()) == "walk"


def test_un_mode_inconnu_leve_une_alarme_sur_front_montant(caplog):
    """A mode outside the hierarchy falls into "Autres modes", which is EXCLUDED from scoring.

    Without an alarm, its mass vanishes from a modal share without breaking anything — the exact
    mechanism of the Téléo and TER defects. A single line per set of modes.
    """
    import logging

    from urban_mobility_agents.utils import move_logger

    move_logger._UNKNOWN_MODES_SEEN.clear()
    with caplog.at_level(logging.ERROR, logger=move_logger.__name__):
        assert _plan_transport_mode(_plan("hovercraft")) == "Autres modes"
        assert _plan_transport_mode(_plan("hovercraft")) == "Autres modes"
    alarmes = [r for r in caplog.records if "[ALARME]" in r.getMessage()]
    assert len(alarmes) == 1, [r.getMessage() for r in alarmes]
    assert "hovercraft" in alarmes[0].getMessage()
    move_logger._UNKNOWN_MODES_SEEN.clear()


def test_les_listes_du_journal_ne_sont_plus_ecrites_a_la_main():
    """They are VIEWS of the frozen hierarchy: five literals can no longer drift."""
    assert {"bus", "metro", "tram", "cableway", "school_bus"} <= _BUS_MODES
    assert _RAIL_MODES == frozenset({"rail"})
    assert _CAR_MODES == frozenset({"car", "__car__"})
    # The P(...) columns keep their display order (archived `moves.csv` files
    # must remain comparable), but their labels come from the hierarchy.
    assert list(_CANONICAL_FR) == ["walking", "cycling", "car", "public_transport",
                                   "train", "motorbike", "other"]
    assert _CANONICAL_FR["train"] == "Train"
    assert _CANONICAL_FR["public_transport"] == "Transports_collectifs"


# ── 2. Main mode ≠ vehicle mode ──────────────────────────────────────────────

def test_la_metrique_agrege_dans_les_quatre_categories_de_l_enquete():
    """`trip_mode_by_purpose_total` counts in walk / bike / car / transit.

    Train is INSIDE public transport: annex p. 53 files ranks 1 to 13
    under that label. Merging `rail` into `transit` is therefore correct here — it is not
    a forgotten mode, it is the published aggregation.
    """
    assert _primary_mode(_plan("rail")) == "transit"
    assert _primary_mode(_plan("bus", "rail")) == "transit"
    assert _primary_mode(_plan("car")) == "car"
    assert _primary_mode(_plan("bicycle")) == "bike"
    assert _primary_mode(_plan("foot")) == "walk"


def test_un_mode_inconnu_nest_plus_compte_en_transports_collectifs():
    """`transit` was the cascade DEFAULT: an unknown mode inflated the transit share."""
    assert _primary_mode(_plan("hovercraft")) == "other"


def test_le_mode_de_vehicule_repond_a_une_autre_question():
    """On a mixed car + bus plan, the two readings diverge — and that is intended.

    The survey classifies this trip as public transport (park-and-ride), so
    `_primary_mode` says "transit". But the car was indeed taken and must be parked
    at the destination, so `_vehicle_mode` says "car". Confusing the two would make the
    return lock lose the car.
    """
    mixte = _plan("car", "bus")
    assert _primary_mode(mixte) == "transit"
    assert _vehicle_mode(mixte) == "car"
    # On non-mixed plans — the only ones OTP produces today — they agree.
    for mode, attendu in (("car", "car"), ("bicycle", "bike"), ("foot", "walk")):
        assert _vehicle_mode(_plan(mode)) == attendu
        assert _primary_mode(_plan(mode)) == attendu
    assert _vehicle_mode(_plan("bus")) == "transit"
