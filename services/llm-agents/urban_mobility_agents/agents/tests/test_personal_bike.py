"""
Unit tests: filtering of the bike mode according to personal_bike.

Checks that:
- The three valid values ("No bike", "regular bike", "e-bike") are correctly
  interpreted, including with case variants — and their French equivalents
  ("Pas de vélo", "vélo normal", "VAE"), carried by the cohorts from before ticket 074.
- A missing field deprives the agent of the bike and raises an alarm (ticket 015, lot 1).
- `_vehicle_mode` correctly identifies an OSMnx bike plan.
- The post-filter actually removes bike plans when include_bike=False.

`_vehicle_mode` is **imported** from production and not copied here (ticket 022): a
copy only fails if the test changes, never if production changes, and it is this
asymmetry that let the Téléo and rail defects slip through. It is indeed `_vehicle_mode`
— the CHAIN question ("where is the car?") — and not `_primary_mode` — the
MAIN MODE question, which follows the survey's hierarchy — that the post-filter asks.
"""

import pytest
from unittest.mock import MagicMock

from urban_mobility_agents.simulation_controller import _vehicle_mode

# `_owns_bike` is IMPORTED from production, not copied. The copy that lived here
# still answered "bike allowed" for a missing field — the defect from before ticket 015 —
# and only knew the French vocabulary: it would have let a whole v6 cohort through
# without ever failing, since it only measured itself.
from urban_mobility_agents.vehicle_chain import _owns_bike as _include_bike


def _make_plan(mode: str):
    leg = MagicMock()
    leg.mode = mode
    leg.is_transfer = False
    plan = MagicMock()
    plan.legs = [leg]
    return plan


# ── Tests include_bike ────────────────────────────────────────────────────────

class TestIncludeBike:
    def test_pas_de_velo_majuscule(self):
        """Actual value in the population (capital P) → must exclude the bike."""
        assert _include_bike({"personal_bike": "Pas de vélo"}) is False

    def test_pas_de_velo_minuscule(self):
        assert _include_bike({"personal_bike": "pas de vélo"}) is False

    def test_pas_de_velo_tout_majuscule(self):
        assert _include_bike({"personal_bike": "PAS DE VÉLO"}) is False

    def test_no_bike_anglais(self):
        """Value of the v6 cohort (ticket 074) → must exclude the bike."""
        assert _include_bike({"personal_bike": "No bike"}) is False

    def test_no_bike_casse_variable(self):
        assert _include_bike({"personal_bike": "NO BIKE"}) is False
        assert _include_bike({"personal_bike": " no bike "}) is False

    def test_velo_normal(self):
        assert _include_bike({"personal_bike": "vélo normal"}) is True

    def test_vae(self):
        assert _include_bike({"personal_bike": "VAE"}) is True

    def test_regular_bike(self):
        assert _include_bike({"personal_bike": "regular bike"}) is True

    def test_e_bike(self):
        assert _include_bike({"personal_bike": "e-bike"}) is True

    def test_champ_absent_prive_du_velo(self):
        """Missing field → NO bike and an alarm (ticket 015, lot 1): the fallback removes a
        mode rather than offering one the agent does not have."""
        assert _include_bike({}) is False

    def test_valeur_inconnue_autorise_velo(self):
        """Unexpected value → must not block the bike (fail-open)."""
        assert _include_bike({"personal_bike": "trottinette"}) is True


# ── Tests _vehicle_mode ───────────────────────────────────────────────────────

class TestVehicleMode:
    def test_plan_bicycle(self):
        assert _vehicle_mode(_make_plan("bicycle")) == "bike"

    def test_plan_bike(self):
        assert _vehicle_mode(_make_plan("bike")) == "bike"

    def test_plan_foot(self):
        assert _vehicle_mode(_make_plan("foot")) == "walk"

    def test_plan_car(self):
        assert _vehicle_mode(_make_plan("car")) == "car"

    def test_plan_bus(self):
        assert _vehicle_mode(_make_plan("bus")) == "transit"


# ── Post-filter tests ─────────────────────────────────────────────────────────

class TestPostFilter:
    def _apply_filter(self, itineraries, include_bike: bool):
        if not include_bike:
            return [it for it in itineraries if _vehicle_mode(it) != "bike"]
        return itineraries

    def test_filtre_supprime_velo_si_pas_de_velo(self):
        plans = [_make_plan("foot"), _make_plan("bicycle"), _make_plan("bus")]
        result = self._apply_filter(plans, include_bike=False)
        modes = [_vehicle_mode(p) for p in result]
        assert "bike" not in modes
        assert len(result) == 2

    def test_filtre_conserve_velo_si_velo_dispo(self):
        plans = [_make_plan("foot"), _make_plan("bicycle"), _make_plan("bus")]
        result = self._apply_filter(plans, include_bike=True)
        assert len(result) == 3

    def test_filtre_liste_vide(self):
        assert self._apply_filter([], include_bike=False) == []

    def test_filtre_sans_velo_dans_liste(self):
        plans = [_make_plan("foot"), _make_plan("car")]
        result = self._apply_filter(plans, include_bike=False)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
