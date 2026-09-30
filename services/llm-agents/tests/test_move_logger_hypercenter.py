"""The hypercentre served is the spec's, and the residence column no longer consults it.

Two decisions coexist in this file, and they must be kept apart.

**The hypercentre** must be the one published by
`scripts/progedo_logit/feature_spec.json` — the same one the mode choice model's
`dist_center_*` derive from. `move_logger.py` carried a second, hard-coded definition
(43.6047 / 1.4442), 820 m away: agents in the intermediate band switched
from one ring to another depending on which module looked at them. This centre now only serves
distances (`dist_center_*` of the model, audit axis A4): from now on, the
**terminal time** also classifies its points by municipality.

**The "Place of residence" column** of `moves.csv`, for its part, is no longer computed at all:
it copies the persona's `residence_zone` trait, set at generation
from the survey's **list-of-municipalities** partition. Classifying by distance
compared 24.4 % of personas to another zone's target and put 45 homes outside the
scope in the 3rd ring. The metric threshold tests remain — but they cover
`geo_reference.residence_zone`, which no longer has any production caller: it is an
audit control, and archived measurement scripts must stay replayable.
"""

import json
from pathlib import Path

import pytest

from mobility_core.geo_reference import (
    FALLBACK_GEO_REFERENCE,
    geo_reference,
    hypercenter,
)
from mobility_core.geo_reference import haversine_km
from mobility_core.geo_reference import residence_zone as classement_metrique
from urban_mobility_agents.utils.move_logger import _residence_zone

FEATURE_SPEC = Path(__file__).resolve().parents[3] / "scripts/progedo_logit/feature_spec.json"

# The old competing constant. It must never again be the centre served.
_LEGACY_CENTER = (43.6047, 1.4442)

SPEC_CENTER = (FALLBACK_GEO_REFERENCE["hypercenter"]["lat"],
               FALLBACK_GEO_REFERENCE["hypercenter"]["lon"])


@pytest.fixture(autouse=True)
def _resolution_fraiche():
    """Resolution is cached for the run: each test replays it."""
    geo_reference.cache_clear()
    yield
    geo_reference.cache_clear()


class TestSourceDeLHypercentre:

    @pytest.mark.skipif(not FEATURE_SPEC.exists(),
                        reason="feature_spec.json missing (restricted PROGEDO data)")
    def test_hypercentre_lu_dans_le_feature_spec(self, monkeypatch):
        monkeypatch.setenv("MODE_CHOICE_FEATURE_SPEC", str(FEATURE_SPEC))
        geo_reference.cache_clear()
        published = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))["geo_reference"]["hypercenter"]
        assert hypercenter() == (published["lat"], published["lon"])

    @pytest.mark.skipif(not FEATURE_SPEC.exists(),
                        reason="feature_spec.json missing (restricted PROGEDO data)")
    def test_le_repli_recopie_bien_la_valeur_publiee(self):
        """Fallback and spec must stay the same value: otherwise the fallback lies."""
        published = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))["geo_reference"]["hypercenter"]
        assert FALLBACK_GEO_REFERENCE["hypercenter"]["lat"] == published["lat"]
        assert FALLBACK_GEO_REFERENCE["hypercenter"]["lon"] == published["lon"]

    def test_spec_absent_replie_sur_la_valeur_publiee_pas_sur_l_ancienne(self, monkeypatch, tmp_path):
        """PROGEDO data is restricted-access: the module must stay usable."""
        # An explicit but missing path: find_repo_file returns None (spec not found).
        monkeypatch.setenv("MODE_CHOICE_FEATURE_SPEC", str(tmp_path / "absent.json"))
        geo_reference.cache_clear()
        assert hypercenter() == SPEC_CENTER
        assert hypercenter() != _LEGACY_CENTER

    def test_spec_illisible_replie_sans_lever(self, monkeypatch, tmp_path):
        casse = tmp_path / "feature_spec.json"
        casse.write_text("{ pas du json", encoding="utf-8")
        monkeypatch.setenv("MODE_CHOICE_FEATURE_SPEC", str(casse))
        geo_reference.cache_clear()
        assert hypercenter() == SPEC_CENTER

    def test_le_spec_prime_sur_le_repli(self, monkeypatch, tmp_path):
        """The file is authoritative, not the copied constant."""
        autre = tmp_path / "feature_spec.json"
        autre.write_text(json.dumps(
            {"geo_reference": {"hypercenter": {"lat": 43.5, "lon": 1.5}}}), encoding="utf-8")
        monkeypatch.setenv("MODE_CHOICE_FEATURE_SPEC", str(autre))
        geo_reference.cache_clear()
        assert hypercenter() == (43.5, 1.5)


class TestClassementMetrique:
    """The distance thresholds — AUDIT CONTROL, with no production caller.

    Up to tt3 they served the terminal time, whose laws were stratified with
    them. Since tt4 the terminal time classifies by municipality, like residence. The thresholds
    stay locked for another reason: three measurement scripts use them
    as a COMPARATOR (`audit_perimetre`, `enrich_residence_zone --check`,
    `measure_couronne_v7`), and an archived trace must stay replayable identically.
    It is no longer the definition of anything — cf. `TestColonneDeResidence`.
    """

    def _point_au_sud(self, km: float) -> tuple[float, float]:
        """Point located `km` south of the spec's hypercentre (same longitude)."""
        return SPEC_CENTER[0] - km / 111.19, SPEC_CENTER[1]

    def test_seuils_mesures_depuis_l_hypercentre_du_spec(self):
        assert classement_metrique(*SPEC_CENTER) == "Toulouse"
        assert classement_metrique(*self._point_au_sud(5)) == "Toulouse"
        assert classement_metrique(*self._point_au_sud(12)) == "1st ring"
        assert classement_metrique(*self._point_au_sud(30)) == "2nd ring"
        assert classement_metrique(*self._point_au_sud(60)) == "3rd ring"

    def test_la_bande_des_820_m_suit_le_spec_et_non_l_ancienne_constante(self):
        """A point 7.99 km from the spec's centre, but more than 8 km from the old one.

        This is exactly the case the two definitions classified differently:
        "Toulouse" for the spec, "1st ring" for the abandoned constant.
        """
        lat, lon = self._point_au_sud(7.99)
        assert haversine_km(*SPEC_CENTER, lat, lon) < 8
        assert haversine_km(*_LEGACY_CENTER, lat, lon) > 8
        assert classement_metrique(lat, lon) == "Toulouse"

    def test_point_inconnu_ne_recoit_pas_de_modalite(self):
        assert classement_metrique(None, None) == ""
        assert classement_metrique(43.6, None) == ""


class TestColonneDeResidence:
    """The log column COPIES the persona's trait. It no longer computes anything."""

    def test_le_trait_est_recopie_tel_quel(self):
        for zone in ("Toulouse", "1st ring", "2nd ring", "3rd ring"):
            assert _residence_zone({"residence_zone": zone}) == zone

    def test_hors_perimetre_est_une_valeur_de_la_colonne(self):
        """Axis A4: a home outside the 453 municipalities is not in the 3rd ring.

        It has no EMC² target, its mass must be counted separately. The column must
        therefore be able to carry the value, otherwise it would vanish into the neighbouring stratum.
        """
        assert _residence_zone({"residence_zone": "outside perimeter"}) == "outside perimeter"

    def test_trait_absent_laisse_la_cellule_vide(self):
        """Population generated before the residence ring, or home without coordinates."""
        assert _residence_zone({}) == ""
        assert _residence_zone({"residence_zone": ""}) == ""
        assert _residence_zone({"residence_zone": None}) == ""

    def test_valeur_hors_referentiel_ramenee_a_vide(self):
        """The synthesis joins this column on the EMC² labels: an exotic value
        would vanish there without being counted. Empty is better, as it is visible."""
        assert _residence_zone({"residence_zone": "4eme couronne"}) == ""
        assert _residence_zone({"residence_zone": "Blagnac"}) == ""

    def test_aucun_repli_a_la_distance_nest_possible(self):
        """The module no longer imports the metric function: the fallback is IMPOSSIBLE.

        This is the heart of the ring by municipality. As long as the import existed, a "reasonable
        fallback" could be restored in one line by inadvertence — and it would have
        reproduced exactly the corrected gap, silently.
        """
        import urban_mobility_agents.utils.move_logger as move_logger

        assert not hasattr(move_logger, "residence_zone")
        # A persona without the trait but with a home right in the centre stays empty: the
        # cell is not filled "as best it can".
        assert _residence_zone({"lat": SPEC_CENTER[0], "lon": SPEC_CENTER[1]}) == ""
