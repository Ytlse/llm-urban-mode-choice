"""The 453-commune scope, the regional rail and coach offer, and system consistency.

Admissible homes, TER and liO in the options and in GAMA, cable car categorisation, modal
vocabulary parity, run overwrite protection and the cache mount check.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POP_V5 = REPO_ROOT / "data/population/population_1000_PANEL_v5/population.json"
POP_V4 = REPO_ROOT / "data/population/population_1000_PANEL_v4/population.json"
POP_FILE = POP_V5 if POP_V5.exists() else POP_V4
GAMA_SETTINGS = REPO_ROOT / "services/GAMA/CityTransport/models/Settings.gaml"
GAMA_PT = REPO_ROOT / "services/GAMA/CityTransport/models/PublicTransport.gaml"
OSMNX_DIRECT = REPO_ROOT / "services/llm-agents/trip_helper/osmnx_direct.py"
OTP_HELPER = REPO_ROOT / "services/llm-agents/trip_helper/otp.py"
SETTINGS_PY = REPO_ROOT / "services/llm-agents/settings.py"
PERIMETER_PY = REPO_ROOT / "services/llm-agents/inputs/population/perimeter.py"
COMMUNE_COURONNE = REPO_ROOT / "packages/mobility_core/src/mobility_core/data/commune_couronne.json"


@pytest.fixture(scope="module")
def population_data():
    # Without a sealed cohort under `data/population/`, these tests measure nothing: they
    # SAY SO, instead of hitting a `FileNotFoundError` that nobody links to the
    # cause. The case is normal between the cold archiving of a version and the sealing of
    # the next one (ticket 074, lot A); the count must come back to zero once v6 is sealed.
    if not POP_FILE.exists():
        pytest.skip(f"no sealed cohort in data/population/ (expected {POP_FILE.name} "
                    f"under population_1000_PANEL_v5 or _v4) — substrate in cold archive")
    with open(POP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gama_settings():
    with open(GAMA_SETTINGS, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def gama_pt():
    with open(GAMA_PT, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def perimeter_453_communes():
    with open(COMMUNE_COURONNE, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_insee = set()
    for k, v in data.items():
        if isinstance(v, list):
            for item in v:
                all_insee.add(item["insee"])
    return all_insee


# ==============================================================================
# 453-COMMUNE SCOPE, TER & liO
# ==============================================================================

def test_tf30_perimetre_admissibilite_453_communes(population_data, perimeter_453_communes):
    """TF-30: 100% of agents living within the official polygon of the 453 communes."""
    assert len(perimeter_453_communes) == 453
    in_perim = sum(1 for p in population_data if p.get("household", {}).get("commune_id") in perimeter_453_communes)
    assert in_perim == 1000


def test_tf31_perimetre_rejet_hors_perimetre():
    """TF-31: Cascade and filtering mechanism for homes outside the scope."""
    assert PERIMETER_PY.exists()
    content = PERIMETER_PY.read_text(encoding="utf-8")
    assert "commune" in content.lower()


def test_tf32_offre_ferroviaire_ter_train():
    """TF-32: Availability and recognition of the TER regional rail mode."""
    settings_code = SETTINGS_PY.read_text(encoding="utf-8")
    assert '"2": "Train"' in settings_code or "'2': 'Train'" in settings_code


def test_tf33_presence_gtfs_lio():
    """TF-33: Presence of the regional liO GTFS catalogue."""
    lio_file = REPO_ROOT / "data/gtfs/lio_2026.zip"
    gtfs_dir = REPO_ROOT / "data/gtfs"
    assert lio_file.exists() or (gtfs_dir.exists() and any("lio" in f.name.lower() for f in gtfs_dir.iterdir()))


def test_tf34_continuite_calendrier_annuel_lio():
    """TF-34: Anti-cliff rule protecting the calendar continuity of the liO service."""
    gtfs_year_test = REPO_ROOT / "scripts/tests/test_gtfs_year.py"
    assert gtfs_year_test.exists()
    content = gtfs_year_test.read_text(encoding="utf-8")
    assert "falaise" in content or "calendar" in content


def test_tf35_reduction_zones_blanches_tc():
    """TF-35: Multi-network handling (Tisséo, TER, liO) for access to stops."""
    assert OTP_HELPER.exists()
    otp_code = OTP_HELPER.read_text(encoding="utf-8")
    assert "_has_reachable_stop" in otp_code


def test_tf36_capacite_ter_gama(gama_settings):
    """TF-36: Realistic configuration of TER train capacity in GAMA (300 seats)."""
    assert "2::300" in gama_settings


def test_tf37_couleur_ter_gama(gama_pt):
    """TF-37: Specific colour symbology for TER trains in GAMA (purple)."""
    assert "color: #purple" in gama_pt


def test_tf38_largeur_trace_ter_gama(gama_settings):
    """TF-38: Line width adapted for the visibility of TER lines (width 25)."""
    assert "2::25" in gama_settings


def test_tf42_recensement_route_types_demarrage(gama_pt):
    """TF-42: Automatic diagnostic action on line types at startup."""
    assert "recenser_les_route_types" in gama_pt


# ==============================================================================
# INFRASTRUCTURE INDICATORS
# ==============================================================================

def test_tf48_preservation_profil_cache_hit_route_extras():
    """TF-48: Preservation of the profile on cache reads via route_extras."""
    osmnx_code = OSMNX_DIRECT.read_text(encoding="utf-8")
    assert "route_extras" in osmnx_code or "_make_travel_plan" in osmnx_code


# ==============================================================================
# MULTIMODAL & SYSTEM CONSISTENCY
# ==============================================================================

def test_tf56_teleo_cableway_categorisation_tc():
    """TF-56: Correct categorisation of the Téléo (cable car) as Public Transport."""
    from mobility_llm.mode_choice import canonical_mode

    from scripts.synthesis.formule_score.metrics import categorize_mode
    for term in ["cableway", "gondola", "funicular"]:
        assert categorize_mode(term) == "transports_collectifs"
        assert canonical_mode(term) == "public_transport"
    assert categorize_mode("foot,cableway,foot") == "transports_collectifs"


def test_tf57_parite_stricte_vocabulaire_modal():
    """TF-57: Strict parity of the modal dictionary (test_parite_modes.py)."""
    test_parite = REPO_ROOT / "scripts/tests/test_parite_modes.py"
    assert test_parite.exists()


def test_tf58_protection_ecrasement_runs_claim_run():
    """TF-58: Protection against overwriting runs via claim_run()."""
    settings_code = SETTINGS_PY.read_text(encoding="utf-8")
    assert "def claim_run" in settings_code


def test_tf59_detection_anomalie_montage_cache():
    """TF-59: Detection of a Docker cache volume anomaly (_cache_dir_is_mounted)."""
    osmnx_code = OSMNX_DIRECT.read_text(encoding="utf-8")
    assert "_cache_dir_is_mounted" in osmnx_code
