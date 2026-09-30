"""What GAMA tells the memory.

Two distinct bugs, measured on `experiments/archive/2026-09-14_23_58`:
the wait before departure was logged as a walk (398 observations out of 412), and
the 253 car trips were rendered to the model as unnamed public transport.

Part B1 lives in GAML, which has no unit test bench here: it is checked on the
SOURCE, as other tests check their structural rules. Part B2 is checked
on the actual rendering, which is what the model reads.
"""

import sys
from pathlib import Path

import pytest

RACINE_DEPOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAML = (
    RACINE_DEPOT / "services" / "GAMA" / "CityTransport" / "models" / "Inhabitant.gaml"
).read_text(encoding="utf-8")

# The file carries TWO `CURRENT_TIMESTAMP < schedule_at` guards: the one in
# `follow_the_vehicle` does not concern departure (the agent is already on board, so already left).
# The B1 rules bear on `follow_the_plan_when_stop`, and on it alone.
REFLEXE_DEPART = GAML.split(
    "reflex follow_the_plan_when_stop"
)[1].split("\n	reflex ")[0]

# The action that RECEIVES the plan, isolated from the attribute declarations.
RECEPTION_DU_PLAN = GAML.split("// Reset the trip state and the metrics")[1].split(
    "do passenger_reset_plan"
)[0]


class TestB1LAttenteNEstPasUneMarche:
    """B1 — the step stopwatch starts at the first movement, not on receipt of the plan."""

    def test_B1_1_le_chronometre_est_repose_apres_la_garde_de_schedule_at(self):
        apres_garde = REFLEXE_DEPART.split("if CURRENT_TIMESTAMP < schedule_at {")[1]
        avant_premier_segment = apres_garde.split("point dest <- list_destination[step_idx];")[0]
        assert "plan_started" in avant_premier_segment, (
            "without a reset after the wait, the first segment absorbs the departure delay"
        )
        assert "step_started_at <- CURRENT_TIMESTAMP" in avant_premier_segment

    def test_B1_2_la_reception_du_plan_reste_un_repli(self):
        """An agent leaving without waiting must behave exactly as before."""
        assert "step_started_at <- CURRENT_TIMESTAMP" in RECEPTION_DU_PLAN
        assert "plan_started <- false" in RECEPTION_DU_PLAN

    def test_B1_3_la_remise_a_zero_n_a_lieu_qu_une_fois_par_trajet(self):
        """Otherwise the following segments would restart from zero at each simulation step."""
        apres_garde = REFLEXE_DEPART.split("if CURRENT_TIMESTAMP < schedule_at {")[1]
        bloc = apres_garde.split("point dest <- list_destination[step_idx];")[0]
        assert "if !plan_started {" in bloc, (
            "the reset must be guarded by the flag, otherwise it applies at "
            "every step and all segment durations drop to zero"
        )

    def test_B1_4_les_segments_suivants_gardent_leur_propre_chronometre(self):
        """The counter restarts at the end of the previous segment, as before the fix."""
        assert "step_started_at <- CURRENT_TIMESTAMP;\n\t\t\tlast_dist_traveled <- 0.0;" in GAML


class TestB2LaVoitureNEstPasUnTransportCollectif:
    """B2 — a personal vehicle has its own observation type."""

    def test_B2_l_aiguillage_gaml_distingue_les_trois_cas(self):
        bloc = GAML.split("bool is_transfer <- _route = _WALK_;")[1].split("step_idx <- step_idx + 1;")[0]
        assert "is_own_vehicle" in bloc
        assert "submit_ob_vehicle" in bloc
        assert "submit_ob_transfer" in bloc
        assert "submit_ob_transit" in bloc

    def test_B2_l_action_est_declaree_virtuelle_et_implementee(self):
        assert "action submit_ob_vehicle(float segment_duration, float dist, string vehicle_mode) virtual: true;" in GAML
        assert GAML.count("action submit_ob_vehicle") == 2, (
            "an unimplemented virtual action makes the model fail at load time"
        )

    def test_B2_l_observation_ne_porte_ni_arret_ni_ligne(self):
        corps = GAML.split("action submit_ob_vehicle(float segment_duration, float dist, string vehicle_mode) {")[1].split("OB_LIST << ob;")[0]
        for champ in ("departure_stop_name", "arrival_stop_name", "by_vehicle_route_id"):
            assert champ not in corps, (
                f"`{champ}` makes no sense for a personal vehicle: it is by "
                f"filling it in that the transit template produced `Unknown Unknown`"
            )

    def test_B2_1_le_rendu_voiture(self):
        from text_helper import env_ob_to_text

        texte = env_ob_to_text(
            "vehicle",
            {"type": "vehicle", "timestamp": 1, "moving_id": "m", "activity_id": "a",
             "mode": "car", "distance": 6512.0, "duration": 487.0},
        )
        assert texte.startswith("[ CAR ]")
        assert "by car" in texte

    def test_B2_2_le_rendu_velo(self):
        from text_helper import env_ob_to_text

        texte = env_ob_to_text(
            "vehicle",
            {"type": "vehicle", "timestamp": 1, "mode": "bike",
             "distance": 2100.0, "duration": 600.0},
        )
        assert texte.startswith("[ BIKE ]")

    def test_B2_un_mode_inattendu_est_refuse(self):
        from pydantic import ValidationError
        from text_helper import parse_ob

        with pytest.raises(ValidationError):
            parse_ob("vehicle", {"type": "vehicle", "timestamp": 1, "mode": "helicopter",
                                 "distance": 1.0, "duration": 1.0})

    def test_B2_4_une_ligne_introuvable_n_est_pas_nommee(self):
        from text_helper import env_ob_to_text
        from text_helper.templates.repository import ROUTES_INCONNUES

        ROUTES_INCONNUES.pop("__DIRECT_CAR__", None)
        texte = env_ob_to_text(
            "transit",
            {"type": "transit", "timestamp": 1, "waiting_time": 0, "distance": 3000.0,
             "duration": 540, "arrival_stop_name": "Gers", "departure_stop_name": "Perget",
             "by_vehicle_route_id": "__DIRECT_CAR__"},
        )
        assert "Unknown" not in texte
        assert "public transport" in texte
        assert ROUTES_INCONNUES.get("__DIRECT_CAR__") == 1, (
            "the case must be counted: a route missing from the reference data is an anomaly, "
            "not a normal state"
        )

    def test_B2_3_une_ligne_connue_est_rendue_comme_avant(self):
        from text_helper import env_ob_to_text
        from text_helper.templates.repository import gtfs_data

        route_id = next(iter(gtfs_data.route_id_map))
        texte = env_ob_to_text(
            "transit",
            {"type": "transit", "timestamp": 1, "waiting_time": 0, "distance": 3000.0,
             "duration": 540, "arrival_stop_name": "Arenes",
             "departure_stop_name": "Brombach", "by_vehicle_route_id": route_id},
        )
        assert texte.startswith("[ PUBLIC TRANSPORT ] Trip by ")
        assert "From: 'Brombach'; To: 'Arenes'" in texte
        assert "Unknown" not in texte

    def test_B2_5_aucune_observation_ne_contient_Unknown_Unknown(self):
        from text_helper import env_ob_to_text

        cas = [
            ("vehicle", {"type": "vehicle", "timestamp": 1, "mode": "car",
                         "distance": 1.0, "duration": 1.0}),
            ("transit", {"type": "transit", "timestamp": 1, "waiting_time": 0,
                         "distance": 1.0, "duration": 1, "arrival_stop_name": "",
                         "departure_stop_name": "", "by_vehicle_route_id": "inexistante"}),
        ]
        for code, ob in cas:
            assert "Unknown Unknown" not in env_ob_to_text(code, ob)
