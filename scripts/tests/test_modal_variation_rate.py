"""Unit tests for the behavioural analysis of modal stability and variation (modal_variation_rate.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis.modal_variation_rate import (
    compute_entropy,
    compute_persona_entropy,
    compute_transitions,
    compute_phase_summary,
    compute_activity_summary,
    compute_transition_matrix,
    extract_chosen_route,
    PHASE_APRES,
    PHASE_AVANT,
    PHASE_EVENEMENT,
    PHASE_HORS,
    PHASE_SANS,
    fenetres_du_run,
    generate_markdown_report,
    phase_de,
    _bits,
)


def test_compute_entropy_properties():
    """Checks the fundamental properties of Shannon entropy."""
    # 1. Deterministic (single-mode) distribution -> H = 0 bit
    det_series = pd.Series(["Voiture Privée"] * 20)
    assert compute_entropy(det_series) == 0.0

    # 2. Equiprobable binary distribution (heads/tails) -> H = 1.0 bit
    bin_series = pd.Series(["Voiture Privée", "Vélo"] * 10)
    assert pytest.approx(compute_entropy(bin_series), 0.001) == 1.0

    # 3. Equiprobable 4-mode distribution -> H = 2.0 bits
    quad_series = pd.Series(["Voiture Privée", "Vélo", "Marche", "Transports_collectifs"] * 5)
    assert pytest.approx(compute_entropy(quad_series), 0.001) == 2.0


def test_extract_chosen_route():
    """Checks extraction of the chosen itinerary from Options (descriptif)."""
    row_nominal = pd.Series({
        "Mode de transport Choisi": "Marche",
        "Index retenu": 2,
        "Options (descriptif)": "0:car:100s:2km | 1:bike:150s:2.1km | 2:foot:600s:1.8km",
    })
    assert extract_chosen_route(row_nominal) == "foot:600s:1.8km"

    # Missing or invalid index -> fall back on the mode
    row_fallback = pd.Series({
        "Mode de transport Choisi": "Voiture Privée",
        "Index retenu": np.nan,
        "Options (descriptif)": "",
    })
    assert extract_chosen_route(row_fallback) == "Voiture Privée"


def test_compute_transitions_synthetic():
    """Checks the computation of transitions for a recurring activity."""
    # Synthetic data for one agent over 4 days
    data = [
        {
            "ID Personne": "101",
            "ID Activité": "act_work",
            "motif_clean": "Travail",
            "Heure de départ": "2026-03-16 08:00:00",
            "depart_dt": pd.to_datetime("2026-03-16 08:00:00"),
            "journee_date": "2026-03-16",
            "jour_simule": 1,
            "phase": "1. Pré-choc (J1-7)",
            "Mode de transport Choisi": "Voiture Privée",
            "chosen_route": "car:1200s",
            "Méthode de sélection": "LLM",
        },
        {
            "ID Personne": "101",
            "ID Activité": "act_work",
            "motif_clean": "Travail",
            "Heure de départ": "2026-03-17 08:00:00",
            "depart_dt": pd.to_datetime("2026-03-17 08:00:00"),
            "journee_date": "2026-03-17",
            "jour_simule": 2,
            "phase": "1. Pré-choc (J1-7)",
            "Mode de transport Choisi": "Vélo",
            "chosen_route": "bike:1800s",
            "Méthode de sélection": "LLM",
        },
        {
            "ID Personne": "101",
            "ID Activité": "act_work",
            "motif_clean": "Travail",
            "Heure de départ": "2026-03-18 08:00:00",
            "depart_dt": pd.to_datetime("2026-03-18 08:00:00"),
            "journee_date": "2026-03-18",
            "jour_simule": 3,
            "phase": "1. Pré-choc (J1-7)",
            "Mode de transport Choisi": "Vélo",
            "chosen_route": "bike:1800s",
            "Méthode de sélection": "LLM",
        },
        {
            "ID Personne": "101",
            "ID Activité": "act_work",
            "motif_clean": "Travail",
            "Heure de départ": "2026-03-24 08:00:00",
            "depart_dt": pd.to_datetime("2026-03-24 08:00:00"),
            "journee_date": "2026-03-24",
            "jour_simule": 9,
            "phase": "2. Péri-choc (J8-9)",
            "Mode de transport Choisi": "Voiture Privée",
            "chosen_route": "car:1200s",
            "Méthode de sélection": "LLM Error (Default index)",
        },
    ]
    df = pd.DataFrame(data)

    transitions = compute_transitions(df, level="activite")
    assert len(transitions) == 3

    # Transition 1 -> 2: modal change Voiture -> Vélo
    assert transitions[0].modal_change is True
    assert transitions[0].route_change is True
    assert transitions[0].is_llm_valid_transition is True

    # Transition 2 -> 3: same mode (Vélo -> Vélo)
    assert transitions[1].modal_change is False
    assert transitions[1].route_change is False
    assert transitions[1].is_llm_valid_transition is True

    # Transition 3 -> 4: Vélo -> Voiture (not a valid LLM one)
    assert transitions[2].modal_change is True
    assert transitions[2].is_llm_valid_transition is False


def test_summaries_and_matrix():
    """Checks the per-phase summary and the transition matrix."""
    # Simulation of a set of 10 transitions
    from scripts.analysis.modal_variation_rate import TransitionObservation
    dummy_trans = [
        TransitionObservation(
            person_id="1", activite_id="a1", motif="Travail",
            jour_prev=1, jour_curr=2, date_prev="2026-03-16", date_curr="2026-03-17",
            phase_curr=PHASE_AVANT,
            mode_prev="Voiture Privée", mode_curr="Vélo",
            route_prev="r1", route_curr="r2",
            modal_change=True, route_change=True, day_gap=1, is_consecutive_day=True,
            selection_prev="LLM", selection_curr="LLM", is_llm_valid_transition=True,
        ),
        TransitionObservation(
            person_id="1", activite_id="a1", motif="Travail",
            jour_prev=2, jour_curr=3, date_prev="2026-03-17", date_curr="2026-03-18",
            phase_curr=PHASE_AVANT,
            mode_prev="Vélo", mode_curr="Vélo",
            route_prev="r2", route_curr="r2",
            modal_change=False, route_change=False, day_gap=1, is_consecutive_day=True,
            selection_prev="LLM", selection_curr="LLM", is_llm_valid_transition=True,
        ),
    ]

    p_sum = compute_phase_summary(dummy_trans)
    assert len(p_sum) == 1
    assert p_sum.iloc[0]["taux_variation_modal"] == 0.5
    assert p_sum.iloc[0]["stabilite_modale"] == 0.5

    a_sum = compute_activity_summary(dummy_trans)
    assert len(a_sum) == 1
    assert a_sum.iloc[0]["motif"] == "Travail"

    mat = compute_transition_matrix(dummy_trans)
    assert mat.loc["Vélo", "Vélo"] == 1.0
    assert mat.loc["Voiture Privée", "Vélo"] == 1.0



# ── The phases come from the run (analysis of 2026-09-25) ─────────────────────────────
# The treated arm 2026-09-24_17_50 — article a09 read on day 11 by household 133048 — came out
# under the title "Choc d'avarie moteur J8-9", cut into four phases frozen on another
# protocol, with a section 2 whose figures came from an old run.
import json


def _run_lu(tmp_path, *, evenement=True):
    run = tmp_path / "run"
    run.mkdir()
    (run / "population_3.json").write_text(json.dumps([
        {"person_id": "1", "household": {"id": "h1"}},
        {"person_id": "2", "household": {"id": "h1"}},
        {"person_id": "3", "household": {"id": "h2"}},
    ]), encoding="utf-8")
    if evenement:
        (run / "evenements.jsonl").write_text(json.dumps({
            "person_id": "1", "horodatage_simule": "2026-03-26T00:00:00", "choc_id": "a09",
            "evenement_id": "a09", "canal": "lu", "moment": "reveil", "jour_run": 11,
            "vecu": "Parks closed.",
        }) + "\n", encoding="utf-8")
        (run / "evenement.yaml").write_text(
            "evenement: a09_vent_autan\nlibelle: Autan gales, parks closed\n", encoding="utf-8")
    return run


def test_les_phases_sont_celles_du_foyer_expose(tmp_path):
    cal = fenetres_du_run(_run_lu(tmp_path))
    # Read on waking on the 26th: the 3 h cutoff does not move the article back to the 25th.
    assert phase_de("1", "2026-03-25", cal) == PHASE_AVANT
    assert phase_de("1", "2026-03-26", cal) == PHASE_EVENEMENT
    assert phase_de("1", "2026-03-27", cal) == PHASE_APRES
    # The co-resident shares the reader's window; the other household is separate.
    assert phase_de("2", "2026-03-27", cal) == PHASE_APRES
    assert cal["fenetres"]["2"].role == "co_resident"
    assert phase_de("3", "2026-03-27", cal) == PHASE_HORS
    assert cal["libelle"] == "a09_vent_autan — Autan gales, parks closed"


def test_un_bras_temoin_est_sans_evenement(tmp_path):
    cal = fenetres_du_run(_run_lu(tmp_path, evenement=False))
    assert phase_de("1", "2026-03-26", cal) == PHASE_SANS


def test_le_rapport_ne_porte_plus_de_chiffres_ecrits_davance(tmp_path):
    cal = fenetres_du_run(_run_lu(tmp_path))
    activite = pd.DataFrame([{
        "motif": "Travail", "transitions_total": 12, "taux_variation_global": 0.25,
        "stabilite_globale": 0.75, "taux_variation_itineraire": 0.3,
        "taux_var_avant": 0.2, "taux_var_apres": None, "rigidite_relative": "Modérée",
    }])
    entropie = pd.DataFrame([{
        "persona_id": "1", "exposition": "foyer exposé", "trajets_total": 20,
        "modes_distincts": 1, "entropie_modale_bits": -0.0, "equitabilite_pielou": -0.0,
        "entropie_avant": 0.0, "entropie_evenement": None, "entropie_apres": None,
    }])
    texte = generate_markdown_report(pd.DataFrame(), activite, entropie, tmp_path, None,
                                     cal).read_text(encoding="utf-8")
    for fige in ("J8-9", "Choc C6", "35.7%", "stabilité 100%", "incident moteur"):
        assert fige not in texte
    assert "a09_vent_autan — Autan gales, parks closed" in texte
    assert "foyer h1 : 2026-03-26" in texte
    assert "`Travail` — 75.0%" in texte
    assert "-0.000" not in texte


def test_zero_bit_negatif_s_ecrit_zero():
    assert _bits(-0.0) == "0.000"
    assert _bits(None) == "—"


def test_les_tables_vont_dans_le_run_analyse_et_pas_dans_current(tmp_path, monkeypatch):
    run = _run_lu(tmp_path)
    colonnes = ["ID Personne", "ID Activité", "Heure de départ", "Mode de transport Choisi",
                "Motifs de déplacement", "Méthode de sélection", "Options (descriptif)",
                "Index retenu"]
    lignes = [",".join(colonnes)]
    for jour, mode in ((24, "Marche"), (25, "Vélo"), (26, "Marche"), (27, "Vélo")):
        lignes.append(f"1,a1,2026-03-{jour} 08:00:00,{mode},work,LLM,,")
    (run / "moves.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    from scripts.analysis import modal_variation_rate as mvr

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mvr", str(run / "moves.csv"), "-o",
                                      str(run / "reports"), "--no-plots"])
    mvr.main()
    assert (run / "mesures" / "taux_variation_par_phase.csv").is_file()
    assert not (tmp_path / "experiments").exists()
