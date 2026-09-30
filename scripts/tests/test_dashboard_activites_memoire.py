"""Tests of the tracking of memory experiments and of their logs in the 📟 Activités en cours tab.

Checks:
1. The detection of running, suspended and finished memory runs.
2. The progress computation (A/B branch, simulated days, percentage).
3. The extraction of the run log for memory and classic experiments.
4. The Streamlit rendering of memory components (progress bar, model card, log, stop/resume button).
5. The resolution of the decision model in _job_decideur for memory targets.
6. The metadata of the memory make targets in makefiles.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

from experiences import memoire as service_memoire  # noqa: E402
from scripts.dashboard import app, experiences, makefiles, memoire as dash_memoire, runner  # noqa: E402


class FauxSt:
    """Lightweight Streamlit simulator to capture the displayed elements."""

    def __init__(self):
        self.barres: list[tuple[float, str]] = []
        self.textes: list[str] = []
        self.boutons: list[str] = []
        self.legendes: list[str] = []
        self.codes: list[tuple[str, str]] = []
        self.toasts: list[str] = []
        self.expanders: list[str] = []

    def progress(self, valeur: float, text: str = ""):
        self.barres.append((valeur, text))

    def markdown(self, texte: str):
        self.textes.append(texte)

    def caption(self, texte: str, **_k):
        self.legendes.append(texte)

    def code(self, texte: str, language: str = ""):
        self.codes.append((texte, language))

    def columns(self, spec, **_k):
        largeur = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * largeur

    def button(self, label: str, **_k):
        self.boutons.append(label)
        return False

    def toast(self, message: str, **_k):
        self.toasts.append(message)

    def expander(self, label: str, **_k):
        self.expanders.append(label)
        return self

    def container(self, **_k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def dossier_memoire_test(tmp_path, monkeypatch):
    """Creates an isolated data/experiences_memoire/ and experiments/ environment."""
    d_mem = tmp_path / "data" / "experiences_memoire"
    d_mem.mkdir(parents=True)
    d_exp = tmp_path / "experiments"
    d_exp.mkdir(parents=True)

    monkeypatch.setattr(service_memoire, "DOSSIER_MEMOIRE", d_mem)
    monkeypatch.setattr(service_memoire, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dash_memoire, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(app, "REPO_ROOT", tmp_path)
    return tmp_path, d_mem, d_exp


def test_detection_activites_memoire_en_cours(dossier_memoire_test):
    """A memory experiment with etat='en_cours' and recently modified is detected."""
    tmp_path, d_mem, d_exp = dossier_memoire_test

    nom_exp = "exp_mem_choc_c1_gem38f_899549_42j"
    dir_exp = d_mem / nom_exp
    dir_exp.mkdir()

    cfg = {
        "nom": nom_exp,
        "canal": "vecu",
        "evenement": "c1_bouchon_rocade",
        "horizon_jours": 42,
        "population": "899549",
        "modeles": {
            "itinary_multi_agent": "gemini-3.8-flash",
            "evenement_jugement": "gemini-3.8-flash",
            "stm_reflection": "gemini-3.8-flash",
        },
    }
    (dir_exp / "experience_memoire.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    etat_data = {
        "nom": nom_exp,
        "etat": "en_cours",
        "branche_active": "treated",
        "debut": "2026-09-25T06:00:00+00:00",
        "traite": {"etat": "en_cours", "debut": "2026-09-25T06:00:00+00:00"},
        "temoin": {"etat": "en_attente"},
    }
    (dir_exp / "etat.json").write_text(json.dumps(etat_data), encoding="utf-8")

    # Simulation of a current run with checkpoints
    archive_dir = d_exp / "archive" / "2026-09-25_06_00"
    archive_dir.mkdir(parents=True)
    for j in range(1, 11):
        (archive_dir / f"population_6_checkpoint_2026-03-{16+j}.json").write_text("{}", encoding="utf-8")
    (archive_dir / "app.log").write_text("[sync] END sim_time=26 March 2026, 20:00 state_update\n", encoding="utf-8")

    link_current = d_exp / "current"
    link_current.symlink_to(archive_dir)

    runs_dir = d_exp / "runs" / f"{nom_exp}_treated"
    runs_dir.mkdir(parents=True)

    en_cours = service_memoire.activites_en_cours()
    assert len(en_cours) == 1, en_cours
    act = en_cours[0]
    assert act["nom"] == nom_exp
    assert act["canal"] == "vecu"
    assert act["modele_decision"] == "gemini-3.8-flash"

    prog = act["progression"]
    assert prog["branche"] == "treated"
    assert "A (Traité)" in prog["branche_label"]
    assert prog["jours_faits"] == 10
    assert prog["horizon_jours"] == 42
    assert prog["total_jours"] == 84
    assert abs(prog["pourcent"] - (10 / 84 * 100)) < 1e-4
    assert "26 March 2026" in prog["derniere_journee"]
    assert "26 March 2026" in act["log_tail"]


def test_rendre_activites_memoire_streamlit(dossier_memoire_test):
    """The rendre_activites component draws the bar, the details, the stop and the log."""
    st = FauxSt()
    activites = [
        {
            "nom": "exp_mem_test_42j",
            "canal": "vecu",
            "evenement": "c1_bouchon",
            "modele_decision": "gemini-3.8-flash",
            "modele_jugement": "gemini-3.8-flash",
            "modele_stm": "gemini-3.8-flash",
            "population": "899549",
            "progression": {
                "branche_label": "A (Traité)",
                "jours_faits": 14,
                "jours_bras": 14,
                "horizon_jours": 42,
                "total_jours": 84,
                "pourcent": 16.7,
                "derniere_journee": "30 March 2026",
                "ecoule_s": 1200.0,
            },
            "log_tail": "2026-09-25 07:00:00 | INFO | decision ok",
            "log_src": "experiments/current/app.log",
        }
    ]

    dash_memoire.rendre_activites(st, activites)

    # 1 progress bar
    assert len(st.barres) == 1
    val, txt = st.barres[0]
    assert abs(val - 0.167) < 0.01
    assert "exp_mem_test_42j" in txt
    assert "Bras A (Traité) en cours" in txt
    assert "14 / 84 jours simulés" in txt

    # Model and time card
    legendes_jointes = " ".join(st.legendes)
    assert "gemini-3.8-flash" in legendes_jointes
    assert "30 March 2026" in legendes_jointes
    assert "en cours depuis 20 min" in legendes_jointes

    # Stop button
    assert any("Arrêter" in b for b in st.boutons)

    # Log expander with the log content
    assert any("Journal du run mémoire" in exp for exp in st.expanders)
    assert any("decision ok" in code for code, _ in st.codes)


def test_journal_semantique_agrege_run_et_detecte_stagnation(dossier_memoire_test):
    """The log translates the technical traces and alerts if the simulated clock is frozen."""
    _, _, d_exp = dossier_memoire_test
    nom = "exp_mem_journal"
    archive = d_exp / "archive" / "journal"
    archive.mkdir(parents=True)
    (d_exp / "current").symlink_to(archive)
    (archive / "identite_run.json").write_text(json.dumps({"rejeu_ab": nom}), encoding="utf-8")
    (archive / "app.log").write_text(
        "\n".join([
            "2026-09-27 10:00:00 | INFO | x - [evenements] « a13 » : 2 lecteur(s) retenu(s) sur 6 agent(s)",
            "2026-09-27 10:00:01 | INFO | x - [evenements] « a13 » chargé — jours 9→13",
            "2026-09-27 10:01:00 | INFO | x - [sync] END sim_time=16 March 2026, 23:25 state_update_duration=0.1s",
            "2026-09-27 10:07:00 | INFO | x - [enquete] J1 · velo : 6 persona(s) interrogé(s) ensemble en 4.0s.",
            "2026-09-27 10:13:00 | INFO | x - [sync] END sim_time=16 March 2026, 23:25 state_update_duration=0.1s",
            "2026-09-27 10:13:01 | INFO | x - [evenements] jour 1 du run (relatif -8) — nominal : 0 exposé(s)",
        ]) + "\n",
        encoding="utf-8",
    )
    (archive / "agent_memory_events.jsonl").write_text(
        '{"context": "shortterm_memory"}\n{"context": "longterm_memory"}\n',
        encoding="utf-8",
    )
    (archive / "trace_rappel.jsonl").write_text(
        '{"servis": [{"doc_id": "a"}]}\n{"servis": []}\n', encoding="utf-8"
    )
    (archive / "llm_exchanges.jsonl").write_text(
        '{\n  "category": "itinary_multi_agent",\n  "tokens_in": 10,\n  "tokens_out": 2\n}\n',
        encoding="utf-8",
    )
    (archive / "llm_errors.jsonl").write_text('{"http_status": 503}\n', encoding="utf-8")
    (archive / "operations_concept.jsonl").write_text(
        '{"operation": "créé"}\n{"operation": "confirmé"}\n', encoding="utf-8"
    )
    (archive / "evenements.jsonl").write_text('{"evenement_id": "a13"}\n', encoding="utf-8")

    progression = {
        "branche_label": "A (Traité)",
        "derniere_journee": "16 March 2026, 23:25",
    }
    journal = service_memoire.journal_semantique(
        nom,
        {"evenement": "a13_punaises_metro", "rejeu_ab": True, "prefixe_commun": True},
        {
            "branche_active": "treated",
            "traite": {"etat": "en_cours"},
            "temoin": {"etat": "en_attente"},
        },
        progression,
    )

    assert journal["compteurs"]["Appels LLM"] == 1
    assert journal["compteurs"]["Erreurs LLM"] == 1
    assert journal["compteurs"]["Entrées STM"] == 1
    assert journal["compteurs"]["Entrées LTM"] == 1
    assert journal["compteurs"]["Souvenirs servis"] == 1
    assert journal["evenement"]["fenetre"] == "9→13"
    assert journal["evenement"]["lecteurs_attendus"] == 2
    assert journal["evenement"]["traces_produites"] == 1
    assert journal["operations_ltm"] == {"créé": 1, "confirmé": 1}
    assert journal["integrite_ab"]["rejeu_actif"] is True
    assert journal["integrite_ab"]["prefixe_commun"] is True
    assert journal["integrite_ab"]["traite"] == "en_cours"
    assert any("figée" in alerte for alerte in journal["alertes"])
    assert any("erreur(s) LLM" in alerte for alerte in journal["alertes"])
    assert any("Enquête" in ligne for ligne in journal["chronologie"])


def test_rendre_activites_affiche_journal_semantique(dossier_memoire_test):
    st = FauxSt()
    activite = {
        "nom": "exp_mem_semantique",
        "canal": "lu",
        "evenement": "a13_punaises_metro",
        "modele_decision": "gemini-3.1-flash-lite",
        "modele_jugement": "gemini-3.1-flash-lite",
        "modele_stm": "gemini-3.5-flash-lite",
        "population": "population_6",
        "progression": {
            "branche_label": "A (Traité)", "jours_faits": 9, "jours_bras": 9,
            "horizon_jours": 25, "total_jours": 50, "pourcent": 18,
            "derniere_journee": "24 March 2026, 22:00", "ecoule_s": 3600,
        },
        "journal_semantique": {
            "icone": "🟠", "phase": "Réflexion STM", "resume": "A (Traité) · J9",
            "alertes": ["Progression simulée figée depuis 12 min."],
            "compteurs": {"Appels LLM": 44, "Erreurs LLM": 1, "Entrées STM": 12,
                          "Entrées LTM": 6, "Souvenirs servis": 18},
            "evenement": {"id": "a13_punaises_metro", "fenetre": "9→13",
                          "lecteurs_attendus": 2, "traces_produites": 1,
                          "dernier_bilan": "jour 9 — 1 lecture servie"},
            "chronologie": ["10:13:00 · 🧠 STM · 3 réflexions"],
            "categories_llm": {"stm_reflection": 3},
            "operations_ltm": {"créé": 2},
            "integrite_ab": {"rejeu_actif": True, "prefixe_commun": True,
                             "traite": "en_cours", "temoin": "en_attente"},
        },
        "log_tail": "ligne brute",
        "log_src": "experiments/current/app.log",
    }

    dash_memoire.rendre_activites(st, [activite])
    rendu = " ".join(st.textes + st.legendes + [c[0] for c in st.codes])
    assert "Réflexion STM" in rendu
    assert "Progression simulée figée" in rendu
    assert "a13_punaises_metro" in rendu
    assert "Intégrité A/B" in rendu
    assert "3 réflexions" in rendu
    assert any("Log brut" in exp for exp in st.expanders)
    assert "ligne brute" in rendu


def test_progression_memoire_passe_du_traite_au_temoin(dossier_memoire_test):
    """The control resumes at 50 % and the simulated date advances the A+B total."""
    _, _, d_exp = dossier_memoire_test
    nom = "exp_mem_test"
    archive = d_exp / "archive" / "temoin"
    archive.mkdir(parents=True)
    (d_exp / "current").symlink_to(archive)
    (archive / "identite_run.json").write_text(json.dumps({"rejeu_ab": nom}), encoding="utf-8")
    (archive / "app.log").write_text(
        "[sync] END sim_time=18 March 2026, 04:45 state_update_duration=0.1s\n",
        encoding="utf-8",
    )
    etat = {
        "branche_active": "control",
        "traite": {"etat": "ok"},
        "temoin": {"etat": "en_cours", "debut": "2026-09-25T06:00:00+00:00"},
    }

    prog = service_memoire.progression_memoire(nom, {"horizon_jours": 4}, etat)
    assert prog["jours_bras"] == 1  # The second day is not finished before 05:00.
    assert prog["jours_faits"] == 5
    assert prog["total_jours"] == 8
    assert prog["pourcent"] == 62.5

    (archive / "app.log").write_text(
        "[sync] END sim_time=18 March 2026, 05:00 state_update_duration=0.1s\n",
        encoding="utf-8",
    )
    prog = service_memoire.progression_memoire(nom, {"horizon_jours": 4}, etat)
    assert prog["jours_faits"] == 6

    # The control has just started, but current still points to the previous arm.
    os.utime(archive / "app.log", (0, 0))
    prog = service_memoire.progression_memoire(nom, {"horizon_jours": 4}, etat)
    assert prog["jours_faits"] == 4

    (archive / "identite_run.json").write_text(
        json.dumps({"rejeu_ab": "une_autre_experience"}), encoding="utf-8"
    )
    prog = service_memoire.progression_memoire(nom, {"horizon_jours": 4}, etat)
    assert prog["jours_faits"] == 4


def test_interrompues_et_terminees_memoire(dossier_memoire_test):
    """Suspended/stopped and finished experiments are correctly identified."""
    tmp_path, d_mem, _ = dossier_memoire_test

    # 1. Suspended experiment
    dir_susp = d_mem / "exp_mem_suspendue"
    dir_susp.mkdir()
    (dir_susp / "experience_memoire.yaml").write_text("nom: exp_mem_suspendue\n", encoding="utf-8")
    (dir_susp / "etat.json").write_text(json.dumps({
        "nom": "exp_mem_suspendue",
        "etat": "suspendue",
        "note": "Quota journalier épuisé",
    }), encoding="utf-8")

    # 2. Finished experiment
    dir_term = d_mem / "exp_mem_terminee"
    dir_term.mkdir()
    (dir_term / "experience_memoire.yaml").write_text("nom: exp_mem_terminee\n", encoding="utf-8")
    (dir_term / "etat.json").write_text(json.dumps({
        "nom": "exp_mem_terminee",
        "etat": "terminee",
        "traite": {"etat": "ok"},
        "temoin": {"etat": "ok"},
    }), encoding="utf-8")
    (dir_term / "traite").mkdir()
    (dir_term / "temoin").mkdir()
    (dir_term / "traite" / "moves.csv").write_text("a,b\n", encoding="utf-8")
    (dir_term / "temoin" / "moves.csv").write_text("a,b\n", encoding="utf-8")

    inter = service_memoire.interrompues()
    assert len(inter) == 1
    assert inter[0]["nom"] == "exp_mem_suspendue"
    assert inter[0]["cause_icone"] == "⏸"
    assert "Quota journalier épuisé" in inter[0]["cause_detail"]

    term = service_memoire.terminees()
    assert len(term) == 1
    assert term[0]["nom"] == "exp_mem_terminee"

    # Check of the rendering of resumable ones with a Reprendre button
    st = FauxSt()
    lances = []
    dash_memoire.rendre_reprenables(st, inter, lancer=lambda cible, val: lances.append((cible, val)))
    assert any("exp_mem_suspendue" in t for t in st.textes)
    assert any("Reprendre" in b for b in st.boutons)

    # Check of the rendering of finished ones
    st_term = FauxSt()
    dash_memoire.rendre_terminees(st_term, term)
    assert any("exp_mem_terminee" in t for t in st_term.textes)


def test_job_decideur_memoire(dossier_memoire_test):
    """_job_decideur extracts the decision model from experience_memoire.yaml."""
    tmp_path, d_mem, _ = dossier_memoire_test
    nom_exp = "exp_mem_test_job"
    dir_exp = d_mem / nom_exp
    dir_exp.mkdir()
    cfg = {
        "nom": nom_exp,
        "modeles": {"itinary_multi_agent": "gemini-3.1-flash-lite"},
    }
    (dir_exp / "experience_memoire.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class FauxJob:
        def __init__(self, label: str, argv: list[str]):
            self.label = label
            self.argv = argv

    job_mem = FauxJob("root:experience-memoire-lancer", ["make", "experience-memoire-lancer", f"EXP={nom_exp}"])
    decideur = app._job_decideur(job_mem)
    assert "gemini-3.1-flash-lite" in decideur
    assert "mémoire" in decideur

    job_nuit = FauxJob("root:experience-memoire-nuit", ["make", "experience-memoire-nuit"])
    assert "campagne mémoire" in app._job_decideur(job_nuit)


def test_makefiles_meta_cibles_memoire():
    """The memory make targets are declared with their flags (long, llm) in makefiles.py."""
    meta = makefiles._META
    assert ("root", "experience-memoire-lancer") in meta
    groupe, flags, vars_opt = meta[("root", "experience-memoire-lancer")]
    assert groupe == "Expériences Mémoire"
    assert "long" in flags
    assert "llm" in flags
    assert "EXP" in vars_opt

    assert ("root", "experience-memoire-nuit") in meta
    groupe_n, flags_n, _ = meta[("root", "experience-memoire-nuit")]
    assert "long" in flags_n
    assert "llm" in flags_n


def test_classic_experience_a_un_volet_log(tmp_path):
    """A classic run now has a log expander in rendre_activites."""
    st = FauxSt()
    exp_dir = tmp_path / "exp_classique" / "executions" / "exec_1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "execution.log").write_text("ligne 1\nligne 2\nligne 3\n", encoding="utf-8")

    act = {
        "executions": [
            {
                "experience": "exp_classique",
                "execution": "exec_1",
                "dossier": str(exp_dir),
                "fournisseur": "google",
                "faits": 50,
                "total": 100,
                "pourcent": 50.0,
                "personnes": 10,
                "personnes_terminees": 5,
                "conditions": "sans conditions",
            }
        ],
        "jeux": [],
        "definies": 1,
        "terminees": 0,
    }

    experiences.rendre_activites(st, act, compact=False)
    # The log expander must be present
    assert any("Journal d'exécution" in exp for exp in st.expanders)
    # The log content must have been extracted
    assert any("ligne 3" in code for code, _ in st.codes)
