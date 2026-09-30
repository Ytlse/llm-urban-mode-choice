""""S'inspirer d'une expérience existante": from the most recently used to the least recent.

Alphabetical order put at the top experiments forgotten for weeks, whereas this
selector is used first of all to start again from what was just done. The criterion is the last
WRITE of an `etat.json`, not the name of the run folder: that name carries the
CREATION time, so a run opened in the morning and resumed in the evening would look old.
"""

import os
import sys
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402


def _experience(racine: Path, nom: str, executions: dict[str, float] | None = None) -> Path:
    d = racine / nom
    d.mkdir(parents=True)
    (d / "experience.yaml").write_text(yaml.safe_dump({"nom": nom}), encoding="utf-8")
    for exec_nom, quand in (executions or {}).items():
        e = d / "executions" / exec_nom
        e.mkdir(parents=True)
        etat = e / "etat.json"
        etat.write_text('{"etat": "terminee"}', encoding="utf-8")
        os.utime(etat, (quand, quand))
    return d


def test_la_plus_recemment_utilisee_vient_en_tete(monkeypatch, tmp_path):
    _experience(tmp_path, "a_vieille", {"2026-09-08_09_00_00": 1_000.0})
    _experience(tmp_path, "z_recente", {"2026-09-08_08_00_00": 9_000.0})
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    assert list(experiences.experiences()) == ["z_recente", "a_vieille"]


def test_une_execution_reprise_compte_comme_recente(monkeypatch, tmp_path):
    """The folder carries the creation time; the state rewritten on resume is what dates the use."""
    _experience(tmp_path, "ouverte_le_matin_reprise_ce_soir", {"2026-09-08_08_00_00": 9_000.0})
    _experience(tmp_path, "ouverte_a_midi_jamais_reprise", {"2026-09-08_12_00_00": 2_000.0})
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    assert list(experiences.experiences())[0] == "ouverte_le_matin_reprise_ce_soir"


def test_les_jamais_utilisees_ferment_la_liste_par_ordre_alphabetique(monkeypatch, tmp_path):
    _experience(tmp_path, "b_jamais")
    _experience(tmp_path, "a_jamais")
    _experience(tmp_path, "m_utilisee", {"2026-09-08_10_00_00": 5_000.0})
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    assert list(experiences.experiences()) == ["m_utilisee", "a_jamais", "b_jamais"]


def test_une_experience_sans_execution_ni_dossier_ne_casse_rien(monkeypatch, tmp_path):
    d = _experience(tmp_path, "sans_dossier_executions")
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    assert experiences.derniere_utilisation(d) == 0.0
    assert list(experiences.experiences()) == ["sans_dossier_executions"]


def test_une_execution_sans_etat_lisible_retombe_sur_son_dossier(monkeypatch, tmp_path):
    d = _experience(tmp_path, "etat_absent")
    (d / "executions" / "2026-09-08_10_00_00").mkdir(parents=True)
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    assert experiences.derniere_utilisation(d) > 0.0, "the folder counts when state is missing"


def test_le_dossier_des_experiences_absent_rend_une_liste_vide(monkeypatch, tmp_path):
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path / "nexiste-pas")
    assert experiences.experiences() == {}


# ── The "Décideur" column names the model family (ticket 043) ──────────────
#
# Four model experiments coexist — booster, logit, random forest, kernel logistic
# regression — and `experience.yaml` only carries the artefact path. The column showed
# `modele` four times, that is, the only thing those four rows had in
# common. The family is DERIVED from the artefact format, as in the run traces.

def _artefact(chemin: Path, format_: str) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text('{\n "format": "%s",\n "spec_version": 2\n}\n' % format_,
                      encoding="utf-8")
    return chemin


def test_le_decideur_modele_nomme_sa_famille(tmp_path):
    art = _artefact(tmp_path / "klr_model.json", "klr_mode_choice_policy")
    label = experiences._decideur_label({"type": "modele", "artefact": str(art)})
    assert label == "modele:klr"


def test_chaque_famille_a_son_libelle(tmp_path):
    attendus = {"lightgbm_mode_choice_policy": "modele:lightgbm",
                "mnl_mode_choice_policy": "modele:mnl",
                "klr_mode_choice_policy": "modele:klr",
                "rf_mode_choice_policy": "modele:rf"}
    for format_, attendu in attendus.items():
        art = _artefact(tmp_path / f"{format_}.json", format_)
        assert experiences._decideur_label(
            {"type": "modele", "artefact": str(art)}) == attendu


def test_un_artefact_illisible_laisse_modele_nu(tmp_path):
    """Inventing a family would be worse than naming none."""
    assert experiences._decideur_label(
        {"type": "modele", "artefact": str(tmp_path / "jamais_estime.json")}) == "modele"
    vide = _artefact(tmp_path / "sans_format.json", "")
    vide.write_text('{"spec_version": 2}', encoding="utf-8")
    assert experiences._decideur_label({"type": "modele", "artefact": str(vide)}) == "modele"


def test_un_format_hors_convention_est_rendu_tel_quel(tmp_path):
    """Better a raw label than a label translated by guesswork."""
    art = _artefact(tmp_path / "exotique.json", "arbre_magique_v9")
    assert experiences._decideur_label(
        {"type": "modele", "artefact": str(art)}) == "modele:arbre_magique_v9"


def test_sans_artefact_c_est_le_booster_par_defaut():
    """`artefact: null` = the default artefact, as at run time."""
    defaut = RACINE / experiences.ARTEFACT_MODELE_DEFAUT
    if not defaut.exists():
        import pytest
        pytest.skip("Booster not trained — `make policy`")
    assert experiences._decideur_label({"type": "modele"}) == "modele:lightgbm"


def test_le_defaut_du_tableau_est_celui_du_decideur():
    """Two constants that drift apart would announce the wrong family.

    Read as text rather than imported: the decision-maker lives in `services/llm-agents/`, another
    package, and importing it here to check a path would cost its dependencies.
    """
    source = (RACINE / "services" / "llm-agents" / "experiences" / "decideur_modele.py").read_text(
        encoding="utf-8")
    fin = experiences.ARTEFACT_MODELE_DEFAUT.split("/")[-1]
    assert f'"{fin}"' in source, (
        f"POLICY_DEFAUT no longer points to {fin}: the dashboard Décideur column "
        "would name the wrong family for an experiment without an explicit artefact.")


def test_une_passerelle_reste_nommee_par_son_modele():
    """The behaviour of the other decision-makers does not change."""
    assert experiences._decideur_label(
        {"type": "passerelle", "modele": "gemini-3.5-flash-lite"}) == "gemini-3.5-flash-lite"
    assert experiences._decideur_label({"type": "aleatoire"}) == "aleatoire"
    assert experiences._decideur_label({"type": "rejeu", "modele": "x"}) == "rejeu:x"
    assert experiences._decideur_label(None) == ""
