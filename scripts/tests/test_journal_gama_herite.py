"""The prefix alarm of a previous attempt does not stop the arm that resumes it (2026-09-28).

A resumed arm writes into the `gama_headless.log` of the attempt it resumes. Its end still
carries "simulation à invalider", written by GAMA when the controller of the previous
attempt stopped. `attendre_fin_du_run` re-read the last 3,000 characters of the
file: on the restart of a13's v5 control arm, at 19:55, it found this alarm there and stopped the
arm before its first step (code 6). A night-time retry failed instead of
resuming. Only what THIS arm writes counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment import run_sequential_cohort as C

ANCIEN = (
    '[gama] {"message": "[ERROR] Received non-JSON HTTP response from controller"}\n'
    '[gama] {"message": "[ALARME] Préfixe commun : /sync a échoué après retentatives, '
    'simulation à invalider.\\n"}\n'
    "[ALARME] Arrêt sur erreur de synchronisation préfixe. Fermeture immédiate du launcher.\n"
)
NOUVEAU = "⏳ GAMA Server pas encore prêt, retry dans 5s...\n[gama] Simulation stopped after 25 days\n"


@pytest.fixture
def banc(tmp_path, monkeypatch):
    """A launcher alive for one round, `make` calls recorded, an application log captured."""
    tours = iter([True, False])
    monkeypatch.setattr(C, "lanceur_en_cours", lambda: next(tours, False))
    monkeypatch.setattr(C, "derniere_journee_simulee", lambda archive: "jour 1")
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    appels: list[list[str]] = []
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **k: appels.append(cmd))
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    archive = tmp_path / "2026-09-28_13_50"
    archive.mkdir()
    yield archive, appels, messages
    logger.remove(sink)


def _reprise(archive: Path) -> int:
    """The previous attempt left its alarm; the resumed arm notes the size, then writes."""
    journal = archive / "gama_headless.log"
    journal.write_text(ANCIEN, encoding="utf-8")
    depuis = C.taille_journal_gama(archive)
    with journal.open("a", encoding="utf-8") as f:
        f.write(NOUVEAU)
    return depuis


def test_l_alarme_heritee_est_ignoree_et_la_fin_propre_de_ce_bras_est_lue(banc):
    archive, _appels, messages = banc
    depuis = _reprise(archive)
    assert depuis > 0

    assert C.attendre_fin_du_run("p", "control", archive, gama_depuis=depuis) is True
    assert not any("Échec de synchronisation" in m for m in messages)
    assert any("hérités d'une tentative précédente" in m for m in messages)
    assert any("Simulation stopped after" in m for m in messages)


def test_contre_epreuve_sans_la_borne_l_alarme_heritee_arrete_le_bras(banc):
    """Without this counter-test, a green test would not prove that it can see the defect."""
    archive, appels, messages = banc
    (archive / "gama_headless.log").write_text(ANCIEN + "⏳ GAMA Server pas encore prêt\n")

    assert C.attendre_fin_du_run("p", "control", archive, gama_depuis=0) is False
    assert any("Échec de synchronisation" in m for m in messages)
    assert ["make", "stop-run"] in appels


def test_une_alarme_ecrite_par_ce_bras_l_arrete(banc):
    archive, _appels, messages = banc
    journal = archive / "gama_headless.log"
    journal.write_text("[gama] jour 3\n", encoding="utf-8")
    depuis = C.taille_journal_gama(archive)
    with journal.open("a", encoding="utf-8") as f:
        f.write(ANCIEN)

    assert C.attendre_fin_du_run("p", "control", archive, gama_depuis=depuis) is False
    assert any("Échec de synchronisation" in m for m in messages)


def test_un_journal_remplace_se_lit_depuis_son_debut(tmp_path):
    """Set aside then recreated, the log is shorter than the bound: all of it is from this arm."""
    journal = tmp_path / "gama_headless.log"
    journal.write_text("simulation à invalider\n", encoding="utf-8")
    assert "simulation à invalider" in C.lire_journal_gama(journal, depuis=10_000, fin=3000)


def test_la_lecture_rend_au_plus_fin_caracteres_de_la_partie_neuve(tmp_path):
    journal = tmp_path / "gama_headless.log"
    journal.write_text("é" * 50 + "neuf-" * 1000, encoding="utf-8")
    depuis = len("é" * 50) * 2  # "é": two bytes in UTF-8
    lu = C.lire_journal_gama(journal, depuis=depuis, fin=3000)
    assert len(lu) == 3000 and "é" not in lu


def test_taille_d_un_journal_absent(tmp_path):
    assert C.taille_journal_gama(None) == 0
    assert C.taille_journal_gama(tmp_path) == 0


def test_la_taille_est_relevee_avant_le_lancement_et_transmise_au_guetteur():
    src = (REPO_ROOT / "scripts" / "experiment" / "run_sequential_cohort.py").read_text(
        encoding="utf-8"
    )
    i_taille = src.find('taille_journal_gama(Path("experiments/archive") / reprise["archive"])')
    i_popen = src.find("process = subprocess.Popen(", i_taille)
    assert 0 < i_taille < i_popen, "the size must be recorded before launching GAMA"
    assert "gama_depuis=gama_depuis," in src
