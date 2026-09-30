"""A job never writes into another job's log ("Activités en cours" tab).

The job registry lives in `st.cache_resource`: a restart of the Streamlit server resets
it, while the jobs it launched are DETACHED and keep writing. The counter restarted
from 1, so a new `001-root-experience-lancer` reopened for WRITING the log of a
`001-root-experience-lancer` from a previous session that was still alive.
Both handles wrote into the same file, each at its own offset: on 2026-09-08, the
console of a "meta/muse-glimmer" job showed the end of a Gemini race.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import runner  # noqa: E402


def test_le_compteur_repart_du_dernier_journal_ecrit(tmp_path):
    for nom in ("001-root-jeu.log", "007-root-experience-lancer.log", "013-root-up-services.log"):
        (tmp_path / nom).write_text("", encoding="utf-8")
    assert runner.dernier_index(tmp_path) == 13


def test_un_dossier_vide_ou_absent_repart_de_zero(tmp_path):
    assert runner.dernier_index(tmp_path) == 0
    assert runner.dernier_index(tmp_path / "nexiste-pas") == 0


def test_les_noms_sans_index_sont_ignores(tmp_path):
    (tmp_path / "app.log").write_text("", encoding="utf-8")
    (tmp_path / "trace-2026.log").write_text("", encoding="utf-8")
    assert runner.dernier_index(tmp_path) == 0


def test_au_dela_de_999_l_index_reste_lisible(tmp_path):
    (tmp_path / "1042-root-experience-lancer.log").write_text("", encoding="utf-8")
    assert runner.dernier_index(tmp_path) == 1042


def _registre(monkeypatch, dossier: Path) -> runner.Registry:
    monkeypatch.setattr(runner, "LOG_DIR", dossier)
    return runner.Registry()


def test_un_nouveau_registre_ne_reprend_pas_les_journaux_existants(monkeypatch, tmp_path):
    """The failure case: the Streamlit server restarts, the previous job is still writing."""
    vivant = tmp_path / "001-root-experience-lancer.log"
    vivant.write_text("sortie du job d'une session precedente, toujours en cours\n", encoding="utf-8")

    registre = _registre(monkeypatch, tmp_path)
    job = registre.launch("root:experience-lancer", ["make", "experience-lancer"], tmp_path)

    assert job.log_path != vivant, "a new job must not reopen another job's log"
    assert job.id == "002-root-experience-lancer"
    assert "session precedente" in vivant.read_text(encoding="utf-8"), "the neighbouring log is intact"
    registre.stop(job.id)


def test_deux_jobs_de_la_meme_cible_ont_des_journaux_distincts(monkeypatch, tmp_path):
    registre = _registre(monkeypatch, tmp_path)
    a = registre.launch("root:experience-lancer", ["make", "experience-lancer"], tmp_path)
    b = registre.launch("root:experience-lancer", ["make", "experience-lancer"], tmp_path)
    assert a.log_path != b.log_path and a.id != b.id
    for j in (a, b):
        registre.stop(j.id)
