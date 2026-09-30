"""The papers repo is found without guessing, and a script never writes to a missing repo.

Ticket 115: paper writing lives in a neighbouring private repo, designated by `PAPER_DIR`.
"""
import logging
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts import depot_papiers  # noqa: E402
from scripts.depot_papiers import (  # noqa: E402
    chemin_papier, exiger_depot_papiers, paper_dir, racine_depot_principal)


def test_la_variable_prime(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "papiers"))
    assert paper_dir() == (tmp_path / "papiers").resolve()
    assert chemin_papier("figures", "x.png") == (tmp_path / "papiers" / "figures" / "x.png").resolve()


def test_sans_variable_le_voisin_du_depot_principal(monkeypatch):
    monkeypatch.delenv("PAPER_DIR", raising=False)
    attendu = racine_depot_principal().parent / "llm-agents-gama-papiers"
    assert paper_dir() == attendu.resolve()


def test_un_worktree_remonte_au_depot_principal(tmp_path):
    principal = tmp_path / "depot"
    (principal / ".git" / "worktrees" / "t115").mkdir(parents=True)
    arbre = principal / ".claude" / "worktrees" / "t115"
    arbre.mkdir(parents=True)
    (arbre / ".git").write_text(f"gitdir: {principal}/.git/worktrees/t115\n", encoding="utf-8")
    assert racine_depot_principal(arbre) == principal


def test_un_depot_ordinaire_reste_lui_meme(tmp_path):
    (tmp_path / ".git").mkdir()
    assert racine_depot_principal(tmp_path) == tmp_path


def test_un_fichier_git_illisible_ne_devine_pas(tmp_path):
    (tmp_path / ".git").write_text("n'importe quoi", encoding="utf-8")
    assert racine_depot_principal(tmp_path) == tmp_path


def test_ecrire_dans_un_depot_absent_arrete_et_alarme(tmp_path, monkeypatch, caplog):
    absent = tmp_path / "absent"
    monkeypatch.setenv("PAPER_DIR", str(absent))
    with caplog.at_level(logging.ERROR, logger="depot_papiers"), pytest.raises(SystemExit):
        exiger_depot_papiers(absent / "figures" / "x.png")
    message = caplog.records[-1].getMessage()
    assert "[ALARME]" in message and "PAPER_DIR" in message and str(absent) in message
    assert not absent.exists(), "the check must create nothing"


def test_une_sortie_hors_du_depot_papiers_passe_toujours(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "absent"))
    exiger_depot_papiers(tmp_path / "docs" / "synthesis" / "x.png")  # does not raise


def test_un_depot_present_journalise_son_succes(tmp_path, monkeypatch, caplog):
    (tmp_path / "papiers").mkdir()
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "papiers"))
    with caplog.at_level(logging.INFO, logger="depot_papiers"):
        exiger_depot_papiers(tmp_path / "papiers" / "figures" / "x.png")
    assert any("Dépôt papiers trouvé" in r.getMessage() for r in caplog.records)


def test_le_module_n_ecrit_rien_a_l_import():
    """Imported by tests without a papers repo: no constant may raise."""
    assert depot_papiers.NOM_PAR_DEFAUT == "llm-agents-gama-papiers"
