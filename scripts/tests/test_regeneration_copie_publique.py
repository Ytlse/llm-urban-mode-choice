"""Every figure regenerates from the public copy, without changing anything in the working repo.

Ticket 113. The public copy files runs under `archive/1_regime_nominal/`, has no
papers repository, and only cites traces with `--avec-traces-citees`. Each fix has
two sides, and both are tested here: the working repository's behaviour is unchanged,
and the copy finds its inputs and writes its outputs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
if str(RACINE / "services" / "llm-agents") not in sys.path:
    sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

from experiences import experience as E  # noqa: E402

from scripts import depot_papiers  # noqa: E402


def _module(rel: str):
    """Imports a script by its path (the script folders are not packages)."""
    chemin = RACINE / rel
    spec = importlib.util.spec_from_file_location(chemin.stem, chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experience(racine: Path, *parties: str, execution: str | None = None) -> Path:
    dossier = racine.joinpath(*parties)
    dossier.mkdir(parents=True)
    (dossier / "experience.yaml").write_text("nom: x\n", encoding="utf-8")
    if execution:
        ex = dossier / "executions" / execution
        ex.mkdir(parents=True)
        (ex / "execution.yaml").write_text("statut: termine\n", encoding="utf-8")
    return dossier


# ── Outputs: the papers repository, otherwise outputs/figures/ ────────────────────────

def test_sortie_dans_le_depot_papiers_quand_il_existe(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPER_DIR", raising=False)
    (tmp_path / "papiers").mkdir()
    monkeypatch.setattr(depot_papiers, "paper_dir", lambda: tmp_path / "papiers")
    assert depot_papiers.sortie_papier("figures") == tmp_path / "papiers" / "figures"


def test_sortie_repliee_sans_depot_papiers_ni_variable(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("PAPER_DIR", raising=False)
    monkeypatch.setattr(depot_papiers, "paper_dir", lambda: tmp_path / "absent")
    with caplog.at_level("WARNING", logger="depot_papiers"):
        dossier = depot_papiers.sortie_papier("figures")
        fichier = depot_papiers.sortie_papier("article-court", "appendices", "images", "H1.png")
    assert dossier == depot_papiers.SORTIE_LOCALE
    assert fichier == depot_papiers.SORTIE_LOCALE / "H1.png"
    assert "sortie repliée" in caplog.text
    assert not (tmp_path / "absent").exists()
    # The fallback is outside the papers repository: `exiger_depot_papiers` lets it through.
    depot_papiers.exiger_depot_papiers(fichier)


def test_une_variable_definie_vers_un_dossier_absent_garde_l_alarme(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "absent"))
    sortie = depot_papiers.sortie_papier("figures")
    assert sortie == tmp_path / "absent" / "figures"
    with pytest.raises(SystemExit):
        depot_papiers.exiger_depot_papiers(sortie)


# ── Reading runs: data/experiences/, otherwise the archive ────────────────────────────

def test_une_racine_sous_archive_trouve_ses_experiences(tmp_path):
    racine = tmp_path / "archive" / "1_regime_nominal"
    attendu = _experience(racine, "jeu_1000", "experiences", "exp_a")
    assert E.trouver_dossier_experience("exp_a", racine=racine) == attendu


def test_une_archive_sous_la_racine_reste_ignoree(tmp_path):
    racine = tmp_path / "data" / "experiences"
    _experience(racine, "regime_nominal", "archive", "exp_b")
    assert E.trouver_dossier_experience("exp_b", racine=racine) is None


def test_la_racine_de_lecture_prefere_data_experiences_quand_il_a_des_executions(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPERIENCES_DIR", raising=False)
    monkeypatch.setattr(E, "_racine", lambda: tmp_path)
    _experience(tmp_path / "data" / "experiences", "regime_nominal", "exp_a", execution="2026-09-21")
    _experience(tmp_path / "archive" / "1_regime_nominal", "jeu", "experiences", "exp_a",
                execution="2026-09-21")
    assert E.racine_lecture_experiences() == tmp_path / "data" / "experiences"


def test_la_racine_de_lecture_se_replie_sur_l_archive(tmp_path, monkeypatch):
    """The public copy publishes `experience.yaml` files without runs: the fallback is decided on
    the runs, not on the definitions."""
    monkeypatch.delenv("EXPERIENCES_DIR", raising=False)
    monkeypatch.setattr(E, "_racine", lambda: tmp_path)
    _experience(tmp_path / "data" / "experiences", "regime_nominal", "exp_a")
    _experience(tmp_path / "archive" / "1_regime_nominal", "jeu", "experiences", "exp_a",
                execution="2026-09-21")
    assert E.racine_lecture_experiences() == tmp_path / "archive" / "1_regime_nominal"


def test_la_variable_experiences_dir_prime(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path / "ailleurs"))
    assert E.racine_lecture_experiences() == tmp_path / "ailleurs"


def test_les_ecritures_ne_visent_jamais_l_archive(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPERIENCES_DIR", raising=False)
    monkeypatch.setattr(E, "_racine", lambda: tmp_path)
    (tmp_path / "archive" / "1_regime_nominal").mkdir(parents=True)
    assert E.dossier_experiences() == tmp_path / "data" / "experiences"


def test_le_depot_de_travail_lit_toujours_data_experiences():
    """Unchanged behaviour: in this repository, `data/experiences/` has runs."""
    if not E.executions_vivantes(RACINE / "data" / "experiences"):
        pytest.skip("no run under data/experiences/ (public copy or fresh clone)")
    ch6 = _module("scripts/analysis/plot_decision_makers.py")
    assert ch6.DOSSIER_EXPERIENCES == RACINE / "data" / "experiences"


# ── Figures G.5 and H.1: explicit input ───────────────────────────────────────────────

def test_fig_g_garde_sa_trace_par_defaut():
    fig_g = _module("scripts/annexes/fig_G_confusion.py")
    assert fig_g.SOURCE == RACINE / "docs/traces/2026-09-22_lot0_entropie_support_unique/audit_12_decideurs.json"


def test_fig_g_sans_entree_alarme_et_n_ecrit_rien(tmp_path, caplog):
    fig_g = _module("scripts/annexes/fig_G_confusion.py")
    sortie = tmp_path / "G5.png"
    with caplog.at_level("ERROR"):
        code = fig_g.main(["--input", str(tmp_path / "absent.json"), "--sortie", str(sortie)])
    assert code == 1
    assert "[ALARME]" in caplog.text
    assert not sortie.exists()


def _paires_fig_h(module) -> dict:
    bloc = {"moy": -1.0, "sd": 0.5, "lo": -2.0, "hi": 0.5, "P": 0.1}
    return {cle: {"comp": bloc, "hcu": bloc, "l1": bloc}
            for _, lignes in module.GROUPES for cle, _ in lignes}


def test_fig_h_garde_sa_trace_par_defaut():
    fig_h = _module("scripts/annexes/fig_H_paired_forest.py")
    assert fig_h.SOURCE == RACINE / "docs/traces/2026-09-21_ticket096_lot2/paired_complet_B2000.json"


@pytest.mark.parametrize("format_", ["trace", "paired_intervals"])
def test_fig_h_lit_la_trace_et_la_sortie_de_paired_intervals(tmp_path, format_):
    fig_h = _module("scripts/annexes/fig_H_paired_forest.py")
    paires = _paires_fig_h(fig_h)
    donnees = paires if format_ == "trace" else {"preset": "chapter6", "replicates": 2000,
                                                 "pairs": paires, "arms": {}}
    entree = tmp_path / "paired.json"
    entree.write_text(json.dumps(donnees), encoding="utf-8")
    sortie = tmp_path / "sous" / "H1.png"
    assert fig_h.main(["--input", str(entree), "--sortie", str(sortie)]) == 0
    assert sortie.is_file() and sortie.stat().st_size > 0


def test_fig_h_une_paire_manquante_alarme(tmp_path, caplog):
    fig_h = _module("scripts/annexes/fig_H_paired_forest.py")
    paires = _paires_fig_h(fig_h)
    paires.pop(next(iter(paires)))
    entree = tmp_path / "paired.json"
    entree.write_text(json.dumps(paires), encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert fig_h.main(["--input", str(entree), "--sortie", str(tmp_path / "H1.png")]) == 1
    assert "[ALARME]" in caplog.text


# ── Memory experiments: the definitions go under config/ ──────────────────────────────

def test_le_constructeur_publie_les_definitions_memoire_sous_config():
    if not (RACINE / "scripts/publication/construire_depot_public.py").is_file():
        pytest.skip("the public-copy builder stays in the working repository")
    constructeur = _module("scripts/publication/construire_depot_public.py")
    assert constructeur.DEFINITIONS_MEMOIRE, "no memory experiment definition found"
    cibles = {cible for source, cible in constructeur.RELOCALISATIONS
              if source.startswith(constructeur.MEMOIRE_PRIVE)}
    assert len(cibles) == len(constructeur.DEFINITIONS_MEMOIRE)
    assert all(c.startswith("config/experiences_memoire/") and c.endswith("/experience_memoire.yaml")
               for c in cibles)
    # The experiments' state (etat.json, traite/) does not go out through the ordinary collection.
    assert constructeur.exclu(f"{constructeur.MEMOIRE_PRIVE}/chocs_reseau/x/etat.json")


def test_le_registre_liste_une_racine_posee_sous_archive(tmp_path):
    from experiences import registre
    racine = tmp_path / "archive" / "1_regime_nominal"
    _experience(racine, "jeu", "experiences", "exp_a", execution="2026-09-21")
    _experience(racine, "jeu", "experiences", "archive", "exp_vieille", execution="2026-01-01")
    lignes = registre.lister(racine)
    assert {ligne["experience"] for ligne in lignes} == {"x"}
    assert len(lignes) == 1
