"""Analysis scripts read an experiment where it lives: in its family (2026-09-28).

Experiments are filed by family (`data/experiences/regime_nominal/<jeu>/<exp>/`).
Two scripts still built the flat path `DOSSIER_EXPERIENCES / <exp>`, which does not exist
for a filed experiment:

- `plot_decision_makers.py` read "0 decision-makers out of 13", produced no figure, and its alarm
  asked to replay decision-makers whose thirteen `scores.json` were on disk;
- `plot_experiences.py` declared every filed experiment as "folder missing";

The folder is resolved by `trouver_dossier_experience` (`experiences` package), like the campaign
and the dashboard: a second resolver would drift from the first.

Two other scripts had the same defect, fixed the same day:

- `audit_unitaire_058.py` returned "no result" on a set whose ten arms are scored;
- `lancer_experience_rf.py` did not see a filed experiment and duplicated it again, which
  rewrites its `experience.yaml` with `executions_connues: []`.

An enumeration of experiments follows the same exclusion rule as the resolver: neither `archive/`
nor `.system_generated/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import experiences.cli as CLI

from scripts.analysis import plot_decision_makers as CH6
from scripts.analysis import plot_experiences as PE
from scripts.progedo_logit import audit_unitaire_058 as AUD
from scripts.progedo_logit import lancer_experience_rf as RF

FAMILLE = Path("regime_nominal") / "jeu_x"


@pytest.fixture
def experiences(tmp_path, monkeypatch):
    racine = tmp_path / "data" / "experiences"
    racine.mkdir(parents=True)
    for module in (CH6, PE):
        monkeypatch.setattr(module, "DOSSIER_EXPERIENCES", racine)
    # The rf launcher derives its path from `RACINE`: we redirect the whole
    # root, so that even the code from before the fix never writes into the repository.
    monkeypatch.setattr(RF, "RACINE", tmp_path)
    monkeypatch.setattr(RF, "DOSSIER_EXPERIENCES", racine, raising=False)
    return racine


def _definir(dossier: Path, nom: str, **champs) -> Path:
    exp = dossier / nom
    exp.mkdir(parents=True, exist_ok=True)
    doc = {"nom": nom, **champs}
    (exp / "experience.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    return exp


def _scorer(exp: Path, execution: str, emd_jsd: float = 4.2, detail: dict | None = None) -> None:
    dossier = exp / "executions" / execution
    dossier.mkdir(parents=True)
    scores = {
        "composite": {"emd_jsd": emd_jsd, "l1": 12.5},
        "couverture": {"taux": 1.0},
        "global": {"n_agents": 1000},
    }
    if detail is not None:
        scores["detail"] = detail
    (dossier / "scores.json").write_text(json.dumps(scores), encoding="utf-8")


# ── plot_decision_makers ───────────────────────────────────────────────────────────


class TestChapitre6:
    def test_le_score_est_lu_dans_la_famille(self, experiences):
        exp = _definir(experiences / FAMILLE, "exp_alea")
        _scorer(exp, "2026-09-21_06_00_00", emd_jsd=7.25)
        score = CH6.dernier_score("exp_alea")
        assert score is not None, "an experiment filed in a family must be readable"
        assert score["_execution"] == "2026-09-21_06_00_00"
        assert score["composite"]["emd_jsd"] == 7.25

    def test_l_epingle_est_respectee_dans_la_famille(self, experiences, monkeypatch):
        exp = _definir(experiences / FAMILLE, "exp_jev")
        _scorer(exp, "2026-09-21_06_25_29", emd_jsd=4.19)
        _scorer(exp, "2026-09-22_10_00_00", emd_jsd=4.14)
        monkeypatch.setattr(CH6, "EXECUTIONS_FIGEES", {"exp_jev": "2026-09-21_06_25_29"})
        score = CH6.dernier_score("exp_jev")
        assert score is not None
        assert score["_execution"] == "2026-09-21_06_25_29"
        assert score["composite"]["emd_jsd"] == 4.19

    def test_une_experience_a_plat_se_lit_toujours(self, experiences):
        exp = _definir(experiences, "exp_plate")
        _scorer(exp, "2026-09-20_00_00_00")
        assert CH6.dernier_score("exp_plate")["_execution"] == "2026-09-20_00_00_00"

    def test_une_experience_introuvable_rend_none(self, experiences):
        assert CH6.dernier_score("exp_fantome") is None


# ── plot_experiences ─────────────────────────────────────────────────────────


class TestPlotExperiences:
    def test_une_experience_rangee_se_lit(self, experiences):
        exp = _definir(experiences / FAMILLE, "exp_lgbm", decideur={"type": "lightgbm"})
        _scorer(exp, "2026-09-21_06_00_00", emd_jsd=5.5)
        point = PE.lire_experience("exp_lgbm")
        assert point["execution"] == "2026-09-21_06_00_00"
        assert point["emd_jsd"] == 5.5
        assert point["est_reference"] is True

    def test_une_experience_introuvable_le_dit(self, experiences):
        with pytest.raises(PE.ExperienceIllisible, match="exp_fantome"):
            PE.lire_experience("exp_fantome")


def _executer(exp: Path, execution: str, **fichiers: dict) -> Path:
    dossier = exp / "executions" / execution
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, contenu in fichiers.items():
        (dossier / f"{nom}.json").write_text(json.dumps(contenu), encoding="utf-8")
    return dossier


# ── audit_unitaire_058 ───────────────────────────────────────────────────────


class TestAudit058:
    JEU = "enquete_058_test_20260316"

    def test_les_experiences_rangees_du_jeu_sont_listees(self, experiences):
        famille = experiences / "regime_nominal" / "jeu_enquete_058_test"
        _definir(famille, "exp_lgbm_058", jeu={"nom": self.JEU})
        _definir(famille, "exp_alea_058", jeu={"nom": self.JEU})
        _definir(experiences / FAMILLE, "exp_lgbm_v6", jeu={"nom": "autre_jeu"})
        _definir(experiences / "archive" / "v1", "exp_vieille_058", jeu={"nom": self.JEU})
        _definir(famille, "brouillon_058", jeu={"nom": self.JEU})
        assert AUD.experiences_du_jeu(experiences, self.JEU) == ["exp_alea_058", "exp_lgbm_058"]

    def test_la_derniere_execution_est_lue_dans_la_famille(self, experiences):
        exp = _definir(experiences / FAMILLE, "exp_lgbm_058")
        _executer(exp, "2026-09-20_08_00_00")
        derniere = _executer(exp, "2026-09-21_08_00_00")
        assert AUD.derniere_execution_de("exp_lgbm_058", experiences) == derniere

    def test_une_experience_introuvable_n_a_pas_d_execution(self, experiences):
        assert AUD.derniere_execution_de("exp_fantome", experiences) is None


# ── lancer_experience_rf ─────────────────────────────────────────────────────


class TestLanceurRf:
    @pytest.fixture
    def cli(self, monkeypatch):
        appels: list[list[str]] = []
        monkeypatch.setattr(CLI, "main", lambda argv: appels.append(argv) or 0)
        return appels

    @staticmethod
    def _lien_ticket(dossier: Path) -> Path:
        texte = (dossier / "LISEZ-MOI.md").read_text(encoding="utf-8")
        lien = texte.split("](", 1)[1].split(")", 1)[0]
        return (dossier / lien).resolve()

    def test_une_experience_rangee_n_est_pas_redupliquee(self, experiences, cli, tmp_path):
        """Duplicating it again would rewrite its experience.yaml with `executions_connues: []`."""
        exp = _definir(experiences / FAMILLE, "exp_rf_x")
        assert RF.definir("exp_rf_x") == 0
        assert not [a for a in cli if a and a[0] == "dupliquer"]
        assert (exp / "LISEZ-MOI.md").is_file()
        assert not (experiences / "exp_rf_x").exists(), "no orphan flat folder"

    def test_le_lien_du_lisez_moi_vit_depuis_la_famille(self, experiences, cli, tmp_path):
        exp = _definir(experiences / FAMILLE, "exp_rf_x")
        RF.definir("exp_rf_x")
        ticket = tmp_path / "docs" / "tickets" / "ticket_044_temoin_random_forest.md"
        assert self._lien_ticket(exp) == ticket.resolve()

    def test_la_copie_est_suivie_la_ou_la_cli_l_a_ecrite(self, experiences, monkeypatch):
        cible = experiences / FAMILLE / "exp_rf_neuf"

        def dupliquer(argv):
            if argv[0] == "dupliquer":
                _definir(cible.parent, cible.name)
            return 0

        monkeypatch.setattr(CLI, "main", dupliquer)
        assert RF.definir("exp_rf_neuf") == 0
        assert (cible / "LISEZ-MOI.md").is_file()

    def test_une_copie_introuvable_est_un_echec(self, experiences, cli):
        assert RF.definir("exp_rf_nulle_part") == 1
