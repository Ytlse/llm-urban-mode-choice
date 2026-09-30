"""`make report` says whether the injected memory reached long-term memory.

On 2026-09-23, the long-term memory base had to be opened by hand and searched for "metro" in
123 documents to discover that the v4 run's shock had never entered it. The report says so
now, and it ALSO says the cases where the memory did get through: a witness whose only visible
outputs are failures cannot tell "it never fires" apart from "it no longer runs".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_report", RACINE / "scripts" / "debug" / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)

PERDU = {
    "sim_ts": 1774620168,
    "sim_day": "2026-03-27",
    "person_id": "861500",
    "evenement_id": "c3_panne_reseau",
    "retrouve": False,
    "seuil": 2,
    "mots_cherches": ["lighting", "announcement", "carriage", "packed", "missed"],
    "mots_retrouves": ["missed"],
}
GARDE = {**PERDU, "retrouve": True, "mots_retrouves": ["lighting", "announcement", "carriage"]}


def _run(tmp_path: Path, lignes: list[dict]) -> Path:
    (tmp_path / "temoin_souvenir.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lignes), encoding="utf-8"
    )
    return tmp_path


def _rendu(run: Path) -> tuple[str, list[str]]:
    out: list[str] = []
    alarms: list[str] = []
    run_report.section_temoin_souvenir(run, out, alarms)
    return "\n".join(out), alarms


class TestLeVerdictSeVoit:
    def test_un_souvenir_perdu_leve_une_alarme_nommant_l_agent(self, tmp_path):
        """THE 2026-09-23 CASE, in miniature."""
        rendu, alarms = _rendu(_run(tmp_path, [PERDU]))

        assert "PERDU" in rendu
        assert len(alarms) == 1
        assert "861500" in alarms[0]
        assert "lighting" in alarms[0], "the alarm must say what was searched for"

    def test_un_souvenir_retrouve_se_voit_sans_lever_d_alarme(self, tmp_path):
        rendu, alarms = _rendu(_run(tmp_path, [GARDE]))

        assert "retrouvé" in rendu
        assert alarms == []

    def test_le_compte_des_deux_verdicts_est_rendu(self, tmp_path):
        rendu, alarms = _rendu(_run(tmp_path, [GARDE, PERDU, GARDE]))

        assert "3 consolidation(s) contrôlée(s)" in rendu
        assert "1 sans trace du souvenir" in rendu
        assert len(alarms) == 1


class TestCeQuiNeCasseRien:
    def test_un_run_sans_injection_ne_rend_aucune_section(self, tmp_path):
        """No event declared: the file does not exist, the section has nothing to say."""
        rendu, alarms = _rendu(tmp_path)

        assert rendu == ""
        assert alarms == []

    def test_un_fichier_vide_ne_rend_aucune_section(self, tmp_path):
        rendu, alarms = _rendu(_run(tmp_path, []))

        assert rendu == ""
        assert alarms == []

    def test_une_ligne_illisible_n_empeche_pas_les_autres(self, tmp_path):
        """A trace truncated by an abrupt stop must not make the report disappear."""
        chemin = tmp_path / "temoin_souvenir.jsonl"
        chemin.write_text(
            json.dumps(PERDU) + "\n{tronqu\n" + json.dumps(GARDE) + "\n", encoding="utf-8"
        )
        rendu, alarms = _rendu(tmp_path)

        assert "2 consolidation(s) contrôlée(s)" in rendu
        assert len(alarms) == 1


class TestLaSectionEstBranchee:
    def test_le_rapport_appelle_la_section(self):
        source = (RACINE / "scripts/debug/run_report.py").read_text(encoding="utf-8")
        assert "    section_temoin_souvenir(run, out, alarms)" in source
