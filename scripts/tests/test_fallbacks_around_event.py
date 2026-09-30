"""The report separates fallbacks BEFORE and AFTER the event.

A global rate says nothing. On 2026-09-23, the baseline of the c3 v4 campaign was
perfectly clean while the measurement window carried more than 10 % of decisions served
by the default index. Aggregated, the two gave a reassuring figure; it was the gap between
the two that made the arm unusable, and `moves.csv` had to be searched by hand
to see it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_report", RACINE / "scripts" / "debug" / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)

EVENEMENT = 1_774_620_168  # 2026-03-27 14:02, the real injection of the c3 v4 campaign
COLONNES = ["Temps simulé", "Mode de transport Choisi", "Méthode de sélection"]
REPLI = "LLM Error (Default index)"


def _run(tmp_path: Path, lignes: list[tuple[int, str]], avec_evenement: bool = True) -> Path:
    """`lignes`: (simulated instant, selection method)."""
    if avec_evenement:
        (tmp_path / "evenements.jsonl").write_text(
            json.dumps({"timestamp": EVENEMENT, "evenement_id": "c3_panne_reseau"}) + "\n",
            encoding="utf-8",
        )
    with (tmp_path / "moves.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES)
        w.writeheader()
        for instant, methode in lignes:
            w.writerow(
                {
                    "Temps simulé": instant,
                    "Mode de transport Choisi": "Train",
                    "Méthode de sélection": methode,
                }
            )
    return tmp_path


def _rendu(run: Path) -> tuple[str, list[str]]:
    out: list[str] = []
    alarms: list[str] = []
    run_report.section_replis_autour_evenement(run, out, alarms)
    return "\n".join(out), alarms


class TestLaSeparationEstFaite:
    def test_une_base_propre_et_une_fenetre_sale_se_distinguent(self, tmp_path):
        """THE CASE OF 2026-09-23, in miniature: 0 % before, 25 % after."""
        run = _run(
            tmp_path,
            [(EVENEMENT - 3600, "LLM")] * 4 + [(EVENEMENT + 3600, "LLM")] * 3 + [(EVENEMENT + 7200, REPLI)],
        )
        rendu, alarms = _rendu(run)

        assert "0/4 = 0.0%" in rendu, "the baseline must be reported clean"
        assert "1/4 = 25.0%" in rendu, "the measurement window must be reported dirty"
        assert any("FENÊTRE DE MESURE" in a for a in alarms)

    def test_l_evenement_est_nomme_et_date(self, tmp_path):
        run = _run(tmp_path, [(EVENEMENT + 60, "LLM")])
        rendu, _ = _rendu(run)

        assert "c3_panne_reseau" in rendu
        assert "2026-03-27" in rendu

    def test_un_repli_juste_avant_l_evenement_compte_avant(self, tmp_path):
        """The boundary is the injection instant, not the start of the day."""
        run = _run(tmp_path, [(EVENEMENT - 1, REPLI), (EVENEMENT, "LLM")])
        rendu, _ = _rendu(run)

        ligne_avant = next(l for l in rendu.splitlines() if l.startswith("| Avant"))
        ligne_apres = next(l for l in rendu.splitlines() if l.startswith("| Après"))
        assert "1/1 = 100.0%" in ligne_avant, ligne_avant
        assert "0/1 = 0.0%" in ligne_apres, ligne_apres


class TestLeDenominateur:
    def test_les_mono_choix_ne_comptent_pas(self, tmp_path):
        """A single-choice is not a decision the model could have missed."""
        run = _run(
            tmp_path,
            [(EVENEMENT + 60, "LLM"), (EVENEMENT + 120, "Un seul itinéraire disponible")],
        )
        rendu, alarms = _rendu(run)

        assert "| 1 |" in rendu, "only the LLM decision enters the denominator"
        assert not alarms

    def test_un_run_sans_repli_ne_leve_aucune_alarme(self, tmp_path):
        run = _run(tmp_path, [(EVENEMENT - 60, "LLM"), (EVENEMENT + 60, "LLM")])
        _, alarms = _rendu(run)

        assert alarms == []


class TestCeQuiNeCasseRien:
    def test_un_run_sans_evenement_ne_rend_aucune_section(self, tmp_path):
        """An ordinary run has no measurement window: the section has nothing to say."""
        run = _run(tmp_path, [(EVENEMENT + 60, REPLI)], avec_evenement=False)
        rendu, alarms = _rendu(run)

        assert rendu == ""
        assert alarms == []

    def test_un_run_sans_moves_ne_casse_pas(self, tmp_path):
        (tmp_path / "evenements.jsonl").write_text(
            json.dumps({"timestamp": EVENEMENT}) + "\n", encoding="utf-8"
        )
        rendu, alarms = _rendu(tmp_path)

        assert rendu == ""
        assert alarms == []

    def test_l_ancien_nom_chocs_jsonl_est_encore_lu(self, tmp_path):
        """Runs predating the event channel only have `chocs.jsonl`."""
        (tmp_path / "chocs.jsonl").write_text(
            json.dumps({"timestamp": EVENEMENT, "choc_id": "c3_panne_reseau"}) + "\n",
            encoding="utf-8",
        )
        run = _run(tmp_path, [(EVENEMENT + 60, "LLM")], avec_evenement=False)
        rendu, _ = _rendu(run)

        assert "c3_panne_reseau" in rendu
