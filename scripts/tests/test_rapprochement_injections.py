"""Tests of the reconciliation of declared vs produced injections — Ticket 108.

Checks that:
1. A run in which all declared injections took place says so explicitly.
2. A run missing an injection raises an [ALARME] naming agent, event and declared day.
3. The plausible reason is identified and named (mode not chosen after shock, idle agent, interrupted run).
4. The report renders the reconciliation in both directions (successes and failures).
5. Intra-household relays `origine: entendu` (Ticket 111) do not count as produced exposures.
6. The real archives `2026-09-23_20_35` and `2026-09-24_00_15` are correctly diagnosed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from scripts.analysis import rapprochement_injections as ri

RACINE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_report", RACINE / "scripts" / "debug" / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)


ARCHIVE_2035 = RACINE / "experiments" / "archive" / "simulations_septembre_2026" / "2026-09-23_20_35"
ARCHIVE_0015 = RACINE / "experiments" / "archive" / "simulations_septembre_2026" / "2026-09-24_00_15"


class TestArchivesReelles:
    @pytest.mark.skipif(not ARCHIVE_2035.is_dir(), reason="Archive 2026-09-23_20_35 missing")
    def test_recette_run_2026_09_23_20_35_signale_jour_16_manquant(self):
        """The founding run of ticket 108: the agent avoids the car after the J15 shock."""
        r = ri.rapprocher(ARCHIVE_2035)
        assert r.declarees == 2
        assert r.produites == 1
        assert r.manquantes == 1
        assert r.conforme is False

        l15 = next(l for l in r.lignes if l.jour_declare == 15)
        assert l15.est_produite is True
        assert l15.person_id == "861500"

        l16 = next(l for l in r.lignes if l.jour_declare == 16)
        assert l16.est_produite is False
        assert l16.person_id == "861500"
        assert "car" in l16.raison or "voiture" in l16.raison
        assert "J15" in l16.raison

        # Checking the alarm
        assert len(r.alarmes) == 1
        alarme = r.alarmes[0]
        assert "861500" in alarme
        assert "c6_voiture_suspecte" in alarme
        assert "jour 16" in alarme
        assert "2026-03-31" in alarme

    @pytest.mark.skipif(not ARCHIVE_0015.is_dir(), reason="Archive 2026-09-24_00_15 missing")
    def test_recette_run_2026_09_24_00_15_signale_jour_13_manquant(self):
        """The second occurrence: the agent does not travel on day 13 (Sat 28 March)."""
        r = ri.rapprocher(ARCHIVE_0015)
        assert r.declarees == 2
        assert r.produites == 1
        assert r.manquantes == 1
        assert r.conforme is False

        l13 = next(l for l in r.lignes if l.jour_declare == 13)
        assert l13.est_produite is False
        assert l13.person_id == "861500"
        assert "déplacé" in l13.raison or "trajet" in l13.raison

        assert len(r.alarmes) == 1
        alarme = r.alarmes[0]
        assert "861500" in alarme
        assert "c3_panne_reseau" in alarme
        assert "jour 13" in alarme


class TestSynthetiques:
    def test_run_nominal_toutes_injections_produites(self, tmp_path):
        """A run where all injections took place is compliant and says so without alarm."""
        (tmp_path / "evenement.yaml").write_text(
            yaml.dump({
                "evenement": "choc_test",
                "canal": "vecu",
                "exposition": {"regle": "agents", "agents": ["agent_1"], "modes": ["car"]},
                "jours": [{"jour": 1, "retard_min": 30}],
            }),
            encoding="utf-8",
        )
        (tmp_path / "evenements.jsonl").write_text(
            json.dumps({"person_id": "agent_1", "jour_run": 1, "retard_injecte_s": 1800}) + "\n",
            encoding="utf-8",
        )

        r = ri.rapprocher(tmp_path)
        assert r.declarees == 1
        assert r.produites == 1
        assert r.manquantes == 0
        assert r.conforme is True
        assert not r.alarmes

        rendu = "\n".join(ri.rendre(r))
        assert "✅ produite" in rendu
        assert "1/1 injection(s) déclarée(s) produite(s)" in rendu

    def test_les_informes_origine_entendu_ne_comptent_pas(self, tmp_path):
        """Ticket 111: `origine: entendu` rows are relays, not exposures."""
        (tmp_path / "evenement.yaml").write_text(
            yaml.dump({
                "evenement": "choc_foyer",
                "canal": "lu",
                "exposition": {"regle": "agents", "agents": ["lecteur_1"]},
                "calendrier": {"fenetre": [1, 3]},
            }),
            encoding="utf-8",
        )
        lignes = [
            {"person_id": "lecteur_1", "jour_run": 1},
            {"person_id": "membre_2", "jour_run": 1, "origine": "entendu"},
        ]
        (tmp_path / "evenements.jsonl").write_text(
            "".join(json.dumps(l) + "\n" for l in lignes),
            encoding="utf-8",
        )

        r = ri.rapprocher(tmp_path)
        # Only lecteur_1 is counted as a produced exposure
        assert r.declarees == 1
        assert r.produites == 1
        assert r.manquantes == 0
        assert r.conforme is True

    def test_run_interrompu_avant_jour_declare(self, tmp_path):
        """If the simulation stops before the declared day, the reason names it explicitly."""
        (tmp_path / "evenement.yaml").write_text(
            yaml.dump({
                "evenement": "choc_test",
                "exposition": {"regle": "agents", "agents": ["agent_1"], "modes": ["car"]},
                "jours": [{"jour": 10, "retard_min": 15}],
            }),
            encoding="utf-8",
        )
        # moves.csv only goes up to day 3 (2026-03-18)
        (tmp_path / "moves.csv").write_text(
            "ID Personne,Heure de départ,Mode de transport Choisi\n"
            "agent_1,2026-03-16 08:00:00,Voiture Privée\n"
            "agent_1,2026-03-18 18:00:00,Voiture Privée\n",
            encoding="utf-8",
        )

        r = ri.rapprocher(tmp_path)
        assert r.manquantes == 1
        assert "arrêté avant ce jour" in r.lignes[0].raison

    def test_agent_sans_deplacement_le_jour_declare(self, tmp_path):
        """If the agent does not move on the declared day, the reason names the absence of a trip."""
        (tmp_path / "evenement.yaml").write_text(
            yaml.dump({
                "evenement": "choc_test",
                "exposition": {"regle": "agents", "agents": ["agent_1"], "modes": ["car"]},
                "jours": [{"jour": 2, "retard_min": 15}],
            }),
            encoding="utf-8",
        )
        # Trips on day 1 (16 March) and day 3 (18 March), nothing on day 2 (17 March)
        (tmp_path / "moves.csv").write_text(
            "ID Personne,Heure de départ,Mode de transport Choisi\n"
            "agent_1,2026-03-16 08:00:00,Voiture Privée\n"
            "agent_1,2026-03-18 18:00:00,Voiture Privée\n",
            encoding="utf-8",
        )

        r = ri.rapprocher(tmp_path)
        assert r.manquantes == 1
        assert "ne s'est pas déplacé" in r.lignes[0].raison or "aucun trajet" in r.lignes[0].raison

    def test_run_sans_evenement_le_dit(self, tmp_path):
        """A run with no declared event reports `rien à rapprocher`."""
        r = ri.rapprocher(tmp_path)
        assert r.aucun_evenement is True
        rendu = "\n".join(ri.rendre(r))
        assert "rien à rapprocher" in rendu


class TestIntegrationRunReport:
    def test_section_run_report_integree(self, tmp_path):
        """The reconciliation section in run_report.py fills the Markdown and the alarms."""
        (tmp_path / "evenement.yaml").write_text(
            yaml.dump({
                "evenement": "c_manquant",
                "exposition": {"regle": "agents", "agents": ["a1"], "modes": ["car"]},
                "jours": [{"jour": 5, "retard_min": 20}],
            }),
            encoding="utf-8",
        )
        out: list[str] = []
        alarms: list[str] = []

        run_report.section_rapprochement_injections(tmp_path, out, alarms)
        texte = "\n".join(out)
        assert "Rapprochement des injections" in texte
        assert "🔴 **manquante**" in texte
        assert any("INJECTION MANQUANTE" in a for a in alarms)
        assert any("a1" in a for a in alarms)
