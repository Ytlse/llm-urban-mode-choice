"""Concept-operation trace, Prometheus series, wiring.

Families D (replay freeze), G (series) and H (activation, safety).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from llm import trace_concepts
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from scripts.analysis.mesures.prometheus import PREFIXE, _familles, publier
from settings import settings
from urban_mobility_agents.utils import mesures_jour, reprise

QUAND = datetime(2026, 3, 16, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A throwaway workdir, trace on, test guard lifted."""
    monkeypatch.setattr(settings.agent, "trace_concepts_enabled", True, raising=False)
    monkeypatch.setattr(settings.app, "trace_concepts_file",
                        str(tmp_path / "operations_concept.jsonl"), raising=False)
    monkeypatch.setattr(trace_concepts, "_run_artifacts_disabled", lambda: False)
    reprise.degeler_si_depasse(0)
    yield tmp_path


def _lignes(workdir: Path) -> list[dict]:
    fichier = workdir / "operations_concept.jsonl"
    if not fichier.is_file():
        return []
    return [json.loads(l) for l in fichier.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestTraceDesConcepts:
    def test_une_operation_est_tracee_avec_son_jour_et_son_agent(self, workdir):
        trace_concepts.tracer_operation("899549", "contredit", QUAND, doc_id="899549_7")
        ligne, = _lignes(workdir)
        assert ligne["person_id"] == "899549"
        assert ligne["operation"] == "contredit"
        assert ligne["sim_day"] == "2026-03-16"
        assert ligne["doc_id"] == "899549_7"

    def test_D5_pendant_le_gel_du_rejeu_rien_n_est_trace(self, workdir):
        """A replayed operation is not an operation: memory is frozen."""
        reprise.geler_jusqu_a(int(QUAND.timestamp()) + 86400, 2)
        try:
            trace_concepts.tracer_operation("899549", "créé", QUAND)
            assert _lignes(workdir) == []
        finally:
            reprise.degeler_si_depasse(int(QUAND.timestamp()) + 172800)
        # Unfrozen, the trace resumes: the freeze suspends, it does not break.
        trace_concepts.tracer_operation("899549", "créé", QUAND)
        assert len(_lignes(workdir)) == 1

    def test_H1_eteinte_par_defaut_aucun_fichier_n_est_ecrit(self, workdir, monkeypatch):
        monkeypatch.setattr(settings.agent, "trace_concepts_enabled", False, raising=False)
        trace_concepts.tracer_operation("899549", "créé", QUAND)
        assert not (workdir / "operations_concept.jsonl").exists()

    def test_H3_une_erreur_d_ecriture_ne_remonte_jamais(self, workdir, monkeypatch):
        monkeypatch.setattr(settings.app, "trace_concepts_file",
                            str(workdir / "introuvable" / "x.jsonl"), raising=False)
        trace_concepts.tracer_operation("899549", "créé", QUAND)  # does not raise

    def test_un_instant_absent_ne_fait_pas_perdre_la_ligne(self, workdir):
        trace_concepts.tracer_operation("899549", "créé", None)
        ligne, = _lignes(workdir)
        assert ligne["sim_ts"] is None and ligne["sim_day"] is None
        assert ligne["operation"] == "créé"

    def test_le_vocabulaire_des_quatre_operations_est_le_MEME_partout(self):
        """Three places name the operations: the trace, the measures, and the reader of the
        `.md` log. A diverging vocabulary would file half of the operations under "out of
        vocabulary" with nothing to flag it — and the contradictions curve, which is the
        very mechanism of the study, would drop to zero while looking right.
        """
        import re

        from scripts.analysis.memoire.sources import _OPERATION
        from scripts.analysis.mesures.calcul import OPERATIONS_CONCEPT

        assert set(trace_concepts.OPERATIONS) == set(OPERATIONS_CONCEPT)
        for operation in trace_concepts.OPERATIONS:
            assert _OPERATION.search(f"opération **{operation}**"), operation

        # And these are indeed the strings that the consolidation chain passes.
        agent = (Path(__file__).resolve().parents[1] / "urban_mobility_agents" / "agents"
                 / "llm_agent.py").read_text(encoding="utf-8")
        # One site traces two operations from a single expression — `confirmé` or `précisé`
        # depending on the operation read — hence reading per call rather than per literal.
        tracees = set()
        for appel in re.finditer(r"tracer_operation\(", agent):
            fenetre = agent[appel.end():appel.end() + 300]
            tracees |= set(re.findall(r'"(créé|confirmé|précisé|contredit)"', fenetre))
        assert tracees == set(trace_concepts.OPERATIONS), tracees


class TestSeries:
    def _mesures(self, tmp_path):
        from test_daily_measures import _point, _run, _souvenir, _t

        return _run(
            tmp_path,
            [_t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="a"),
             _t("1", "2026-03-17", "08:00:00", "Voiture Privée", activite="a"),
             _t("1", "2026-03-18", "08:00:00", "Marche", activite="a")],
            rappels=[{"sim_ts": 1773637201, "sim_day": "2026-03-17", "person_id": "1",
                      "candidats": 7, "servis": [{"doc_id": "1_0"}]}],
            operations=[{"sim_day": "2026-03-17", "person_id": "1", "operation": "contredit"}],
            points={3: ("2026-03-18T03:00:00", {"1": [_souvenir("1_0", "concept", 11.0)]})},
        )

    def _publier(self, tmp_path):
        from scripts.analysis.mesures import calculer

        registre = CollectorRegistry()
        gauges = _familles(lambda n, a, l: Gauge(n, a, l, registry=registre))
        jour = publier(calculer(self._mesures(tmp_path)), gauges)
        return jour, generate_latest(registre).decode()

    def test_G1_les_series_du_dernier_jour_CLOS_sont_publiees(self, tmp_path):
        """The current day would see its shares rise as departures come in."""
        jour, sortie = self._publier(tmp_path)
        assert jour == 2
        assert f"{PREFIXE}_dernier_jour_clos 2.0" in sortie
        assert f'{PREFIXE}_jour_trajets{{person_id="1"}} 1.0' in sortie
        assert f'{PREFIXE}_reprise_veille{{person_id="1"}} 1.0' in sortie
        assert f'{PREFIXE}_vivier_rappel{{person_id="1"}} 7.0' in sortie
        assert f'{PREFIXE}_operations_concept{{operation="contredit",person_id="1"}} 1.0' in sortie

    def test_G2_toutes_les_familles_sont_des_gauges(self, tmp_path):
        """An incremented Counter would double up on the replay of a resume."""
        registre = CollectorRegistry()
        _familles(lambda n, a, l: Gauge(n, a, l, registry=registre))
        types = {m.type for m in registre.collect()}
        assert types == {"gauge"}

    def test_G1bis_une_mesure_absente_n_est_pas_publiee_a_zero(self, tmp_path):
        from test_daily_measures import _run, _t

        from scripts.analysis.mesures import calculer

        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche", activite="a"),
                              _t("1", "2026-03-17", "08:00:00", "Marche", activite="b")])
        registre = CollectorRegistry()
        gauges = _familles(lambda n, a, l: Gauge(n, a, l, registry=registre))
        publier(calculer(run), gauges)
        sortie = generate_latest(registre).decode()
        # No activity of day 1 has an antecedent: the series must not exist at 0.
        assert f"{PREFIXE}_reprise_veille{{" not in sortie
        assert f"{PREFIXE}_conformite_habitude{{" not in sortie

    def test_un_run_d_un_seul_jour_ne_publie_rien(self, tmp_path):
        from test_daily_measures import _run, _t

        from scripts.analysis.mesures import calculer

        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")])
        registre = CollectorRegistry()
        gauges = _familles(lambda n, a, l: Gauge(n, a, l, registry=registre))
        assert publier(calculer(run), gauges) == 0


class TestBranchement:
    def test_H1_eteint_le_module_ne_lit_ni_n_ecrit_rien(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings.agent, "mesures_jour_enabled", False, raising=False)
        assert mesures_jour.actif() is False
        assert mesures_jour.ecrire_mesures_du_jour(tmp_path) == 0
        assert not (tmp_path / "mesures").exists()

    def test_H3_un_workdir_illisible_leve_une_alarme_et_rend_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings.agent, "mesures_jour_enabled", True, raising=False)
        monkeypatch.setattr(mesures_jour, "_run_artifacts_disabled", lambda: False)
        assert mesures_jour.ecrire_mesures_du_jour(tmp_path / "nexiste_pas") == 0

    def test_les_csv_et_les_series_sortent_du_MEME_calcul(self, tmp_path, monkeypatch):
        from test_daily_measures import _run, _t

        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="a"),
            _t("1", "2026-03-17", "08:00:00", "Voiture Privée", activite="a"),
            _t("1", "2026-03-18", "08:00:00", "Voiture Privée", activite="a"),
        ])
        monkeypatch.setattr(settings.agent, "mesures_jour_enabled", True, raising=False)
        monkeypatch.setattr(settings.app, "mesures_dir", "mesures", raising=False)
        monkeypatch.setattr(mesures_jour, "_run_artifacts_disabled", lambda: False)
        mesures_jour.reinitialiser()
        jour = mesures_jour.ecrire_mesures_du_jour(run)
        assert jour == 2
        csv_ = (run / "mesures" / "choix_modal_par_jour.csv").read_text(encoding="utf-8")
        assert "2,2026-03-17,1,1,1,1,1" in csv_.replace(" ", "")
        mesures_jour.reinitialiser()


class TestTableauDeBord:
    """Contract: family G. The dashboard must not drift from the code that feeds it."""

    CHEMIN = (Path(__file__).resolve().parents[3] / "infra" / "grafana" / "dashboards"
              / "09_memoire_personas.json")

    def _tableau(self):
        return json.loads(self.CHEMIN.read_text(encoding="utf-8"))

    def _familles_exposees(self) -> set[str]:
        registre = CollectorRegistry()
        _familles(lambda n, a, l: Gauge(n, a, l, registry=registre))
        return {m.name for m in registre.collect()}

    def test_le_tableau_est_un_json_valide_et_provisionne(self):
        tableau = self._tableau()
        assert tableau["uid"] == "memoire-personas"
        assert "sim" in tableau["tags"], "without this tag, it leaves the simulation dashboards menu"
        assert tableau["panels"]

    def test_G3_toute_famille_citee_est_reellement_exposee_par_le_code(self):
        """A query citing a vanished family draws a flat curve, not an error.

        It is the worst of both worlds: the dashboard looks like it works.
        """
        import re

        exposees = self._familles_exposees()
        citees = set()
        for panneau in self._tableau()["panels"]:
            for cible in panneau.get("targets", []):
                citees |= set(re.findall(rf"{PREFIXE}_[a-z_]+", cible["expr"]))
        assert citees, "no query in the dashboard: it shows nothing"
        assert citees <= exposees, citees - exposees

    def test_G4_les_couleurs_de_modes_suivent_la_palette_officielle(self):
        """Repository palette: car red, bike purple, public transport green, walking cyan."""
        attendues = {"car": "red", "cycling": "purple", "public_transport": "green",
                     "walking": "#00CCCC"}
        surcharges = {}
        for panneau in self._tableau()["panels"]:
            for surcharge in panneau.get("fieldConfig", {}).get("overrides", []):
                motif = surcharge["matcher"]["options"]
                for propriete in surcharge["properties"]:
                    if propriete["id"] == "color":
                        surcharges[motif] = propriete["value"]["fixedColor"]
        for mode, couleur in attendues.items():
            assert surcharges.get(f".*{mode}.*") == couleur, mode

    def test_G5_le_tableau_dit_que_son_axe_est_le_temps_reel_et_que_le_CSV_fait_foi(self):
        textes = " ".join(
            panneau.get("options", {}).get("content", "")
            for panneau in self._tableau()["panels"] if panneau["type"] == "text")
        assert "temps **réel**" in textes
        assert "CSV fait foi" in textes
