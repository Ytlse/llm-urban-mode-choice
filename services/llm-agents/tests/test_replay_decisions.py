"""The replay trace spares decisions already taken.

On a hot resume, GAMA replays the days already lived to rebuild its state. Memory is
frozen, but decisions are REMADE and, with the cache off, paid for again. Measured on
2026-09-16: eight days replayed, about a hundred decisions, forty-five minutes of network waiting
to get back to an already known state.
"""

from pathlib import Path

import pytest
from urban_mobility_agents.utils import rejeu_decisions as R

RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _propre():
    R.reinitialiser()
    yield
    R.reinitialiser()


def _tracer_une(workdir, personne="899549", activite="act-1", instant=1000.0, code="car#1"):
    R.charger(workdir)
    R.tracer(
        personne, activite, instant,
        code_plan=code, raison="raison", fournisseur="google_gemini31_key1",
        distribution={"car": 0.7, "bus": 0.3},
    )


class TestTracerEtResservir:
    def test_A1_une_decision_vivante_est_tracee_sous_sa_cle(self, tmp_path):
        _tracer_une(tmp_path)
        assert R.chemin(tmp_path).is_file()
        assert R.charger(tmp_path) == 1

    def test_A4_la_reponse_resservie_est_identique_a_celle_tracee(self, tmp_path):
        _tracer_une(tmp_path)
        R.charger(tmp_path)
        enr = R.chercher("899549", "act-1", 1000.0)
        assert enr is not None
        assert enr["code_plan"] == "car#1"
        assert enr["raison"] == "raison"
        assert enr["distribution"] == {"car": 0.7, "bus": 0.3}

    def test_A5_une_cle_inconnue_rend_None_sans_lever(self, tmp_path):
        _tracer_une(tmp_path)
        R.charger(tmp_path)
        assert R.chercher("899549", "act-INCONNUE", 1000.0) is None

    def test_la_cle_distingue_deux_instants_de_la_meme_activite(self, tmp_path):
        """An activity that recurs every day must not re-serve the previous day's decision."""
        _tracer_une(tmp_path, instant=1000.0, code="car#1")
        R.tracer("899549", "act-1", 87400.0, code_plan="bus#2", raison="r", fournisseur="p",
                 distribution={})
        R.charger(tmp_path)
        assert R.chercher("899549", "act-1", 1000.0)["code_plan"] == "car#1"
        assert R.chercher("899549", "act-1", 87400.0)["code_plan"] == "bus#2"


class TestLaFenetreDeRejeu:
    def test_seules_les_decisions_prises_avant_le_point_sont_indexees(self, tmp_path):
        """Beyond the resume point, the decision must be TAKEN AGAIN by the model, not re-served:
        that is where the run comes alive again. The criterion is the time of the DECISION."""
        R.charger(tmp_path)
        R.noter_horloge(900.0)
        R.tracer("899549", "act-1", 1000.0, code_plan="car#1", raison="r", fournisseur="p",
                 distribution={})
        R.noter_horloge(8000.0)
        R.tracer("899549", "act-2", 9000.0, code_plan="x", raison="r", fournisseur="p",
                 distribution={})
        assert R.charger(tmp_path, jusqu_a=5000.0) == 1
        assert R.chercher("899549", "act-1", 1000.0) is not None
        assert R.chercher("899549", "act-2", 9000.0) is None

    def test_une_decision_prise_la_veille_pour_une_activite_posterieure_au_point_est_gardee(
        self, tmp_path
    ):
        """The case of the a13 v5 control (2026-09-28): 1320712 chooses on 25/03 at 19:45 its
        trip of 26/03 at 18:16; the resume point is on 26/03 at 12:45. Discarded, the decision was
        asked again of the model with the frozen 26/03 memory — a choice never made that evening."""
        point = 1774529100
        R.charger(tmp_path)
        R.noter_horloge(1774467900)
        R.tracer("1320712", "a6b3f493", 1774549009, code_plan="__DIRECT_BICYCLE__^^",
                 raison="r", fournisseur="google_gemini31_key1", distribution={})
        assert R.charger(tmp_path, jusqu_a=point) == 1
        enr = R.chercher("1320712", "a6b3f493", 1774549009)
        assert enr is not None and enr["code_plan"] == "__DIRECT_BICYCLE__^^"

    def test_l_heure_passee_par_l_appelant_prime_sur_l_horloge_du_pas(self, tmp_path):
        """The agent reads the clock before waiting for the model: the response may arrive a step
        later, the decision stays dated from the step where it was requested."""
        R.charger(tmp_path)
        R.noter_horloge(6000.0)
        R.tracer("899549", "act-1", 7000.0, code_plan="car#1", raison="r", fournisseur="p",
                 distribution={}, decide_a=4500.0)
        assert R.charger(tmp_path, jusqu_a=5000.0) == 1

    def test_une_ligne_sans_heure_de_decision_est_gardee_et_comptee(self, tmp_path):
        """Trace written before 2026-09-28: no `decide_a`. It is only consulted during
        the freeze, where only decisions from before the point are asked again — keeping it thus
        re-serves nothing the run did not decide before the point."""
        import json

        from loguru import logger

        R.chemin(tmp_path).write_text(
            json.dumps({"personne": "1", "activite": "a", "instant": 9000.0,
                        "code_plan": "x", "raison": "r", "fournisseur": "p",
                        "distribution": {}}) + "\n",
            encoding="utf-8",
        )
        lignes: list[str] = []
        sink = logger.add(lambda m: lignes.append(str(m)), level="INFO")
        try:
            assert R.charger(tmp_path, jusqu_a=5000.0) == 1
        finally:
            logger.remove(sink)
        assert R.chercher("1", "a", 9000.0) is not None
        assert any("1 without a decision time" in m for m in lignes)

    def test_tracer_consigne_l_horloge_du_pas(self, tmp_path):
        import json

        R.charger(tmp_path)
        R.noter_horloge(1234.0)
        R.tracer("1", "a", 2000.0, code_plan="x", raison="r", fournisseur="p", distribution={})
        enr = json.loads(R.chemin(tmp_path).read_text(encoding="utf-8").splitlines()[0])
        assert enr["decide_a"] == 1234.0


class TestComptageEtAlarme:
    def test_B1_servies_et_manquees_sont_comptees_separement(self, tmp_path):
        _tracer_une(tmp_path)
        R.charger(tmp_path)
        R.chercher("899549", "act-1", 1000.0)
        R.chercher("899549", "absente", 1.0)
        assert "1 décision(s) resservie(s)" in R.bilan()
        assert "1 manquée(s)" in R.bilan()

    def test_B2_un_rejeu_divergent_leve_une_alarme(self, tmp_path, caplog):
        """A replay that does not find its own choices does not rebuild the state one believes."""
        from loguru import logger

        messages: list[str] = []
        puits = logger.add(lambda m: messages.append(str(m)), level="ERROR")
        try:
            R.charger(tmp_path)
            for i in range(25):
                R.chercher("899549", f"absente-{i}", float(i))
        finally:
            logger.remove(puits)
        trace = "\n".join(messages)
        assert "[ALARME]" in trace and "divergent" in trace

    def test_B2_une_alarme_ne_se_repete_pas(self, tmp_path):
        from loguru import logger

        messages: list[str] = []
        puits = logger.add(lambda m: messages.append(str(m)), level="ERROR")
        try:
            R.charger(tmp_path)
            for i in range(60):
                R.chercher("899549", f"absente-{i}", float(i))
        finally:
            logger.remove(puits)
        assert sum("[ALARME]" in m for m in messages) == 1, "rising edge, not a flood"


class TestCeQuiNeDoitPasCasser:
    def test_C1_sans_trace_le_comportement_est_celui_davant(self, tmp_path):
        assert R.charger(tmp_path) == 0
        assert R.chercher("899549", "act-1", 1000.0) is None

    def test_C2_une_trace_tronquee_est_ignoree_ligne_a_ligne(self, tmp_path):
        _tracer_une(tmp_path)
        with R.chemin(tmp_path).open("a", encoding="utf-8") as f:
            f.write('{"personne": "x", "activite"\n')  # line cut off mid-flight
        assert R.charger(tmp_path) == 1, "the sound line remains usable"

    def test_tracer_sans_chargement_prealable_ne_leve_pas(self):
        """Tracing must never bring a run down, even when badly initialised."""
        R.tracer("a", "b", 1.0, code_plan="c", raison="r", fournisseur="p", distribution={})


class TestLaChaineEstBranchee:
    """Chain guards: the module may be perfect and be called nowhere."""

    def _source(self) -> str:
        return (RACINE / "urban_mobility_agents" / "agents" / "llm_agent.py").read_text(
            encoding="utf-8"
        )

    def test_A3_la_relecture_est_conditionnee_au_gel(self):
        src = self._source()
        assert "rejeu_decisions" in src, "the module is called nowhere"
        assert "gel_actif" in src, (
            "re-serving outside the freeze window would turn the trace into a permanent cache"
        )

    def test_A1_la_decision_vivante_est_tracee(self):
        assert "rejeu_decisions.tracer(" in self._source().replace("R.", "rejeu_decisions.")

    def test_la_trace_est_ouverte_sur_TOUT_run_pas_seulement_a_la_reprise(self):
        """It is written during normal life. Opening it only on resume leaves it empty,
        hence inert — observed on 2026-09-17: zero lines after two simulated days."""
        src = (RACINE / "handle" / "application.py").read_text(encoding="utf-8")
        i_ouverture = src.find("rejeu_decisions.charger(_workdir)")
        i_garde = src.find('if _reprise:\n        from urban_mobility_agents.utils.reprise import')
        assert i_ouverture > 0, "the trace is never opened outside a resume"
        assert i_garde < 0 or i_ouverture < i_garde, (
            "the unconditional opening must precede the bounded loading of the resume"
        )

    def test_la_decision_est_datee_du_pas_ou_elle_est_demandee(self):
        """2026-09-28 — the agent reads the clock before waiting for the model and passes it to
        the trace; the controller sets it at each sync, before the thaw."""
        src = self._source()
        assert "_decide_a = rejeu_decisions.horloge()" in src
        assert "decide_a=_decide_a" in src
        ctrl = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text(
            encoding="utf-8"
        )
        i_horloge = ctrl.find("rejeu_decisions.noter_horloge(timestamp)")
        i_degel = ctrl.find("if degeler_si_depasse(timestamp):")
        assert 0 < i_horloge < i_degel, "the step clock must be recorded before the thaw"
