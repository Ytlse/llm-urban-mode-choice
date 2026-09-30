"""A default fallback is not a decision, and an experiment run stops.

What these tests would have caught on 2026-09-23: the c3 v4 campaign let four decisions
be made by `itineraries[0]` inside the measurement window — 0/48 fallbacks before the shock, 4/39
after — without anything stopping or flagging it. A safeguard already existed but
only looked at `genre_erreur == "quota_journalier"`, and an upstream saturation (54 × HTTP 503
"high demand" that day, zero RESOURCE_EXHAUSTED) returns NO genre_erreur.

The tests cover the stopping LOGIC and what triggers it, read in the source: the
full path requires GAMA, a model and a pipeline, and a test that set all of that up would
no longer tell which of the three broke.
"""

from __future__ import annotations

from pathlib import Path

from settings import settings
from urban_mobility_agents import simulation_controller as sc

SOURCE = Path(sc.__file__).read_text(encoding="utf-8")
# The tests run from `services/llm-agents/`: the repository root is walked up to, it is not
# assumed. Without that, two tests read a relative path that did not exist.
RACINE = Path(__file__).resolve().parents[3]


# ── The lock: armed in an experiment, silent elsewhere ───────────────────────────────────


class TestVerrouDExperience:
    def test_muet_par_defaut(self, monkeypatch):
        """An ordinary run keeps its behaviour: it falls back, it does not stop."""
        for nom in sc._VERROUS_ARRET_EXPERIENCE:
            monkeypatch.delenv(nom, raising=False)
        assert sc._arret_sur_repli_arme() is False

    def test_arme_par_le_nom_neuf(self, monkeypatch):
        for nom in sc._VERROUS_ARRET_EXPERIENCE:
            monkeypatch.delenv(nom, raising=False)
        monkeypatch.setenv("EXPERIMENT_STOP_ON_FALLBACK", "1")
        assert sc._arret_sur_repli_arme() is True

    def test_l_ancien_nom_reste_admis(self, monkeypatch):
        """`EXPERIMENT_HIBERNATE_ON_QUOTA` is written in campaign scripts already launched."""
        for nom in sc._VERROUS_ARRET_EXPERIENCE:
            monkeypatch.delenv(nom, raising=False)
        monkeypatch.setenv("EXPERIMENT_HIBERNATE_ON_QUOTA", "1")
        assert sc._arret_sur_repli_arme() is True

    def test_une_valeur_autre_que_1_n_arme_pas(self, monkeypatch):
        for nom in sc._VERROUS_ARRET_EXPERIENCE:
            monkeypatch.delenv(nom, raising=False)
        monkeypatch.setenv("EXPERIMENT_STOP_ON_FALLBACK", "true")
        assert sc._arret_sur_repli_arme() is False

    def test_la_campagne_pose_le_verrou(self):
        """Every campaign is protected by default: no lever to remember at launch."""
        chemin = RACINE / "scripts/experiment/run_sequential_cohort.py"
        cohorte = chemin.read_text(encoding="utf-8")
        assert 'env["EXPERIMENT_STOP_ON_FALLBACK"] = "1"' in cohorte

    def test_le_verrou_traverse_le_conteneur(self):
        """A variable not declared in compose never reaches the controller."""
        compose = (RACINE / "infra/docker-compose.yml").read_text(encoding="utf-8")
        assert "EXPERIMENT_STOP_ON_FALLBACK:" in compose
        assert "EXPERIMENT_HIBERNATE_ON_QUOTA:" in compose, (
            "the alias must still be passed through"
        )


# ── The stopping logic, read in the source ───────────────────────────────────────────────


class TestLogiqueDArret:
    @staticmethod
    def _bloc_repli() -> str:
        """Isolates the decision branch; consolidation uses the same lock further up."""
        return SOURCE.split('selection_method = "LLM"')[1].split("plan: TravelPlan")[0]

    def test_le_compteur_est_declare(self):
        assert "self._replis_consecutifs: int = 0" in SOURCE

    def test_une_decision_reussie_casse_la_serie(self):
        """Without a reset, three fallbacks spread over a whole run would end up stopping it."""
        bloc = SOURCE.split('selection_method = "LLM"')[1][:300]
        assert "self._replis_consecutifs = 0" in bloc

    def test_le_compteur_monte_avant_toute_decision_d_arret(self):
        bloc = self._bloc_repli()
        assert bloc.index("self._replis_consecutifs += 1") < bloc.index(
            "if _arret_sur_repli_arme():"
        )

    def test_le_premier_echec_non_absorbe_suspend_avant_repli(self):
        bloc = self._bloc_repli()
        assert "_declencher_hibernation_propre" in bloc
        assert 'motif="decision_absente"' in bloc
        assert bloc.index('motif="decision_absente"') < bloc.index("plan_index = 0")

    def test_le_chemin_quota_du_077_est_preserve(self):
        """It alone knows the reopening time: it must not be absorbed by the new one."""
        garde = 'if _genre in ("quota_journalier", "surcharge_fournisseur"):'
        assert '_genre = _trace_decision.get("genre_erreur")' in SOURCE
        assert garde in SOURCE
        bloc = SOURCE.split(garde)[1][:500]
        assert "motif=_genre" in bloc, "the marker says which of the two stopped the run"
        assert "reprise_a" in bloc, "the reopening time must keep travelling"

    def test_une_surcharge_qualifiee_arrete_au_premier_echec(self):
        """2026-09-25 — the worker returns the batch `surcharge_fournisseur` before the client gives up.

        Without this kind, three silent `Timeout expiré` first entered the measurement as fallbacks
        (2026-09-23: 16 March 05:00, 07:48, then the third) before the threshold stopped the run.
        """
        bloc = self._bloc_repli()
        assert "surcharge_fournisseur" in bloc
        assert bloc.index("surcharge_fournisseur") < bloc.index(
            'motif="decision_absente"'
        ), "the qualified overload keeps its reason and its resume time"

    def test_l_arret_pour_surcharge_le_dit_en_alarme(self):
        bloc = SOURCE.split('elif motif == "surcharge_fournisseur":')[1][:600]
        assert "[ALARME] [hibernation]" in bloc

    def test_les_deux_arrets_rendent_la_main_sans_choisir(self):
        """A `return None, None`: above all not a plan_index=0 after deciding to stop."""
        bloc = self._bloc_repli()
        assert bloc.count("return None, None") == 2

    def test_une_consolidation_absente_suspend_aussi_l_experience(self):
        assert "ConsolidationMemoryUnavailable" in SOURCE
        assert 'motif="consolidation_memoire"' in SOURCE


# ── Visibility ───────────────────────────────────────────────────────────────────────────


class TestLeRepliSeVoit:
    def test_le_repli_est_journalise_en_alarme(self):
        """It was at `debug`: moves.csv had to be searched by hand to discover it."""
        assert "[ALARME] [repli]" in SOURCE

    def test_l_alarme_porte_de_quoi_agir(self):
        bloc = SOURCE.split("[ALARME] [repli]")[1][:600]
        for attendu in ("agent=", "instant=", "consecutifs="):
            assert attendu in bloc, attendu

    def test_le_repli_n_est_plus_en_debug(self):
        assert "No suitable plan found for person" not in SOURCE


# ── The stop marker ──────────────────────────────────────────────────────────────────────


class TestMarqueurDArret:
    def test_le_marqueur_porte_le_motif(self):
        assert '"motif": motif' in SOURCE

    def test_une_saturation_n_invente_pas_d_heure_de_reouverture(self):
        """Writing the string "None" into `resume_at` would suggest a date."""
        assert "if resume_at is None" in SOURCE
        assert '"replis_consecutifs": self._replis_consecutifs' in SOURCE

    def test_l_arret_pour_decision_absente_le_dit_en_alarme(self):
        bloc = SOURCE.split('if motif == "decision_absente":')[1][:600]
        assert "[ALARME] [hibernation]" in bloc
        assert "Stop at the first failure" in bloc

    def test_l_arret_reste_un_sigterm(self):
        """`sys.exit` is swallowed into a request error under ASGI; the orchestrator expects a 0."""
        assert "os.kill(os.getpid(), signal.SIGTERM)" in SOURCE


# ── The bench: a qualified overload stays transient ──────────────────────────────────────


def test_le_banc_range_la_surcharge_qualifiee_en_passerelle_occupee():
    """R2: transient, never `epuise` — and without depending on the wording of the reason."""
    import inspect

    from experiences import decideurs as D

    source = inspect.getsource(D)
    branche = source.split('genre_erreur") == "surcharge_fournisseur"')[1][:600]
    assert '"passerelle_occupee: "' in branche
    assert "epuise" not in branche.split("elif")[0].replace("jamais `epuise`", "")
