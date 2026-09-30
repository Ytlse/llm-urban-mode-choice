"""Did the injected memory reach long-term memory?

What these tests would have caught on 2026-09-23: campaign c3 v4 injected a metro breakdown
rated `grave` (0.75) into the short-term memory of 861500, the evening consolidation wrote *« Today
went very smoothly overall »*, and none of the run's 123 long-term memory documents mentioned the
metro, the lighting or an announcement. The run went on for 2 h 22 measuring the effect of a memory that
did not exist.

The texts used here are the REAL ones: that of `config/evenements/c3_panne_reseau.yaml` and the
reflections actually written by runs `11_10` (which keeps the shock) and `13_54` (which
loses it). A test on invented text would say nothing about the corpus the witness must sort.
"""

from __future__ import annotations

from pathlib import Path

from llm.evenements import temoin
from llm.evenements.injection import PREFIXE_LU, PREFIXE_VECU
from settings import settings

RACINE = Path(__file__).resolve().parents[3]

TEXTE_C3 = (
    "The metro stopped between two stations and the lighting went out. No announcement was made "
    "for 25 minutes. When the service restarted the carriage was packed full. I missed the "
    "appointment I was travelling to, and arrived 28 minutes after the time I had planned. It is "
    "the fourth time this month that this line has stopped."
)

# Run 11_10 — the shock got through. Real excerpt of the consolidated reflection.
GARDE = (
    "The metro stopped between two stations and the lighting went out, with no announcement for "
    "25 minutes. When the service restarted the carriage was packed full and I missed the "
    "appointment I was travelling to."
)

# Run 13_54 — the shock is lost. Real excerpt of the consolidation that followed the injection.
PERDU = (
    "Today went very smoothly overall, with all my trips adhering closely to schedule. My morning "
    "public transport trip using Bus 35 went well, with only a short walk between errands."
)


# ── Spotting the injected entry ────────────────────────────────────────────────────────────


class TestLAncrage:
    def test_une_consolidation_ordinaire_ne_declenche_rien(self):
        """The vast majority of consolidations: no injected entry, no check."""
        assert temoin.texte_injecte(["I drove to work.", "I walked home."]) is None

    def test_le_prefixe_du_vecu_est_reconnu(self):
        entree = f"Arrived on time.\n{PREFIXE_VECU} {TEXTE_C3}"
        assert temoin.texte_injecte(["I walked home.", entree]) == TEXTE_C3

    def test_le_prefixe_du_lu_est_reconnu(self):
        """The `lu` channel goes through the same witness."""
        entree = f"{PREFIXE_LU} I read in the paper: « {TEXTE_C3} »"
        assert TEXTE_C3 in (temoin.texte_injecte([entree]) or "")

    def test_le_texte_est_pris_apres_le_prefixe_pas_avant(self):
        """The arrival observation is JOINED to the text: confusing the two would skew everything."""
        entree = f"Arrived on time. Distance 4.2 km.\n{PREFIXE_VECU} {TEXTE_C3}"
        assert "Distance" not in (temoin.texte_injecte([entree]) or "")


# ── The distinctive words ─────────────────────────────────────────────────────────────────


class TestMotsDistinctifs:
    def test_le_vocabulaire_de_l_incident_est_retenu(self):
        mots = temoin.mots_distinctifs(TEXTE_C3)
        for attendu in ("lighting", "announcement", "carriage", "packed", "missed"):
            assert attendu in mots, attendu

    def test_le_vocabulaire_banal_du_domaine_est_ecarte(self):
        """« minutes », « line », « metro » are in every reflection, shock or not."""
        mots = temoin.mots_distinctifs(TEXTE_C3)
        for banal in ("minutes", "line", "metro", "time", "stopped", "arrived", "month"):
            assert banal not in mots, banal

    def test_between_est_ecarte(self):
        """THE CALIBRATED CASE: it was the only word the lost run found again, in
        « a short walk between errands » — a sentence unrelated to the breakdown."""
        assert "between" not in temoin.mots_distinctifs(TEXTE_C3)

    def test_l_ordre_du_texte_est_conserve(self):
        """An alarm that names the words must read back in the order of the injected sentence."""
        mots = temoin.mots_distinctifs(TEXTE_C3)
        assert mots.index("lighting") < mots.index("carriage") < mots.index("appointment")

    def test_pas_de_doublon(self):
        assert len(set(temoin.mots_distinctifs(TEXTE_C3))) == len(
            temoin.mots_distinctifs(TEXTE_C3)
        )


# ── The search in what was written ─────────────────────────────────────────────────


class TestLaRecherche:
    def test_le_run_qui_garde_le_choc_est_reconnu(self):
        mots = temoin.mots_distinctifs(TEXTE_C3)
        trouves = temoin.mots_retrouves(mots, [GARDE])
        assert len(trouves) >= settings.agent.temoin_souvenir_mots_min
        assert "lighting" in trouves

    def test_le_run_qui_perd_le_choc_est_reconnu(self):
        """THE 2026-09-23 CASE: the consolidation describes an uneventful day."""
        mots = temoin.mots_distinctifs(TEXTE_C3)
        trouves = temoin.mots_retrouves(mots, [PERDU])
        assert len(trouves) < settings.agent.temoin_souvenir_mots_min, trouves

    def test_une_flexion_compte(self):
        """« announcements » must count as « announcement »: the model paraphrases, it does not copy."""
        assert "announcement" in temoin.mots_retrouves(
            ["announcement"], ["There were no announcements at all."]
        )

    def test_une_sous_chaine_ne_compte_pas(self):
        """« rain » must not be recognised in « train » — the classic trap."""
        assert temoin.mots_retrouves(["rain"], ["I took the train."]) == []

    def test_la_casse_est_indifferente(self):
        assert temoin.mots_retrouves(["carriage"], ["The CARRIAGE was full."]) == [
            "carriage"
        ]


# ── The full check ──────────────────────────────────────────────────────────────────


class TestLeControle:
    def _entrees(self):
        return [f"Arrived on time.\n{PREFIXE_VECU} {TEXTE_C3}"]

    def test_aucune_injection_aucun_constat(self):
        assert (
            temoin.controler(
                person_id="861500",
                sim_ts=1774620168,
                contenus_courts=["I drove to work."],
                textes_longs=["A quiet day."],
                seuil=2,
            )
            is None
        )

    def test_le_souvenir_retrouve_rend_un_constat_positif(self):
        constat = temoin.controler(
            person_id="861500",
            sim_ts=1774620168,
            contenus_courts=self._entrees(),
            textes_longs=[GARDE],
            seuil=2,
        )
        assert constat is not None and constat.retrouve is True
        assert constat.verdict == "retrouvé"

    def test_le_souvenir_perdu_rend_un_constat_negatif(self):
        constat = temoin.controler(
            person_id="861500",
            sim_ts=1774620168,
            contenus_courts=self._entrees(),
            textes_longs=[PERDU],
            seuil=2,
        )
        assert constat is not None and constat.retrouve is False
        assert constat.verdict == "PERDU"
        assert "lighting" in constat.mots_cherches

    def test_le_controle_ne_leve_jamais(self):
        """A witness that brought down a consolidation would cost more than the defect it watches."""
        assert (
            temoin.controler(
                person_id="x",
                sim_ts=0,
                contenus_courts=None,  # type: ignore[arg-type]
                textes_longs=None,  # type: ignore[arg-type]
                seuil=2,
            )
            is None
        )


# ── The wiring and the setting ─────────────────────────────────────────────────────────


class TestLeBranchement:
    def test_le_controle_est_appele_apres_l_ecriture_en_memoire_longue(self):
        """Before the write, it would query an intention, not what was written."""
        source = (
            RACINE / "services/llm-agents/urban_mobility_agents/agents/llm_agent.py"
        ).read_text(encoding="utf-8")
        assert "temoin.controler(" in source
        avant = source.split("temoin.controler(")[0]
        assert "await self.aadd_long_term_memory(context, entry)" in avant

    def test_le_seuil_est_reglable_et_vaut_deux(self):
        assert settings.agent.temoin_souvenir_mots_min == 2

    def test_le_temoin_n_arrete_pas_le_run(self):
        """Decision of 2026-09-23: it raises an ALARM. The witness is heuristic where the fallback one
        is certain; stopping waits until its false alarm rate has been observed."""
        source = (
            RACINE / "services/llm-agents/llm/evenements/temoin.py"
        ).read_text(encoding="utf-8")
        assert "SIGTERM" not in source
        assert "_declencher_hibernation" not in source

    def test_l_echec_est_une_alarme_et_le_succes_se_dit_aussi(self):
        source = (
            RACINE / "services/llm-agents/llm/evenements/temoin.py"
        ).read_text(encoding="utf-8")
        assert "[ALARME] [temoin]" in source
        assert "logger.info" in source, "un témoin muet ne se distingue pas d'un témoin mort"


class TestLIdentifiantDeLEvenement:
    """On 2026-09-24, the first trace written under real conditions carried
    `"evenement_id": ""`: the verdict was right, but the line did not say which shock
    it was about. The evening consolidation does not know the event joined to a morning
    arrival — the identifier lives only in the run registry."""

    def test_le_registre_nomme_l_evenement_quand_l_appelant_se_tait(self, monkeypatch):
        class _Evenement:
            evenement_id = "c6_voiture_suspecte"

        class _Registre:
            evenement = _Evenement()

        import llm.evenements as evenements_module

        monkeypatch.setattr(evenements_module, "registre", lambda: _Registre())
        constat = temoin.controler(
            person_id="861500",
            sim_ts=1774849500,
            contenus_courts=[f"On time. {PREFIXE_VECU} The engine stalled on the expressway."],
            textes_longs=["The engine stalled on the expressway near the exit."],
            seuil=2,
        )
        assert constat is not None
        assert constat.evenement_id == "c6_voiture_suspecte"

    def test_l_appelant_garde_le_dernier_mot(self, monkeypatch):
        """The day a run declares two events, the call site will have to pass
        the identifier — and what it passes must win over the fallback."""

        class _Evenement:
            evenement_id = "declare_par_le_run"

        class _Registre:
            evenement = _Evenement()

        import llm.evenements as evenements_module

        monkeypatch.setattr(evenements_module, "registre", lambda: _Registre())
        constat = temoin.controler(
            person_id="861500",
            sim_ts=1774849500,
            contenus_courts=[f"On time. {PREFIXE_VECU} The engine stalled on the expressway."],
            textes_longs=["The engine stalled on the expressway near the exit."],
            seuil=2,
            evenement_id="passe_par_l_appelant",
        )
        assert constat is not None
        assert constat.evenement_id == "passe_par_l_appelant"

    def test_sans_registre_la_trace_reste_lisible(self, monkeypatch):
        """With no declared event, the witness does not complain and breaks nothing: it simply
        names less."""
        import llm.evenements as evenements_module

        monkeypatch.setattr(evenements_module, "registre", lambda: None)
        constat = temoin.controler(
            person_id="861500",
            sim_ts=1774849500,
            contenus_courts=[f"On time. {PREFIXE_VECU} The engine stalled on the expressway."],
            textes_longs=["The engine stalled on the expressway near the exit."],
            seuil=2,
        )
        assert constat is not None
        assert constat.evenement_id == ""
