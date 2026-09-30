"""Single-itinerary trips and the habits journal.

Two distinct defects, both measured on `experiments/archive/2026-09-14_23_58`:
116 trips out of 514 wrote no decision entry, and the habits journal
went back to the last `axe_objet` of the buffer — hence to the previous trip.
"""

import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from llm.noyau import bloc_habitudes, cle_journal, noter_trajet

CONTROLEUR = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text(
    encoding="utf-8"
)
AGENT = (
    RACINE / "urban_mobility_agents" / "agents" / "llm_agent.py"
).read_text(encoding="utf-8")


class TestC1EcritureDeLaDecisionContrainte:
    """C1 — a trip with no alternative still writes its decision entry."""

    def test_la_branche_itineraire_unique_note_la_decision(self):
        branche = CONTROLEUR.split('selection_method = "Un seul itinéraire disponible"')[1]
        # The recording must happen BEFORE leaving the branch (the final `plan:`).
        avant_plan = branche.split("plan: TravelPlan = itineraries[plan_index]")[0]
        assert "note_decision_contrainte" in avant_plan, (
            "a single-itinerary trip must write its decision entry: without it, "
            "memory does not know the trip took place"
        )

    def test_l_ecriture_porte_les_axes_de_la_decision(self):
        corps = AGENT.split("def note_decision_contrainte")[1].split("\n    async def")[0]
        assert "_axes_de_la_decision" in corps, (
            "without axes, the entry is invisible to the basket and to the habits journal"
        )

    def test_l_ecriture_est_sans_effet_sans_plan(self):
        corps = AGENT.split("def note_decision_contrainte")[1].split("\n    async def")[0]
        assert re.search(r"if plan is None:\s*\n\s*return", corps), (
            "a missing plan must not bring planning down"
        )


class TestC2LeTexteDitQueLeChoixEtaitContraint:
    """C2 — the agent must be able to tell what it decided from what it underwent."""

    def test_le_message_ne_pretend_pas_qu_un_modele_a_choisi(self):
        corps = AGENT.split("def note_decision_contrainte")[1].split("\n    async def")[0]
        message = corps.split("stm_msg = (")[1].split(")")[0]
        assert "chosen by gateway LLM" not in message
        assert "only available itinerary" in message

    def test_le_message_dit_que_le_mode_n_est_pas_prefere(self):
        corps = AGENT.split("def note_decision_contrainte")[1].split("\n    async def")[0]
        assert "not preferred" in corps, (
            "without this the agent draws a preference from a lack of options"
        )


class TestC3C4JournalDesHabitudes:
    """C3 and C4 — the arrival is recorded under ITS purpose, or not at all."""

    def _bloc_journal(self) -> str:
        return CONTROLEUR.split("# Ticket 071, batch 4 — the completed trip enters the JOURNAL")[
            1
        ].split("_weather = get_weather")[0]

    def test_C3_le_mode_est_appariee_sur_l_activite(self):
        bloc = self._bloc_journal()
        assert "(person.person_id, str(observation.activity_id))" in bloc, (
            "without matching on the activity, the arrival is attributed to the previous trip"
        )

    def test_C3_le_mode_ne_vient_PAS_du_tampon_de_memoire_courte(self):
        """Regression of 2026-09-15: the buffer is emptied by consolidations.

        Looking for the decision in short-term memory on arrival found almost
        nothing any more — 2 trips logged for 40 arrivals — and the "Mes habitudes" block
        had disappeared from ALL decision prompts. The mode is recorded at decision time.
        """
        bloc = self._bloc_journal()
        assert "get_short_term_memory" not in bloc, (
            "the short buffer is emptied between the decision and the arrival: it cannot "
            "serve as a source for the habits journal"
        )
        assert "_mode_par_activite" in bloc

    def test_C3_le_mode_est_note_a_la_decision(self):
        assert "self._mode_par_activite[(person.person_id, str(next_activity.id))]" in (
            CONTROLEUR
        ), "the mode must be recorded where it is certain: at the time of the decision"
        assert "mode_canonique(plan.mode_label())" in CONTROLEUR

    def test_C3_la_table_est_bornee_par_purge_a_la_lecture(self):
        """Otherwise it would grow without end on a long run."""
        bloc = self._bloc_journal()
        assert "_mode_par_activite.pop(" in bloc, (
            "reading without removing would grow the table at each trip"
        )

    def test_C3_le_motif_vient_de_l_arrivee_pas_de_la_decision(self):
        bloc = self._bloc_journal()
        assert "_decision.axe_motif" not in bloc, (
            "the purpose of the earlier decision filed the return trips under the outbound purpose"
        )
        assert 'observation.data.get("purpose")' in bloc

    def test_C3_le_creneau_vient_de_l_arrivee(self):
        bloc = self._bloc_journal()
        assert "_decision.axe_creneau" not in bloc
        assert "creneau_de(wall_clock(observation.timestamp))" in bloc

    def test_C4_une_arrivee_sans_mode_connu_n_enregistre_rien(self):
        bloc = self._bloc_journal()
        avant_note = bloc.split("noter_trajet")[0]
        assert "if _mode_retenu is None:" in avant_note
        assert "NOT logged" in avant_note, (
            "the gap must show: a wrong habit is worse than a missing habit"
        )


class TestC5C6ComportementDuJournal:
    """C5 and C6 — what the journal must contain, checked on the function itself."""

    def test_C5_un_agent_dont_les_retours_sont_contraints_porte_ses_deux_motifs(self):
        """The exact scenario of agent 609: outbound decided, return constrained.

        Before the fix, the 58 returns were filed under "work" and the block carried
        no `home`. Here, each arrival is recorded under its own purpose.
        """
        journal = {}
        for _ in range(20):
            noter_trajet(journal, "work", "matin", "car")
            noter_trajet(journal, "home", "soir", "car")

        assert cle_journal("work", "matin") in journal
        assert cle_journal("home", "soir") in journal
        assert journal[cle_journal("home", "soir")]["total"] == 20

        motifs = {ligne.split(" ")[0] for ligne in bloc_habitudes(journal)}
        assert motifs == {"work", "home"}

    def test_C6_le_compte_du_journal_suit_les_arrivees_notees(self):
        journal = {}
        for i in range(10):
            noter_trajet(journal, "shop", "midi", "public_transport", retard_s=0)
        assert sum(e["total"] for e in journal.values()) == 10

    def test_C6_un_mode_non_resolu_ne_fait_pas_deriver_le_denominateur(self):
        journal = {}
        noter_trajet(journal, "work", "matin", "car")
        noter_trajet(journal, "work", "matin", None)
        assert journal[cle_journal("work", "matin")]["total"] == 1
