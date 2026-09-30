"""What a household member tells the others.

Rules R32 to R42, plus the four guards against the loop requested by the author on
2026-09-22 (G1 to G4).

Everything here is PURE: no simulator, no model, no network call.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import foyer
from llm.memory import MemoryEntry, MemoryType
from settings import settings

RACINE = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 3, 16, 22, 0)


@dataclass
class FauxIdentite:
    traits_json: dict


@dataclass
class FauxAgent:
    person_id: str
    household_id: str | None = None
    immobile: bool = False
    identity: FauxIdentite = field(default_factory=lambda: FauxIdentite({}))


@dataclass
class FausseLTM:
    """The minimum that `foyer.py` reads: `user_metadata[pid]["entries"]`."""

    user_metadata: dict = field(default_factory=dict)

    def ajouter(self, person_id: str, entree: MemoryEntry) -> MemoryEntry:
        self.user_metadata.setdefault(person_id, {"entries": []})["entries"].append(entree)
        return entree


# Household 5177: Constance, 12 years old, without a licence or a bike, and Jacques, 49 years old,
# licence and bike. What Jacques learns about the car must NEVER become a belief of
# Constance — it is the case that grounds R4.
CONSTANCE = FauxAgent(
    "12", "5177",
    identity=FauxIdentite({
        "name": "Constance Ledoux", "age": 12,
        "has_driving_license": False, "number_of_cars": 0, "personal_bike": "No bike",
    }),
)
JACQUES = FauxAgent(
    "49", "5177",
    identity=FauxIdentite({
        "name": "Jacques Aubert", "age": 49,
        "has_driving_license": True, "number_of_cars": 1, "personal_bike": "Own bike",
    }),
)
XAVIER = FauxAgent(
    "40", "312",
    identity=FauxIdentite({"name": "Xavier Briand", "age": 40}),
)


def _reflexion(pid: str, texte: str, quand: datetime) -> MemoryEntry:
    return MemoryEntry(
        content=texte, timestamp=quand, memory_type=MemoryType.REFLECTION, person_id=pid,
    )


def _concept(pid: str, texte: str, quand: datetime, *, observations=2,
             axe_objet="public_transport", origine=None, contre=0) -> MemoryEntry:
    return MemoryEntry(
        content=json.dumps([texte, "", "", "", ""], ensure_ascii=False),
        timestamp=quand, memory_type=MemoryType.CONCEPT, person_id=pid,
        observations=observations, contre_exemples=contre,
        axe_objet=axe_objet, origine=origine,
    )


@pytest.fixture(autouse=True)
def _foyer_propre():
    foyer.reinitialiser()
    settings.agent.memoire__partage_foyer_enabled = True
    yield
    foyer.reinitialiser()
    settings.agent.memoire__partage_foyer_enabled = False


def _indexer(*agents):
    foyer.initialiser(list(agents))


# ── R32. Le drapeau ──────────────────────────────────────────────────────────────────────
def test_R32_drapeau_eteint_aucun_bloc():
    """Everything measured before this batch stays comparable: it is a switch."""
    settings.agent.memoire__partage_foyer_enabled = False
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Busy day, the ring road was slow.", T0))
    assert foyer.bloc_du_soir(ltm, CONSTANCE, T0 + timedelta(hours=1)) == ""


def test_R32bis_le_defaut_est_eteint():
    from settings import AgentConfig

    assert AgentConfig().memoire__partage_foyer_enabled is False


# ── R33, R34. The evening summary ───────────────────────────────────────────────────────────
def test_R33_le_recit_cite_le_bilan_que_lautre_a_ecrit():
    """« No new information » (author, 2026-09-22): the account QUOTES, it does not write."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    bilan = "The ring road was slow again this morning; I left ten minutes earlier."
    ltm.ajouter("49", _reflexion("49", bilan, T0))
    lignes = foyer.recit_du_soir(ltm, "12", T0 + timedelta(hours=1))
    assert len(lignes) == 1
    assert bilan in lignes[0]
    assert "Jacques Aubert (49)" in lignes[0]


def test_R33bis_le_libelle_ne_dit_jamais_un_lien_de_parente():
    """EMC² and eqasim give the household, the age and the gender — NOT the parentage.

    Inventing « your son » would fabricate a datum, and it would fall back into the prompts
    as a fact.
    """
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("12", _reflexion("12", "School bus was full.", T0))
    ligne = foyer.recit_du_soir(ltm, "49", T0 + timedelta(hours=1))[0]
    assert "Constance Ledoux (12)" in ligne
    for parente in ("daughter", "son", "father", "mother", "fille", "fils", "père", "mère"):
        assert parente not in ligne.lower()


def test_R34_le_recit_nest_jamais_ecrit_dans_la_memoire_du_receveur():
    """The block is an ENTRY of the call. The memory does not grow from having listened."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Long day.", T0))
    ltm.user_metadata.setdefault("12", {"entries": []})
    avant = len(ltm.user_metadata["12"]["entries"])
    foyer.bloc_du_soir(ltm, CONSTANCE, T0 + timedelta(hours=1))
    assert len(ltm.user_metadata["12"]["entries"]) == avant


# ── G1. An account is never served twice ────────────────────────────────────────────
def test_G1_un_bilan_deja_entendu_ne_repart_pas():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "The ring road was slow.", T0))
    premier = foyer.recit_du_soir(ltm, "12", T0 + timedelta(hours=1))
    assert len(premier) == 1
    second = foyer.recit_du_soir(ltm, "12", T0 + timedelta(days=1))
    assert second == [], "le même bilan servi deux fois est la boucle qu'on veut éviter"


def test_G1bis_ce_qui_arrive_apres_le_passage_nest_pas_perdu():
    """The sharing is continuous, with a lag of at most one night, never a loss."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Day one.", T0))
    foyer.recit_du_soir(ltm, "12", T0 + timedelta(hours=1))
    ltm.ajouter("49", _reflexion("49", "Day two.", T0 + timedelta(days=1)))
    suite = foyer.recit_du_soir(ltm, "12", T0 + timedelta(days=1, hours=1))
    assert len(suite) == 1 and "Day two" in suite[0]


def test_G1ter_le_repere_est_PAR_receveur():
    """Without that, the order of the consolidations would decide who hears what."""
    _indexer(CONSTANCE, JACQUES, FauxAgent("99", "5177",
                                           identity=FauxIdentite({"name": "Tiers", "age": 30})))
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Shared day.", T0))
    assert foyer.recit_du_soir(ltm, "12", T0 + timedelta(hours=1))
    assert foyer.recit_du_soir(ltm, "99", T0 + timedelta(hours=2)), (
        "le passage de Constance ne doit pas priver le troisième membre"
    )


def test_le_repere_survit_a_une_reprise_a_chaud():
    """Otherwise a resume would make the whole household hear again nights already heard."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "A day.", T0))
    foyer.recit_du_soir(ltm, "12", T0 + timedelta(hours=1))
    sauvegarde = foyer.etat_pour_reprise()

    foyer.reinitialiser()
    _indexer(CONSTANCE, JACQUES)
    foyer.charger_etat(sauvegarde)
    assert foyer.recit_du_soir(ltm, "12", T0 + timedelta(days=1)) == []


# ── R38. The six household rules ───────────────────────────────────────────────────────────
def test_R38_R1_un_concept_non_ancre_ne_circule_pas():
    """One does not tell the household what one has not checked oneself."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Bus 62 is often late", T0, observations=0))
    lignes, refus = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert lignes == [] and refus["R1"] == 1


def test_R38_R3_un_concept_sans_axe_ne_circule_pas():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Things are slow", T0, axe_objet=None))
    lignes, refus = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert lignes == [] and refus["R3"] == 1


def test_R38_R4_le_cas_Constance_et_Jacques():
    """What Jacques learns about the CAR never becomes a belief of Constance."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "The ring road jams at 8am", T0, axe_objet="car"))
    lignes, refus = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert lignes == [] and refus["R4"] == 1
    # And the same concept passes to an adult who drives.
    _indexer(CONSTANCE, JACQUES)
    ltm2 = FausseLTM()
    ltm2.ajouter("12", _concept("12", "The ring road jams at 8am", T0, axe_objet="car"))
    lignes2, _ = foyer.croyances_partagees(ltm2, JACQUES, T0)
    assert len(lignes2) == 1


def test_R38_R4bis_marche_et_TC_traversent_pour_tout_le_monde():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Metro A is packed at 8", T0, axe_objet="public_transport"))
    lignes, _ = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert len(lignes) == 1


def test_R38_R4ter_un_mode_hors_table_ne_traverse_pas():
    """No permissive fallback: an unknown word would make the licence lock decorative."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Something", T0, axe_objet="teleportation"))
    lignes, refus = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert lignes == [] and refus["R4"] == 1


def test_R38_R2_une_confirmation_ne_fait_pas_repartir_un_concept():
    """Otherwise a belief confirmed every day would be served every evening to the whole family."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    concept = ltm.ajouter("49", _concept("49", "Metro A is packed at 8", T0))
    assert len(foyer.croyances_partagees(ltm, CONSTANCE, T0)[0]) == 1
    concept.observations += 1  # une simple confirmation
    lignes, refus = foyer.croyances_partagees(ltm, CONSTANCE, T0 + timedelta(days=1))
    assert lignes == [] and refus["R2"] == 1


def test_R38_R2bis_une_precision_le_fait_repartir():
    """The content has changed: it is no longer the same sentence."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    concept = ltm.ajouter("49", _concept("49", "Metro A is packed", T0))
    foyer.croyances_partagees(ltm, CONSTANCE, T0)
    concept.content = json.dumps(["Metro A is packed between 8 and 8:30", "", "", "", ""])
    lignes, _ = foyer.croyances_partagees(ltm, CONSTANCE, T0 + timedelta(days=1))
    assert len(lignes) == 1


def test_R38_R6_un_agent_seul_nentend_rien_et_ne_dit_rien():
    _indexer(CONSTANCE, JACQUES, XAVIER)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Metro A is packed", T0))
    lignes, refus = foyer.croyances_partagees(ltm, XAVIER, T0)
    assert lignes == [] and refus["R6"] == 1
    assert foyer.autres_membres("40") == []


def test_R42_un_membre_immobile_ne_raconte_rien():
    immobile = FauxAgent("77", "5177", immobile=True,
                         identity=FauxIdentite({"name": "Immobile", "age": 80}))
    _indexer(CONSTANCE, JACQUES, immobile)
    assert {m.person_id for m in foyer.autres_membres("12")} == {"49"}


# ── G3. Un seul saut ─────────────────────────────────────────────────────────────────────
def test_G3_une_croyance_entendue_ne_repart_jamais():
    """D2, and Q2 settled on 2026-09-21: even confirmed later, it does not go out again."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    entendu = ltm.ajouter(
        "49", _concept("49", "The school bus is always full", T0, origine="entendu")
    )
    assert foyer.croyances_partagees(ltm, CONSTANCE, T0)[0] == []
    # Confirmed fifteen times afterwards: it still does not go out again.
    entendu.observations = 15
    assert foyer.croyances_partagees(ltm, CONSTANCE, T0 + timedelta(days=10))[0] == []


def test_G3bis_une_croyance_vecue_sur_le_meme_sujet_circule():
    """D2 does not sterilise the receiver: its own day creates a belief which, for its part, goes out."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "The school bus is always full", T0, origine="entendu"))
    ltm.ajouter("49", _concept("49", "The school bus is full after 4pm", T0, origine="vecu"))
    lignes, _ = foyer.croyances_partagees(ltm, CONSTANCE, T0)
    assert len(lignes) == 1 and "after 4pm" in lignes[0]


def test_G3ter_une_entree_anterieure_au_ticket_est_tenue_pour_vecue():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Metro A is packed", T0, origine=None))
    assert len(foyer.croyances_partagees(ltm, CONSTANCE, T0)[0]) == 1


# ── G4. Le détecteur de reformulation ────────────────────────────────────────────────────
def test_G4_le_detecteur_signale_sans_jamais_couper():
    """A READING indicator. The memory already refuses elsewhere to let a similarity decide."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    for i, texte in enumerate((
        "The metro is unreliable in the morning",
        "The metro is unreliable during morning peak",
        "Morning metro service is unreliable",
    )):
        ltm.ajouter("49", _concept("49", texte, T0 + timedelta(days=i)))
    signales = foyer.detecter_reformulation(ltm, "5177")
    assert signales, "trois reformulations dans le même panier doivent se voir"
    _panier, combien, mots = signales[0]
    assert combien == 3
    assert "unreliable" in mots and "metro" in mots
    # And nothing was cut: the three concepts are still there.
    assert len(ltm.user_metadata["49"]["entries"]) == 3


def test_G4bis_un_panier_a_un_seul_concept_ne_signale_rien():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Metro A is packed", T0))
    assert foyer.detecter_reformulation(ltm, "5177") == []


# ── R35. No more bound on the members (ticket 118) ─────────────────────────────────────
def test_R35_tous_les_membres_sont_racontes_meme_nombreux():
    """The old bound `memoire__recit_soir_max` postponed the surplus members: it no longer exists."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    for i in range(4):
        ltm.ajouter("49", _reflexion("49", f"Day {i}.", T0 + timedelta(days=i)))
    lignes = foyer.recit_du_soir(ltm, "12", T0 + timedelta(days=9))
    assert len(lignes) == 1 and all(f"Day {i}." in lignes[0] for i in range(4))
    assert not hasattr(settings.agent, "memoire__recit_soir_max")


# ── R41. The provenance field, in both arms ──────────────────────────────────────
def test_R41_le_schema_demande_la_provenance_et_la_version_a_bouge():
    from urban_mobility_agents.agents.llm_agent import SCHEMA_REFLEXION_VERSION

    assert SCHEMA_REFLEXION_VERSION == 5, (
        "une réponse mémoïsée sans accusé de lecture pourrait ignorer le service presse"
    )
    schema = json.loads(
        (
            RACINE.parents[1] / "packages" / "mobility_llm" / "src" / "mobility_llm"
            / "categories" / "stm_reflection" / "output_schema.json"
        ).read_text("utf-8")
    )
    concept = schema["properties"]["agents"]["items"]["properties"]["concepts"]["items"]
    assert concept["properties"]["source"]["enum"] == ["lived", "heard"]
    assert "source" in concept["required"], (
        "demandé dans les DEUX bras : c'est ce qui garde une version de schéma unique"
    )
    agent = schema["properties"]["agents"]["items"]
    assert "press_service_considered" in agent["required"]


def test_la_provenance_rendue_par_le_modele_est_traduite():
    from urban_mobility_agents.agents.llm_agent import _normaliser_origine

    assert _normaliser_origine("heard") == "entendu"
    assert _normaliser_origine("lived") == "vecu"
    # The least inventive fallback, but it leaves a trace: hearsay taken for lived experience
    # WOULD GO OUT AGAIN into the household, which D2 forbids.
    assert _normaliser_origine("dunno") == "vecu"
    assert _normaliser_origine(None) == "vecu"


# ── L'index des ménages ──────────────────────────────────────────────────────────────────
def test_lindex_alarme_quand_aucun_agent_ne_porte_de_menage(caplog):
    assert foyer.initialiser([FauxAgent("1"), FauxAgent("2")]) == 0
    assert foyer.autres_membres("1") == []


def test_le_bloc_complet_porte_les_deux_choses():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "The ring road was slow.", T0))
    ltm.ajouter("49", _concept("49", "Metro A is packed at 8", T0))
    bloc = foyer.bloc_du_soir(ltm, CONSTANCE, T0 + timedelta(hours=1))
    assert bloc.startswith("Tonight at home")
    assert "told you about their day" in bloc
    assert "they have seen it" in bloc


# ── Measure no. 1 of the household study, which comes before all the others ─────────────────────────
def test_les_refus_par_regle_sont_COMPTES_et_non_jetes():
    """They were computed then thrown away: the first measure was emitted nowhere.

    « If this number is close to zero, the channel is empty and nothing else makes sense to
    measure » — household study, measure no. 1.
    """
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _concept("49", "Bus 62 is often late", T0, observations=0))
    ltm.ajouter("49", _concept("49", "The ring road jams", T0, axe_objet="car"))
    foyer.bloc_du_soir(ltm, CONSTANCE, T0)
    c = foyer.compteurs()
    assert c["candidats"] == 2
    assert c["refus_R1"] == 1, "le concept non ancré"
    assert c["refus_R4"] == 1, "la voiture, chez une enfant de 12 ans"
    assert c["receveurs"] == 1


def _capturer(niveau: str = "INFO") -> tuple[list, int]:
    """Loguru does not feed `caplog`: we plug in a sink, like the rest of the repository."""
    from loguru import logger as _logger

    messages: list[str] = []
    jeton = _logger.add(lambda m: messages.append(str(m)), level=niveau)
    return messages, jeton


def test_le_bilan_se_journalise_meme_a_zero():
    """A mute counter does not tell « nothing to say » from « the mechanism is not running »."""
    from loguru import logger as _logger

    _indexer(CONSTANCE, JACQUES)
    messages, jeton = _capturer()
    try:
        foyer.journaliser_compteurs()
    finally:
        _logger.remove(jeton)
    assert any("bilan du run" in m for m in messages)


def test_des_blocs_servis_sans_aucune_croyance_levent_une_alarme():
    """The case that the matched campaign makes likely: 225 concepts out of 231 left at zero observation."""
    from loguru import logger as _logger

    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "A quiet day.", T0))
    ltm.ajouter("49", _concept("49", "Bus 62 is often late", T0, observations=0))
    foyer.bloc_du_soir(ltm, CONSTANCE, T0)
    messages, jeton = _capturer("ERROR")
    try:
        foyer.journaliser_compteurs()
    finally:
        _logger.remove(jeton)
    assert any("[ALARME]" in m and "R1 d'abord" in m for m in messages)


# ── The anti-loop guard is counted, and the count adds up ─────────────────────────────────
def test_G3_les_concepts_entendus_sont_COMPTES_et_pas_seulement_ecartes():
    """A guard that nobody knows whether it bit cannot be checked.

    Measured on the run of the read channel of 2026-09-22: 426 concepts examined, 296 outcomes counted.
    The 130 missing were the `origine: entendu` concepts, discarded by G3 without a single
    line saying so. The single hop is a CLAIM of the paper; it must be read in a
    journal, not deduced from a subtraction.
    """
    source = (RACINE / "llm" / "foyer.py").read_text("utf-8")
    assert 'refus["G3"] += 1' in source, "l'écart par G3 doit être compté"
    assert '"G3": 0' in source, "le compteur G3 doit exister dès l'initialisation"
    assert "refus_G3" in source, "et remonter dans le bilan du run"


def test_le_bilan_du_foyer_alarme_si_le_compte_ne_tombe_pas_juste():
    """Every exit of the loop must be counted somewhere — otherwise a refusal is invisible."""
    source = (RACINE / "llm" / "foyer.py").read_text("utf-8")
    assert "le compte ne tombe pas juste" in source
    assert "quittent la boucle sans être comptés" in source


# ── The account is given IN THE EVENING, once (analysis of 2026-09-25) ───────────────────────────
# On the treated arm of 2026-09-24_17_50, 18 [ALARME] « evening account TRUNCATED »: the block was
# served at EACH consolidation (three per agent and per day at the median, seven at most), each
# consolidation writes a summary, and a receiver who rarely consolidated found 9 to 11 of them from
# the others. The bound did not bite because of a broken marker, but because « one summary per
# member and per night » was false. The paper says « in the evening »: the code says so too.
SOIR = datetime(2026, 3, 16, 20, 0)


def test_le_foyer_ne_parle_pas_en_journee_et_rien_nest_perdu():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Morning rush on line A.", SOIR - timedelta(hours=11)))
    midi = SOIR - timedelta(hours=7, minutes=20)
    assert foyer.bloc_du_soir(ltm, CONSTANCE, midi) == ""
    assert foyer.compteurs()["hors_soir"] == 1
    assert foyer.compteurs().get("receveurs", 0) == 0, "une consolidation de jour n'examine rien"
    bloc = foyer.bloc_du_soir(ltm, CONSTANCE, SOIR)
    assert "Morning rush on line A." in bloc, "ce qui n'est pas raconté à midi l'est le soir"


def test_un_seul_recit_par_soir_le_reste_attend_le_lendemain():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "First.", SOIR - timedelta(hours=2)))
    assert "First." in foyer.bloc_du_soir(ltm, CONSTANCE, SOIR)
    ltm.ajouter("49", _reflexion("49", "Second.", SOIR + timedelta(hours=1)))
    assert foyer.bloc_du_soir(ltm, CONSTANCE, SOIR + timedelta(hours=2)) == ""
    # 01:30 appartient encore à la journée simulée de la veille (frontière à 3 h).
    assert foyer.bloc_du_soir(ltm, CONSTANCE, SOIR + timedelta(hours=5, minutes=30)) == ""
    assert foyer.compteurs()["deja_servi_ce_soir"] == 2
    lendemain = foyer.bloc_du_soir(ltm, CONSTANCE, SOIR + timedelta(days=1))
    assert "Second." in lendemain and "First." not in lendemain


def test_un_soir_sans_rien_a_dire_ne_ferme_pas_la_soiree():
    """Nothing new at 6 p.m. does not deprive the receiver of what the other will tell at 9 p.m."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    assert foyer.bloc_du_soir(ltm, CONSTANCE, SOIR - timedelta(hours=2)) == ""
    ltm.ajouter("49", _reflexion("49", "Late news.", SOIR + timedelta(hours=1)))
    assert "Late news." in foyer.bloc_du_soir(ltm, CONSTANCE, SOIR + timedelta(hours=2))


def test_une_ligne_par_membre_qui_cite_tous_ses_bilans():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    for i, h in enumerate((8, 13, 19)):
        ltm.ajouter("49", _reflexion("49", f"Bilan {i}.", SOIR.replace(hour=h)))
    lignes = foyer.recit_du_soir(ltm, "12", SOIR + timedelta(hours=1))
    assert len(lignes) == 1
    assert all(f"Bilan {i}." in lignes[0] for i in range(3))
    assert lignes[0].index("Bilan 0.") < lignes[0].index("Bilan 2."), "dans l'ordre du jour"


def test_le_repere_suit_le_dernier_bilan_cite_pas_lheure_du_receveur():
    """A summary written AFTER the receiver's turn but dated before is no longer lost.

    The EDF queue does not serve the consolidations in simulated order: the old marker, set at
    the receiver's time, swallowed any summary of another member dated before that time and
    arrived afterwards.
    """
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Early.", SOIR))
    assert foyer.recit_du_soir(ltm, "12", SOIR + timedelta(hours=2))
    ltm.ajouter("49", _reflexion("49", "Arrived late in the queue.", SOIR + timedelta(hours=1)))
    suite = foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=1))
    assert suite and "Arrived late in the queue." in suite[0]


def test_aucune_troncature_tout_est_cite_et_lexces_salarme_sur_front_montant():
    """Ticket 118, O1 — the evening account is NEVER truncated (author, 2026-09-29).

    Counter-check: with the old bound of 8 summaries per member, 20 pending summaries would have
    been spread over three evenings. Here everything goes the first evening, nothing waits for the next one, and
    the excess only produces an alarm, on a rising edge.
    """
    from loguru import logger as _logger

    _indexer(CONSTANCE, JACQUES)
    messages, jeton = _capturer("ERROR")
    try:
        ltm = FausseLTM()
        for i in range(20):
            ltm.ajouter("49", _reflexion("49", f"Day {i}.", SOIR + timedelta(days=i)))
        lignes = foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=21))
        assert len(lignes) == 1
        assert all(f'"Day {i}."' in lignes[0] for i in range(20)), "tout est cité, dans l'ordre"
        # A second evening in excess, right after, does not trigger the alarm again.
        for i in range(20, 40):
            ltm.ajouter("49", _reflexion("49", f"Day {i}.", SOIR + timedelta(days=22, minutes=i)))
        foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=23))
        premier_front = len([m for m in messages if "anormalement long" in m])
        # An evening with nothing pending re-arms; the next excess is a new edge.
        assert foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=24)) == [], "rien n'attend"
        for i in range(40, 50):
            ltm.ajouter("49", _reflexion("49", f"Day {i}.", SOIR + timedelta(days=25, minutes=i)))
        foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=26))
    finally:
        _logger.remove(jeton)
    assert premier_front == 1, "deux soirs en excès de suite ne font qu'une alarme"
    alarmes = [m for m in messages if "[ALARME]" in m and "anormalement long" in m]
    assert len(alarmes) == 2, "après un soir normal, un nouvel excès est un nouveau front"
    assert foyer.compteurs()["exces"] == 3
    assert foyer.compteurs()["bilans_cites"] == 50
    assert "troncatures" not in foyer.compteurs()


def test_sous_le_seuil_dalerte_aucune_alarme():
    from loguru import logger as _logger

    _indexer(CONSTANCE, JACQUES)
    messages, jeton = _capturer("ERROR")
    try:
        ltm = FausseLTM()
        for i in range(8):
            ltm.ajouter("49", _reflexion("49", f"Day {i}.", SOIR + timedelta(hours=i)))
        assert foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=1))
    finally:
        _logger.remove(jeton)
    assert not [m for m in messages if "anormalement long" in m]


def test_letat_du_foyer_se_relit_depuis_le_point_restaure():
    """Ticket 118, O1 — the factory resets the household; the restored point then gives it back.

    On the a13 v5 control, the state was written in each point but never reread: the first evening
    after a resume quoted all the summaries since the first day (112 against 14).
    """
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    for i in range(5):
        ltm.ajouter("49", _reflexion("49", f"Old {i}.", SOIR - timedelta(days=5 - i)))
    assert foyer.recit_du_soir(ltm, "12", SOIR - timedelta(hours=1))
    meta = {"jour_simule": 6, "foyer": foyer.etat_pour_reprise()}
    # What the factory does at each /init: the index starts again, the state with it.
    foyer.reinitialiser()
    _indexer(CONSTANCE, JACQUES)
    foyer.restaurer_depuis_point(meta)
    ltm.ajouter("49", _reflexion("49", "Tonight only.", SOIR + timedelta(hours=1)))
    lignes = foyer.recit_du_soir(ltm, "12", SOIR + timedelta(hours=2))
    assert lignes and "Tonight only." in lignes[0]
    assert not any(f"Old {i}." in lignes[0] for i in range(5)), "déjà entendu avant la reprise"


def test_un_point_sans_etat_du_foyer_salarme(monkeypatch):
    from loguru import logger as _logger

    monkeypatch.setattr(settings.agent, "memoire__partage_foyer_enabled", True)
    messages, jeton = _capturer("ERROR")
    try:
        foyer.restaurer_depuis_point({"jour_simule": 4})
        foyer.restaurer_depuis_point({"jour_simule": 1, "foyer": {}})
    finally:
        _logger.remove(jeton)
    alarmes = [m for m in messages if "[ALARME]" in m and "ne porte pas l'état du foyer" in m]
    assert len(alarmes) == 1, "un état vide est légitime, une clé absente ne l'est pas"


def test_la_restauration_du_foyer_est_cablee_apres_la_fabrique():
    """Relu avant `init_dynamic_scenario`, l'état serait effacé par `foyer.reinitialiser()`."""
    source = (RACINE / "handle" / "application.py").read_text("utf-8")
    init = source[source.index("async def init(request"):]
    assert init.index("init_dynamic_scenario(") < init.index("restaurer_depuis_point(")
    assert init.index("_point_restaure = restaurer_si_demande(") < init.index("init_dynamic_scenario(")


def test_un_repere_de_lancienne_forme_se_relit_encore():
    """A resume point written before 2026-09-25 carries a marker per receiver alone."""
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "Already heard.", SOIR))
    ltm.ajouter("49", _reflexion("49", "New one.", SOIR + timedelta(days=1)))
    foyer.charger_etat({"lu_jusqu_a": {"12": (SOIR + timedelta(hours=1)).isoformat()}})
    lignes = foyer.recit_du_soir(ltm, "12", SOIR + timedelta(days=1, hours=1))
    assert lignes and "New one." in lignes[0] and "Already heard." not in lignes[0]


def test_la_soiree_servie_survit_a_une_reprise():
    _indexer(CONSTANCE, JACQUES)
    ltm = FausseLTM()
    ltm.ajouter("49", _reflexion("49", "A day.", SOIR - timedelta(hours=1)))
    assert foyer.bloc_du_soir(ltm, CONSTANCE, SOIR)
    sauvegarde = foyer.etat_pour_reprise()
    foyer.reinitialiser()
    _indexer(CONSTANCE, JACQUES)
    foyer.charger_etat(sauvegarde)
    ltm.ajouter("49", _reflexion("49", "Another.", SOIR + timedelta(hours=1)))
    assert foyer.bloc_du_soir(ltm, CONSTANCE, SOIR + timedelta(hours=2)) == ""


def test_les_reglages_du_soir_ont_leurs_defauts():
    from settings import AgentConfig

    assert AgentConfig().memoire__recit_soir_heure == 18
    assert AgentConfig().memoire__recit_soir_alerte_par_membre == 8
