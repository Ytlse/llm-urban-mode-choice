"""FUNCTIONAL tests of the severity chain, with test doubles.

What is doubled: the LLM gateway (the test decides what the model answers, including
answering badly), the vector index, the memoisation store and the disk. What stays real:
severity, writing to short-term memory, accumulation per disruption, the maximum rule, the
construction of long-term entries, decay, reinforcement and cleanup.
"""

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.gravite import force_initiale, gravite_deterministe
from llm.longterm import MultiUserLongTermMemory
from llm.memory import MemoryEntry, MemoryType
from llm.shortterm import UserShortTermMemory
from settings import settings
from sim_clock import gama_timestamp, wall_clock
from urban_mobility_agents.agents.llm_agent import (
    SCHEMA_REFLEXION_VERSION,
    _gravite_de_la_contrainte,
    _normaliser_concept,
)

T0 = 1773637200  # 16 March 2026, 05:00 in simulation WALL-CLOCK time

# The five scenarios of the specification, in deterministic severity.
I_NOMINAL = gravite_deterministe(0)[0]
I_QUART = gravite_deterministe(900)[0]
I_DEMI = gravite_deterministe(1800)[0]
I_PETIT = gravite_deterministe(540)[0]
I_PANNE = gravite_deterministe(2700, correspondance_ratee=True, mode_contraint=True)[0]


# ══════════════════════════ test doubles ════════════════════════════════════════


class _Noeud:
    def __init__(self, doc_id, ts, person_id="42", score=0.9):
        self.text = doc_id
        self.score = score
        self.metadata = {
            "person_id": person_id,
            "timestamp": ts.isoformat(),
            "memory_type": "reflection",
            "tags": "bus 401",
            "doc_id": doc_id,
        }


class _IndexDouble:
    def __init__(self, noeuds):
        self._noeuds = noeuds

    def as_retriever(self, **_k):
        class _R:
            async def aretrieve(_self, _q):
                return list(self._noeuds)

        return _R()


class _LtmDouble(MultiUserLongTermMemory):
    """Real long-term memory, without index or disk."""

    def __init__(self, noeuds=()):
        self.shared_index = _IndexDouble(noeuds)
        self.user_metadata = {}
        self.metadata_access_times = {}
        self.long_term_memory_filter_by_datetime = False
        self.metrics = {"queries": 0, "cache_hits": 0, "cache_misses": 0}
        self._dirty = set()

    def ensure_user_initialized(self, person_id):
        self.user_metadata.setdefault(person_id, {"entries": []})

    def _save_user_metadata(self, person_id):
        pass

    def _schedule_flush(self):
        pass


@pytest.fixture()
def fenetre_large():
    origine = settings.agent.long_term_max_days_query
    settings.agent.long_term_max_days_query = 60
    yield
    settings.agent.long_term_max_days_query = origine


# ═════════════ C. Decay since the last recall, reinforcement ════════════════════


def _ltm_avec_entree(force=None, ecrit_il_y_a=10, rappele_il_y_a=None):
    base = wall_clock(T0)
    ltm = _LtmDouble([_Noeud("42_0", base - timedelta(days=ecrit_il_y_a))])
    entree = MemoryEntry(
        content="42_0",
        timestamp=base - timedelta(days=ecrit_il_y_a),
        memory_type=MemoryType.REFLECTION,
        person_id="42",
        doc_id="42_0",
        force=force,
        dernier_rappel=(
            None if rappele_il_y_a is None else base - timedelta(days=rappele_il_y_a)
        ),
    )
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [entree]
    return ltm, entree, base


def _servir(ltm, quand, top_k=10):
    return asyncio.run(
        ltm.aquery_user_memories(
            person_id="42",
            query="bus 401",
            top_k=top_k,
            max_past_days=60,
            query_at=gama_timestamp(quand),
        )
    )


def test_C7_la_decroissance_part_du_dernier_rappel(fenetre_large):
    """A ten-day-old memory, recalled yesterday, weighs like a one-day-old memory.

    Without the `dernier_rappel` field, the gap between 0.77 and 0.03 is invisible: that is the
    whole point of the distinction, and it is what justifies the field.
    """
    ltm, entree, base = _ltm_avec_entree(force=3.8, ecrit_il_y_a=10, rappele_il_y_a=1)
    poids = ltm._time_decay_score(None, gama_timestamp(base), entree=entree)
    assert poids == pytest.approx(0.77, abs=0.01)

    sans_rappel = MemoryEntry(
        content="x", timestamp=base - timedelta(days=10),
        memory_type=MemoryType.REFLECTION, person_id="42", force=2.8,
    )
    assert ltm._time_decay_score(None, gama_timestamp(base), entree=sans_rappel) == pytest.approx(
        0.03, abs=0.01
    )


def test_C8_l_horodatage_d_origine_ne_bouge_jamais_au_rappel(fenetre_large):
    """It is shown in the prompt: shifting it would rewrite the agent's history."""
    ltm, entree, base = _ltm_avec_entree(force=2.8, ecrit_il_y_a=3)
    ecrit_le = entree.timestamp
    for _ in range(3):
        _servir(ltm, base)
    assert entree.timestamp == ecrit_le
    assert entree.rappels == 3
    assert entree.force == pytest.approx(2.8 + 3)
    assert entree.dernier_rappel == base


def test_C9_seules_les_entrees_servies_sont_renforcees(fenetre_large):
    """A candidate left out of the top-K has not been recalled."""
    base = wall_clock(T0)
    noeuds = [_Noeud(f"42_{i}", base - timedelta(days=1), score=0.9 - i * 0.1) for i in range(4)]
    ltm = _LtmDouble(noeuds)
    ltm.ensure_user_initialized("42")
    entrees = [
        MemoryEntry(
            content=f"42_{i}", timestamp=base - timedelta(days=1),
            memory_type=MemoryType.REFLECTION, person_id="42",
            doc_id=f"42_{i}", force=2.8,
        )
        for i in range(4)
    ]
    ltm.user_metadata["42"]["entries"] = entrees

    servis = _servir(ltm, base, top_k=2)
    ids_servis = {r.metadata["doc_id"] for r in servis}
    assert len(ids_servis) == 2

    for e in entrees:
        if e.doc_id in ids_servis:
            assert e.rappels == 1, f"{e.doc_id} was served and must be reinforced"
        else:
            assert e.rappels == 0, f"{e.doc_id} was not served"


def test_C10_le_renforcement_survit_au_tour_suivant(fenetre_large):
    """Two successive recalls add up: the strength rises by two days, not one."""
    ltm, entree, base = _ltm_avec_entree(force=2.8, ecrit_il_y_a=1)
    _servir(ltm, base)
    _servir(ltm, base)
    assert entree.force == pytest.approx(4.8)
    assert entree.rappels == 2


# ═══════════════════ D. Triggering by disruption ════════════════════════════════


def _tampon(*gravites) -> UserShortTermMemory:
    mem = UserShortTermMemory("42")
    base = wall_clock(T0)
    for i, g in enumerate(gravites):
        mem.add_message(f"obs {i}", base + timedelta(minutes=i), activity_id="a1", importance=g)
    return mem


def _declenche_par_rupture(mem: UserShortTermMemory) -> bool:
    """The controller's exact condition, on the real buffer."""
    return (
        len(mem.recent_entries) > 0
        and mem.gravite_cumulee() >= settings.agent.memoire__theta_gravite_cumulee
    )


@pytest.mark.parametrize(
    ("nom", "gravites", "attendu"),
    [
        ("D1 deux quarts d'heure", (I_QUART, I_QUART), False),          # 0.50
        ("D2 trois quarts d'heure", (I_QUART, I_QUART, I_QUART), True), # 0.75
        ("D3 la panne, dès la première entrée", (I_PANNE,), True),      # 0.80
        ("D4 une demi-heure et un petit retard", (I_DEMI, I_PETIT), False),  # 0.65
        ("D5 exactement au seuil", (I_DEMI, 0.20), True),               # 0.70
        ("D6 dix trajets nominaux", tuple([I_NOMINAL] * 10), False),    # 0.00
    ],
)
def test_D_le_cumul_de_gravite_declenche_ou_non(nom, gravites, attendu):
    assert _declenche_par_rupture(_tampon(*gravites)) is attendu, nom


def test_D5bis_le_seuil_est_atteint_et_non_depasse():
    """`I_cumul >= Θ`: at exactly Θ, it triggers."""
    mem = _tampon(settings.agent.memoire__theta_gravite_cumulee)
    assert mem.gravite_cumulee() == pytest.approx(settings.agent.memoire__theta_gravite_cumulee)
    assert _declenche_par_rupture(mem) is True


def test_D8_un_tampon_vide_ne_declenche_rien():
    assert _declenche_par_rupture(UserShortTermMemory("42")) is False


def test_D10_la_consommation_du_tampon_remet_le_cumul_a_zero():
    """An agent in disruption must not loop on itself."""
    mem = _tampon(I_PANNE)
    assert _declenche_par_rupture(mem) is True
    mem.remove_batch(list(mem.recent_entries))
    assert mem.gravite_cumulee() == 0.0
    assert _declenche_par_rupture(mem) is False


def test_D9_l_alarme_de_rarete_part_sur_front_montant(caplog):
    """Less than one trigger per agent and per simulated week, otherwise Θ is too low."""
    from urban_mobility_agents.simulation_controller import SimulationLoopV1

    loop = object.__new__(SimulationLoopV1)
    loop._ruptures = []
    loop._rupture_alarme_on = False

    with caplog.at_level("ERROR"):
        loop._compter_ruptures(T0, {f"p{i}" for i in range(3)}, agents_total=10)
    assert loop._rupture_alarme_on is False, "0.3 per agent: rare regime, no alarm"

    with caplog.at_level("ERROR"):
        loop._compter_ruptures(T0 + 3600, {f"q{i}" for i in range(8)}, agents_total=10)
    assert loop._rupture_alarme_on is True, "1.1 per agent: the regime is no longer exceptional"

    avant = len([r for r in caplog.records if "[ALARME]" in r.message])
    loop._compter_ruptures(T0 + 7200, {"r1"}, agents_total=10)
    apres = len([r for r in caplog.records if "[ALARME]" in r.message])
    assert apres == avant, "rising edge: the alarm does not repeat"


def test_D11_la_fenetre_de_rarete_glisse_en_temps_simule():
    """Disruptions older than one simulated week leave the count."""
    from urban_mobility_agents.simulation_controller import SimulationLoopV1

    loop = object.__new__(SimulationLoopV1)
    loop._ruptures = []
    loop._rupture_alarme_on = False
    loop._compter_ruptures(T0, {f"p{i}" for i in range(12)}, agents_total=10)
    assert loop._rupture_alarme_on is True, "1.2 per agent: the regime is no longer rare"
    # eight days later, the twelve disruptions are out of the window
    loop._compter_ruptures(T0 + 8 * 86400, set(), agents_total=10)
    assert loop._ruptures == []
    assert loop._rupture_alarme_on is False


# ═══════════ G. Forward and backward compatibility ═════════════════════════════


def test_G1_une_entree_d_avant_le_lot_1_se_relit(fenetre_large):
    """None of the new fields, and yet it reloads and is served."""
    ancien = {
        "content": "le bus 401 est ponctuel",
        "timestamp": wall_clock(T0).isoformat(),
        "memory_type": "concept",
        "person_id": "42",
        "activity_id": None,
        "tags": "bus 401",
        "doc_id": "42_0",
    }
    e = MemoryEntry.from_dict(ancien)
    assert e.importance == 0.0
    assert e.force is None
    assert e.rappels == 0
    assert e.dernier_rappel is None
    assert e.horodatage_de_reference == e.timestamp


def test_G2_le_retour_en_arriere_reste_possible():
    """An entry written with severity, read back by code that ignores its new fields."""
    e = MemoryEntry(
        content="x", timestamp=wall_clock(T0), memory_type=MemoryType.CONCEPT,
        person_id="42", importance=0.8, force=16.24, rappels=3,
    )
    brut = e.to_dict()
    brut["un_champ_d_une_autre_version"] = {"quelconque": True}
    relu = MemoryEntry.from_dict(brut)
    assert relu.importance == pytest.approx(0.8)
    assert relu.force == pytest.approx(16.24)
    assert relu.content == "x"


def test_G2bis_le_round_trip_conserve_le_dernier_rappel():
    base = wall_clock(T0)
    e = MemoryEntry(
        content="x", timestamp=base, memory_type=MemoryType.REFLECTION,
        person_id="42", dernier_rappel=base + timedelta(days=2),
    )
    relu = MemoryEntry.from_dict(e.to_dict())
    assert relu.dernier_rappel == base + timedelta(days=2)
    assert relu.horodatage_de_reference == base + timedelta(days=2)


def test_G3_les_deux_formats_de_concept_se_lisent():
    """A RESUMED run reads back concepts and a cache written in the former format."""
    ancien = ["Bus 401 is reliable", "bus 401", "Bus 401", "morning", "work"]
    lu = _normaliser_concept(ancien)
    assert lu.cinq == ancien
    assert lu.niveau is None, "no level had been requested: it will fall back on I_det"
    assert lu.valence == "neutre"
    assert lu.mode is None, "the former format carries no mode"
    assert lu.operation == "creer", "ni opération ni cible : le repli le moins destructeur"
    assert lu.cible == ""

    nouveau = {
        "content": "Bus 401 is reliable", "keywords": "bus 401",
        "spatial_scope": "Bus 401", "temporal_scope": "morning", "purpose": "work",
        "mode": "public_transport", "severity": "serious", "valence": "negative",
        "operation": "contradict", "target_id": "K2",
    }
    lu2 = _normaliser_concept(nouveau)
    assert lu2.cinq == ancien
    assert lu2.niveau == "serious"
    assert lu2.valence == "negative"
    assert lu2.mode == "public_transport"
    assert lu2.operation == "contredire"
    assert lu2.cible == "K2"

    # "any" is an ANSWER, not an absence: the concept speaks of no mode, and its axis
    # stays empty rather than matching all modes.
    assert _normaliser_concept({**nouveau, "mode": "any"}).mode is None


def test_G4_la_cle_de_memoisation_porte_la_version_de_schema():
    """A response memoised under the old schema must MISS the cache, not be served."""
    from llm.reflection_store import ReflectionMemoStore

    commun = dict(
        person_id="42", category="stm_reflection", identity="id",
        context_text="vécu", guidelines="", departure_timestamp=1.0, llm_params={},
    )
    assert ReflectionMemoStore.make_key(**commun, schema_version=0) != (
        ReflectionMemoStore.make_key(**commun, schema_version=SCHEMA_REFLEXION_VERSION)
    )


# ═════════════════ H. Severity of the chain constraint ═════════════════════════


class _ContexteFictif:
    def __init__(self, contrainte):
        self.data = {"type": "travel_plan", "contrainte_chaine": contrainte}


@pytest.mark.parametrize(
    ("contrainte", "attendu"),
    [
        ("", 0.0),
        ("retour_force", 0.10),
        ("sortie_bloquee", 0.10),
        ("passager", 0.0),  # a household arrangement, not a suffered degradation
    ],
)
def test_H3_la_contrainte_de_chaine_donne_sa_gravite_a_la_decision(contrainte, attendu):
    assert _gravite_de_la_contrainte(_ContexteFictif(contrainte)) == pytest.approx(attendu)


def test_H4_un_contexte_sans_contrainte_ne_leve_pas():
    class _Vide:
        data = None

    assert _gravite_de_la_contrainte(_Vide()) == 0.0


# ═════════════════ H1. The chain, one agent day ════════════════════════════════


def test_H1_de_l_observation_a_la_duree_de_vie_du_concept():
    """The complete path of severity, without going through the gateway.

    Four observations including the breakdown, a buffer that crosses the threshold, a model that
    UNDERESTIMATES, and the resulting lifetime.
    """
    mem = _tampon(I_NOMINAL, I_QUART, I_PANNE, I_NOMINAL)

    # 1. the day constitutes a disruption
    # ⚠ The 45-minute breakdown is worth 0.94 and no longer 0.80: above the reference
    # delay, the delay component no longer plateaus. The day's accumulation therefore rises
    # too, and the disruption is crossed earlier.
    assert mem.gravite_cumulee() == pytest.approx(0.25 + 0.9426990406, abs=1e-9)
    assert _declenche_par_rupture(mem) is True

    # 2. the floor is the worst of the day, not the last nor the average
    plancher = mem.gravite_maximale()
    assert plancher == pytest.approx(0.9426990406, abs=1e-9)

    # 3. the model underestimates: the fact takes over again
    from llm.gravite import gravite_concept, gravite_jugee

    importance = gravite_concept(gravite_jugee("noticeable"), plancher)
    assert importance == pytest.approx(
        0.9426990406, abs=1e-9
    ), "the judgement cannot downgrade a measured fact"

    # 4. the lifetime follows the retained severity
    assert force_initiale(importance) == pytest.approx(18.64, abs=0.01)


def test_H2_un_jour_sans_rien_ne_declenche_ni_ne_qualifie():
    """A device silent when all is well cannot tell "it works" from "it no longer runs"."""
    mem = _tampon(*([I_NOMINAL] * 4))
    assert mem.gravite_cumulee() == 0.0
    assert mem.gravite_maximale() == 0.0
    assert _declenche_par_rupture(mem) is False
    # and the lifetime of an ordinary day's memory stays the default one
    assert force_initiale(mem.gravite_maximale()) == pytest.approx(2.8)
