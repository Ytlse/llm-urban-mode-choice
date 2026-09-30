"""FUNCTIONAL tests of the launch and recall chain (age window), with test doubles.

The tests of `test_memory_constants.py` lock values. These ones lock
a BEHAVIOUR, end to end and through the same calls as production:

- LAUNCH chain: experiment definition → `appliquer_fenetre_age` → configuration;
- RECALL chain: configuration → `aquery_user_memories` → memories actually served.

The vector index and the embedding model are doubled: they are not the subject of the test
and loading them would cost several seconds per case. Everything else is production code,
including the age filter, the working-day filter and the ranking.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.longterm import MultiUserLongTermMemory
from settings import settings
from sim_clock import gama_timestamp, wall_clock

# 16 March 2026, 05:00 in simulation WALL-CLOCK time.
T0 = 1773637200


# ─────────────────────────────── test doubles ───────────────────────────────────


class _ExperienceFictive:
    """The bare minimum of what `appliquer_fenetre_age` reads from an experiment."""

    def __init__(self, horizon_jours: int, memoire: bool = True):
        self.horizon_jours = horizon_jours
        self.memoire = memoire


class _Noeud:
    """What the LlamaIndex retriever returns: a text, a score, metadata."""

    def __init__(self, texte: str, score: float, ts, person_id: str):
        self.text = texte
        self.score = score
        self.metadata = {
            "person_id": person_id,
            "timestamp": ts.isoformat(),
            "memory_type": "reflection",
            "tags": "bus 401",
            "doc_id": f"{person_id}_{texte}",
        }


class _RetrieverDouble:
    def __init__(self, noeuds):
        self._noeuds = noeuds

    async def aretrieve(self, _query):
        return list(self._noeuds)


class _IndexDouble:
    """Double of the shared index: returns fixed nodes, without embedding or ChromaDB."""

    def __init__(self, noeuds):
        self._noeuds = noeuds
        self.requetes = 0

    def as_retriever(self, **_kwargs):
        self.requetes += 1
        return _RetrieverDouble(self._noeuds)


class _LtmDouble(MultiUserLongTermMemory):
    """Real long-term memory, plugged into a doubled index and without disk."""

    def __init__(self, noeuds):
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


@pytest.fixture()
def fenetre_restauree():
    """The window is a process setting: the test must not leave it modified."""
    origine = settings.agent.long_term_max_days_query
    yield
    settings.agent.long_term_max_days_query = origine


# ──────────────────── 1. launch chain, with a doubled experiment ────────────────


def test_le_lancement_regle_la_fenetre_sur_l_horizon(fenetre_restauree, capsys):
    from experiences.cli import appliquer_fenetre_age

    retenue = appliquer_fenetre_age(_ExperienceFictive(horizon_jours=5))

    assert retenue == 5
    assert settings.agent.long_term_max_days_query == 5


def test_le_lancement_plafonne_un_horizon_trop_long(fenetre_restauree):
    from experiences.cli import appliquer_fenetre_age

    retenue = appliquer_fenetre_age(_ExperienceFictive(horizon_jours=90))

    assert retenue == settings.agent.memoire__fenetre_age_max_jours == 60


def test_le_lancement_annonce_la_fenetre_retenue(fenetre_restauree, capsys):
    """A silent setting cannot be told apart from a missing setting.

    The window decides what the agent can recall: it must be readable in the
    launch output, without having to reread the code.
    """
    from experiences.cli import appliquer_fenetre_age

    appliquer_fenetre_age(_ExperienceFictive(horizon_jours=60))
    sortie = capsys.readouterr().out

    assert "fenêtre d'âge au rappel" in sortie
    assert "60 j" in sortie


def test_la_fenetre_est_reglee_meme_si_la_memoire_est_coupee(fenetre_restauree, capsys):
    """An inert value is better than a wrong value if memory is switched back on."""
    from experiences.cli import appliquer_fenetre_age

    assert appliquer_fenetre_age(_ExperienceFictive(horizon_jours=7, memoire=False)) == 7
    assert settings.agent.long_term_max_days_query == 7
    assert "coupée" in capsys.readouterr().out


# ──────────────── 2. recall chain, from configuration to memories served ────────


def _memoire_a_trois_ages(person_id="42"):
    """Three memories: 10, 45 and 70 days before the query."""
    base = wall_clock(T0)
    return _LtmDouble(
        [
            _Noeud("recent", 0.9, base - timedelta(days=10), person_id),
            _Noeud("second_mois", 0.9, base - timedelta(days=45), person_id),
            _Noeud("hors_horizon", 0.9, base - timedelta(days=70), person_id),
        ]
    )


@pytest.mark.asyncio
async def test_un_run_de_soixante_jours_sert_son_second_mois(fenetre_restauree):
    """This is the defect the age window fixes, checked by failure.

    At 30 days — the previous value — the 45-day memory was excluded from recall, hence
    invisible to the decision, without any log line saying so.
    """
    from experiences.cli import appliquer_fenetre_age

    ltm = _memoire_a_trois_ages()
    appliquer_fenetre_age(_ExperienceFictive(horizon_jours=60))

    servis = await ltm.aquery_user_memories(
        person_id="42",
        query="bus 401",
        top_k=10,
        max_past_days=settings.agent.long_term_max_days_query,
        query_at=gama_timestamp(wall_clock(T0)),
    )
    contenus = {r.content for r in servis}

    assert "recent" in contenus
    assert "second_mois" in contenus, "the second month is recalled again"


@pytest.mark.asyncio
async def test_le_plafond_de_soixante_jours_ecarte_toujours_au_dela(fenetre_restauree):
    """The cap remains a real filter: the window opens, it does not disappear."""
    from experiences.cli import appliquer_fenetre_age

    ltm = _memoire_a_trois_ages()
    appliquer_fenetre_age(_ExperienceFictive(horizon_jours=60))

    servis = await ltm.aquery_user_memories(
        person_id="42",
        query="bus 401",
        top_k=10,
        max_past_days=settings.agent.long_term_max_days_query,
        query_at=gama_timestamp(wall_clock(T0)),
    )

    assert "hors_horizon" not in {r.content for r in servis}


@pytest.mark.asyncio
async def test_un_horizon_court_referme_la_fenetre(fenetre_restauree):
    """The window follows the horizon IN BOTH DIRECTIONS.

    In a five-day experiment, a forty-five-day-old memory has no business being
    in the recall: it would come from another run.
    """
    from experiences.cli import appliquer_fenetre_age

    ltm = _memoire_a_trois_ages()
    appliquer_fenetre_age(_ExperienceFictive(horizon_jours=5))

    servis = await ltm.aquery_user_memories(
        person_id="42",
        query="bus 401",
        top_k=10,
        max_past_days=settings.agent.long_term_max_days_query,
        query_at=gama_timestamp(wall_clock(T0)),
    )

    assert {r.content for r in servis} == set()


@pytest.mark.asyncio
async def test_le_rappel_n_expose_jamais_les_souvenirs_d_un_autre_agent(fenetre_restauree):
    """Isolation invariant, re-checked on the Python side even when the index gets it wrong.

    The double deliberately returns a node belonging to another agent: this is the case
    that a filter delegated to the vector store would not catch on its own.
    """
    base = wall_clock(T0)
    ltm = _LtmDouble(
        [
            _Noeud("a_moi", 0.9, base - timedelta(days=2), "42"),
            _Noeud("a_quelqu_un_d_autre", 0.99, base - timedelta(days=2), "77"),
        ]
    )
    settings.agent.long_term_max_days_query = 60

    servis = await ltm.aquery_user_memories(
        person_id="42",
        query="bus 401",
        top_k=10,
        max_past_days=settings.agent.long_term_max_days_query,
        query_at=gama_timestamp(wall_clock(T0)),
    )

    assert {r.content for r in servis} == {"a_moi"}
