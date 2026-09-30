"""Four defects of long-term memory recall and cleanup.

Each test fails on the behaviour from BEFORE the fix. The four defects come
from an external review and were checked in the code before being fixed:

A. cleanup compared memories in simulated time with the machine clock;
B. entries removed from the metadata stayed in the vector index, and
   the document identifier collided after a cleanup;
C. the working-day filter short-circuited the age window;
D. a single-word tag capped at 0.70 on the lexical score.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.longterm import MultiUserLongTermMemory
from llm.memory import MemoryEntry, MemoryType
from sim_clock import wall_clock

# 16 March 2026, 05:00 in simulation WALL time. Memories carry NAIVE
# `datetime`s with wall-clock fields: building them otherwise would bring back the
# process time zone.
T0 = 1773637200


class _Ltm(MultiUserLongTermMemory):
    """Bare instance: no index, no disk, only the logic is tested."""

    def __init__(self):
        self.user_metadata = {}
        self.shared_index = None
        self.long_term_memory_filter_by_datetime = False
        self.metrics = {"queries": 0}
        self._dirty = set()

    # neutralises disk I/O
    def ensure_user_initialized(self, person_id):
        self.user_metadata.setdefault(person_id, {"entries": []})

    def _save_user_metadata(self, person_id):
        pass


@pytest.fixture()
def ltm():
    return _Ltm()


def _entry(ts, kind=MemoryType.CONVERSATION, pid="42", doc_id=None):
    """Test entry, EPISODIC by default.

    ⚠ The default was `CONCEPT` before gravity was introduced. It changed because the retention
    rule changed: a concept is never purged by the clock any more (semantic
    register), so a concept can no longer serve as a guinea pig for cleanup tests.
    These tests are about the cleanup MECHANISM, not about concept retention: they
    therefore take an episodic entry, and the semantic regime has its own tests.
    """
    return MemoryEntry(
        content="le bus 401 est ponctuel",
        timestamp=ts,
        memory_type=kind,
        person_id=pid,
        tags="bus 401",
        doc_id=doc_id,
    )


# ------------------------------------------------------- A. cleanup clock


def test_le_nettoyage_utilise_le_temps_simule_et_non_la_machine(ltm):
    """A memory from yesterday in simulated time survives, even if the machine is months later.

    This is the central defect: with `datetime.now()`, all memories of a run replaying
    a past date fell outside the window and were deleted en masse.
    """
    base = wall_clock(T0)
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [
        _entry(base - timedelta(days=1)),  # recent in SIMULATED time
        _entry(base - timedelta(days=40)),  # old, must go
    ]
    ltm.cleanup_user_memories("42", days_threshold=30)
    restants = ltm.user_metadata["42"]["entries"]
    assert len(restants) == 1, (
        "yesterday's memory was deleted: machine clock still in play"
    )
    assert restants[0].timestamp == base - timedelta(days=1)


def test_le_nettoyage_abandonne_s_il_ne_peut_pas_dater(ltm):
    """With no timestamped memory, nothing is cleaned rather than falling back on the machine."""
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = []
    ltm.cleanup_user_memories("42", days_threshold=30)
    assert ltm.user_metadata["42"]["entries"] == []
    assert "last_cleanup" not in ltm.user_metadata["42"], (
        "an abandoned cleanup is not dated"
    )


def test_le_nettoyage_separe_l_episodique_du_semantique(ltm):
    """The retention rule has CHANGED, and this test says how.

    BEFORE: a common age threshold, plus an exemption by TYPE — reflections and
    summaries never went, whatever their age.

    NOW, two regimes (Tulving, 1972):
      - semantic (concepts, summaries): never purged by the clock, it is lost through
        contradiction;
      - episodic (raw entries, reflections): purged when its temporal weight drops below
        the threshold. A hundred-day reflection with zero gravity weighs 1e-16: it goes.

    The consequence to accept is that reflections no longer pile up endlessly. Those
    that carry a high gravity survive anyway, through their longer lifetime —
    that is the point of the final assertion.
    """
    base = wall_clock(T0)
    ltm.ensure_user_initialized("42")
    marquante = _entry(base - timedelta(days=100), MemoryType.REFLECTION)
    marquante.importance, marquante.force = 1.0, 19.6
    ltm.user_metadata["42"]["entries"] = [
        _entry(base - timedelta(days=100), MemoryType.REFLECTION),  # ordinary: goes
        _entry(base - timedelta(days=100), MemoryType.SUMMARY),     # semantic: stays
        _entry(base - timedelta(days=100), MemoryType.CONCEPT),     # semantic: stays
        _entry(base),                                               # recent: stays
    ]
    ltm.cleanup_user_memories("42", days_threshold=30)
    restants = ltm.user_metadata["42"]["entries"]
    types = {str(e.memory_type) for e in restants}

    assert "summary" in types and "concept" in types, "semantic memory is not forgotten by clock"
    assert "reflection" not in types, "an ordinary hundred-day reflection weighs 1e-16"
    assert len(restants) == 3

    # Two reflections of the SAME age, thirty days, that only gravity tells apart. This is the
    # heart of gravity: a memory's lifetime depends on what it tells.
    #   - ordinary, lifetime 2.8 d  → weight exp(-30/2.8)  = 2.2e-5, below the threshold: goes
    #   - striking, lifetime 19.6 d → weight exp(-30/19.6) = 0.22, well above: stays
    # The anchor entry sets the simulated "now", which is the agent's most recent
    # memory: without it, both reflections would be zero days old and nothing would move.
    marquante.timestamp = base - timedelta(days=30)
    banale = _entry(base - timedelta(days=30), MemoryType.REFLECTION)
    ancre = _entry(base)
    ltm.user_metadata["42"]["entries"] = [banale, marquante, ancre]

    ltm.cleanup_user_memories("42", days_threshold=7)
    restants = ltm.user_metadata["42"]["entries"]

    assert marquante in restants, "gravity must decide survival, not the type"
    assert banale not in restants, "an ordinary month-old reflection no longer weighs anything"
    assert restants == [marquante, ancre]


def test_la_date_de_nettoyage_est_en_temps_simule(ltm):
    base = wall_clock(T0)
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [_entry(base)]
    ltm.cleanup_user_memories("42", days_threshold=30)
    assert ltm.user_metadata["42"]["last_cleanup"] == base.isoformat()


# ------------------------------------------- B. deletion from the vector index


class _IndexEspion:
    def __init__(self):
        self.supprimes = []

    def delete_ref_doc(self, doc_id, delete_from_docstore=False):
        self.supprimes.append(doc_id)


def test_les_entrees_supprimees_sortent_aussi_de_l_index(ltm):
    """Without this, a forgotten memory keeps being served back to the model."""
    base = wall_clock(T0)
    idx = _IndexEspion()
    ltm.shared_index = idx
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [
        _entry(base - timedelta(days=40), doc_id="42_0"),
        _entry(base, doc_id="42_1"),
    ]
    ltm.cleanup_user_memories("42", days_threshold=30)
    assert idx.supprimes == ["42_0"], "the purged memory stayed in the index"


def test_une_suppression_qui_echoue_ne_perd_pas_les_metadonnees(ltm):
    class _IndexCasse:
        def delete_ref_doc(self, *a, **k):
            raise RuntimeError("index indisponible")

    base = wall_clock(T0)
    ltm.shared_index = _IndexCasse()
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [
        _entry(base - timedelta(days=40), doc_id="42_0"),
        _entry(base, doc_id="42_1"),
    ]
    ltm.cleanup_user_memories("42", days_threshold=30)  # must not raise
    assert len(ltm.user_metadata["42"]["entries"]) == 1


def test_les_entrees_sans_identifiant_ne_font_pas_echouer_le_nettoyage(ltm):
    """Entries written without an identifier: not addressable, hence not deletable.

    They leave the metadata and stay in the index. This is the accepted residue of
    the fix, and it is logged rather than silent.
    """
    base = wall_clock(T0)
    idx = _IndexEspion()
    ltm.shared_index = idx
    ltm.ensure_user_initialized("42")
    ltm.user_metadata["42"]["entries"] = [
        _entry(base - timedelta(days=40), doc_id=None),   # old, without identifier
        _entry(base, doc_id="42_9"),                      # recent: sets the "now"
    ]
    ltm.cleanup_user_memories("42", days_threshold=30)
    assert idx.supprimes == [], "an entry without identifier is not addressable"
    assert len(ltm.user_metadata["42"]["entries"]) == 1


def test_le_maintenant_est_le_souvenir_le_plus_recent_de_l_agent(ltm):
    """Property of `_sim_now`: an agent whose history stops two months ago does not
    get its memory purged for all that. Its "now" is its latest experience."""
    base = wall_clock(T0)
    ltm.ensure_user_initialized("42")
    vieux = [_entry(base - timedelta(days=60 + j)) for j in range(3)]
    ltm.user_metadata["42"]["entries"] = list(vieux)
    ltm.cleanup_user_memories("42", days_threshold=30)
    assert len(ltm.user_metadata["42"]["entries"]) == 3


def test_l_identifiant_de_document_est_monotone(ltm):
    """After a cleanup the list gets shorter: a counter based on its length
    would reuse identifiers already indexed."""
    meta = {"entries": [], "next_doc_index": 0}
    ltm.user_metadata["42"] = meta
    vus = []
    for _ in range(3):
        i = meta.get("next_doc_index", len(meta["entries"]))
        meta["next_doc_index"] = i + 1
        vus.append(f"42_{i}")
        meta["entries"].append(_entry(wall_clock(T0), doc_id=vus[-1]))
    meta["entries"] = meta["entries"][:1]  # cleanup: the list gets shorter
    i = meta.get("next_doc_index", len(meta["entries"]))
    assert f"42_{i}" not in vus, "identifier collision after cleanup"


def test_memory_entry_conserve_son_identifiant_au_round_trip():
    e = _entry(wall_clock(T0), doc_id="42_7")
    assert MemoryEntry.from_dict(e.to_dict()).doc_id == "42_7"


def test_memory_entry_relit_un_ancien_format_sans_identifiant():
    d = _entry(wall_clock(T0)).to_dict()
    d.pop("doc_id")
    assert MemoryEntry.from_dict(d).doc_id is None


# --------------------------------------------------- C. both filters combined


def _filtre(ltm, entry_dt, query_dt, max_past_days, par_date):
    """Replicates the `filter_message` decision, isolated from the vector query."""
    ltm.long_term_memory_filter_by_datetime = par_date
    if max_past_days >= 0 and not ltm._filter_memory_by_past_days(
        entry_dt, query_dt, max_past_days
    ):
        return False
    if par_date and query_dt:
        return ltm._filter_memory_by_working_day(
            entry_dt, query_dt
        ) and ltm._filter_memory_by_peak_time(entry_dt, query_dt)
    return True


def test_la_fenetre_d_age_s_applique_meme_avec_le_filtre_par_jour(ltm):
    """Defect C: a three-year-old memory passed if it fell on the right weekday."""
    query = wall_clock(T0)
    vieux = query - timedelta(days=364)  # same weekday, very old
    assert vieux.weekday() == query.weekday(), (
        "the test case must isolate the weekday"
    )
    assert _filtre(ltm, vieux, query, 30, par_date=True) is False


def test_la_fenetre_d_age_s_applique_sans_le_filtre_par_jour(ltm):
    query = wall_clock(T0)
    assert _filtre(ltm, query - timedelta(days=40), query, 30, par_date=False) is False
    assert _filtre(ltm, query - timedelta(days=5), query, 30, par_date=False) is True


# ------------------------------------------------ D. tag ceiling


def test_une_etiquette_d_un_seul_mot_peut_atteindre_un(ltm):
    """Defect D: it capped at 0.70 for lack of bigrams."""
    assert ltm._bleu_score("je prends le métro ce soir", "métro") == pytest.approx(1.0)


def test_une_etiquette_de_deux_mots_atteint_toujours_un(ltm):
    assert ltm._bleu_score("le bus 401 est ponctuel", "bus 401") == pytest.approx(1.0)


def test_une_etiquette_mono_terme_absente_vaut_zero(ltm):
    assert ltm._bleu_score("je prends le tram", "métro") == pytest.approx(0.0)


def test_les_bigrammes_comptent_encore_quand_ils_existent(ltm):
    """The 0.7 / 0.3 weighting still applies as soon as a bigram is available."""
    s = ltm._bleu_score(
        "bus 401 ponctualité", "401 bus"
    )  # unigrams yes, bigram no
    assert s == pytest.approx(0.7)
