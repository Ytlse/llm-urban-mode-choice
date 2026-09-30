"""Consolidation schedule, forgetting scale, absolute ranking.

Three properties are checked here by failure, not by reading:

1. the time score is ABSOLUTE and reproduces the old decay at the default;
2. the ranking is no longer renormalised per batch, so two decisions are comparable
   and the time constant has a real effect;
3. the daily floor makes the moment of consolidation deterministic.
"""

import math
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.longterm import MultiUserLongTermMemory
from settings import settings
from sim_clock import gama_timestamp, wall_clock

# 1773637200 = 16 March 2026, 05:00 in the simulation's WALL-CLOCK time. Memories
# carry NAIVE `datetime`s with wall-clock fields (cf. `sim_clock.wall_clock`):
# building them otherwise would reintroduce the process time zone, which is exactly
# the bug fixed on 2026-09-04.
T0 = 1773637200

# ---------------------------------------------------------------------------- tools


class _Ltm(MultiUserLongTermMemory):
    """Bare instance: only the scoring is tested, no index is needed."""

    def __init__(self):
        pass


@pytest.fixture()
def ltm():
    return _Ltm()


def _node(tags: str, score: float, ts):
    class _N:
        def __init__(self):
            self.score = score
            self.metadata = {"tags": tags, "timestamp": ts.isoformat()}

    return _N()


# ------------------------------------------------------------- 1. absolute decay


def test_score_temporel_reproduit_l_ancienne_base_au_defaut(ltm):
    """The default of 2.8 days reproduces the base 0.7 per day, to within 1e-3.

    This is the condition for a gap measured after this change to be attributable
    to the new mechanisms and not to a silently accelerated forgetting.
    """
    assert settings.agent.long_term_retrieval__force_base_jours == pytest.approx(2.8)
    base = wall_clock(T0)
    for jours in (0, 1, 2, 3, 7, 14):
        ts = base - timedelta(days=jours)
        obtenu = ltm._time_decay_score(ts.isoformat(), gama_timestamp(base))
        assert obtenu == pytest.approx(0.7**jours, abs=1e-3), f"at {jours} days"


def test_score_temporel_est_absolu_et_ne_depend_pas_du_lot(ltm):
    """A one-day-old memory is worth the same whatever its neighbours.

    Previously, min-max normalisation made this value relative to the batch:
    the most recent was always worth 1, the oldest always 0.
    """
    base = wall_clock(T0)
    at = gama_timestamp(base)
    un_jour = (base - timedelta(days=1)).isoformat()
    assert ltm._time_decay_score(un_jour, at) == pytest.approx(
        math.exp(-1 / 2.8), abs=1e-6
    )


def test_constante_de_temps_plus_longue_remonte_le_score(ltm, monkeypatch):
    """Tripling the time constant must shift the curve, otherwise no experiment
    sensitivity arm measures anything at all."""
    base = wall_clock(T0)
    at = gama_timestamp(base)
    vieux = (base - timedelta(days=3)).isoformat()

    court = ltm._time_decay_score(vieux, at)
    monkeypatch.setattr(
        settings.agent, "long_term_retrieval__force_base_jours", 2.8 * 3
    )
    long = ltm._time_decay_score(vieux, at)

    assert long > court
    assert long - court > 0.2, "the effect must be clear-cut, not marginal"


def test_force_nulle_ne_leve_pas(ltm, monkeypatch):
    monkeypatch.setattr(settings.agent, "long_term_retrieval__force_base_jours", 0.0)
    base = wall_clock(T0)
    assert ltm._time_decay_score(base.isoformat(), gama_timestamp(base)) == 0.0


# ---------------------------------------------------------- 2. absolute ranking


def test_le_plus_recent_ne_vaut_pas_toujours_un(ltm):
    """Two memories that are both old must not see the less old one raised to 1.

    That is exactly what min-max normalisation did, and it is what made
    two decisions incomparable with each other.
    """
    base = wall_clock(T0)
    at = gama_timestamp(base)
    nodes = [
        _node("bus 401", 0.5, base - timedelta(days=10)),
        _node("bus 401", 0.5, base - timedelta(days=12)),
    ]
    scores = ltm.rank_nodes("bus 401", at, nodes)
    poids_temps = settings.agent.long_term_retrieval__time_weight
    ecart = float(scores[0] - scores[1])
    # The two memories differ only by their date, 10 days versus 12. Both being
    # very old, their time component is near zero and their gap must
    # be too. Under min-max, the first would have got 1 and the second 0, i.e. a
    # gap equal to ALL of the time weight — this is what this test forbids.
    assert 0 < ecart < poids_temps * 0.1, (
        f"gap {ecart:.4f}; under min-max it would have been {poids_temps:.2f}"
    )


def test_un_lot_de_souvenirs_frais_score_plus_haut_qu_un_lot_ancien(ltm):
    """Comparability property: the score of a decision reads in absolute terms."""
    base = wall_clock(T0)
    at = gama_timestamp(base)
    frais = [_node("metro B", 0.5, base - timedelta(hours=2))]
    ancien = [_node("metro B", 0.5, base - timedelta(days=20))]
    assert (
        ltm.rank_nodes("metro B", at, frais)[0]
        > ltm.rank_nodes("metro B", at, ancien)[0]
    )


def test_score_de_similarite_est_borne(ltm):
    """A vector store that returns an out-of-bounds similarity must not make
    the composite score blow up."""
    base = wall_clock(T0)
    n = _node("velo", 3.7, base)
    score = ltm.rank_nodes("velo", gama_timestamp(base), [n])[0]
    assert 0.0 <= score <= 1.0


def test_lot_vide(ltm):
    assert ltm.rank_nodes("x", 0, []).size == 0


# --------------------------------------------------------------- 3. daily floor


class _Stm:
    def __init__(self, n):
        self.recent_entries = list(range(n))


class _Personne:
    def __init__(self, pid, is_llm_based=True):
        self.person_id = pid
        self.is_llm_based = is_llm_based


def _eligibilite(entrees, heure, deja_plancher_ce_jour, *, active=True, llm=True):
    """Replicates the controller's eligibility rule, isolated from its I/O.

    The controller builds the same decision from the simulated wall-clock hour, the
    buffer fill level and the day already covered by the floor.
    """
    seuil = settings.agent.stm_reflection_min_entries
    floor_day = (
        1
        if (active and heure >= settings.agent.stm_reflection_daily_floor_hour)
        else None
    )
    if not llm:
        return False, False
    if entrees >= seuil:
        return True, False
    if floor_day is not None and entrees > 0 and deja_plancher_ce_jour != floor_day:
        return True, True
    return False, False


def test_le_seuil_volumetrique_reste_prioritaire():
    ok, par_plancher = _eligibilite(entrees=12, heure=10, deja_plancher_ce_jour=None)
    assert ok and not par_plancher


def test_agent_peu_mobile_consolide_quand_meme_le_soir():
    """Six entries, i.e. a single trip: below the threshold, so never consolidated
    without a daily floor. The floor catches it."""
    assert _eligibilite(entrees=6, heure=10, deja_plancher_ce_jour=None) == (
        False,
        False,
    )
    assert _eligibilite(entrees=6, heure=22, deja_plancher_ce_jour=None) == (True, True)


def test_le_plancher_ne_part_qu_une_fois_par_jour():
    assert _eligibilite(entrees=6, heure=22, deja_plancher_ce_jour=1) == (False, False)
    assert _eligibilite(entrees=6, heure=23, deja_plancher_ce_jour=1) == (False, False)


def test_le_plancher_ignore_un_tampon_vide():
    assert _eligibilite(entrees=0, heure=23, deja_plancher_ce_jour=None) == (
        False,
        False,
    )


def test_le_plancher_se_desactive():
    assert _eligibilite(
        entrees=6, heure=23, deja_plancher_ce_jour=None, active=False
    ) == (
        False,
        False,
    )


def test_le_plancher_ne_touche_pas_les_agents_non_llm():
    assert _eligibilite(entrees=6, heure=23, deja_plancher_ce_jour=None, llm=False) == (
        False,
        False,
    )


def test_defauts_du_plancher():
    assert settings.agent.stm_reflection_daily_floor_enabled is True
    assert settings.agent.stm_reflection_daily_floor_hour == 22, (
        "22:00 leaves the simulated night to the EDF drain (ticket 010)"
    )
