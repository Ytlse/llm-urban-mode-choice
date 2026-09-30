"""The "what changed recently" window.

Why this file exists. On run `2026-09-19_07_31`, the car disappears then comes back
exactly fourteen days after the shock — the value of `FENETRE_CHANGEMENTS_JOURS`, hard-coded.
The report attributes this return to the decay of the memory; the window predicts the same
date. Nothing made it possible to decide. This lot makes the parameter tunable (hence
ablatable) and its exit visible in the run log.

It changes NO memory rule: same default values, same mechanism.

⚠ TICKET 095 UPDATE — what this file describes has become the `fixe` mode. The repository
default is now `derivee`, where the duration is computed memory by memory from the severity.
All cases below therefore explicitly set the `fixe` mode: they remain the contract of the
methodological control arm, the one that must reproduce the campaigns of 19 and 20 September
2026. The contract of the derived mode lives in `test_shock_memory_duration.py`.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import noyau as noyau_module
from llm.memory import MemoryEntry, MemoryType
from llm.noyau import bloc_changements
from settings import settings
from sim_clock import wall_clock

T0 = 1773637200


def _choc(texte, gravite=0.9, jours=1):
    return MemoryEntry(
        content=texte,
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.REFLECTION,
        person_id="899549",
        importance=gravite,
        force=19.6,
    )


def _banal(texte, jours=1):
    return MemoryEntry(
        content=texte,
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.REFLECTION,
        person_id="899549",
        importance=0.0,
        force=2.8,
    )


@pytest.fixture(autouse=True)
def _journal_propre():
    noyau_module.reinitialiser()
    yield
    noyau_module.reinitialiser()


@pytest.fixture(autouse=True)
def _mode_fixe(monkeypatch):
    """This whole file describes the `fixe` mode — it declares it rather than inheriting it."""
    monkeypatch.setattr(
        settings.agent, "memoire__mode_fenetre_changements", noyau_module.MODE_FIXE,
        raising=False,
    )


@pytest.fixture
def journal():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO")
    yield lignes
    logger.remove(sink)


@pytest.fixture
def fenetre(monkeypatch):
    def _poser(jours=None, maxi=None):
        if jours is not None:
            monkeypatch.setattr(settings.agent, "memoire__fenetre_changements_jours", jours, raising=False)
        if maxi is not None:
            monkeypatch.setattr(settings.agent, "memoire__changements_max", maxi, raising=False)
    return _poser


# ══════════════════════ I1 to I4 — the setting ══════════════════════════════════


def test_I1_les_valeurs_par_defaut_ne_bougent_pas():
    """I1 — the historical behaviour in `fixe` mode: 14 days, 3 lines.

    The repository DEFAULT, for its part, has changed since: `derivee`. It is checked right here, so
    that a silent return to `fixe` by default shows in this file rather than in a run.
    """
    assert settings.agent.memoire__fenetre_changements_jours == 14
    assert settings.agent.memoire__changements_max == 3
    assert (
        type(settings.agent).model_fields["memoire__mode_fenetre_changements"].default
        == noyau_module.MODE_DERIVEE
    )
    assert bloc_changements([_choc("panne", jours=13)], wall_clock(T0))
    assert bloc_changements([_choc("panne", jours=15)], wall_clock(T0)) == []


def test_I2_la_fenetre_se_regle(fenetre):
    """I2 — this setting is what makes the ablation arm possible."""
    souvenir = _choc("panne sur voie rapide", jours=10)
    fenetre(jours=7)
    assert bloc_changements([souvenir], wall_clock(T0)) == []
    fenetre(jours=21)
    assert bloc_changements([souvenir], wall_clock(T0))


def test_I3_la_valeur_est_relue_a_chaque_appel(fenetre):
    """I3 — frozen at import, an environment override would be useless."""
    souvenir = _choc("panne", jours=10)
    fenetre(jours=7)
    assert bloc_changements([souvenir], wall_clock(T0)) == []
    fenetre(jours=14)
    assert bloc_changements([souvenir], wall_clock(T0))


def test_I4_le_nombre_de_lignes_se_regle(fenetre):
    """I4 — the block is capped, and the cap is declared."""
    entrees = [_choc("panne A", jours=1), _choc("panne B", jours=2), _choc("panne C", jours=3)]
    fenetre(maxi=1)
    assert len(bloc_changements(entrees, wall_clock(T0))) == 1
    fenetre(maxi=3)
    assert len(bloc_changements(entrees, wall_clock(T0))) == 3


def test_I5_fenetre_a_zero_est_la_valeur_d_ablation(fenetre):
    """I5 — the "no window" arm: no shock episodic memory enters the block."""
    fenetre(jours=0)
    assert bloc_changements([_choc("panne", jours=0)], wall_clock(T0)) == []
    assert bloc_changements([_choc("panne", jours=13)], wall_clock(T0)) == []


def test_I6_fenetre_negative_est_ramenee_a_zero_et_journalisee(fenetre, journal):
    """I6 — a meaningless value must not read as an accepted setting."""
    fenetre(jours=-3)
    assert bloc_changements([_choc("panne", jours=1)], wall_clock(T0)) == []
    assert [m for n, m in journal if n == "WARNING" and "négative" in m]


# ══════════════════════ I7 to I10 — the window exit shows ═══════════════════════


def test_I7_la_sortie_de_fenetre_est_journalisee(journal, fenetre):
    """I7 — the event that, on 13 April, coincides with the return of the car."""
    fenetre(jours=14)
    bloc_changements([_choc("panne sur voie rapide", jours=15)], wall_clock(T0), person_id="899549")
    sorties = [m for n, m in journal if "sorti" in m and "899549" in m]
    assert sorties, "the window exit left no trace in the log"
    assert "14" in sorties[0], "the applied window is not stated"


def test_I8_un_souvenir_encore_dans_la_fenetre_ne_journalise_rien(journal, fenetre):
    """I8 — the event is the EXIT, not the presence."""
    fenetre(jours=14)
    assert bloc_changements([_choc("panne", jours=3)], wall_clock(T0), person_id="899549")
    assert not [m for _, m in journal if "sorti" in m]


def test_I9_un_agent_sans_choc_ne_journalise_rien(journal, fenetre):
    """I9 — never having had a shock is not exiting one."""
    fenetre(jours=14)
    bloc_changements([_banal("trajet ordinaire", jours=30)], wall_clock(T0), person_id="899549")
    assert not [m for _, m in journal if "sorti" in m]


def test_I10_front_montant_une_seule_ligne_par_jour(journal, fenetre):
    """I10 — otherwise the line repeats at each decision and drowns what it announces."""
    fenetre(jours=14)
    entrees = [_choc("panne", jours=15)]
    for _ in range(4):
        bloc_changements(entrees, wall_clock(T0), person_id="899549")
    assert len([m for _, m in journal if "sorti" in m]) == 1


def test_I10b_le_lendemain_ne_rejournalise_pas(journal, fenetre):
    """I10 (continued) — the exit is a dated event, not a daily state."""
    fenetre(jours=14)
    entrees = [_choc("panne", jours=15)]
    bloc_changements(entrees, wall_clock(T0), person_id="899549")
    bloc_changements(entrees, wall_clock(T0) + timedelta(days=1), person_id="899549")
    assert len([m for _, m in journal if "sorti" in m]) == 1
