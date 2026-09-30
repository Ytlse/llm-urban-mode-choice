"""The judgement of an EXPERIENCED event is awaited, not launched in the background.

The initial design put the judgement "in the evening queue": it went off as a background task and
raised the importance of the short-term entry before the consolidation consumed it. The
reasoning assumed an EVENING consolidation. There is none — the reflection fires at the ENTRY
THRESHOLD, and the event's entry is often the one that crosses this threshold: it
then triggers the consolidation that consumes it, while its judgement is in flight.

Measured on the 2026-09-23 run, campaign c3:

    10:24:08  [gravite] CHOC pour 861500 à 27 March 2026, 14:02 : I_det=0.87
    10:24:09  [reflexion-stm] agent=861500 concepts=2        ← consolidation, 1 s later
    10:24:14  « c3_panne_reseau » jugé : grave (0.75)        ← 5 s too late

The event was rated on the measured fact (0.87) instead of the judged severity (0.75),
that is, under the regime that decision D7 abandoned.
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
CTRL = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")


def _bloc_du_site_d_appel() -> str:
    debut = CTRL.index("if _jugement_attendu:")
    return CTRL[debut : debut + 600]


def test_R11_le_jugement_est_attendu():
    bloc = _bloc_du_site_d_appel()
    assert "await self._juger_evenement_subi(" in bloc


def test_R12_le_jugement_ne_part_plus_en_tache_de_fond():
    """THIS is the regression to prevent: a `_spawn` reopens the race, silently."""
    bloc = _bloc_du_site_d_appel()
    assert "_spawn" not in bloc, (
        "running the judgement as a background task makes it lose the race every time "
        "the injected entry crosses the reflection threshold"
    )


def test_R13_la_consolidation_part_bien_au_seuil_d_entrees():
    """The cause. If this setting disappeared, the reasoning above would have to be redone.

    `stm_reflection_min_entries` is what makes the race UNWINNABLE for a deferred
    judgement: the reflection does not fire at a time, it fires at an entry count, and
    the event's entry is the one that can push the count over.
    """
    from settings import settings

    seuil = settings.agent.stm_reflection_min_entries
    assert isinstance(seuil, int) and seuil > 0, (
        "a positive threshold means triggering by ENTRY COUNT, not by clock"
    )


def test_R14_la_garde_du_jugement_tardif_reste_en_place():
    """Waiting makes the race impossible; the guard stays, it costs nothing.

    A future caller going back to deferred mode would find the alarm rather than a silence.
    """
    debut = CTRL.index("async def _juger_evenement_subi")
    fin = CTRL.index("async def _injecter_evenements_du_reveil")
    methode = CTRL[debut:fin]
    assert "arrived AFTER the consolidation" in methode
    assert methode.count("registre.tracer(") == 4


def test_R15_l_alarme_ne_dit_plus_du_soir():
    """The label referred to a moment that does not exist, and sent me looking for a latency."""
    assert "APRÈS la consolidation du soir" not in CTRL
