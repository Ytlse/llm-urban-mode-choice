"""When the event takes place — ticket 100, lot 1.

Two x-axes, and both are needed.

`jour_du_run` counts the days since the start of the run, 1 for the first. It is read from the
ANCHOR of ticket 075 and **never** from the first observed timestamp: after a hot resume, GAMA
replays from its t0, and re-anchoring would move the event back several days in the middle of
the run.

`jour_relatif` counts the days elapsed since the first day of the event: −2, −1, 0, +1…
It is defined on **every** day of the run, including long before and long after: it is
the x-axis of all the drop-off and return curves, and an x-axis that existed only on
event days would plot nothing.

The window drawn per household (059 Q5) arrives in lot 2; this module carries here the only form
delivered by 079, the list of declared days.
"""

from __future__ import annotations


def jour_du_run(timestamp: int) -> int:
    """1 for the first simulated day of the run."""
    from urban_mobility_agents.utils.ancre_run import jours_ecoules

    return int(jours_ecoules(int(timestamp))) + 1


def jour_tire(graine: int, evenement_id: str, cible: str, fenetre: tuple[int, int]) -> int:
    """The publication day of THIS target, drawn within the declared window (059 Q5).

    Two households do not read on the same day. This is what prevents a calendar effect — a
    Monday, the eve of a holiday, a rainy day — from being confused with the effect of the
    article: if all read on the same morning, the two would be strictly inseparable.

    Deterministic and stable from one run to the next, like the exposure draw and for the same
    reason: two replays of the same scenario must be the same experiment.
    """
    from llm.evenements.exposition import tirage_stable

    debut, fin = int(fenetre[0]), int(fenetre[1])
    largeur = fin - debut + 1
    rang = int(tirage_stable(graine, evenement_id, f"parution:{cible}") * largeur)
    return debut + min(rang, largeur - 1)


def date_du_jour_run(jour: int):
    """The wall-clock date of day `jour` of the run (1 = first day), or `None` without anchor.

    Read from the anchor of ticket 075, like `jour_du_run`: the same x-axis in both directions.
    """
    from datetime import timedelta

    from sim_clock import wall_clock
    from urban_mobility_agents.utils.ancre_run import ancre

    debut = ancre()
    if debut is None:
        return None
    return wall_clock(int(debut)).date() + timedelta(days=int(jour) - 1)


def jours_de_service(debut, n: int, sans_week_end: bool) -> tuple:
    """The `n` TRAVEL days starting at `debut` — ticket 111, decision D3.

    `debut` counts as day 1 if it is a working day. When no departure takes place at weekends
    (`agent.no_weekend_departures`), Saturday and Sunday are skipped: an article read on a
    Thursday is served Thursday, Friday, Monday, Tuesday and Wednesday. Without this rule, five
    calendar days would give only three days of decisions to a household reading on a Thursday.

    WALL-CLOCK calendar, deterministic: the result does not depend on when the decision is
    pre-computed, and this is what makes the served line independent of the rolling horizon.
    """
    from datetime import timedelta

    jours = []
    courant = debut
    while len(jours) < int(n):
        if not (sans_week_end and courant.weekday() >= 5):
            jours.append(courant)
        courant = courant + timedelta(days=1)
    return tuple(jours)
