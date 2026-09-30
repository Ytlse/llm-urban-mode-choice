"""A person's activity chain is **cyclic** — a single implementation.

The last activity of the day is followed by a return to the first, which is home
in nearly all cases. A day of n activities therefore counts n trips, not
n − 1: the return home is a trip, and it must be decided like the others.

This module exists because two implementations of this rule diverged (ticket 045,
alert A1). The simulation controller closed the cycle with `(i + 1) % n`; the experiment
platform enumerated consecutive pairs and stopped at the last activity. Result:
the platform measured 2,693 decisions where the simulation played 3,693 on cohort v1,
i.e. **27% of the day never decided**. And the omission was not random — it removed
exactly the trip where the vehicle chain constraint bites hardest (an agent who left by
car comes back by car), which structurally pushed modal shares towards the modes
of the outbound trip.

The controller's rule is authoritative, and both callers now go through here.

**Times.** Nothing to arbitrate: sealed populations already encode the loop. For every
activity, `scheduled_start_time` of activity `(i + 1) % n` equals `end_time` of activity `i`,
**including at the loop** — checked on 2026-09-11 on the 1,894 persons of cohorts v1 and v5,
without a single discrepancy. The platform's convention (scheduled time of the destination) and
the controller's (end of the origin activity) therefore designate the same instant.
"""

from __future__ import annotations

from collections.abc import Sequence

from models import Activity

__all__ = ["activite_suivante", "paires_de_la_journee"]


def activite_suivante(
    activites: Sequence[Activity], courante: Activity
) -> Activity | None:
    """The activity following `courante` in the cyclic chain, or `None` if there is no trip.

    Returns `None` in three cases, which are those the three controller sites were already
    testing each on their own:

    - the person has zero or a single activity — a one-link chain does not loop
      onto itself, otherwise we would fabricate a trip from an activity to itself;
    - `courante` does not belong to the chain (matched by `id`, never by object
      equality: two activities can be equal field by field);
    - one of the two ends has no location, so no trip can be computed.

    A return to the SAME place, however, is indeed a trip: it is enumerated here and it is up to
    the set to classify it as unusable (`origine_egale_destination`). Singling it out here would
    make it disappear from the raw expected trips, where it must count.
    """
    n = len(activites)
    if n <= 1:
        return None
    idx = next((i for i, a in enumerate(activites) if a.id == courante.id), None)
    if idx is None:
        return None
    suivante = activites[(idx + 1) % n]
    if courante.location is None or suivante.location is None:
        return None
    return suivante


def paires_de_la_journee(
    activites: Sequence[Activity],
) -> list[tuple[Activity, Activity]]:
    """The (origin, destination) pairs of the day, **closing trip included**.

    A day of n located activities returns n pairs, the last one going from the last
    activity to the first.
    """
    paires: list[tuple[Activity, Activity]] = []
    for courante in activites:
        suivante = activite_suivante(activites, courante)
        if suivante is not None:
            paires.append((courante, suivante))
    return paires
