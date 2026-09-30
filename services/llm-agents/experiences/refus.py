"""Classify a launch refusal: is it temporary, or final?

WHY THIS MODULE. On 2026-09-16, four arms of a campaign were declared broken when they were
simply waiting for quota. The launcher refuses to create a run when all the instances of a
model are exhausted; it exits with the same code as an invalid configuration, and the
campaign, which launches in the background without reading that code, sees only a missing
state. Its log then reports "launched 3 times without ever writing a state", which sends one
looking for a failure where there is only a quota window to wait for.

THREE CLASSES, AND THREE DIFFERENT ACTIONS.

- `quota_epuise` — the instances do serve the model, their daily quota is used up. Tomorrow
  the same launch goes through. A campaign must POSTPONE, not fail, and must not count an
  attempt: nothing was attempted.
- `quota_insuffisant` — the load exceeds the daily quota, whatever the time. Waiting changes
  nothing: it takes a better-provisioned model, a smaller set, or accepting to spread over
  several days with `--ignorer-aptitude`. Postponing would loop forever, so it is a failure —
  but a failure that SAYS what it is.
- `definitif` — everything else: invalid configuration, stale set, incompatible mode. No
  amount of waiting resolves it.

The classification works on the TEXT of the refusal, and this is a deliberate choice: refusals
are produced in four different places, all already written to be read by a human. Adding a
code everywhere would mean touching those four places for no gain here; what matters is that
the rule sits in ONE place, named, and tested.
"""

from __future__ import annotations

QUOTA_EPUISE = "quota_epuise"
QUOTA_INSUFFISANT = "quota_insuffisant"
DEFINITIF = "definitif"

#: Marker written next to the launch log when a launch is refused.
SUFFIXE_MARQUEUR = ".refus.json"

# Order matters: `quota_insuffisant` is tested BEFORE `quota_epuise`, because a load refusal
# can mention both ("1000 requêtes/jour contre 9695 sollicitations" names the quota without
# being a momentary shortage).
_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (QUOTA_INSUFFISANT, ("hors d'atteinte", "jours de quota")),
    (QUOTA_EPUISE, ("momentanément épuisée", "momentanément épuisées")),
)


def classer(motifs: list[str]) -> str:
    """The class of a batch of refusals. The most severe wins: a final refusal accompanied
    by a quota shortage stays final, since waiting would not lift it."""
    if not motifs:
        return DEFINITIF
    classes = set()
    for motif in motifs:
        texte = (motif or "").lower()
        for classe, signatures in _SIGNATURES:
            if any(s.lower() in texte for s in signatures):
                classes.add(classe)
                break
        else:
            classes.add(DEFINITIF)
    if DEFINITIF in classes:
        return DEFINITIF
    if QUOTA_INSUFFISANT in classes:
        return QUOTA_INSUFFISANT
    return QUOTA_EPUISE


def est_reportable(classe: str) -> bool:
    """Postponing only makes sense for what waiting resolves. The rest is a failure, and saying
    so at once beats discovering it after two recovery passes."""
    return classe == QUOTA_EPUISE


__all__ = ["QUOTA_EPUISE", "QUOTA_INSUFFISANT", "DEFINITIF", "SUFFIXE_MARQUEUR",
           "classer", "est_reportable"]
