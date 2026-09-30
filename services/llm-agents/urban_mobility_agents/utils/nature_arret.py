"""Nature of an experiment stop: transient, quota or defect (2026-09-29).

The controller stops rather than serve a decision the model did not take (tickets 077, 105).
The night chain then relaunches the experiment an hour later. That is only right if the cause
PASSES: a provider overload. A defect of the protocol or of the code repeats identically at each
relaunch — the control arm of a13 v6 stopped three times at the same instant, on the same 409
`rejeu_obligatoire_absent`, before it was stopped by hand.

The reason alone does not decide: `decision_absente`, `consolidation_memoire` and
`enquete_incomplete` cover either case depending on the error KIND the gateway qualified. Hence:

- `passager` (transient) — the cause is qualified as an overload: new attempt after the wait;
- `quota` — daily quota exhausted: no new attempt on the same key before it reopens;
- `defaut` (defect) — everything else, including a missing or unknown cause: stop, no new attempt.

When in doubt, stop. One stop too many costs a manual relaunch; one attempt too many costs a
whole night, and fixes nothing.

Dependency-free module: the night chain imports it to read an old marker, written before the
controller recorded the nature in it.
"""

from __future__ import annotations

PASSAGER = "passager"
QUOTA = "quota"
DEFAUT = "defaut"
NATURES = (PASSAGER, QUOTA, DEFAUT)

# Error kinds that say « the provider is saturated, it will come back ». `panne_simulee` is the
# bench's survey outage (`EXPERIMENT_SURVEY_PANNE`), which plays an overload.
GENRES_SURCHARGE = frozenset({"surcharge_fournisseur", "panne_simulee"})
GENRES_QUOTA = frozenset({"quota_journalier"})

# Reasons whose cause is the overload by construction: the controller only sets them after a
# failure the gateway qualified (`surcharge_fournisseur`), or when a decision did not come back
# in time (`decision_en_retard`, a consequence of the provider's slowness).
MOTIFS_PASSAGERS = frozenset({"surcharge_fournisseur", "decision_en_retard"})
MOTIFS_QUOTA = frozenset({"quota_journalier"})
# Reasons that do not tell the cause: the error kind decides.
MOTIFS_SELON_GENRE = frozenset(
    {"decision_absente", "consolidation_memoire", "enquete_incomplete"}
)


def nature_arret(motif: str | None, genre: str | None = None) -> str:
    """`passager`, `quota` or `defaut`, for a stop reason and the error kind that caused it.

    >>> nature_arret("surcharge_fournisseur")
    'passager'
    >>> nature_arret("decision_absente", "surcharge_fournisseur")
    'passager'
    >>> nature_arret("decision_absente")        # cause not qualified
    'defaut'
    >>> nature_arret("prefixe_commun", "rejeu_obligatoire_absent")
    'defaut'
    """
    if motif in MOTIFS_QUOTA or (motif in MOTIFS_SELON_GENRE and genre in GENRES_QUOTA):
        return QUOTA
    if motif in MOTIFS_PASSAGERS:
        return PASSAGER
    if motif in MOTIFS_SELON_GENRE and genre in GENRES_SURCHARGE:
        return PASSAGER
    return DEFAUT


def nature_du_marqueur(marqueur: dict | None) -> str:
    """The nature of an `en_attente_quota.json` marker: the one it carries, otherwise recomputed.

    A marker written before 2026-09-29 has no `nature` field: it is recomputed from the reason and
    the kind (`cause`), and an empty or unreadable marker is a defect.
    """
    if not marqueur:
        return DEFAUT
    porte = marqueur.get("nature")
    if porte in NATURES:
        return str(porte)
    return nature_arret(marqueur.get("motif"), marqueur.get("cause"))


__all__ = [
    "DEFAUT",
    "GENRES_QUOTA",
    "GENRES_SURCHARGE",
    "MOTIFS_PASSAGERS",
    "MOTIFS_QUOTA",
    "MOTIFS_SELON_GENRE",
    "NATURES",
    "PASSAGER",
    "QUOTA",
    "nature_arret",
    "nature_du_marqueur",
]
