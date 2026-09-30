"""Concept consolidation — ticket 071, lot 3.

A concept no longer piles up: it gets **corrected**. It is confirmed, refined or
contradicted, with a counter behind it. This is the extract-then-update principle of
Mem0 (Chhikara et al., 2025) and of the note evolution of A-MEM (Xu et al., 2025).

**There is no deletion**, and this is a deliberate divergence from Mem0, whose set of
operations includes a `DELETE`. An outdated concept is marked and dated, never deleted: the
reason is scientific, not technical — this dated setting-aside **is** the readable trace
of a habit change, and it is the observable the hysteresis experiment looks for.

**A concept is not forgotten by the clock.** That a line is saturated on rainy days between
8:00 and 8:30 does not become false because ten days have passed. This is Tulving's (1972)
separation between episodic and semantic, and the two-system architecture of McClelland,
McNaughton & O'Reilly (1995).
"""

from __future__ import annotations

from loguru import logger

from settings import settings

# The four operations, in the model's vocabulary (English) and in the specification's
# (French). Mapping to Mem0: `creer` = ADD, `preciser` = UPDATE,
# `confirmer` = NOOP plus a counter, `contredire` replaces DELETE.
CREER = "creer"
CONFIRMER = "confirmer"
PRECISER = "preciser"
CONTREDIRE = "contredire"
OPERATIONS = (CREER, CONFIRMER, PRECISER, CONTREDIRE)

ALIAS_OPERATIONS: dict[str, str] = {
    "create": CREER,
    "confirm": CONFIRMER,
    "refine": PRECISER,
    "update": PRECISER,
    "contradict": CONTREDIRE,
}

# Number of concepts of one basket shown to the model. The "no mode" basket would otherwise
# gather every general concept of a purpose, and showing it whole would bloat the prompt.
CONCEPTS_MONTRES_PAR_PANIER = 5


def normaliser_operation(valeur: str | None) -> str:
    """Operation requested by the model, mapped to the internal vocabulary.

    An unknown or missing value falls back to `creer`: it is the least destructive fallback —
    we add a concept rather than touch an existing one on the strength of an answer we did not
    understand. But it leaves a TRACE, otherwise a model that never honoured the
    contract would go unnoticed.
    """
    if not valeur:
        return CREER
    clef = str(valeur).strip().lower()
    clef = ALIAS_OPERATIONS.get(clef, clef)
    if clef not in OPERATIONS:
        logger.warning(
            f"[concepts] opération INCONNUE « {valeur} » — hors de {OPERATIONS} ; "
            f"repli sur « {CREER} », rien n'est modifié dans l'existant"
        )
        return CREER
    return clef


def panier_de(axe_objet: str | None, axe_motif: str | None) -> tuple:
    """Basket key: the mode-purpose pair.

    The basket designates a small SET of candidates, never a single slot. A unique identity
    per pair would condemn the agent to a single thought per mode and per purpose, each new
    concept destroying the previous one: a cycling commuter would lose "the canal path
    is protected" when learning "the bike shelter is full at 8:30".
    """
    return (axe_objet, axe_motif)


def confiance(observations: int, contre_exemples: int) -> float:
    """Laplace's rule of succession (1814), so that a single observation is not worth
    certainty.

    A never-observed concept is worth 0.5: neither believed nor dismissed.
    """
    obs = max(0, int(observations or 0))
    contre = max(0, int(contre_exemples or 0))
    return (obs + 1) / (obs + contre + 2)


def est_hors_service(observations: int, contre_exemples: int) -> bool:
    """Does the concept stop being served to the model?

    Exact reading of the threshold under Laplace: the concept has been contradicted **more
    often** than it has been confirmed. It is NOT deleted for all that — its dated
    setting-aside is the observable the experiment looks for.

    ⚠ This is the third exception to the non-exclusion rule of lot 2, together with the
    agent's identity and the age window. It is written as an exception rather than merged
    into the rule: the wording "nothing filters" carries over to the paper, where it would
    become false without mention.
    """
    return confiance(observations, contre_exemples) < float(
        settings.agent.memoire__confiance_seuil_service
    )


def est_depasse(observations: int, contre_exemples: int) -> bool:
    """Is the concept outdated? TWO conditions, not one.

    At least three contradictions **and** more contradictions than confirmations. Neither
    three contradictions against twenty confirmations, nor a majority of contradictions over
    two observations.
    """
    return int(contre_exemples or 0) >= int(
        settings.agent.memoire__contre_exemples_seuil
    ) and est_hors_service(observations, contre_exemples)
