"""Stage 1 tally — ticket 059, lot 3.

WHAT THIS MODULE COMPUTES, AND WHAT IT REFUSES TO COMPUTE
---------------------------------------------------------
Three quantities, and two guards.

- **The modal share gap** between two conditions, paired on the SAME trips. A gap
  computed on two different samples would measure the difference between the samples.
- **The sign agreement rate** against the frozen grid, with its binomial test. On twenty
  predictions, the bar is at fifteen agreements ($p = 0.021$); fourteen are not enough.
- **The weighted kappa** on the ordinal intensity, published without being tested: twenty items
  do not allow it.

The first guard is **vacuity**. A mode absent from the sample returns "non
concluant", never 0.0. In this repository, a missing measurement yields the perfect score, and this
pattern has lied before: a zero gap and an unmeasurable gap are both written `0.0` if nobody
pays attention.

The second is **noise**. A gap smaller than the decision-maker's noise floor is
not an effect. The floor is declared with the result, it is not kept quiet — measured at 3.2 % on
the memory agent (ticket 095, § 7 bis), and to be established by identical replay for each
decision-maker measured here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from scripts.analysis.presse.grille import Grille

# The verdict returned when the count does not allow a conclusion. It is a RESULT, not a failure,
# and it is distinct from a zero: `slope_verdict` of ticket 031 set this precedent.
NON_CONCLUANT = "non concluant"

# Minimum number of trips carrying a mode for its gap to be interpreted. Below it, a
# single trip that flips moves the share by more than ten points.
EFFECTIF_MIN_PAR_MODE = 30


@dataclass(frozen=True)
class EcartModal:
    article: str
    mode: str
    part_reference: float | None
    part_condition: float | None
    n_apparies: int
    verdict: str | None = None

    @property
    def ecart(self) -> float | None:
        if self.verdict or self.part_reference is None or self.part_condition is None:
            return None
        return self.part_condition - self.part_reference

    # No `signe` property here, on purpose: the sign of a gap is not a
    # property of the gap, it depends on the decision-maker's NOISE FLOOR. Putting it on this
    # class would make it a default, and a default gets forgotten — every non-zero gap
    # would then take a sign, including that of a decision flipping at random. The sign
    # is computed by `signes_observes`, which requires that floor.


def _part(decisions: list[str], mode: str) -> float | None:
    return decisions.count(mode) / len(decisions) if decisions else None


def ecart_apparie(
    reference: dict[str, str], condition: dict[str, str], *, article: str, mode: str
) -> EcartModal:
    """Share gap of `mode`, on the trips present in BOTH conditions.

    `reference` and `condition` map a trip identifier to the chosen mode.
    The intersection is taken explicitly: a trip carried by only one condition is
    set aside and counted, never filled in with a default.
    """
    communs = sorted(set(reference) & set(condition))
    if len(communs) < EFFECTIF_MIN_PAR_MODE:
        return EcartModal(article, mode, None, None, len(communs), verdict=NON_CONCLUANT)
    ref = [reference[d] for d in communs]
    con = [condition[d] for d in communs]
    return EcartModal(article, mode, _part(ref, mode), _part(con, mode), len(communs))


def signes_observes(ecarts: list[EcartModal], *, plancher_de_bruit: float) -> dict[tuple[str, str], str | None]:
    """The sign of each gap, '0' when it fits within the noise, None when it is not readable.

    `plancher_de_bruit` is a SHARE (0.032 for 3.2 %), and it is required: it has no
    default, because a default would get it forgotten. A gap smaller than the decision-maker's
    own noise is not an effect, and calling it '+' because it is positive would amount to scoring
    chance against a prediction.
    """
    if plancher_de_bruit < 0:
        raise ValueError("the noise floor is a positive share")
    observes: dict[tuple[str, str], str | None] = {}
    for e in ecarts:
        if e.ecart is None:
            observes[(e.article, e.mode)] = None
            continue
        if abs(e.ecart) <= plancher_de_bruit:
            observes[(e.article, e.mode)] = "0"
        else:
            observes[(e.article, e.mode)] = "+" if e.ecart > 0 else "-"
    return observes


@dataclass(frozen=True)
class AccordDeSigne:
    concordants: int
    lisibles: int
    non_lisibles: int
    # ⚠ `p_binomial` is KEPT but is now always `None`: the binomial test was
    # removed from chapter 7 (author's review, 2026-09-21). The field stays so that
    # reports already written can be read back without an exception, and so that one SEES it is
    # empty rather than believing it never existed. It goes at the next rework of the module.
    p_binomial: float | None = None
    # The interval of the sign agreement, by resampling GROUPED BY EVENT.
    intervalle: tuple[float, float] | None = None
    verdict: str | None = None

    @property
    def taux(self) -> float | None:
        return self.concordants / self.lisibles if self.lisibles else None


# Number of resamples. 2,000 is enough to stabilise a 95 % interval to the hundredth on
# five groups; going higher costs time without moving the bound.
REECHANTILLONNAGES = 2000
GRAINE_BOOTSTRAP = 59


def intervalle_groupe_par_evenement(
    concordances: list[tuple[str, bool]],
    *,
    tirages: int = REECHANTILLONNAGES,
    graine: int = GRAINE_BOOTSTRAP,
) -> tuple[float, float] | None:
    """95 % interval of the agreement rate, resampling EVENTS, not cells.

    ⚠ **This is what replaces the binomial test, and the reason is structural.** The binomial
    assumed twenty independent draws. They are not: the modal shares of one
    event **sum to one**, so that a single behaviour — the agent leaves the bike for the
    car — mechanically produces two, even four agreements. Counting these cells as
    separate observations inflates the count and narrows the interval by a factor that has
    no counterpart in the data.

    The resampling unit is therefore the **event**: five articles are drawn with replacement, and
    each article takes its four cells along as a block. The resulting interval says what
    the sample of articles allows one to claim — and on five articles, it is wide. It is a
    result, not a flaw of the method: five events do not carry the precision that
    twenty cells claimed to give.

    Returns `None` below two distinct events: an interval drawn on a single group
    resamples nothing.
    """
    import random as _random

    par_evenement: dict[str, list[bool]] = {}
    for evenement, concorde in concordances:
        par_evenement.setdefault(evenement, []).append(concorde)
    groupes = [par_evenement[k] for k in sorted(par_evenement)]
    if len(groupes) < 2:
        return None

    alea = _random.Random(graine)
    taux: list[float] = []
    for _ in range(tirages):
        tire = [alea.choice(groupes) for _ in groupes]
        plats = [c for g in tire for c in g]
        if plats:
            taux.append(sum(plats) / len(plats))
    if not taux:
        return None
    taux.sort()
    bas = taux[int(0.025 * (len(taux) - 1))]
    haut = taux[int(0.975 * (len(taux) - 1))]
    return (round(bas, 4), round(haut, 4))


def accord_de_signe(
    grille: Grille, observes: dict[tuple[str, str], str | None]
) -> AccordDeSigne:
    """The sign agreement rate and its interval, on the READABLE cells only.

    ⚠ The denominator is the number of readable cells, not the grid's twenty. Counting an
    unmeasurable cell as a disagreement would blame the effect for the sample's fault;
    counting it as an agreement would be worse. The report publishes both numbers.

    ⚠ **The binomial test is removed** (chapter 7, review of 2026-09-21). It assumed twenty
    independent cells; they are not. The uncertainty goes through
    `intervalle_groupe_par_evenement`, which resamples articles and not cells.
    """
    lisibles = [
        (c, observes.get((c.article, c.mode)))
        for c in grille.cellules
        if observes.get((c.article, c.mode)) is not None
    ]
    n_non_lisibles = len(grille.cellules) - len(lisibles)
    if not lisibles:
        return AccordDeSigne(0, 0, n_non_lisibles, verdict=NON_CONCLUANT)
    concordants = sum(1 for c, o in lisibles if c.concorde(o))
    return AccordDeSigne(
        concordants=concordants,
        lisibles=len(lisibles),
        non_lisibles=n_non_lisibles,
        p_binomial=None,
        intervalle=intervalle_groupe_par_evenement(
            [(c.article, c.concorde(o)) for c, o in lisibles]
        ),
    )


def kappa_pondere(
    grille: Grille, intensites_observees: dict[tuple[str, str], int | None]
) -> float | str:
    """Cohen's kappa with quadratic weighting on the ordinal intensity 0-3.

    Published WITHOUT being tested: twenty items do not allow testing its significance, and an
    interval on twenty cells would mostly tell the size of the sample.

    Returns `NON_CONCLUANT` when fewer than ten cells are readable, or when one of the two
    series is constant — a kappa on a constant series is 0 by construction, and that zero
    says nothing about agreement.
    """
    paires = [
        (c.intensite, intensites_observees[(c.article, c.mode)])
        for c in grille.cellules
        if intensites_observees.get((c.article, c.mode)) is not None
    ]
    if len(paires) < 10:
        return NON_CONCLUANT
    attendues, observees = zip(*paires, strict=True)
    if len(set(attendues)) == 1 or len(set(observees)) == 1:
        return NON_CONCLUANT

    n = len(paires)
    categories = sorted(set(attendues) | set(observees))
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    poids = [[1 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]

    observe = 0.0
    for a, o in paires:
        observe += poids[index[a]][index[o]]
    observe /= n

    ca, co = Counter(attendues), Counter(observees)
    hasard = 0.0
    for a, na in ca.items():
        for o, no in co.items():
            hasard += poids[index[a]][index[o]] * (na / n) * (no / n)
    if math.isclose(hasard, 1.0):
        return NON_CONCLUANT
    return (observe - hasard) / (1 - hasard)
