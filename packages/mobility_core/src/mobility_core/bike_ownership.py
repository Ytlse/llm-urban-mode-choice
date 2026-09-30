"""
core/bike_ownership.py — The persona's bike, learned from EMC² and not imputed to the household.

`personal_bike` was drawn by eqasim with `p = min(1, donor_bikes / size)`, where
the number of bikes is **copied** from an ENTD 2008 household matched without household
size or housing among the matching attributes. The total came out roughly right
(53.3% holders for ~51% expected) and the distribution was wrong: the household-size
gradient was **inverted** (76% holders among single persons against
33% observed, 36% in 4-person households against 65%). This module replaces this copy
with three stages learned from the EMC² Toulouse 2023 survey (ticket 015).

**The number of bikes is a household trait, not a per-person draw.** This is the core
point of the ticket and what fixes the order of the stages. In 4-person households,
the survey sees 15.7% of families with no bike at all and 40.1% with one bike per head; an
independent individual draw **calibrated on the same mean** produces 1.4% and 18.3%,
piling everything up in the middle (variance 0.91 against 2.13 observed, overdispersion ×2.4).
The two laws have exactly the same mean: no adjustment on the mean can therefore
bring them closer. We draw `k` first, and there is nothing left to adjust.

The three stages:

1. **How many bikes in the household** (`stock_law`) — multinomial logit on `k = M21`
   capped at `4+`, learned on the survey's 10,783 households with `COE0` weighting.
2. **Who in the household holds them** (`assign`) — weighted draw without replacement by a
   propensity learned on `P20` (« à vélo, conducteur », `COEP` weighting, respondents
   `PENQ = 1`), Efraimidis–Spirakis scheme. `k` decides **how many**, the propensity
   only decides **who**.
3. **Which type of bike** (`VAE_SHARE`) — draw per assigned bike, 7.7% of the fleet.

Three safeguards, the same as `housing_type`:

- **Determinism by hashing.** No RNG: the key is the home address for
  stage 1, the address plus the person's index for stages 2 and 3, and the salt is
  versioned. Two runs, two machines, two moments give the same fleet.
- **Outside the layer, no guessing.** A home without a fine zone has no law: the trait
  is `None`, and it must show. A massive `personal_bike = None` must make the validation
  **fail**, not pass.
- **No silent fallback.** Missing resource ⇒ error at load time.

## What stage 1 is conditioned on, and why not housing

The ticket left a trap to settle: the persona's `housing_type` is itself
**imputed** from the law of its fine zone (`core/housing_type.py`). Does conditioning `k`
on it amount to conditioning on the zone, or does it add something?

Measured on the survey: imputed housing matches observed housing only **47.6%**
of the time. Three consequences, in this order:

1. Conditioning `k` on (zone, size, car ownership) **without** housing reproduces the
   equipment curve by imputed housing to **within 0.6 point** (61.2 → 43.3% against
   61.6 → 42.7% expected).
2. Conditioning on imputed housing **degrades** this result (65.6 → 39.3%, i.e. +4.0 to
   −4.6 points of gap): it applies a coefficient learned on an observed variable
   to a variable that is wrong half the time, which inflates the amplitude beyond what
   the noisy variable can carry.
3. Above all, it would create a dependency between **two imputations** whose joint law
   is neither the true one nor one that can be measured — an artefact.

Decision: **stage 1 is conditioned on the zone, household size and car
ownership, never on housing.** Housing is not here "a rewrite of the
zone" accepted for convenience: it is a less informative variable than the zone,
since it is drawn from it.

A corollary not to be hidden, and it is an amendment to the ticket: the criterion "71% detached
house → 38% large apartment block (± 4 pts)" is **unreachable by construction** on a
synthetic population, whatever the quality of the model. Crossing the survey's **true**
number of bikes with **imputed** housing already gives 61.6 → 42.7%, i.e. 19 points
of amplitude instead of the 33.4 published: this is regression dilution, and it
caps what the measurement can see. The enforceable target is therefore the diluted curve, which
the exporter computes and publishes next to the published curve. Aiming at the 33.4 points
would amount to over-correcting the model to compensate for noise on the measurement axis.

`M2` (housing tenure status), which the ticket cited as a candidate covariate, is
discarded for another reason, that of the repository's feature contract: the persona does not
carry it. A variable that cannot be computed at application time does not go in.

The resource (`mobility_core/data/bike_ownership.json`) is produced by
`scripts/progedo_logit/export_bike_ownership.py` (`make bike-ownership`) from the
restricted-access microdata. Like the fine-zone layer, it is **outside the repository**.

Inputs/outputs are confined to `BikeOwnershipModel.load`; the rest of the module is
pure, in line with the `mobility_core` architecture contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mobility_core.population_reference import occupation_enquete

logger = logging.getLogger(__name__)

_occupations_inconnues: set[str] = set()


def _alarme_occupation_inconnue(valeur: object, occupations: Sequence[str]) -> None:
    """Rising edge per value: the alarm fires only once per category."""
    texte = str(valeur)
    if texte in _occupations_inconnues:
        return
    _occupations_inconnues.add(texte)
    logger.error(
        "[ALARME] occupation %r outside the vocabulary of the bike equipment law — "
        "indicators at zero, hence REFERENCE CATEGORY. Expected: %s.",
        texte, list(occupations))

# ── Output contract ──────────────────────────────────────────────────────────
# The three labels carried by `traits_json`, which no consumer may
# rebuild by hand. `None` (trait missing) is not a fourth category:
# it means "outside the fine-zone layer", and it must show.
TRAIT_KEY = "personal_bike"

# Trait labels, in ENGLISH since the v6 cohort (ticket 074). They are written into
# `traits_json` and read back by string comparison — not by a key — in
# `vehicle_chain.has_personal_bike` and `model_on_common_set.has_bike`. Both know
# BOTH vocabularies: cohorts before v6 and their archived runs still carry
# the French labels, and knowing only one would return "has a bike" for a
# persona who has none, across a whole cohort, without a single line missing.
NO_BIKE = "No bike"
PLAIN_BIKE = "regular bike"
ELECTRIC_BIKE = "e-bike"

#: The pre-v6 labels, kept to read back archived cohorts and traces.
LABELS_FR: tuple[str, ...] = ("Pas de vélo", "vélo normal", "VAE")

LABELS: tuple[str, ...] = (NO_BIKE, PLAIN_BIKE, ELECTRIC_BIKE)

# Draw salt. Versioned: changing it reshuffles the whole fleet, which must be a
# deliberate and dated act, not a side effect.
DRAW_SALT = "personal_bike_v1"

# Capping of `k`. Beyond 4 bikes the survey counts no longer carry anything
# (57 households out of 10,783 report 7 or more) and the distinction has no effect in
# simulation: a 4-person household cannot ride 7 bikes.
K_MAX = 4
K_CLASSES: tuple[int, ...] = tuple(range(K_MAX + 1))

# Capping of household size in the model indicators, aligned with the
# ticket's reference tables (1 / 2 / 3 / 4+).
SIZE_MAX = 4

# Minimum age to hold a bike. It is the scope of survey question `P20`,
# and it structurally prevents assigning the household bike to a three-year-old.
MIN_AGE_ELIGIBLE = 5

# Minimum age for an e-bike. Taken from the existing safeguard (ticket 008, A1.a): the draw
# without an age filter assigned electric-assist bikes to schoolchildren.
MIN_AGE_ELECTRIC = 14

# Share of e-bikes **in the fleet**: `ML21 / M21` in the survey, 7.67%. The draw is
# therefore made per assigned bike, not per person.
#
# The error this figure corrects: eqasim applied 14.8%, which is the share of
# *equipped households* owning at least one e-bike (8% of households / 54% equipped) —
# hence 1.7× too many e-bikes. A share of households is not a share of the fleet.
#
# Not to be confused either with the **12% of bike trips** made by e-bike (AUAT
# report p. 26): the 7.7 → 12% gap is a usage effect — an e-bike rides more than a
# pedal bike — not a stock effect. Aiming at 12% here would be a level error.
VAE_SHARE = 0.0767


# ── Deterministic draws ──────────────────────────────────────────────────────

def address_key(lat: float, lon: float) -> str:
    """Stable identifier of an address, to 10⁻⁶ degree (~0.1 m).

    Same key as `core.housing_type.address_key`, and on purpose: the address is
    already the repository's household key, the one that makes a household share a housing
    type. Stage 1 therefore introduces no new assumption.
    """
    return f"{float(lat):.6f},{float(lon):.6f}"


def uniform(key: str) -> float:
    """Deterministic uniform on [0, 1), derived from a stable hash.

    Python's `hash()` is randomised per process: it would give a different fleet on
    every run. SHA-256 depends neither on the version, nor on the platform, nor on
    `PYTHONHASHSEED`.
    """
    digest = hashlib.sha256(f"{DRAW_SALT}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def draw_index(probabilities: Sequence[float], u: float) -> int | None:
    """Inverse of the cumulative distribution function. `None` if the law is empty or degenerate.

    Drawing from nothing would return `0` — "no bike" — which is a plausible value
    and therefore undetectable. It is exactly the silent fallback this module refuses.

    The check is on a **finite and strictly positive** total, not only on
    `total > 0`: a softmax of all-infinite scores returns `nan`s, with which all
    comparisons are false. The loop then fell through to its rounding net and
    returned the **last** class — i.e. `k = 4`, a four-bike household out of an
    empty law, without a single log line. It is the worst possible silence; it is
    detected by the enrichment tests.
    """
    total = sum(probabilities)
    if not probabilities or not math.isfinite(total) or total <= 0:
        return None
    threshold = u * total
    cumulated = 0.0
    for index, probability in enumerate(probabilities):
        cumulated += probability
        if threshold < cumulated:
            return index
    return len(probabilities) - 1  # floating-point rounding net: u < 1


# ── Logit models, evaluated in pure Python ───────────────────────────────────
# Both stages are served as **coefficients**, not cell tables:
# a table crossing zone × size × car ownership × k would be sparse (785
# zones) and unreadable, whereas thirty-odd coefficients can be reread and checked
# by eye. Cell counts are published by the exporter in the resource's
# `validation` block — they serve for checking, not for prediction.

def _softmax(scores: Sequence[float]) -> list[float]:
    """Numerically stable softmax (shifting by the max keeps `exp` from overflowing)."""
    top = max(scores)
    exponentials = [math.exp(s - top) for s in scores]
    total = sum(exponentials)
    return [e / total for e in exponentials]


def _logistic(score: float) -> float:
    """Sigmoid, written never to overflow on very negative scores."""
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class LogitModel:
    """A logit (binary or multinomial) reduced to what is needed to predict.

    `features` fixes the order of the design vector; `coefficients` carries one row
    per class (multinomial) or a single one (binary). The order is a **contract**: a
    resource written for features other than the module's is rejected at
    load time rather than aligning coefficients on the wrong columns.
    """

    features: tuple[str, ...]
    intercepts: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.intercepts):
            raise ValueError(
                f"Inconsistent logit: {len(self.coefficients)} coefficient rows "
                f"for {len(self.intercepts)} intercepts."
            )
        for row in self.coefficients:
            if len(row) != len(self.features):
                raise ValueError(
                    f"Inconsistent logit: row of {len(row)} coefficients for "
                    f"{len(self.features)} features {self.features}."
                )

    def scores(self, design: dict[str, float]) -> list[float]:
        """Linear scores, one value per coefficient row.

        A feature missing from the design is 0 — that is the meaning of an inactive
        indicator, and the only case where it happens. Continuous features are
        always supplied by the module's callers.
        """
        return [
            intercept + sum(c * design.get(name, 0.0)
                            for name, c in zip(self.features, row))
            for intercept, row in zip(self.intercepts, self.coefficients)
        ]

    def probabilities(self, design: dict[str, float]) -> list[float]:
        """Multinomial law over `classes`."""
        return _softmax(self.scores(design))

    def probability(self, design: dict[str, float]) -> float:
        """Probability of the positive class of a binary logit."""
        if len(self.coefficients) != 1:
            raise ValueError("probability() only makes sense for a binary logit.")
        return _logistic(self.scores(design)[0])

    @classmethod
    def from_doc(cls, doc: dict) -> LogitModel:
        return cls(
            features=tuple(str(f) for f in doc["features"]),
            intercepts=tuple(float(v) for v in doc["intercepts"]),
            coefficients=tuple(tuple(float(v) for v in row)
                               for row in doc["coefficients"]),
            classes=tuple(int(c) for c in doc.get("classes") or (1,)),
        )


# ── Design vectors — THE contact point between training and application ──────
# These two functions are the module's reason to exist: they are called **at
# training** by `export_bike_ownership.py` and **at application** by
# `enrich_personal_bike.py`. The features are written once, so no drift is
# possible between the two sides — the lesson of ticket 005 (§3, feature
# contract) applied here.

def stock_design(household_size: int,
                 number_of_cars: float | None,
                 density_hh_km2: float | None,
                 dist_center_km: float) -> dict[str, float]:
    """Design vector of stage 1 (how many bikes in the household).

    No individual covariate: `k` is a household attribute. `density_hh_km2`
    is missing for the 81 fine zones (out of 785) with no surveyed household; it is replaced
    by the scope median, carried by the resource — not zero, which would describe a desert.
    """
    size = max(1, min(int(household_size), SIZE_MAX))
    cars = 0.0 if number_of_cars is None else float(number_of_cars)
    design = {
        "size2": 1.0 if size == 2 else 0.0,
        "size3": 1.0 if size == 3 else 0.0,
        "size4p": 1.0 if size >= 4 else 0.0,
        "cars1": 1.0 if 0.5 <= cars < 1.5 else 0.0,
        "cars2p": 1.0 if cars >= 1.5 else 0.0,
        "log_density": math.log1p(max(0.0, float(density_hh_km2 or 0.0))),
        "log_dist_center": math.log1p(max(0.0, float(dist_center_km))),
    }
    return design


def propensity_design(k: int,
                      household_size: int,
                      age: float | None,
                      gender: str | None,
                      main_occupation: str | None,
                      density_hh_km2: float | None,
                      dist_center_km: float,
                      occupations: Sequence[str]) -> dict[str, float]:
    """Design vector of stage 2 (propensity to ride a bike).

    **No trip distance**, and it is a decision, not an oversight: a stock
    must be invariant to the trip. Otherwise the same agent has a bike for the bakery and
    none for work, and the vehicle chain lock loses its meaning. (`D12` is
    moreover endogenous to the mode and already flagged "contaminated" by the mode
    choice policy; `DP15` is 0 for 54.6% of persons.)

    `occupations` fixes the order of the occupation indicators: it comes from the resource,
    so that the vector stays aligned with the coefficients even if the repository recoding
    gains a category. The VOCABULARY also comes from it — the resource is a frozen
    artefact, which names its variables `occ_Travail à plein temps` — whereas the persona says
    `Full-time worker` since ticket 074: the occupation therefore goes through
    `occupation_enquete`, otherwise all indicators would stay at zero and
    the whole cohort would fall into the reference category, with nothing reported.
    """
    size = max(1, min(int(household_size), SIZE_MAX))
    stock = max(0, min(int(k), K_MAX))
    design = {
        "k1": 1.0 if stock == 1 else 0.0,
        "k2": 1.0 if stock == 2 else 0.0,
        "k3": 1.0 if stock == 3 else 0.0,
        "k4p": 1.0 if stock >= 4 else 0.0,
        "size2": 1.0 if size == 2 else 0.0,
        "size3": 1.0 if size == 3 else 0.0,
        "size4p": 1.0 if size >= 4 else 0.0,
        # Age in hundreds of years, with its square: cycling is not
        # monotonic in age (it peaks among pupils and young workers).
        "age": 0.0 if age is None else float(age) / 100.0,
        "age2": 0.0 if age is None else (float(age) / 100.0) ** 2,
        "female": 1.0 if (gender or "") == "Female" else 0.0,
        "log_density": math.log1p(max(0.0, float(density_hh_km2 or 0.0))),
        "log_dist_center": math.log1p(max(0.0, float(dist_center_km))),
    }
    enquete = occupation_enquete(main_occupation)
    if enquete is None and main_occupation not in (None, ""):
        _alarme_occupation_inconnue(main_occupation, occupations)
    for occupation in occupations:
        design[f"occ_{occupation}"] = 1.0 if enquete == occupation else 0.0
    return design


# ── Stage 2 — the weighted draw without replacement ──────────────────────────

@dataclass(frozen=True)
class Member:
    """A household member, as the assignment needs to see it.

    `present = False` denotes an **absent seat**: a member of the nominal household that the
    spatial filter did not keep in the population file. These seats take part
    in the draw and may take a bike, but nothing is written for them.
    """

    index: int
    propensity: float
    eligible: bool
    present: bool = True


def assign(members: Sequence[Member], k: int, household_key: str) -> set[int]:
    """Who, among `members`, holds one of the household's `k` bikes.

    Draw without replacement weighted by propensity, **Efraimidis–Spirakis** scheme:
    each eligible member receives a key `u ** (1 / p)` with `u` deterministic uniform,
    members are ranked by decreasing key, and the first `min(k, eligible)` are served.

    Three intended properties:

    - the number assigned is **exactly** the household stock — `k` sets the
      level, the ranking only orders;
    - the probability of being served **grows** with propensity;
    - there is **no deterministic order**: no "always the eldest", no sorting
      artefact on ties. The index only enters the hash.

    The last bikes therefore go to low-propensity members: **these are the
    dormant bikes**, and it is right to represent them. ~11 points of the population
    will hold a bike without riding it (≈ 51% holders for 39.5% riders).
    Their holder will not use them — it is up to the mode choice model and the agent to
    decide not to take them, not up to the imputation to make them disappear.

    Consequence to accept: the actual inclusion probability of this scheme is not
    exactly `p_i`, it is distorted by the counting constraint. The table
    `P(riding | k, size)` is therefore not an identity but a **validation
    criterion** — we check afterwards that the mechanism reproduces it.

    If `k` exceeds the number of eligible members, the surplus is held by no one: a bike
    is a household object, and since the JSON only carries individuals, a bike without
    a holder simply does not appear in it.
    """
    if k <= 0:
        return set()
    keyed: list[tuple[float, int]] = []
    for member in members:
        if not member.eligible:
            continue
        u = uniform(f"bike-holder:{household_key}:{member.index}")
        # `p = 0` must not raise: a zero propensity gives the lowest key,
        # hence service as a very last resort — which is the intended meaning, not an
        # exclusion. `u = 0` (zero measure but reachable on 64 bits) likewise.
        propensity = max(1e-12, float(member.propensity))
        keyed.append((u ** (1.0 / propensity), member.index))
    keyed.sort(key=lambda pair: (-pair[0], pair[1]))
    return {index for _, index in keyed[:k]}


def electric_probability(under_age_holder_share: float) -> float:
    """E-bike probability to apply **to eligible holders**, so that the fleet comes out
    at `VAE_SHARE`.

    The age filter and the fleet share contradict each other if applied naively.
    `VAE_SHARE` is a share of the **whole** fleet, children included: `ML21 / M21` does not
    distinguish whose bike it is. But no e-bike is assigned under 14 (safeguard
    of ticket 008: the unfiltered draw put electric-assist bikes under
    schoolchildren). Applying 7.67% to those aged 14 and over only therefore brings the fleet
    **below** the target, in proportion to the bikes held by children — measured at
    11.8% of holders on `toulouse_population_1000.json`, i.e. a fleet capped at
    6.8% instead of 7.7%.

    We renormalise: `p = VAE_SHARE / (1 − share_of_ineligible_holders)`. It is also
    the fairest reading of reality — an e-bike is almost never a child's
    bike, so the 7.67% of the fleet is in fact concentrated on adult bikes.

    The ineligible share comes from the resource — measured by the exporter by replaying
    the assignment **on the survey** (16.2%) — and not from the population being enriched,
    whose observed share differs (11.8% on `toulouse_population_1000.json`). This choice
    is deliberate: a probability recomputed on each file would make a persona's bike type
    depend on the **file it is in**, so that the same
    household would come out with an e-bike in the 1,000-agent population and a pedal bike in
    the 10,000 one. Determinism by hashing would lose all meaning. We prefer a
    half-point gap on the fleet to a trait that moves with the context.

    Capped at 0.5: beyond that the renormalisation would exceed 15% and would signal an
    aberrant population rather than a legitimate adjustment.
    """
    share = min(0.5, max(0.0, float(under_age_holder_share)))
    return min(1.0, VAE_SHARE / (1.0 - share))


def bike_label(household_key: str, member_index: int, age: float | None,
               electric_p: float = VAE_SHARE) -> str:
    """Stage 3: which bike, for a holder already selected by stage 2.

    The draw is on **the bike**, via the (household, member) key that identifies it —
    since a holder holds exactly one bike. Salt distinct from the assignment's:
    otherwise the stage 2 draw rank and the bike type would be correlated, and
    e-bikes would systematically go to high propensities.

    `electric_p` is the probability **conditional on eligible holders**, as
    `electric_probability` computes it. The `VAE_SHARE` default is the degraded "no
    renormalisation" case: it under-produces e-bikes, and it is only there so that the function
    stays callable on its own in a test.
    """
    if age is not None and float(age) < MIN_AGE_ELECTRIC:
        return PLAIN_BIKE
    u = uniform(f"bike-kind:{household_key}:{member_index}")
    return ELECTRIC_BIKE if u < electric_p else PLAIN_BIKE


# ── The resource ─────────────────────────────────────────────────────────────

DEFAULT_RESOURCE = Path(__file__).resolve().parent / "data" / "bike_ownership.json"

# Accepted resource version, on the same pattern as `residence_zone.RESOURCE_VERSION`
# and `housing_type.MIN_RESOURCE_VERSION`: a resource written for another schema
# (new covariate, different capping) must be explicitly rejected at
# load time, not silently loaded with misaligned coefficients.
RESOURCE_VERSION = 1

# Expected features of each stage. The module rejects a resource written for
# others: coefficients aligned on the wrong columns do not produce an
# error, they produce a wrong fleet.
STOCK_FEATURES: tuple[str, ...] = (
    "size2", "size3", "size4p", "cars1", "cars2p", "log_density", "log_dist_center",
)
PROPENSITY_BASE_FEATURES: tuple[str, ...] = (
    "k1", "k2", "k3", "k4p", "size2", "size3", "size4p",
    "age", "age2", "female", "log_density", "log_dist_center",
)


@dataclass(frozen=True)
class BikeOwnershipModel:
    """The three stages, as they can be applied outside the source data."""

    stock: LogitModel
    propensity: LogitModel
    occupations: tuple[str, ...]
    median_density: float
    under_age_holder_share: float
    validation: dict
    meta: dict

    @property
    def electric_p(self) -> float:
        """E-bike probability to apply to holders aged 14 and over."""
        return electric_probability(self.under_age_holder_share)

    @classmethod
    def load(cls, resource: Path | None = None) -> BikeOwnershipModel:
        """Loads the resource (the module's only I/O point).

        Missing = explicit error. A fallback to an overall law would produce a fleet
        decorrelated from geography and household size, i.e. exactly
        the bias this module exists to correct.
        """
        path = Path(resource) if resource else DEFAULT_RESOURCE
        if not path.exists():
            raise FileNotFoundError(
                f"Bike equipment model missing: {path}. Produce it with "
                "`make bike-ownership` (python -m scripts.progedo_logit."
                "export_bike_ownership) — it requires the PROGEDO data under "
                "'data/PROGEDO 2023/' (restricted access lil-1750)."
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        version = int(doc.get("version") or 0)
        if version != RESOURCE_VERSION:
            raise ValueError(
                f"Resource {path} is at version {version}, the module expects "
                f"{RESOURCE_VERSION}. Rerun `make bike-ownership`."
            )
        stock = LogitModel.from_doc(doc["stock"])
        propensity = LogitModel.from_doc(doc["propensity"])
        occupations = tuple(str(o) for o in doc.get("occupations") or ())

        if stock.features != STOCK_FEATURES:
            raise ValueError(
                f"Resource {path}: stage 1 written for features other than the "
                f"module's.\n  resource : {stock.features}\n  module   : "
                f"{STOCK_FEATURES}\nRe-export it (make bike-ownership)."
            )
        expected = PROPENSITY_BASE_FEATURES + tuple(f"occ_{o}" for o in occupations)
        if propensity.features != expected:
            raise ValueError(
                f"Resource {path}: stage 2 written for features other than the "
                f"module's.\n  resource : {propensity.features}\n  module   : "
                f"{expected}\nRe-export it (make bike-ownership)."
            )
        if stock.classes != K_CLASSES:
            raise ValueError(
                f"Resource {path}: stage 1 written for classes {stock.classes}, "
                f"the module expects {K_CLASSES} (capping K_MAX={K_MAX})."
            )
        return cls(
            stock=stock,
            propensity=propensity,
            occupations=occupations,
            median_density=float(doc.get("median_density") or 0.0),
            under_age_holder_share=float(doc.get("under_age_holder_share") or 0.0),
            validation=doc.get("validation") or {},
            meta=doc.get("meta") or {},
        )

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def stock_probabilities(self, household_size: int,
                            number_of_cars: float | None,
                            density_hh_km2: float | None,
                            dist_center_km: float) -> list[float]:
        """Law of `k` over `K_CLASSES` for a household."""
        density = self.median_density if density_hh_km2 is None else density_hh_km2
        return self.stock.probabilities(
            stock_design(household_size, number_of_cars, density, dist_center_km))

    def draw_stock(self, household_size: int,
                   number_of_cars: float | None,
                   density_hh_km2: float | None,
                   dist_center_km: float,
                   household_key: str) -> int | None:
        """Number of bikes of a household, drawn by hashing its key (the address).

        One draw **per household**, replacing the copy from the ENTD donor: the
        number of bikes stops being independent of the household that receives it, and that is
        the whole point of the ticket.
        """
        probabilities = self.stock_probabilities(
            household_size, number_of_cars, density_hh_km2, dist_center_km)
        index = draw_index(probabilities, uniform(f"bike-stock:{household_key}"))
        return None if index is None else K_CLASSES[index]

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def propensity_of(self, k: int, household_size: int, age: float | None,
                      gender: str | None, main_occupation: str | None,
                      density_hh_km2: float | None,
                      dist_center_km: float) -> float:
        """Propensity of a person to ride a bike, in the `P20` sense."""
        density = self.median_density if density_hh_km2 is None else density_hh_km2
        return self.propensity.probability(propensity_design(
            k, household_size, age, gender, main_occupation,
            density, dist_center_km, self.occupations))
