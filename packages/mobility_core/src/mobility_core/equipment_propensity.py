"""equipment_propensity.py — Individual propensity for a mobility equipment item.

Machinery **shared** by two traits that tickets 016 and 017 specify
separately but of which they say explicitly that "batches 1 and 2 are common:
same source file, same `PENQ = 1` restriction, same `COEP` weighting, same cause
(the 15-29 age class of the ENTD matching), same correction pattern". A single
loader, a single design vector, two targets:

| Trait | EMC² target | Resource |
|---|---|---|
| `has_pt_subscription` | `P12 == 6` (PT pass valid yesterday) | `pt_subscription.json` |
| `has_driving_license` | `P7 == 1` (car driving licence) | `driving_license.json` |

**Why one module and not two.** The design vector is identical for both traits,
and it is the only place in the repository where training and application must see
*exactly* the same vector. Two copies would drift at the first recoding
change — which is why `export_bike_ownership` already goes through
`bike_ownership.propensity_design` rather than copying the formula.

## What the module does not do

- **No household stage.** A pass and a licence are personal: no shared
  stock, no draw without replacement, no household to rebuild. That is what makes
  these two traits cheaper than the bike (ticket 015). Within-household correlation
  is carried by **household car ownership**, a covariate that is observed and correct in the
  synthetic population (1.28 simulated cars against 1.25 measured).
- **No trip covariate.** An equipment item is a **stock**, it must be
  invariant to the trip: otherwise the same agent has a pass to go to work and
  no longer has one to go shopping, and the trait stops being a trait. Same argument
  as ticket 015 on `D12`.
- **No income.** `M22` (household income) is delivered **empty** — 0 non-null values out of
  10,783 households. The eligibility filter of means-tested fares
  (free travel for seniors, solidarity fare for jobseekers, student grant levels) is therefore
  unobservable, and this module does not approximate it. It does not need to: its target is
  "holds a pass", not "benefits from such a discount" — the survey only serves
  two categories of `P12` (yes without detail / no) and does not allow the second
  question.

## Where pricing comes in, and where it does not

It **never comes in as a quantity**: no amount, no level, no ridership
rate. It comes in only as a **break location** in the age
curve — the `under_26`, `age_62p`, `age_65p` steps of `FEATURE_KNOTS`. In a logit
age enters the log-odds linearly and *cannot* produce the cliff that
the survey measures (15-17: 64.0% · 18-24: 63.3% · 25-29: 29.3%); the Tisséo rule
"under 26 and students" says where to place the knot, the survey says whether it is worth
it. The export script fits **with and without** these steps and publishes both
out-of-sample AUCs: a step that does not improve cross-validation grouped by
household is removed, and the removal is printed.

## The draw

Bernoulli of the propensity, hash key `(home address, person index,
versioned salt)` — the determinism of `housing_type.py`. Two runs, two machines,
two moments give the same result; changing `DRAW_SALT` is a dated act.

Unlike housing, the draw is on **the person** and not on the address: a
pass and a licence are individual, two flatmates have no reason to
share theirs. The address stays in the key so that two populations drawn from two
different subsets give the same value to the same person.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mobility_core.population_reference import occupation_enquete

logger = logging.getLogger(__name__)

RESOURCE_DIR = Path(__file__).resolve().parent / "data"

# Legal driving age. It is not a model parameter: no propensity is
# evaluated below it, the trait is `False` by construction. The survey confirms it
# (0.0% licence holders at 15-17), and a legal threshold is not fitted.
DRIVING_AGE = 18

# Minimum age of the `P12` scope ("PT pass valid yesterday") and minimum age of the
# exported population. No agent therefore escapes the pass imputation.
PT_MIN_AGE = 5


@dataclass(frozen=True)
class TraitSpec:
    """What distinguishes the two traits — everything else is common."""

    key: str                    # key in `traits_json`
    resource: str               # resource file name
    salt: str                   # draw salt, versioned
    target: str                 # description of the EMC² target, written into the resource
    min_age: int                # below it, no propensity evaluated
    below_min_age: bool         # value imposed below `min_age`


PT_SUBSCRIPTION = TraitSpec(
    key="has_pt_subscription",
    resource="pt_subscription.json",
    salt="pt_subscription_v1",
    target="P12 == 6 — possession d'un abonnement TC valide hier",
    min_age=PT_MIN_AGE,
    below_min_age=False,
)

DRIVING_LICENSE = TraitSpec(
    key="has_driving_license",
    resource="driving_license.json",
    salt="driving_license_v1",
    # `P7 == 3` means "supervised driving and driving lessons": 266 surveyed
    # persons, median age 18, of whom 155 adults. They are NOT licence holders.
    # A bare `== 1` would assert it silently; it is written here to be read.
    target="P7 == 1 — titulaire du permis voiture (P7 == 3, conduite accompagnée, "
           "compte NON)",
    min_age=DRIVING_AGE,
    below_min_age=False,
)

TRAITS = {"pt_subscription": PT_SUBSCRIPTION, "driving_license": DRIVING_LICENSE}


# ── Design vector ────────────────────────────────────────────────────────────
#
# Fixed order. The resource coefficients are aligned on it, and
# `design_vector` is the ONLY definition: training calls it just as
# application does.

# Terms always present.
FEATURE_BASE = (
    "age10",            # age / 10 — continuous slope
    "age10_sq",         # curvature: the propensity for a pass is not monotonic
    "female",
    "cars0",            # household without a car — 61.8% pass holders against 16.1% at 2+
    "cars2p",           # household with two cars or more (reference: one car)
    "log_density",      # household density of the fine zone, in log
    "dist_center10",    # distance to the hypercentre / 10 km
)

# Fare steps. Present only if cross-validation keeps them — the
# resource carries the list actually fitted in `features`.
FEATURE_KNOTS = (
    "under_26",         # Tisséo youth fare ("under 26")
    "age_62p",          # senior eligibility for retirees
    "age_65p",          # general senior eligibility
)


def _knot_values(age: float) -> dict[str, float]:
    return {
        "under_26": 1.0 if age < 26 else 0.0,
        "age_62p": 1.0 if age >= 62 else 0.0,
        "age_65p": 1.0 if age >= 65 else 0.0,
    }


_occupations_inconnues: set[str] = set()


def _alarme_occupation_inconnue(valeur: object, occupations: Sequence[str]) -> None:
    """Says once that an occupation has fallen into the reference category.

    Rising edge per value: without it, the alarm would fire for each of the 11,329
    personas of a pool and would drown `make error`.
    """
    texte = str(valeur)
    if texte in _occupations_inconnues:
        return
    _occupations_inconnues.add(texte)
    logger.error(
        "[ALARME] occupation %r outside the vocabulary of the fitted law — all its "
        "indicators stay at zero, hence the REFERENCE CATEGORY. Expected "
        "categories: %s. A whole cohort in the reference pulls the trait towards the "
        "mean without anything missing anywhere.", texte, list(occupations))


def design_vector(age: float | None,
                  gender: str | None,
                  main_occupation: str | None,
                  number_of_cars: float | None,
                  density_hh_km2: float | None,
                  dist_center_km: float | None,
                  occupations: Sequence[str],
                  features: Sequence[str],
                  median_density: float) -> list[float]:
    """Design vector, in the order of ``features``.

    ``features`` comes from the resource: it is the resource that fixes the order AND says which
    steps were kept at fitting. Reading the order anywhere other than here
    (a module constant, for example) would misalign the vector with the coefficients
    as soon as a step was removed.

    A missing value is never guessed "as the most frequent": density
    falls back on its scope median (published in the resource), distance on
    zero, and the absence is counted by the caller.

    THE OCCUPATION GOES THROUGH THE SURVEY VOCABULARY. The resource names its variables
    `occ_Travail à plein temps`: these are the categories the coefficients
    were fitted on, and they are part of the frozen artefact. The persona, for its part, says
    `Full-time worker` since ticket 074. Comparing the two strings as they are
    left ALL indicators at zero — the reference category for the whole
    cohort, without a single log line; measured on v6, the gap of
    `permis_adultes` to its target went from 2.52 to 5.23 points.

    A category that neither vocabulary knows still leaves its indicators
    at zero — it is the reference category, not an invented one — but it now
    **says so**, once per unknown value.
    """
    a = float(age or 0.0)
    density = median_density if density_hh_km2 is None or (
        isinstance(density_hh_km2, float) and math.isnan(density_hh_km2)
    ) else float(density_hh_km2)
    cars = float(number_of_cars or 0.0)
    values: dict[str, float] = {
        "age10": a / 10.0,
        "age10_sq": (a / 10.0) ** 2,
        "female": 1.0 if (gender or "") == "Female" else 0.0,
        "cars0": 1.0 if cars <= 0 else 0.0,
        "cars2p": 1.0 if cars >= 2 else 0.0,
        "log_density": math.log1p(max(0.0, density)),
        "dist_center10": float(dist_center_km or 0.0) / 10.0,
        **_knot_values(a),
    }
    enquete = occupation_enquete(main_occupation)
    if enquete is None and main_occupation not in (None, ""):
        _alarme_occupation_inconnue(main_occupation, occupations)
    for occupation in occupations:
        values[f"occ_{occupation}"] = 1.0 if enquete == occupation else 0.0
    missing = [f for f in features if f not in values]
    if missing:
        raise KeyError(f"Unknown design vector variables: {missing}")
    return [values[f] for f in features]


# ── Deterministic draw ───────────────────────────────────────────────────────

def uniform(salt: str, key: str) -> float:
    """Deterministic uniform on [0, 1), derived from a stable hash.

    Python's `hash()` is randomised per process: it would give different traits
    on every run. SHA-256 depends neither on the version, nor on the
    platform, nor on `PYTHONHASHSEED`.
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def draw_key(lat: float | None, lon: float | None, person_id: Any) -> str:
    """Draw key: home address **and** person identifier.

    The address alone would make all members of a household draw identically — wrong for
    a personal trait. The identifier alone would make the trait depend on the population
    draw, and so change at every regeneration.
    """
    if lat is None or lon is None:
        return f"nohome/{person_id}"
    return f"{float(lat):.6f},{float(lon):.6f}/{person_id}"


# ── Resource ─────────────────────────────────────────────────────────────────

@dataclass
class PropensityLaw:
    """A propensity law loaded from its resource, ready to apply."""

    spec: TraitSpec
    features: tuple[str, ...]
    occupations: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    median_density: float
    meta: dict

    @classmethod
    def load(cls, spec: TraitSpec, resource: Path | None = None) -> PropensityLaw:
        """Loads the resource. Its absence is an **explicit error**.

        Never a silent fallback to an overall propensity: a missing law must
        stop the enrichment with the message saying which command produces it, not
        blindly impute a trait that three consumers will read as measured.
        """
        path = Path(resource) if resource else RESOURCE_DIR / spec.resource
        if not path.exists():
            raise FileNotFoundError(
                f"Propensity law missing: {path}. Produce it with "
                f"`make equipment-propensity` (PROGEDO data required, restricted "
                f"access lil-1750).")
        raw = json.loads(path.read_text(encoding="utf-8"))
        law = raw.get("law", raw)
        features = tuple(law["features"])
        coefficients = tuple(float(v) for v in law["coefficients"])
        if len(features) != len(coefficients):
            raise ValueError(
                f"{path}: {len(features)} variables for {len(coefficients)} "
                f"coefficients. Resource corrupted or truncated.")
        return cls(
            spec=spec,
            features=features,
            occupations=tuple(law["occupations"]),
            intercept=float(law["intercept"]),
            coefficients=coefficients,
            median_density=float(law["median_density"]),
            meta=raw.get("meta", {}),
        )

    def propensity(self, age: float | None, gender: str | None,
                   main_occupation: str | None, number_of_cars: float | None,
                   density_hh_km2: float | None,
                   dist_center_km: float | None) -> float | None:
        """P(trait) for a person, or ``None`` below the trait's scope age."""
        if age is None:
            return None
        if float(age) < self.spec.min_age:
            return None
        x = design_vector(age, gender, main_occupation, number_of_cars,
                          density_hh_km2, dist_center_km, self.occupations,
                          self.features, self.median_density)
        z = self.intercept + sum(c * v for c, v in zip(self.coefficients, x))
        return 1.0 / (1.0 + math.exp(-z))

    def value(self, age: float | None, gender: str | None,
              main_occupation: str | None, number_of_cars: float | None,
              density_hh_km2: float | None, dist_center_km: float | None,
              lat: float | None, lon: float | None,
              person_id: Any) -> tuple[bool, str]:
        """Drawn value of the trait, and the **reason** — so that the report counts it.

        Reasons: ``sous_age_champ`` (value imposed by the threshold), ``tirage`` (Bernoulli
        of the propensity). A reason is returned rather than logged here: the module
        does not know whether it runs in a script, a notebook or a test.
        """
        p = self.propensity(age, gender, main_occupation, number_of_cars,
                            density_hh_km2, dist_center_km)
        if p is None:
            return self.spec.below_min_age, "sous_age_champ"
        u = uniform(self.spec.salt, draw_key(lat, lon, person_id))
        return (u < p), "tirage"


def write_resource(path: Path, spec: TraitSpec, law: dict, validation: dict,
                   source: dict) -> None:
    """Writes the resource: the law, its recipe, and its provenance.

    **No microdata** — coefficients and aggregated tables only, like
    `bike_ownership.json`. That is what allows committing the resource although its
    source is restricted-access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "equipment_propensity/v1",
        "trait": spec.key,
        "law": law,
        "validation": validation,
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "target": spec.target,
            "draw_salt": spec.salt,
            "min_age": spec.min_age,
            **source,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
