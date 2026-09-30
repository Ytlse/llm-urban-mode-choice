"""
core/population_reference.py — The framing of the surveyed population, read and enforceable.

The repository's whole measurement chain compares simulated modal shares with the targets of
`cerema_values.yaml`. This comparison only makes sense if both sides talk about
the **same population** and the **same counted object**. `population_emc2_2023.yaml`
describes precisely this population: scope, minimum age, adjustment weights,
split into rings, household structure.

**This module exists because nobody read this file** (ticket 020,
finding C1). It described a framing, it looked like data, and
most of it lay dormant in comments. It is the "vacuity" pattern the project hunts down:
a framing value without a reader is a false value waiting to happen.

**What this module guarantees, and what it does not.** It guarantees that the file
is readable, complete, and consistent with itself (the rings sum to 453, the
distributions to 100%, the household size is the inhabitants/households ratio).
It does NOT guarantee that the values are the survey's: that is checked by
recomputing them from the ProGEDO microdata, which
`scripts/data/population/audit_perimetre.py --recompute` does — and the microdata are
restricted-access, hence absent from an ordinary machine or container.

**Two notions never to be confused**, and it is the heart of axis A3 of ticket 020:

- a **PERSON** target (modal share, age, occupation) is compared with a count of
  persons or trips — `COEP` weighting on the survey side;
- a **HOUSEHOLD** target (size, car ownership) is compared with a count of households —
  `COE0` weighting on the survey side. A synthetic population is a sample of
  persons: large households appear in it in proportion to their size.
  Comparing its raw mean with a household target produces a PHANTOM gap. Measured on
  `toulouse_population_1000.json`: 2.71 persons per household raw against 2.01 when
  weighting each person by `1/size`, for a target of 2.08.
  :func:`household_weight` is this weighting, and it has a name so that it is not
  rediscovered a third time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from mobility_core.resources import find_repo_file

# The framing is not copied into the package: it lives in the repository
# (`scripts/data/population/population_emc2_2023.yaml`) and is looked up via
# `resources.find_repo_file` — environment variable, designated root, walk up from
# the current directory, then /app (`controller` container).
REFERENCE_RELATIVE = "scripts/data/population/population_emc2_2023.yaml"
REFERENCE_ENV = "POPULATION_EMC2_REFERENCE"

# Ring categories. They are EXACTLY the `lieu_residence` keys of
# `cerema_values.yaml`, up to whitespace normalisation (`normalize_place`): the
# classification of agents must be identical to them, otherwise modal shares by zone are
# compared with targets that do not designate the same territories.
# ENGLISH since ticket 074, and for a specific reason: the residence ring is
# SERVED TO THE MODEL in the persona narrative ("Lives in: 3rd ring"). As long as it was
# only a join key to the survey, its language committed no one; served to the model,
# it falls under the same rule as the rest of the prompt.
#
# The keys of `cerema_values.yaml` (`1ere_couronne`), for their part, stay FRENCH: they
# cross a repository boundary (`prompt_calibration/` reads them) and are never seen
# by the model. The translation happens at the boundary, in `scripts/synthesis/frames.py`.
COURONNES: tuple[str, ...] = ("Toulouse", "1st ring", "2nd ring", "3rd ring")

#: The pre-v6 categories, in the SAME ORDER. Two uses, and two only: reading back an archived
#: cohort or trace, and reading back a resource generated before the switch.
COURONNES_FR: tuple[str, ...] = ("Toulouse", "1ere couronne", "2eme couronne",
                                 "3eme couronne")

#: Old category → canonical category.
COURONNE_DEPUIS_FR: dict[str, str] = dict(zip(COURONNES_FR, COURONNES))

#: Canonical category → identifier in `cerema_values.yaml` (`lieu_residence`).
COURONNE_VERS_CEREMA: dict[str, str] = {
    "Toulouse": "Toulouse", "1st ring": "1ere_couronne",
    "2nd ring": "2eme_couronne", "3rd ring": "3eme_couronne",
}


def couronne_canonique(valeur: str | None) -> str | None:
    """Canonical ring category, whatever the language of the source.

    Reading a pre-switch cohort without translating would drop every agent into
    "unknown category": a whole mass counted outside the reference data, without a single
    line missing anywhere.
    """
    if valeur is None:
        return None
    texte = str(valeur).strip()
    return COURONNE_DEPUIS_FR.get(texte, texte or None)

# Out of scope is NOT a ring. It is a fifth category, and giving it
# a name is the point: before ticket 020, a home 100 km from the Capitole was
# classed "3eme couronne" by the metric classification and compared with the target of a
# territory it does not live in.
OUT_OF_PERIMETER = "outside perimeter"

#: The same category before the switch (ticket 074).
OUT_OF_PERIMETER_FR = "hors périmètre"

# Persona `main_occupation` → SURVEY category, as the fitted artefacts
# name it (`driving_license.json`, `pt_subscription.json`, `bike_ownership.json`, whose
# variables are literally called `occ_Travail à plein temps`).
#
# WHY THIS TABLE EXISTS. These three laws are FROZEN artefacts: their coefficients
# were fitted on the survey, and the names of their variables are part of the fit.
# When `main_occupation` switched to English (ticket 074), the design vector stopped
# recognising A SINGLE category: all indicators fell to zero, hence
# the whole cohort into the reference category. Nothing crashed — measured: the gap of
# `permis_adultes` to its target went from 2.52 to 5.23 points and that of `abonnement_tc`
# from 2.65 to 8.45, without a single log line.
#
# The VALUES are therefore frozen with the artefacts; only the keys follow the persona.
OCCUPATION_ENQUETE: dict[str, str] = {
    # v6 and later
    "Pupil (up to Baccalaureate)": "Scolaire (jusqu'au Bac)",
    "Student": "Étudiant",
    "Full-time worker": "Travail à plein temps",
    "Part-time worker": "Travail à temps partiel",
    "Unemployed / job seeker": "Chômeur/recherche d'emploi",
    "Homemaker": "Personne au foyer",
    "Retired": "Retraité",
    # v5 and earlier: identity, so that archived cohorts read back directly.
    "Scolaire (jusqu'au Bac)": "Scolaire (jusqu'au Bac)",
    "Étudiant": "Étudiant",
    "Travail à plein temps": "Travail à plein temps",
    "Travail à temps partiel": "Travail à temps partiel",
    "Chômeur/recherche d'emploi": "Chômeur/recherche d'emploi",
    "Personne au foyer": "Personne au foyer",
    "Retraité": "Retraité",
    "Autre": "Autre",
}


def occupation_enquete(valeur: object) -> str | None:
    """Survey occupation category, whatever the persona's language.

    `None` for a missing or out-of-vocabulary value — the caller then decides, and
    SAYS SO; it is precisely what the silent zero of the indicators did not do.
    """
    if valeur is None:
        return None
    return OCCUPATION_ENQUETE.get(str(valeur).strip())


# Activity purpose → label of the persona's `travel_purposes`, SERVED to the model.
#
# ONE SINGLE definition: it lived twice, in the eqasim generator and in
# `align_minor_traits`, each with a comment saying "the same mapping as the other". Translating
# them on one side only was enough to make them diverge — the generator set `['Work']`, the
# pre-imputation rewrote it as `['Travail']`. `mobility_core` is mounted in the eqasim
# image: both sides read the same table instead of promising to stay in agreement.
PURPOSE_LABEL: dict[str, str] = {
    "work": "Work",
    "education": "Education",
    "shop": "Shopping",
}

#: Pre-v6 labels, to read back archived populations.
PURPOSE_LABEL_FR: dict[str, str] = {
    "work": "Travail",
    "education": "Etude",
    "shop": "Achats",
}

# Minimum age of the survey's target population. The modal share classes
# start at `5-9`: a 3-year-old agent falls into this class without anything
# reporting it (`frames.age_to_cat` tests `a <= 9`). Hence the explicit check.
MIN_AGE = 5


class PopulationReferenceError(ValueError):
    """Framing missing, unreadable, or inconsistent. Never a silent fallback.

    There is no reasonable fallback value for a framing: serving
    rings that do not sum to 453, or a household size that does not add up,
    would let the whole chain compare different populations silently.
    """


def find_reference() -> Path | None:
    """First `population_emc2_2023.yaml` found at the standard locations (cf. `resources`)."""
    return find_repo_file(REFERENCE_RELATIVE, env_var=REFERENCE_ENV)


def _require(node: dict, path: str):
    """Walks down a dotted path, and raises naming what is missing."""
    cursor = node
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise PopulationReferenceError(
                f"incomplete framing: key '{path}' missing")
        cursor = cursor[part]
    return cursor


def _check_sums_to(node: dict, path: str, expected: float, tolerance: float) -> None:
    total = sum(float(v) for v in _require(node, path).values())
    if abs(total - expected) > tolerance:
        raise PopulationReferenceError(
            f"inconsistent framing: '{path}' sums to {total:g}, "
            f"expected {expected:g} (± {tolerance:g})")


def validate(reference: dict) -> dict:
    """Internal checks of the framing. Raises `PopulationReferenceError` at the first failure.

    The tolerances are not for comfort: the values published by CEREMA are
    rounded to the percentage point and to the thousand, and requiring exact equality
    would make a correct file fail.
    """
    _check_sums_to(reference, "population.repartition_par_classe_age", 100, 1.0)
    _check_sums_to(reference, "population.repartition_par_occupation_principale",
                   100, 1.0)
    _check_sums_to(reference,
                   "menages_equipement_voiture.perimetre_2023.repartition_motorisation",
                   100, 1.0)
    _check_sums_to(reference, "enquete.localisation_deplacements", 100, 0.5)

    # The rings cover the scope, no more, no less.
    decoupage = _require(reference, "territoire.decoupage_concentrique")
    # The reference data is a SURVEY file: its categories are those of
    # `cerema_values.yaml`, kept in French (they cross the boundary of
    # `prompt_calibration/`). They are mapped to the canonical vocabulary before comparing, rather
    # than requiring the reference data to speak the persona's language.
    noms = [couronne_canonique(str(z["nom"])) for z in decoupage]
    if tuple(noms) != COURONNES:
        raise PopulationReferenceError(
            f"inconsistent framing: rings {noms} instead of {list(COURONNES)} — "
            "these are the `lieu_residence` categories of cerema_values.yaml")
    communes = sum(int(z["communes"]) for z in decoupage)
    attendu = int(_require(reference, "territoire.perimetre_2023.communes"))
    if communes != attendu:
        raise PopulationReferenceError(
            f"inconsistent framing: the rings total {communes} communes, "
            f"the scope declares {attendu}")

    # The mean household size must follow from the inhabitants / households ratio, otherwise
    # one of the three values is a figure copied independently of the other two.
    totaux = _require(reference, "population.totaux_perimetre_2023")
    ratio = float(totaux["habitants"]) / float(totaux["nombre_menages"])
    declaree = float(totaux["taille_moyenne_menage"])
    if abs(ratio - declaree) > 0.05:
        raise PopulationReferenceError(
            f"inconsistent framing: declared household size {declaree:.2f}, "
            f"inhabitants/households = {ratio:.2f}")

    # The target population (aged 5 and over) is a subset of the inhabitants, and the
    # breakdown by ring adds up to it.
    cible = float(totaux["habitants_5_ans_et_plus"])
    if not 0.8 * float(totaux["habitants"]) <= cible <= float(totaux["habitants"]):
        raise PopulationReferenceError(
            f"inconsistent framing: population aged 5 and over ({cible:g}) outside "
            f"the plausible range under {totaux['habitants']:g} inhabitants")
    par_zone = sum(float(z["habitants_5_ans_et_plus"])
                   for z in _require(reference,
                                     "population.repartition_par_territoire").values())
    if abs(par_zone - cible) / cible > 0.02:
        raise PopulationReferenceError(
            f"inconsistent framing: the breakdown by ring totals {par_zone:g} "
            f"inhabitants aged 5 and over, the total declares {cible:g}")

    if int(_require(reference, "enquete.methodologie.age_minimum_enquete")) != MIN_AGE:
        raise PopulationReferenceError(
            "inconsistent framing: `age_minimum_enquete` diverges from "
            f"`MIN_AGE` = {MIN_AGE}, which is the bound tested by the age check")

    return reference


@lru_cache(maxsize=1)
def population_reference() -> dict:
    """Validated framing of the surveyed population. Cached, raises if it is missing.

    `population_reference.cache_clear()` lets tests replay the resolution.
    """
    path = find_reference()
    if path is None:
        raise PopulationReferenceError(
            "framing not found: scripts/data/population/population_emc2_2023.yaml. "
            "Unlike the feature spec, this file is VERSIONED in the "
            "repository — its absence is an anomaly, not a normal case.")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PopulationReferenceError(f"unreadable framing ({path}): {exc}") from exc
    if not isinstance(loaded, dict):
        raise PopulationReferenceError(f"empty or malformed framing ({path})")
    validate(loaded)
    logger.info(
        f"Population framing read from {path} | "
        f"{_require(loaded, 'territoire.perimetre_2023.communes')} communes, "
        f"{_require(loaded, 'population.totaux_perimetre_2023.habitants_5_ans_et_plus'):,} "
        "inhabitants aged 5 and over")
    return loaded


# ── Named accessors ───────────────────────────────────────────────────────────
# A named accessor rather than a dotted path copied into each caller: it is what
# allows renaming a YAML key without silently breaking three consumers.

def survey_window() -> tuple[str, str]:
    """Collection window `(start, end)` in ISO — the season the targets carry."""
    node = _require(population_reference(), "enquete.periode_enquete")
    return str(node["debut"]), str(node["fin"])


def surveyed_weekdays() -> tuple[int, ...]:
    """Days of the surveyed previous day, 1 = Monday. The survey counts no weekend."""
    return tuple(int(d) for d in
                 _require(population_reference(),
                          "enquete.methodologie.jours_enquetes"))


def couronne_commune_counts() -> dict[str, int]:
    """`ring → number of communes`, from the survey's split.

    The keys are CANONICAL (English): the reference data speaks the survey's vocabulary,
    its callers the persona's. Translating here rather than in each of them prevents one
    half of the repository from indexing by "1ere couronne" and the other by "1st ring" — two
    dictionaries that never join, with nothing saying that they do not.
    """
    return {couronne_canonique(str(z["nom"])): int(z["communes"]) for z in
            _require(population_reference(), "territoire.decoupage_concentrique")}


def couronne_population_shares() -> dict[str, float]:
    """`ring → share of the population aged 5 and over (in %)`.

    It is the spatial representativeness target (axis A9): an excess of Toulouse
    pulls the car share down by more than 30 points without any choice
    model being at fault, since the `voiture` target is 31% in Toulouse and 64%
    in the 1st ring.
    """
    node = _require(population_reference(), "population.repartition_par_territoire")
    order = dict(zip(("toulouse", "premiere_couronne", "deuxieme_couronne",
                      "troisieme_couronne"), COURONNES))
    total = sum(float(v["habitants_5_ans_et_plus"]) for v in node.values())
    return {order[k]: 100.0 * float(v["habitants_5_ans_et_plus"]) / total
            for k, v in node.items() if k in order}


def household_targets() -> dict[str, float]:
    """HOUSEHOLD targets: mean size, car ownership. To be weighted by `1/size`.

    See :func:`household_weight` — comparing these values with a raw mean of a
    synthetic population produces a phantom gap of around 30%.
    """
    reference = population_reference()
    totaux = _require(reference, "population.totaux_perimetre_2023")
    equip = _require(reference,
                     "menages_equipement_voiture.perimetre_2023")
    motor = equip["repartition_motorisation"]
    return {
        "taille_moyenne_menage": float(totaux["taille_moyenne_menage"]),
        "voitures_par_menage": float(equip["voitures_par_menage_moyen"]),
        "sans_voiture_pct": float(motor["sans_voiture"]),
        "une_voiture_pct": float(motor["une_voiture"]),
        "deux_voitures_plus_pct": float(motor["deux_voitures_et_plus"]),
    }


def household_weight(household_size: float | None) -> float:
    """Weight of a PERSON in a count of HOUSEHOLDS: `1 / size`.

    A synthetic population samples persons. A household of 5 appears in it
    5 times, a household of 1 only once: the raw mean of a household attribute
    is therefore biased by size. Weighting each person by the inverse of
    their household size gives each household a weight of 1.

    Missing or non-positive size → `0.0`, that is *this person does not count
    in a household statistic*. It is not a degraded fallback: without a household
    size, the person has no household base, and inventing one (1, for
    example) would fabricate a one-person household that does not exist.
    """
    try:
        size = float(household_size)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / size if size > 0 else 0.0
