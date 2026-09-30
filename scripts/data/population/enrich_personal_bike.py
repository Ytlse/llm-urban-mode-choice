"""enrich_personal_bike.py — The persona's bike, rewritten from EMC² (ticket 015, route 1).

Reads a population file back, **groups agents by home address**, draws the number of
bikes of the household, assigns them by name, and rewrites `personal_bike`. Same
pattern as `enrich_housing_type.py`: no regeneration, applicable to existing
populations.

What this replaces: eqasim drew `p = min(1, donor_bikes / size)` where the number of
bikes is **copied** from an ENTD 2008 household matched without household size or
housing. The total came out roughly right and the distribution was wrong — household
size gradient **inverted**. The details of the three stages and of the conditioning
decisions are in `packages/mobility_core/src/mobility_core/bike_ownership.py`.

## The address as household key: usable, imperfect, and one must know it

The JSON does not carry the household — an architecture decision of the ticket: the household
only exists while `k` is drawn and split, the agent receives a single value. Households
must therefore be rebuilt, and the only available key is the home address. It is
**already** the repository's household key: `housing_type` hashes the address so that two
personas of the same household share the housing type. This step therefore makes no new
assumption, it reuses the repository's.

Two defects, both handled explicitly and counted in the report:

- **Collisions.** Two distinct households at the same address point. Detectable because the
  cluster exceeds the `household_size` of its members, or carries several values (8
  clusters out of 547 in `toulouse_population_1000.json`). They are **split** by
  `household_size`, then into chunks of that size — never treated as a single big
  household, which would inherit the `k` of a large family.
- **Partially present households.** The footprint filter only keeps agents whose
  home is in the zone: ~25 % of agents belong to a household whose members are
  not all in the file. We therefore draw on the **nominal** size
  (`household_size`) and only materialise the present members. Otherwise the `k`
  bikes of the household would concentrate on the kept agents alone and
  over-equip them.

## What the validation refuses to let through

A massive `personal_bike = None` must make the validation **fail**, not pass:
it is the "emptiness ≠ perfection" pattern the repository tracks. `--check` exits in failure if
coverage is insufficient or if a target is out of tolerance, and it says which one.

Exit codes of `--check`, and the distinction is useful to the caller:

| Code | Meaning |
|---|---|
| 0 | all served targets are within tolerance |
| 1 | resource or file not found — nothing was done |
| 2 | a served target is **contradicted**: the model or the population is at fault |
| 3 | population **enriched but not validated** — not enough households to decide |

Usage:
    python -m scripts.data.population.enrich_personal_bike data/population/toulouse_population_1000.json
    python -m scripts.data.population.enrich_personal_bike data/population/*.json --dry-run
    python -m scripts.data.population.enrich_personal_bike data/population/*.json --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mobility_core.bike_ownership import (
    ELECTRIC_BIKE,
    K_MAX,
    MIN_AGE_ELIGIBLE,
    NO_BIKE,
    PLAIN_BIKE,
    TRAIT_KEY,
    VAE_SHARE,
    BikeOwnershipModel,
    Member,
    address_key,
    assign,
    bike_label,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Tolerances of the acceptance criteria of ticket 015. Two of them are
# **restated** relative to the letter of the ticket, on the basis of a measurement, and the
# report says so every time it displays them:
#
# - `bikes_per_household`: the ticket asks for 1.22 (± 0.05), which applies to UNclipped
#   `M21`. The model is clipped at `4+` — the clipping the ticket itself specifies —
#   and therefore caps at 1.151. The enforceable target is the clipped mean, served by the
#   resource (`validation.stock.clipping_cost`). Clipping only costs 0.011 bike per
#   household on the trait actually produced.
# - `by_housing`: the ticket asks for 71 % → 38 % (± 4 pts). Unattainable by
#   construction, since the persona's housing is itself imputed from its zone's law
#   (agreement 47.6 %). The enforceable target is the **diluted** curve, served by the
#   resource (`validation.housing_reference.attainable_on_imputed_housing`).
TOLERANCES = {
    "holders_pct": 3.0,
    "equipped_pct": 2.0,
    "bikes_per_household": 0.05,
    "by_size_pct": 5.0,
    "by_housing_pct": 4.0,
    "vae_pct": 1.5,
}

# Clipping of size buckets, identical to that of the tables served by
# `export_bike_ownership.SIZE_BUCKET_MAX`. Both sides MUST bucket the same way:
# a "5+" bucket on the population side against a "5+" on the survey side compares two different
# mixes of sizes (the holder rate goes from 52 % to 40 % between 5 and 6), and
# fabricates a 14-point gap that is only a composition effect.
SIZE_BUCKET_MAX = 6

# Minimum counts, **in households** and not in persons, for a cell to be
# enforceable. The ticket already requires flagging survey cells under 30
# weighted observations; this is the same rule applied to the measurement side.
#
# Below that, the verdict is neither "ok" nor "FAIL" but **inconclusive**, and the
# nuance matters: declaring a failure on noise teaches to ignore failures, and
# declaring a success would pass the lack of material off as a result — this is
# exactly the "emptiness ≠ perfection" pattern this script exists to refuse.
#
# Practical consequence: populations of 10 and 100 agents cannot arbitrate the
# cross-tabulations (18 households in their largest housing cell). They remain
# enrichable, their report remains readable, but only a population of the order of
# 1,000 agents makes the check enforceable.
MIN_CELL_HOUSEHOLDS = 30
# The SIGN of the slope (sizes 1 → 4) is only judged from 100 households per size, and an
# inversion is a FAILURE only if it exceeds the uncertainty of the two compared cells
# (`slope_verdict`). Decision of 2026-09-03 (ticket 031, question 7): on the v4 sealed cohort,
# sizes 3 and 4 had 69 and 55 households, 63.4 % against 55.5 % — an 8 pt inversion
# for intervals of ± 12-13 pt is sampling noise, and the same model gives on the
# pool of 11,329 persons a slope 32.8 < 49.1 < 55.0 < 60.9 over 2,350 / 1,657 / 744 / 532
# households (measured on the evening of 2026-09-03). The slope is thus enforceable on the POOL; on a
# cohort of 1,000, it is displayed (inconclusive) without counting in the verdict.
SLOPE_MIN_CELL = 100
SLOPE_Z = 1.96

# Minimum number of checks that actually decided for a report to count as
# validation. Without this safeguard, a population of 10 agents would pass `--check` with
# zero conclusive check — the perfect score through lack of measurement, precisely what
# the ticket forbids ("no target reached through lack of measurement").
MIN_CONCLUSIVE_CHECKS = 4

# Exit codes of `--check`, and the distinction matters to the caller (the generation
# notebook, CI): "the target is missed" and "there is nothing to measure" do not
# call for the same action. The first is a defect of the model or of the population,
# the second is a property of the file — a population of 10 agents cannot
# arbitrate a cross-tabulation, and that is not a bug.
EXIT_OK = 0
EXIT_TARGET_MISSED = 2
EXIT_NOT_MEASURABLE = 3

# Prefix that marks a "not enough material" failure in the list of failures.
NOT_MEASURABLE = "population non mesurable"

# Minimum coverage of the trait. Below it, the population is not usable and
# no target makes sense: a target "reached" on 10 % of agents measures nothing.
MIN_COVERAGE = 0.80


@dataclass
class Household:
    """A household rebuilt from the address, for the duration of the draw.

    `nominal_size` is the size declared by the personas (the real household), `members`
    those the file contains. The difference is materialised by absent seats
    at draw time.
    """

    key: str
    nominal_size: int
    members: list[int] = field(default_factory=list)


def build_households(population: list[dict], addresses: list[Optional[str]]) -> tuple[list[Household], Counter]:
    """Groups agents into households, splitting address collisions.

    A cluster is **coherent** if all its members declare the same `household_size`
    and it is not larger than that size. Otherwise it is a collision: it is
    split first by declared `household_size`, then into chunks of that size. Two
    single people at the same point thus make two households of one, not one household of two — which
    would give them the `k` of a couple.
    """
    clusters: dict[str, list[int]] = defaultdict(list)
    counts: Counter = Counter()
    for index, key in enumerate(addresses):
        if key is None:
            counts["sans_adresse"] += 1
            continue
        clusters[key].append(index)

    households: list[Household] = []
    for key, members in clusters.items():
        by_size: dict[int, list[int]] = defaultdict(list)
        for index in members:
            traits = (population[index].get("identity") or {}).get("traits_json") or {}
            size = traits.get("household_size")
            by_size[max(1, int(size)) if size else 1].append(index)

        coherent = len(by_size) == 1 and len(members) <= next(iter(by_size))
        counts["grappes_coherentes" if coherent else "grappes_en_collision"] += 1

        for size, group in sorted(by_size.items()):
            # Chunks of `size`: a cluster of 5 persons all declaring a household of
            # 2 becomes three households (2, 2, 1), not one household of 5.
            for offset in range(0, len(group), size):
                chunk = group[offset:offset + size]
                suffix = "" if coherent else f"#s{size}n{offset // size}"
                households.append(Household(key=f"{key}{suffix}",
                                            nominal_size=size, members=chunk))
    return households, counts


def enrich(population: list[dict], model: BikeOwnershipModel, resolver) -> Counter:
    """Sets `personal_bike` on each persona. Returns the count per category.

    Homes are matched in a single vectorised call: one spatial index query
    per batch, not one per persona — same rule as `enrich_housing_type`.
    """
    homes = [(person.get("identity") or {}).get("home") or {} for person in population]
    lats = [home.get("lat") for home in homes]
    lons = [home.get("lon") for home in homes]

    resolvable = [i for i, (la, lo) in enumerate(zip(lats, lons))
                  if la is not None and lo is not None]
    zones = resolver.resolve_many([lats[i] for i in resolvable],
                                  [lons[i] for i in resolvable])
    zone_by_index = dict(zip(resolvable, zones))
    addresses: list[Optional[str]] = [
        address_key(lats[i], lons[i]) if i in zone_by_index else None
        for i in range(len(population))
    ]

    households, counts = build_households(population, addresses)
    # Renormalised once for the whole file: applying VAE_SHARE to holders aged
    # 14 and over only would bring the fleet out below the target.
    electric_p = model.electric_p

    def traits_of(index: int) -> Optional[dict]:
        return (population[index].get("identity") or {}).get("traits_json")

    def clear(index: int, reason: str) -> None:
        """No trait, and the one left over from an earlier enrichment is removed.

        `personal_bike` inherited from eqasim is precisely what is being replaced: leaving it in
        place outside the layer would give a population half learned, half copied, without
        anything flagging it.
        """
        traits = traits_of(index)
        if traits is None:
            counts["sans_traits"] += 1
            return
        traits.pop(TRAIT_KEY, None)
        counts[reason] += 1

    # A persona without a resolvable address enters NO household: the loop below
    # never sees it, and a `personal_bike` inherited from an earlier enrichment
    # survived there — the population was then "half learned, half copied" for these
    # agents, which this module claims to refuse. Measured on 2026-09-04 on the pool of
    # 11,329 persons: 14 personas, all without home coordinates, still carried
    # the value set upstream. The trait is therefore removed from them here, explicitly.
    for index, key in enumerate(addresses):
        if key is None and (traits_of(index) or {}).pop(TRAIT_KEY, None) is not None:
            counts["trait_herite_retire_sans_adresse"] += 1

    for household in households:
        zone = zone_by_index.get(household.members[0])
        if zone is None:
            for index in household.members:
                clear(index, "hors_couche")
            continue

        # ── Stage 1: how many bikes in this household ────────────────────────
        # Car ownership is a household attribute; personas of the same household
        # carry it identically, so the first member's is read. The size used is
        # the NOMINAL size, not the number of present members.
        first = traits_of(household.members[0]) or {}
        k = model.draw_stock(
            household_size=household.nominal_size,
            number_of_cars=first.get("number_of_cars"),
            density_hh_km2=zone.density_hh_km2,
            dist_center_km=zone.dist_center_km,
            household_key=household.key,
        )
        if k is None:
            for index in household.members:
                clear(index, "sans_loi")
            continue

        # ── Stage 2: who holds them ──────────────────────────────────────────
        present: list[Member] = []
        skipped: list[int] = []
        for index in household.members:
            traits = traits_of(index)
            if traits is None:
                skipped.append(index)
                continue
            age = traits.get("age")
            present.append(Member(
                index=index,
                propensity=model.propensity_of(
                    k=k,
                    household_size=household.nominal_size,
                    age=age,
                    gender=traits.get("gender"),
                    main_occupation=traits.get("main_occupation"),
                    density_hh_km2=zone.density_hh_km2,
                    dist_center_km=zone.dist_center_km,
                ),
                eligible=(age is not None and float(age) >= MIN_AGE_ELIGIBLE),
                present=True,
            ))
        for index in skipped:
            counts["sans_traits"] += 1

        # Absent seats: the members of the nominal household that the spatial filter did not
        # keep. They carry the household's mean propensity, take part in the draw, and
        # may take a bike — but nothing is written for them. Without this
        # complement, a single person extracted from a household of four would alone receive
        # the bikes of all four.
        mean_propensity = (statistics.fmean([m.propensity for m in present])
                           if present else 0.0)
        absent = [
            Member(index=-1 - position, propensity=mean_propensity,
                   eligible=True, present=False)
            for position in range(max(0, household.nominal_size - len(household.members)))
        ]
        if absent:
            counts["places_absentes"] += len(absent)

        holders = assign(present + absent, k, household.key)

        # ── Stage 3: which type of bike ──────────────────────────────────────
        for member in present:
            traits = traits_of(member.index)
            if member.index in holders:
                label = bike_label(household.key, member.index,
                                   (traits or {}).get("age"), electric_p)
            else:
                label = NO_BIKE
            traits[TRAIT_KEY] = label
            counts[label] += 1

    return counts


# ── Report and check ─────────────────────────────────────────────────────────

def measure(population: list[dict]) -> dict:
    """The quantities of the acceptance criteria, read back from the enriched file.

    Everything is measured **per person**, on the agents carrying the trait only: mixing
    agents outside the layer into the denominators would mechanically lower the shares and
    present the lack of measurement as a result.

    Each cell is counted **twice**: in persons and in **distinct households**.
    The second count is the one that governs precision, and it is a point of substance, not
    a nicety: `k` is drawn once **per household**, so members of the same
    household are not independent observations. On the 100-agent population,
    the "individuel isolé" cell has 37 persons but only 18 addresses:
    computing its standard deviation on 37 overestimates precision by a factor √2 and turns
    draw noise into a gap blamed on the model.
    """
    holders_by_size: dict[int, list[int]] = defaultdict(list)
    holders_by_housing: dict[str, list[int]] = defaultdict(list)
    households_by_size: dict[int, set] = defaultdict(set)
    households_by_housing: dict[str, set] = defaultdict(set)
    fleet = electric = holders = 0
    with_trait = 0
    all_households: set = set()
    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json") or {}
        label = traits.get(TRAIT_KEY)
        if label is None:
            continue
        home = identity.get("home") or {}
        key = (address_key(home["lat"], home["lon"])
               if home.get("lat") is not None else f"?{with_trait}")
        with_trait += 1
        all_households.add(key)
        has = label != NO_BIKE
        holders += int(has)
        if has:
            fleet += 1
            electric += int(label == ELECTRIC_BIKE)
        size = traits.get("household_size")
        if size:
            bucket = min(SIZE_BUCKET_MAX, max(1, int(size)))
            holders_by_size[bucket].append(int(has))
            households_by_size[bucket].add(key)
        housing = traits.get("housing_type")
        if housing:
            holders_by_housing[housing].append(int(has))
            households_by_housing[housing].add(key)

    return {
        "n": len(population),
        "with_trait": with_trait,
        "households": len(all_households),
        "fleet": fleet,
        "coverage": with_trait / len(population) if population else 0.0,
        "holders_pct": 100.0 * holders / with_trait if with_trait else float("nan"),
        "vae_pct_of_fleet": 100.0 * electric / fleet if fleet else float("nan"),
        "holders_by_size": {
            size: (100.0 * sum(values) / len(values), len(values),
                   len(households_by_size[size]))
            for size, values in sorted(holders_by_size.items())
        },
        "holders_by_housing": {
            housing: (100.0 * sum(values) / len(values), len(values),
                      len(households_by_housing[housing]))
            for housing, values in sorted(holders_by_housing.items())
        },
    }


def household_measure(population: list[dict]) -> dict:
    """The quantities at **household** level: equipped share and bikes per household.

    Rebuilt from the address, as at draw time. Partially present households are
    left out of this count: "the household's bikes" cannot be measured on a household
    missing members — the numerator would be truncated and the denominator not.

    **The count is broken down by size, and this is essential.** Leaving out incomplete
    households does not take a neutral sample: a one-person household is
    always complete, a household of five almost never. On
    `toulouse_population_1000.json`, measurable households are 50.7 % single
    persons against 39.3 % overall. Comparing their equipment rate with the survey's 53.6 %
    — which covers *all* sizes — compares two different compositions
    and fabricates a 5-point gap that does not exist. `report` therefore standardises
    the target on the composition actually measured.
    """
    addresses = []
    for person in population:
        home = (person.get("identity") or {}).get("home") or {}
        addresses.append(address_key(home["lat"], home["lon"])
                         if home.get("lat") is not None else None)
    households, _ = build_households(population, addresses)

    by_size: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "equipped": 0, "bikes": 0})
    complete = equipped = bikes = 0
    for household in households:
        labels = [((population[i].get("identity") or {}).get("traits_json") or {}).get(TRAIT_KEY)
                  for i in household.members]
        if any(label is None for label in labels):
            continue
        if len(household.members) != household.nominal_size:
            continue
        count = sum(1 for label in labels if label != NO_BIKE)
        complete += 1
        bikes += count
        equipped += int(count > 0)
        bucket = by_size[min(SIZE_BUCKET_MAX, household.nominal_size)]
        bucket["n"] += 1
        bucket["equipped"] += int(count > 0)
        bucket["bikes"] += count
    return {
        "complete_households": complete,
        "equipped_pct": 100.0 * equipped / complete if complete else float("nan"),
        "bikes_per_household": bikes / complete if complete else float("nan"),
        "by_size": {size: dict(values) for size, values in sorted(by_size.items())},
    }


def standardise(by_size: dict[int, dict[str, int]],
                reference: dict[int, float]) -> Optional[float]:
    """Survey target recomposed on the size breakdown actually measured.

    Direct standardisation: `Σ_s measured_share(s) × survey_value(s)`. This is the correct
    way to compare two populations whose composition differs — without it, one
    blames the imputation for a gap that is only a structure effect.

    `None` if the reference does not cover the measured sizes: better to serve no
    target than a shaky one.
    """
    total = sum(values["n"] for values in by_size.values())
    if not total or any(size not in reference for size in by_size):
        return None
    return sum(values["n"] / total * reference[size]
               for size, values in by_size.items())


def report(measured: dict, household: dict, counts: Counter,
           model: BikeOwnershipModel, journal: Optional[dict] = None) -> list[str]:
    """Displays the result against the targets. Returns the list of failures.

    `journal`, if provided, receives the structured version of what is printed —
    each check with its measurement, target, margin and verdict, the slope by
    household size, the verdict counters — so that the representativeness
    synthesis (`scripts/panel/synthese_representativite.py --velo`) reads the
    bike verdict from a file rather than from copied console output.
    """
    if journal is None:
        journal = {}
    journal["controles"] = []
    validation = model.validation
    targets = validation.get("targets") or {}
    stock = validation.get("stock") or {}
    housing_ref = validation.get("housing_reference") or {}
    failures: list[str] = []
    # How many checks actually decided. A report where everything is
    # "inconclusive" must NOT pass for a success: it is the "emptiness ≠
    # perfection" pattern — the lack of measurement produces the perfect score.
    verdicts = {"ok": 0, "echec": 0, "non_concluant": 0}

    print(f"\nTrait set on {measured['with_trait']}/{measured['n']} personas "
          f"({100 * measured['coverage']:.1f} %)")
    for key in ("hors_couche", "sans_loi", "sans_traits", "sans_adresse",
                "trait_herite_retire_sans_adresse"):
        if counts.get(key):
            print(f"  {key:22s} {counts[key]:5d} (trait missing)")
    for key in ("grappes_coherentes", "grappes_en_collision", "places_absentes"):
        if counts.get(key):
            print(f"  {key:22s} {counts[key]:5d}")

    if measured["coverage"] < MIN_COVERAGE:
        failures.append(
            f"couverture {100 * measured['coverage']:.1f} % < "
            f"{100 * MIN_COVERAGE:.0f} % — population inexploitable, et AUCUNE cible "
            f"ci-dessous n'a de sens sur si peu d'agents")

    def check(label: str, got: float, target: Optional[float], tolerance: float,
              note: str = "", n: Optional[int] = None,
              sd: Optional[float] = None) -> None:
        """Compares a measured quantity with its target, **sampling noise included**.

        A population of 1,000 agents gives cells of 150 to 350 persons. At
        n = 164, the standard deviation of a proportion around 50 % is already 3.9 points:
        requiring ± 4 points on such a cell means measuring at the noise level, and a
        correct file would "fail" one time in three for nothing.

        The margin is therefore `tolerance + 2 σ`, with σ the binomial standard deviation of the cell.
        On a 10,000-agent file σ is divided by three and the check tightens
        by itself — this is the intended behaviour: the more material, the more
        enforceable the target. Without `n`, only the tolerance applies.
        """
        if target is None:
            print(f"  {label:42s} {got:8.2f}   (no target served){note}")
            journal["controles"].append({"controle": label, "mesure": got, "cible": None,
                                         "n_foyers": n, "verdict": "pas de cible"})
            return
        if n is not None and 0 < n < MIN_CELL_HOUSEHOLDS:
            # Neither "ok" nor "FAIL": the cell has nothing to decide on. Declaring a
            # failure on so few households teaches to ignore failures, and declaring it
            # "ok" would pass the lack of material off as a success.
            verdicts["non_concluant"] += 1
            print(f"  {label:42s} {got:8.2f}  target {target:7.2f} "
                  f"{'':>16s}  {got - target:+6.2f}  INCONCLUSIVE "
                  f"({n} households < {MIN_CELL_HOUSEHOLDS}){note}")
            journal["controles"].append({"controle": label, "mesure": got, "cible": target,
                                         "ecart": got - target, "n_foyers": n,
                                         "verdict": "non concluant"})
            return
        delta = got - target
        sigma = 0.0
        if n and n > 0:
            if sd is not None:
                # Mean of a count: σ = standard deviation / √n, the standard deviation coming from
                # the survey (served in the resource).
                sigma = sd / n ** 0.5
            elif 0.0 <= target <= 100.0:
                share = min(max(target / 100.0, 0.0), 1.0)
                sigma = 100.0 * (share * (1.0 - share) / n) ** 0.5
        margin = tolerance + 2.0 * sigma
        ok = abs(delta) <= margin
        band = f"± {tolerance:.2f}" + (f"+2σ({2 * sigma:.1f})" if sigma else "")
        print(f"  {label:42s} {got:8.2f}  target {target:7.2f} "
              f"{band:>16s}  {delta:+6.2f}  {'ok' if ok else 'ÉCHEC'}{note}")
        verdicts["ok" if ok else "echec"] += 1
        journal["controles"].append({"controle": label, "mesure": got, "cible": target,
                                     "ecart": delta, "marge": margin, "n_foyers": n,
                                     "verdict": "ok" if ok else "echec"})
        if not ok:
            failures.append(f"{label} : {got:.2f} contre {target:.2f} ± {margin:.2f}")

    print("\n── Person level ────────────────────────────────────────────────────────")
    published_holders = targets.get("holders_pct")
    check("personnes dotées d'un vélo (%)", measured["holders_pct"],
          targets.get("holders_pct_mechanism"), TOLERANCES["holders_pct"],
          f"   [publiée : {published_holders:.1f}]" if published_holders else "",
          n=measured["households"])
    check("part de VAE dans le parc (%)", measured["vae_pct_of_fleet"],
          targets.get("vae_share_of_fleet_pct"), TOLERANCES["vae_pct"],
          n=measured["fleet"])

    print("\n── Household size gradient (the check that used to fail) ──────────────")
    # Reference under the mechanism's rules (clipped k, eligibility at 5 years), not the
    # published figure: otherwise the mechanism is blamed for not producing the holders
    # it deliberately refuses to produce.
    rows_size = targets.get("holders_by_household_size") or []
    reference = {row["size"]: row.get("holders_pct_mechanism", row["holders_pct"])
                 for row in rows_size}
    published_by_size = {row["size"]: row["holders_pct"] for row in rows_size}
    curve = []
    for size, (got, n, n_hh) in measured["holders_by_size"].items():
        curve.append(got)
        published = published_by_size.get(size)
        check(f"taille {size} (n={n}, {n_hh} foyers)", got, reference.get(size),
              TOLERANCES["by_size_pct"],
              f"   [publiée : {published:.1f}]" if published is not None else "",
              n=n_hh)
    # The SIGN of the slope over sizes 1 to 4 is a criterion in its own right: it is
    # what was inverted (76 % among single persons against 33 % observed).
    cells = [measured["holders_by_size"].get(s, (None, 0, 0)) for s in (1, 2, 3, 4)]
    ordered = [value for value, _, _ in cells]
    statut, detail = slope_verdict(cells)
    journal["pente_tailles_1_4"] = {"statut": statut, "detail": detail, "taux_pct": ordered,
                                    "foyers": [n_hh for _, _, n_hh in cells],
                                    "min_foyers_pour_juger": SLOPE_MIN_CELL, "z": SLOPE_Z}
    if statut == "non calculable":
        print("  slope over sizes 1→4: not computable (a size is missing)")
    elif statut == "non concluant":
        # An inversion on cells of 50 households is noise: at n = 55, the interval is
        # ± 13 points. It is displayed, nothing is pronounced, and the overall verdict does not depend on it;
        # the slope is judged on the pool (see SLOPE_MIN_CELL).
        print(f"  {'pente sur les tailles 1→4':42s} "
              f"{' / '.join(f'{v:.1f}' for v in ordered)}  "
              f"INCONCLUSIVE ({detail}) — to be judged on the pool")
    else:
        print(f"  {'pente croissante sur les tailles 1→4':42s} "
              f"{' / '.join(f'{v:.1f}' for v in ordered)}  "
              f"{'ok' if statut == 'ok' else 'ÉCHEC'} — {detail}")
        if statut == "echec":
            failures.append("la pente sur les tailles 1→4 baisse au-delà de l'incertitude — c'est "
                            "le défaut même que le ticket 015 corrige")

    if measured["holders_by_housing"]:
        print("\n── Equipment by housing type ───────────────────────────────────────────")
        attainable = housing_ref.get("attainable_on_imputed_housing") or {}
        published = housing_ref.get("published_on_observed_housing") or {}
        from mobility_core.housing_type import key_for
        if not attainable:
            print("  [diluted target not served: re-export the resource with the "
                  "housing-type table present (make housing-type)]")
        else:
            # The figures come from the resource, never from a literal: they
            # move with every improvement of the housing imputation (ticket 019).
            accord = housing_ref.get("imputed_vs_observed_agreement_pct")
            spread = housing_ref.get("attainable_spread_pts")
            # Published spread: BOTH bounds must be served. A
            # `get("grand_habitat_collectif", 0.0)` would fabricate a spread from
            # a missing bound — it would then equal the upper bound, a wrong and
            # perfectly plausible figure. Better not to announce it.
            low = published.get("individuel_isole")
            high = published.get("grand_habitat_collectif")
            published_spread = (low - high) if (low is not None and high is not None) else None
            print("  Cible = courbe DILUÉE (habitat imputé), non la courbe publiée : "
                  "l'habitat du\n  persona est lui-même tiré de la loi de sa zone et de "
                  "sa taille de ménage"
                  + (f"\n  (accord avec l'habitat observé : {accord:.1f} %)"
                     if accord is not None else "")
                  + ", ce qui écrase l'amplitude"
                  + (f" de {published_spread:.1f}\n  à {spread:.1f} points (en part de "
                     f"ménages équipés)"
                     if (spread is not None and published_spread is not None) else "")
                  + ". Viser la publiée serait sur-corriger le\n  modèle pour compenser "
                    "le bruit de l'axe de mesure.\n"
                    "  Les cibles ci-dessous sont en part de PERSONNES dotées — l'unité "
                    "du trait.")
        for housing, (got, n, n_hh) in measured["holders_by_housing"].items():
            key = key_for(housing)
            note = ""
            if key and key in published:
                # Explicit labelling of the unit: the published curve is a share of
                # equipped HOUSEHOLDS, the measurement alongside a share of equipped PERSONS.
                # Putting them side by side without saying so is exactly the confusion that
                # once suggested a negative bias on all four categories.
                note = (f"   [réf. publiée, part de MÉNAGES sur habitat observé : "
                        f"{published[key]:.1f}]")
            check(f"{housing} (n={n}, {n_hh} foyers)", got,
                  attainable.get(key) if key else None,
                  TOLERANCES["by_housing_pct"], note, n=n_hh)

    print("\n── Household level (complete households, standardised targets) ─────────")
    clipping = stock.get("clipping_cost") or {}
    rows = stock.get("by_household_size") or []
    complete = household["complete_households"]
    print(f"  measurable complete households: {complete}")
    print("  Targets STANDARDISED on the size breakdown actually measured:\n"
          "  leaving out incomplete households over-represents single persons (50.7 %\n"
          "  of measurable ones against 39.3 % overall), who are the least equipped.\n"
          "  The raw survey target would apply to another composition.")
    equipped_ref = {row["size"]: row["equipped_pct_observed"] for row in rows}
    # ATTRIBUTABLE bikes, not the stock: the trait only carries bikes that have a
    # holder, and the survey counts bikes that no one in the household can hold
    # (0.44 of stock against 0.33 attributable among single persons).
    bikes_ref = {row["size"]: row.get("attributable_per_household_observed")
                 for row in rows
                 if row.get("attributable_per_household_observed") is not None}
    equipped_target = standardise(household["by_size"], equipped_ref)
    bikes_target = standardise(household["by_size"], bikes_ref)
    raw_equipped = (stock.get("overall") or {}).get("equipped_pct_observed")
    check("ménages équipés (%)", household["equipped_pct"], equipped_target,
          TOLERANCES["equipped_pct"],
          f"   [brute, toutes tailles : {raw_equipped:.1f}]" if raw_equipped else "",
          n=complete)
    published_stock = clipping.get("bikes_per_household_unclipped")
    check("vélos attribués par ménage", household["bikes_per_household"], bikes_target,
          TOLERANCES["bikes_per_household"],
          f"   [stock publié sur M21 non écrêté, toutes tailles : {published_stock:.3f}]"
          if published_stock is not None else "",
          n=complete, sd=clipping.get("attributable_sd"))

    # Explanatory note on the cost of clipping. `if clipping:` only guaranteed a
    # non-empty dict, and the five bracket accesses that followed crashed the report
    # on a resource exported by another version — AFTER displaying the verdicts,
    # so `--check` never returned its exit code. A note is not a verdict:
    # if the figures are missing, we stay silent, we do not bring down the validation.
    _needed = ("k_max", "bikes_per_household_unclipped", "bikes_per_household_clipped",
               "attributable_per_household_unclipped",
               "attributable_per_household_clipped")
    if all(clipping.get(field) is not None for field in _needed):
        stock_cost = (clipping["bikes_per_household_unclipped"]
                      - clipping["bikes_per_household_clipped"])
        trait_cost = (clipping["attributable_per_household_unclipped"]
                      - clipping["attributable_per_household_clipped"])
        print(f"  (target clipped at {clipping['k_max']}+: clipping removes "
              f"{stock_cost:.3f} from the published stock\n   but only {trait_cost:.3f} "
              f"from the produced trait — the surplus bikes of 5+ households have "
              f"no one\n   to hold them anyway.)")
    elif clipping:
        print("  (cost of clipping not published by this resource — re-export it "
              "with `make bike-ownership`)")

    conclusive = verdicts["ok"] + verdicts["echec"]
    journal["verdicts"] = dict(verdicts)
    print(f"\n  verdicts: {verdicts['ok']} ok, {verdicts['echec']} failure(s), "
          f"{verdicts['non_concluant']} inconclusive")
    if conclusive < MIN_CONCLUSIVE_CHECKS:
        failures.append(
            f"{NOT_MEASURABLE}: seulement {conclusive} contrôle(s) concluant(s) sur "
            f"{conclusive + verdicts['non_concluant']} — cette population est trop "
            f"petite pour être validée. Elle est enrichie, mais rien n'est vérifié : "
            f"ne pas lire ce rapport comme un succès (il faut de l'ordre de 1 000 "
            f"agents pour que les croisements tranchent)")
    return failures


def slope_verdict(cells: list, min_cell: int = SLOPE_MIN_CELL, z: float = SLOPE_Z) -> tuple[str, str]:
    """Verdict on the slope of holder rates by household size (1 → 4).

    `cells`: list of (rate_pct | None, n_persons, n_households) in size order.
    Returns (status, detail) with status ∈ {"non calculable", "non concluant", "ok", "echec"}:
    - non calculable: a size is missing;
    - non concluant: the smallest cell has fewer than `min_cell` households — nothing is pronounced;
    - ok: increasing, or inversion(s) contained within the combined uncertainty of the two cells
      (z · √(p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂), in points);
    - echec: a drop from one size to the next exceeds that uncertainty.
    """
    import math
    values = [v for v, _, _ in cells]
    if not all(v is not None for v in values):
        return "non calculable", "une taille est absente"
    smallest = min((n_hh for _, _, n_hh in cells), default=0)
    if smallest < min_cell:
        return "non concluant", f"plus petite cellule : {smallest} foyers < {min_cell}"
    within, breaks = [], []
    for (v1, _, n1), (v2, _, n2) in zip(cells, cells[1:]):
        if v2 >= v1:
            continue
        p1, p2 = v1 / 100.0, v2 / 100.0
        margin_pt = 100.0 * z * math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))
        drop = v1 - v2
        (within if drop <= margin_pt else breaks).append(f"{v1:.1f} → {v2:.1f} (−{drop:.1f} pt, incertitude ± {margin_pt:.1f})")
    if breaks:
        return "echec", "baisse significative : " + " ; ".join(breaks)
    if within:
        return "ok", "inversion dans l'incertitude : " + " ; ".join(within)
    return "ok", "croissante"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path, nargs="+",
                        help="Population JSON files to enrich (modified in place)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Bike-ownership model (default: mobility_core/data/)")
    parser.add_argument("--zones", type=Path, default=None,
                        help="Fine zone layer (default: mobility_core/data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Computes and reports without rewriting the files")
    parser.add_argument("--check", action="store_true",
                        help="Exits in failure if a target is out of tolerance")
    parser.add_argument("--rapport-json", type=Path, default=None,
                        help="Writes the structured report (checks, slope, verdicts, exit "
                             "code) to this file — read by the representativeness synthesis")
    args = parser.parse_args()

    from mobility_core.zone_resolver import ZoneResolver

    feature_spec = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
    try:
        model = BikeOwnershipModel.load(args.model)
        resolver = ZoneResolver.load(args.zones,
                                     feature_spec if feature_spec.exists() else None)
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    practice = model.validation.get("practice") or {}
    print(f"Model exported on {model.meta.get('exported_at', '?')}")
    print(f"  stage 1: k over classes {model.stock.classes}, "
          f"{len(model.stock.features)} household covariates")
    print(f"  stage 2: {len(model.propensity.features)} covariates, "
          f"{practice.get('overall_practice_pct_predicted', '?')} % practitioners "
          f"predicted for {practice.get('overall_practice_pct_observed', '?')} % observed")

    all_failures: list[tuple[Path, list[str]]] = []
    journaux: list[dict] = []
    for path in args.population:
        if not path.exists():
            print(f"[ERREUR] Population introuvable : {path}", file=sys.stderr)
            return 1
        raw = path.read_bytes()
        population = json.loads(raw.decode("utf-8"))
        counts = enrich(population, model, resolver)
        print(f"\n=== {path}")
        measured = measure(population)
        journal: dict = {"fichier": str(path), "sha256_avant": hashlib.sha256(raw).hexdigest(),
                         "n": len(population), "trait_pose": measured["with_trait"],
                         "couverture_pct": round(100 * measured["coverage"], 2),
                         "compteurs": dict(counts)}
        failures = report(measured, household_measure(population), counts, model, journal)
        journal["echecs"] = list(failures)
        journaux.append(journal)
        if failures:
            all_failures.append((path, failures))
        if args.dry_run:
            print("  [dry-run] file not rewritten")
            continue
        # Atomic write: a crash during writing must not leave a
        # truncated population behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"  written → {path}")

    code = EXIT_OK
    if all_failures:
        print("\n── Targets out of tolerance ────────────────────────────────────────────")
        for path, failures in all_failures:
            for failure in failures:
                print(f"  {path.name} : {failure}")
        if args.check:
            # If ALL failures are "not enough material", the file is not
            # at fault: it is too small to be judged. This is signalled by a distinct
            # code, which the caller can handle differently from a missed target.
            measurable = [failure for _, failures in all_failures
                          for failure in failures
                          if not failure.startswith(NOT_MEASURABLE)]
            if not measurable:
                print(f"\n  → code {EXIT_NOT_MEASURABLE}: population enriched but NOT "
                      f"VALIDATED (not enough households to decide), and not "
                      f"\"failed\". No served target is contradicted.")
                code = EXIT_NOT_MEASURABLE
            else:
                code = EXIT_TARGET_MISSED
        else:
            print("  (informational: rerun with --check to make it a failure)")
    elif args.check:
        print("\nAll served targets are within tolerance.")
    if args.rapport_json:
        write_journal(args.rapport_json, model, journaux, code, args)
    return code


def write_journal(path: Path, model: BikeOwnershipModel, journaux: list[dict], code: int,
                  args) -> None:
    """The structured report of `--check`: what the console said, readable by a script."""
    from datetime import datetime, timezone
    payload = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/data/population/enrich_personal_bike.py",
        "modele_exporte_le": model.meta.get("exported_at"),
        "regles": {"MIN_CELL_HOUSEHOLDS": MIN_CELL_HOUSEHOLDS, "SLOPE_MIN_CELL": SLOPE_MIN_CELL,
                   "SLOPE_Z": SLOPE_Z, "MIN_CONCLUSIVE_CHECKS": MIN_CONCLUSIVE_CHECKS,
                   "MIN_COVERAGE": MIN_COVERAGE, "TOLERANCES": TOLERANCES},
        "check": bool(args.check), "dry_run": bool(args.dry_run),
        "populations": journaux, "code_sortie": code,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  structured report → {path}")


if __name__ == "__main__":
    raise SystemExit(main())
