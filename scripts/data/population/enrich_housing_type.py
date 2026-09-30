"""enrich_housing_type.py — The persona's "housing type" trait (action A2).

Last enrichment step of the synthetic population, in the same vein as
step 3bis of the notebook (urban/peri-urban/rural zone from the INSEE tables):
it rereads a population file, adds a field to `traits_json`, and rewrites it.

What it adds: `traits_json["housing_type"]`, the dwelling type in the EMC² sense
("Individuel isolé", "Individuel accolé", "Petit habitat collectif", "Grand
habitat collectif", "Autres"). Without it, the "Type de logement" column of the trip
log is written empty and the corresponding axis of the summary page stays
at zero.

**The trait is imputed, not observed.** No source of the generation chain
carries it: neither eqasim nor the INSEE tables used by the notebook. It is therefore drawn
from the law the EMC² survey observes **for the home's fine zone, corrected for
household size** (see `packages/mobility_core/src/mobility_core/housing_type.py` for the details, the size
lever of ticket 019 and the safeguards). Three consequences never to keep quiet:

- the page's breakdown by housing type measures an **imputed** axis, whose
  marginal law comes from the survey that also serves as the target: it says whether the
  simulation chooses the same modes *for a given housing type*, not whether it places
  people correctly in dwellings;
- a home outside the fine-zone layer has no type: the trait is absent,
  and it must stay so. The log column is then empty, which is not a
  category. A persona **without `household_size`** is in the same case since ticket
  019: imputing on the zone alone would let back in through the window the flattened gradient that
  the size lever corrects;
- the size used is the **nominal** size declared by the persona, never the
  number of members present in the file. 118 of the 498 address clusters of
  `toulouse_population_1000.json` are partial (bbox filtering): counting those
  present would put families of four into single-person laws.

Two restricted-access resources are required (`make zones`, `make housing-type`).
Their absence is a normal case: the command then fails with the message saying
which one is missing and how to produce it, without ever imputing blindly.

`--check` compares the result with the targets of ticket 019 (share of detached houses by
household size, **sign** of the slope, trait coverage) and exits with a failure if one
of them is out of tolerance. A massive `None` fails there explicitly: a population
whose trait is missing everywhere does not "pass" the acceptance test, it invalidates it.

Usage:
    python -m scripts.data.population.enrich_housing_type data/population/toulouse_population_1000.json
    python -m scripts.data.population.enrich_housing_type data/population/*.json --dry-run
    python -m scripts.data.population.enrich_housing_type data/population/toulouse_population_1000.json --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from mobility_core.housing_type import (
    MODALITY_KEYS,
    SIZE_MAX,
    SIZE_TRAIT_KEY,
    TRAIT_KEY,
    HousingTypeTable,
    address_key,
    key_for,
    size_bucket,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Minimum coverage of the trait. Below it, the population is not usable and
# no target makes sense — this is the criterion "a massive None must make the
# validation fail, not pass" of ticket 019.
MIN_COVERAGE = 0.80

# Tolerance on the detached-house share of a size class, in points. The ticket
# asks for ± 4 pts; the cell's sampling noise is added to it (2 σ).
TOLERANCE_PT = 4.0

# Below this number of addresses, a cell does not decide. Counting it as "met" would be
# the "vacuity ≠ perfection" pattern: the absence of measurement would yield the perfect score.
MIN_CELL_ADDRESSES = 20

# Exit codes, the same as `enrich_personal_bike` — the notebook and the Makefile
# distinguish them: a population of 10 agents is not at fault, it is too small to
# be judged, and that must not stop an enrichment chain.
EXIT_OK = 0
EXIT_RESOURCE_MISSING = 1
EXIT_TARGET_MISSED = 2
EXIT_NOT_MEASURABLE = 3

# Prefix of the failures that only question the file's size, not the imputation.
NOT_MEASURABLE = "population non mesurable"


def enrich(population: list[dict], table: HousingTypeTable, resolver) -> Counter:
    """Sets `housing_type` on each persona. Returns the count per category.

    Homes are matched to zones in a single vectorised call: it is one
    spatial-index query per batch, not one per persona.
    """
    homes = [(person.get("identity") or {}).get("home") or {} for person in population]
    lats = [home.get("lat") for home in homes]
    lons = [home.get("lon") for home in homes]

    resolvable = [i for i, (la, lo) in enumerate(zip(lats, lons))
                  if la is not None and lo is not None]
    zones = resolver.resolve_many([lats[i] for i in resolvable],
                                  [lons[i] for i in resolvable])
    zone_by_index: dict[int, Optional[object]] = dict(zip(resolvable, zones))

    counts: Counter = Counter()
    for i, person in enumerate(population):
        traits = (person.get("identity") or {}).get("traits_json")
        if traits is None:
            counts["sans_traits"] += 1
            continue
        zone = zone_by_index.get(i)
        if zone is None:
            # Outside the layer, or home without coordinates: nothing is invented. The trait
            # is removed if it lingered from an earlier enrichment.
            traits.pop(TRAIT_KEY, None)
            counts["hors_couche"] += 1
            continue
        # NOMINAL household size, as the persona declares it: the file does not
        # always contain all its members (see the docstring).
        size = size_bucket(traits.get(SIZE_TRAIT_KEY))
        if size is None:
            traits.pop(TRAIT_KEY, None)
            counts["sans_taille"] += 1
            continue
        label = table.housing_type(zone.zf, lats[i], lons[i],
                                   traits.get(SIZE_TRAIT_KEY))
        if label is None:
            traits.pop(TRAIT_KEY, None)
            counts["sans_loi"] += 1
            continue
        traits[TRAIT_KEY] = label
        counts[key_for(label) or "inconnu"] += 1
        # Fallback level served: published at each enrichment, as ticket 019
        # requires. A population mostly served by the scope would have
        # no geographic conditioning at all any more.
        counts[f"repli_{table.level_for(zone.zf)}"] += 1
    return counts


def measure_by_size(population: list[dict]) -> dict[int, dict]:
    """Share of detached houses by household size class, and its effective sample size.

    The sample size that matters for sampling noise is the number of distinct
    **addresses**, not of personas: the draw is made on the home, and the six members
    of a household share a single draw.
    """
    rows: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "isolated": 0, "addresses": set()})
    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json") or {}
        label = traits.get(TRAIT_KEY)
        size = size_bucket(traits.get(SIZE_TRAIT_KEY))
        if label is None or size is None:
            continue
        home = identity.get("home") or {}
        row = rows[size]
        row["n"] += 1
        # By the KEY, never by the label: the label changed language with v6
        # (ticket 074), and a literal comparison would have counted zero detached houses
        # over the whole population — a slope of +0.0 pt where the survey measures +38.2.
        row["isolated"] += int(key_for(label) == "individuel_isole")
        if home.get("lat") is not None and home.get("lon") is not None:
            row["addresses"].add((address_key(home["lat"], home["lon"]), size))
    return {size: {"n": row["n"],
                   "isolated_pct": 100.0 * row["isolated"] / row["n"],
                   "n_addresses": len(row["addresses"])}
            for size, row in sorted(rows.items()) if row["n"]}


def report(counts: Counter, table: HousingTypeTable, population: list[dict]) -> list[str]:
    """Distribution obtained, size gradient, fallbacks. Returns the list of failures.

    The gap to the survey's law over the whole scope is not a defect in itself: the
    simulated population does not occupy the survey scope uniformly. It must
    nevertheless be read, because a massive gap would signal a zone matching that
    went wrong. The **size gradient**, on the other hand, is a target: it is the subject of ticket
    019, and its sign is a criterion in its own right.
    """
    n = len(population)
    attributed = sum(counts.get(key, 0) for key in MODALITY_KEYS)
    failures: list[str] = []

    print(f"\nHousing types imputed: {attributed}/{n} personas "
          f"({100.0 * attributed / n if n else 0:.1f} %)")
    for key in ("hors_couche", "sans_taille", "sans_loi", "sans_traits"):
        if counts.get(key):
            print(f"  {key:26s} {counts[key]:5d} (trait absent, empty column)")
    levels = [(level, counts.get(f"repli_{level}", 0))
              for level in ("zone", "secteur", "perimetre")]
    if any(value for _, value in levels):
        print("  law level served: " + ", ".join(
            f"{level} {value}" for level, value in levels))
        if attributed and counts.get("repli_zone", 0) / attributed < 0.5:
            print("  [avertissement] fewer than half of the personas receive the law "
                  "of THEIR zone: the geographic conditioning has largely fallen back")

    if not n or attributed / n < MIN_COVERAGE:
        failures.append(
            f"couverture {100.0 * attributed / n if n else 0:.1f} % < "
            f"{100 * MIN_COVERAGE:.0f} % — trait absent sur trop d'agents : la "
            f"population n'est pas exploitable, et aucune cible ci-dessous n'a de sens")

    if not attributed:
        return failures

    print(f"\n  {'modalité':26s} {'simulée':>9s} {'EMC² (périmètre)':>18s} {'écart':>8s}")
    for key, share in zip(MODALITY_KEYS, table.global_shares):
        got = 100.0 * counts.get(key, 0) / attributed
        target = 100.0 * share
        print(f"  {key:26s} {got:8.2f}% {target:17.2f}% {got - target:+7.2f}")
    l1 = sum(abs(100.0 * counts.get(k, 0) / attributed - 100.0 * s)
             for k, s in zip(MODALITY_KEYS, table.global_shares))
    print(f"  {'écart L1 cumulé':26s} {l1:8.2f} points")
    print("  (the population does not occupy the survey scope uniformly: this "
          "gap\n   is read, not validated — see the size gradient below)")

    return failures + check_size_gradient(measure_by_size(population), table)


def check_size_gradient(measured: dict[int, dict],
                        table: HousingTypeTable) -> list[str]:
    """The acceptance criterion of ticket 019: the size gradient, values AND sign."""
    targets = table.observed_isolated_share_by_size()
    failures: list[str] = []
    print("\n── Share of detached houses by household size (ticket 019) ─────────────")
    if not targets:
        print("  [targets not served: re-export the table (make housing-type) so "
              "that its validation block is present]")
        return ["la table ne porte pas les parts observées par taille : le critère de "
                "recette du ticket 019 ne peut pas être évalué"]

    print(f"  {'taille':>7s} {'simulée':>9s} {'EMC²':>8s} {'écart':>7s} "
          f"{'marge':>7s} {'adresses':>9s}")
    conclusive = 0
    for size, row in measured.items():
        target = targets.get(size)
        if target is None:
            print(f"  {size:>7d} {row['isolated_pct']:8.1f}%   (no target)")
            continue
        addresses = row["n_addresses"]
        # Margin = ticket tolerance + 2 σ of the cell, binomial σ over the ADDRESSES:
        # at 42 addresses, the standard deviation of a proportion around 55 % is already 7.7 pts.
        p = target / 100.0
        sigma = 100.0 * math.sqrt(p * (1 - p) / addresses) if addresses else float("inf")
        margin = TOLERANCE_PT + 2 * sigma
        gap = row["isolated_pct"] - target
        if addresses < MIN_CELL_ADDRESSES:
            verdict = f"NON CONCLUANT ({addresses} adresses)"
        elif abs(gap) <= margin:
            verdict, conclusive = "ok", conclusive + 1
        else:
            verdict, conclusive = "ÉCHEC", conclusive + 1
            failures.append(
                f"taille {size} : {row['isolated_pct']:.1f} % d'individuel isolé pour "
                f"{target:.1f} % observés (écart {gap:+.1f} pt, marge ±{margin:.1f})")
        print(f"  {size:>7d} {row['isolated_pct']:8.1f}% {target:7.1f}% {gap:+7.1f} "
              f"{margin:7.1f} {addresses:9d}   {verdict}")

    # The sign of the slope is a criterion in its own right: it is what was wrong
    # before ticket 019. It is judged only if BOTH targets are served and they
    # describe a non-zero slope — a default `targets.get(size, 0)` would fabricate
    # an expected slope from a missing target, and would judge against nothing.
    low, high = measured.get(1), measured.get(SIZE_MAX)
    expected = (None if targets.get(1) is None or targets.get(SIZE_MAX) is None
                else targets[SIZE_MAX] - targets[1])
    if low is not None and high is not None and expected:
        slope = high["isolated_pct"] - low["isolated_pct"]
        ok = slope > 0 if expected > 0 else slope < 0
        print(f"  slope single person → household of {SIZE_MAX}+: {slope:+.1f} pt "
              f"(observed {expected:+.1f} pt)   {'ok' if ok else 'ÉCHEC'}")
        if not ok:
            failures.append(
                f"la pente entre la personne seule et le ménage de {SIZE_MAX}+ vaut "
                f"{slope:+.1f} pt pour {expected:+.1f} pt observés — c'est le défaut "
                f"même que le ticket 019 corrige, et son signe est un critère à part")
        conclusive += 1
    else:
        print(f"  pente personne seule → ménage de {SIZE_MAX}+ : non jugée (une des deux "
              f"classes de taille est absente, ou la pente observée est nulle)")
    if not conclusive:
        failures.append(
            f"{NOT_MEASURABLE} : aucun contrôle n'a tranché sur le gradient de taille. "
            f"Elle est enrichie, mais rien n'est vérifié — ne pas lire ce rapport comme "
            f"un succès")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path, nargs="+",
                        help="JSON population files to enrich (modified in place)")
    parser.add_argument("--table", type=Path, default=None,
                        help="Housing type table (default: mobility_core/data/)")
    parser.add_argument("--zones", type=Path, default=None,
                        help="Fine-zone layer (default: mobility_core/data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Computes and reports without rewriting the files")
    parser.add_argument("--check", action="store_true",
                        help="Exits with a failure if a target of ticket 019 is out of tolerance")
    args = parser.parse_args()

    from mobility_core.zone_resolver import ZoneResolver

    feature_spec = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
    try:
        table = HousingTypeTable.load(args.table)
        resolver = ZoneResolver.load(args.zones,
                                     feature_spec if feature_spec.exists() else None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return EXIT_RESOURCE_MISSING

    print(f"Table exported on {table.meta.get('exported_at', '?')} — "
          f"{table.meta.get('conditioning', 'conditionnement non documenté')}")

    all_failures: list[tuple[Path, list[str]]] = []
    for path in args.population:
        if not path.exists():
            print(f"[ERREUR] Population introuvable : {path}", file=sys.stderr)
            return EXIT_RESOURCE_MISSING
        population = json.loads(path.read_text(encoding="utf-8"))
        counts = enrich(population, table, resolver)
        print(f"\n=== {path}")
        failures = report(counts, table, population)
        if failures:
            all_failures.append((path, failures))
        if args.dry_run:
            print("  [dry-run] file not rewritten")
            continue
        # Atomic write: a crash in the middle of writing must not leave a
        # truncated population behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"  written → {path}")

    if all_failures:
        print("\n── Targets out of tolerance ────────────────────────────────────────────")
        for path, failures in all_failures:
            for failure in failures:
                print(f"  [{path.name}] {failure}")
        if args.check:
            # If ALL failures are "not enough material", the file is not
            # at fault: it is too small to be judged. A distinct code, which the caller
            # can handle differently from a contradicted target.
            missed = [failure for _, failures in all_failures for failure in failures
                      if not failure.startswith(NOT_MEASURABLE)]
            if not missed:
                print(f"\n  → code {EXIT_NOT_MEASURABLE} : population enrichie mais NON "
                      f"VALIDÉE (pas assez d'adresses par classe de taille pour "
                      f"trancher), et non « en échec ». Aucune cible servie n'est "
                      f"démentie.")
                return EXIT_NOT_MEASURABLE
            return EXIT_TARGET_MISSED
        print("  (informatif : relancez avec --check pour en faire un échec)")
    elif args.check:
        print("\nAll served targets are within tolerance.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
